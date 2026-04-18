import os
import gc
import uuid
import tempfile
from datetime import datetime, timezone
import torch
import re
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TextItem, TableItem, PictureItem
from app.utils.logger import logger
from app.services.ocr_service import extract_text_from_image


def clean_pua_chars(text: str) -> str:
    """Replace common Private Use Area (PUA) Unicode characters mapping to PDF custom fonts."""
    replacements = {
        "\ue081": "(", "\ue082": ")", "\ue083": " ✅", "\ue084": "",
        "\ue085": "'", "\ue086": '"', "\ue087": '"', "\ue088": "-",
        "\ue089": "•", "\ue08a": "→", "\ue08b": "—", "\ue08c": "…",
        "\ue08d": "'", "\ue08e": "'", "\ue08f": "–", "\ue090": "×",
        "\ue091": "÷", "\ue092": ":",
    }
    for pua, replacement in replacements.items():
        text = text.replace(pua, replacement)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)
    return text


# ──────────────────────────────────────────────────────────────────────
# Hardware Strategy:
#   - GTX 1650 (4GB VRAM) cannot hold the RT-DETR layout model (~3.5GB)
#     plus page image tensors. CUDA is NOT viable for Docling on this card.
#   - CPU mode works, but the entire 43-page PDF loaded at once exceeds
#     system RAM during the "preprocess" rasterization stage.
#   - Solution: Force CPU + process the PDF in small page-range chunks
#     (default 5 pages). Each chunk is a temporary mini-PDF. After each
#     chunk, we garbage-collect so memory never accumulates.
# ──────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 5  # Pages per processing batch — tune lower if RAM is still tight

device = AcceleratorDevice.CPU
logger.info(f"Docling using accelerator: {device}")

accel_options = AcceleratorOptions(num_threads=1, device=device)

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 1.0      # 1x native resolution (default 2x eats 4× more RAM)
pipeline_options.accelerator_options = accel_options

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def _make_chunk(document_id, page_num, content, content_type, source_filename, element_type=None):
    """Create a standardized chunk dictionary with full production metadata.

    Every chunk gets:
      - chunk_id:   unique UUID for vector DB indexing
      - source:     original uploaded filename for traceability
      - timestamp:  ISO 8601 processing time
      - element_type: Docling's structural label (SectionHeaderItem, ListItem, etc.)
    """
    return {
        "document_id": document_id,
        "chunk_id": str(uuid.uuid4()),
        "page": page_num,
        "content": content,
        "content_type": content_type,
        "metadata": {
            "element_type": element_type or content_type,
            "source": source_filename,
            "page_number": page_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }


def _extract_chunks_from_result(result, document_id, page_offset, source_filename):
    """Extract text, table, and image chunks from a Docling conversion result.

    Uses isinstance() checks against Docling's actual class hierarchy so that
    ALL subclasses of TextItem are captured (SectionHeaderItem, ListItem,
    TitleItem, CodeItem, FormulaItem, etc.) — not just TextItem itself.

    Args:
        result: Docling ConversionResult object
        document_id: UUID string for this document
        page_offset: Page number offset to map chunk pages back to the original PDF
        source_filename: Original uploaded filename for metadata

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for item, level in result.document.iterate_items():
        # Map page number back to original document
        page_num = page_offset + 1  # default fallback
        if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
            chunk_page = getattr(item.prov[0], "page_no", 1)
            page_num = page_offset + chunk_page

        # ── Table (check FIRST — TableItem may also have .text) ──
        if isinstance(item, TableItem):
            table_text = ""
            if hasattr(item, "export_to_markdown"):
                table_text = item.export_to_markdown(result.document)
            else:
                table_text = str(getattr(item, "data", ""))

            if table_text and table_text.strip():
                table_text = clean_pua_chars(table_text)
                chunks.append(_make_chunk(
                    document_id, page_num, table_text,
                    "table", source_filename, "TableItem"
                ))

        # ── Image / Picture ──
        elif isinstance(item, PictureItem):
            logger.info(f"Found image block on page {page_num}. Extracting via OCR...")
            try:
                pil_img = None
                if hasattr(item, "get_image"):
                    pil_img = item.get_image(result.document)
                elif hasattr(item, "image") and hasattr(item.image, "pil_image"):
                    pil_img = item.image.pil_image

                if pil_img:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        pil_img.save(tmp, format="PNG")
                        tmp_path = tmp.name

                    ocr_text = extract_text_from_image(tmp_path)
                    os.remove(tmp_path)

                    if ocr_text and ocr_text.strip():
                        chunks.append(_make_chunk(
                            document_id, page_num, ocr_text,
                            "image", source_filename, "PictureItem"
                        ))
                else:
                    logger.warning(f"Image block on page {page_num}: PIL image not extractable.")
            except Exception as e:
                logger.error(f"Failed to OCR image on page {page_num}: {e}")

        # ── Text (catches ALL subclasses: TitleItem, SectionHeaderItem, 
        #    ListItem, CodeItem, FormulaItem, FieldHeadingItem, FieldValueItem) ──
        elif isinstance(item, TextItem):
            text = clean_pua_chars(getattr(item, "text", ""))
            item_label = type(item).__name__
            
            if text and text.strip():
                chunks.append(_make_chunk(
                    document_id, page_num, text,
                    "text", source_filename, item_label
                ))

        # ── Fallback: any unknown item type that has a .text attribute ──
        else:
            text = clean_pua_chars(getattr(item, "text", ""))
            item_label = type(item).__name__
            
            if text and text.strip():
                logger.info(f"Captured unknown item type '{item_label}' on page {page_num}")
                chunks.append(_make_chunk(
                    document_id, page_num, text,
                    "text", source_filename, item_label
                ))

    return chunks


def parse_document(file_path, document_id):
    """Parse a PDF document in memory-safe page-range chunks.

    Splits the PDF into mini-PDFs of CHUNK_SIZE pages, converts each
    independently with Docling, and merges all extracted chunks.
    """
    logger.info("Parsing Document using Docling...")

    # Extract original filename from the saved path (format: uuid_filename.pdf)
    base_name = os.path.basename(file_path)
    # Strip the leading document_id + underscore to recover original filename
    source_filename = "_".join(base_name.split("_")[1:]) if "_" in base_name else base_name

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    logger.info(f"Document has {total_pages} pages. Processing in chunks of {CHUNK_SIZE}...")

    all_chunks = []

    for start in range(0, total_pages, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_pages)
        logger.info(f"Processing pages {start + 1}-{end} of {total_pages}...")

        # Build a temporary mini-PDF containing only this page range
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        tmp_pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                writer.write(tmp_pdf)
                tmp_pdf_path = tmp_pdf.name

            result = converter.convert(tmp_pdf_path)
            batch_chunks = _extract_chunks_from_result(
                result, document_id, page_offset=start, source_filename=source_filename
            )
            all_chunks.extend(batch_chunks)

            logger.info(
                f"Pages {start + 1}-{end} complete — "
                f"{len(batch_chunks)} chunks extracted "
                f"({len(all_chunks)} total so far)"
            )

        except Exception as e:
            logger.error(f"Failed to process pages {start + 1}-{end}: {e}")

        finally:
            # Clean up temp file + force garbage collection between chunks
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Document processing complete. Total chunks extracted: {len(all_chunks)}")
    return all_chunks
