import os
import gc
import tempfile
import torch
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from app.utils.logger import logger
from app.services.ocr_service import extract_text_from_image

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


def _extract_chunks_from_result(result, document_id, page_offset):
    """Extract text, table, and image chunks from a Docling conversion result.
    
    Args:
        result: Docling ConversionResult object
        document_id: UUID string for this document
        page_offset: Page number offset to map chunk pages back to the original PDF
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for item, level in result.document.iterate_items():
        item_type = type(item).__name__.lower()

        # Map page number back to original document
        page_num = page_offset + 1  # default fallback
        if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
            chunk_page = getattr(item.prov[0], "page_no", 1)
            page_num = page_offset + chunk_page

        # ── Text ──
        if "textitem" in item_type:
            text = getattr(item, "text", "")
            if text.strip():
                chunks.append({
                    "document_id": document_id,
                    "page": page_num,
                    "content": text,
                    "content_type": "text",
                    "metadata": {}
                })

        # ── Table ──
        elif "tableitem" in item_type:
            table_text = ""
            if hasattr(item, "export_to_markdown"):
                table_text = item.export_to_markdown(result.document)
            else:
                table_text = str(getattr(item, "data", ""))

            if table_text.strip():
                chunks.append({
                    "document_id": document_id,
                    "page": page_num,
                    "content": table_text,
                    "content_type": "table"
                })

        # ── Image / Picture ──
        elif "pictureitem" in item_type or "image" in item_type:
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
                        chunks.append({
                            "document_id": document_id,
                            "page": page_num,
                            "content": ocr_text,
                            "content_type": "image"
                        })
                else:
                    logger.warning(f"Image block on page {page_num}: PIL image not extractable.")
            except Exception as e:
                logger.error(f"Failed to OCR image on page {page_num}: {e}")

    return chunks


def parse_document(file_path, document_id):
    """Parse a PDF document in memory-safe page-range chunks.
    
    Splits the PDF into mini-PDFs of CHUNK_SIZE pages, converts each
    independently with Docling, and merges all extracted chunks.
    """
    logger.info("Parsing Document using Docling...")

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
            batch_chunks = _extract_chunks_from_result(result, document_id, page_offset=start)
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
