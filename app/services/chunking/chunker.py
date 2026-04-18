"""Core chunking engine."""
from app.services.chunking.tokenizer import SimpleTokenizer
from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
import uuid
from datetime import datetime, timezone

class Chunker:

    def __init__(self):
        self.tokenizer = SimpleTokenizer()

    def chunk_document_elements(self, document_id: str, elements: list, source: str = "pdf") -> list:
        """
        Process the raw Docling JSON elements into semantic chunks.
        Applies Hybrid Structure-First Chunking (Group Headers, Fast-Pass Tables).
        """
        final_chunks = []
        current_section = "Unknown"
        
        # Buffer for semantic grouping of text/lists
        text_buffer = ""
        current_tokens = 0
        current_pages = set()
        
        def _flush_buffer():
            nonlocal text_buffer, current_tokens, current_pages
            if not text_buffer.strip():
                return
            
            # If the buffer exceeds CHUNK_SIZE, we split it using token overlap strategy
            text_tokens = self.tokenizer.encode(text_buffer)
            start = 0
            while start < len(text_tokens):
                end = start + CHUNK_SIZE
                chunk_tokens = text_tokens[start:end]
                
                final_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": self.tokenizer.decode(chunk_tokens).strip(),
                    "content_type": "text",
                    "page": list(current_pages)[0] if current_pages else 1,
                    "section": current_section,
                    "token_count": len(chunk_tokens),
                    "metadata": {
                        "element_type": "SemanticGroup",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "pages": list(current_pages)
                    }
                })
                if end >= len(text_tokens):
                    break
                # Advance with overlap
                start = end - CHUNK_OVERLAP
                
            # Reset buffer
            text_buffer = ""
            current_tokens = 0
            current_pages = set()

        for el in elements:
            content_type = el.get("content_type")
            content = el.get("content", "").strip()
            page = el.get("page", 1)
            el_type = el.get("metadata", {}).get("element_type", "TextItem")
            
            if not content:
                continue

            # Step 1: Update Section Header tracking
            if el_type == "SectionHeaderItem":
                _flush_buffer() # Flush existing buffer before new section
                current_section = content
                # Add header to the buffer so following text is grouped WITH it
                text_buffer += f"Section: {content}\n"
                current_tokens += self.tokenizer.count_tokens(text_buffer)
                current_pages.add(page)
                continue
                
            # Step 2: Table Preservation (Fast-Pass)
            if content_type == "table":
                _flush_buffer() # flush before table
                table_tokens = self.tokenizer.count_tokens(content)
                final_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": content,
                    "content_type": "table",
                    "page": page,
                    "section": current_section,
                    "token_count": table_tokens,
                    "metadata": {
                        "element_type": "TableItem",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                })
                continue
                
            # Step 3: Image OCR Handling (Multimodal)
            if content_type == "image":
                _flush_buffer()
                img_tokens = self.tokenizer.count_tokens(content)
                final_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": content,
                    "content_type": "image",
                    "page": page,
                    "section": current_section,
                    "token_count": img_tokens,
                    "metadata": {
                        "element_type": "PictureItem",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                })
                continue
                
            # Step 4: Semantic Grouping for Text/List Items
            if el_type == "ListItem":
                append_text = f"\n- {content}"
            else:
                append_text = f"\n{content}"
                
            item_tokens = self.tokenizer.count_tokens(append_text)
            
            if current_tokens > 0 and current_tokens + item_tokens > CHUNK_SIZE:
                _flush_buffer()
                # Re-add header context to the new buffer if we split mid-section
                if current_section != "Unknown":
                    text_buffer += f"Section: {current_section}\n"
                    current_tokens += self.tokenizer.count_tokens(text_buffer)
            
            text_buffer += append_text
            current_tokens += item_tokens
            current_pages.add(page)
            
        # Flush any remaining buffer at the end of document
        _flush_buffer()
        
        total_chunks = len(final_chunks)
        for idx, chunk in enumerate(final_chunks):
            chunk["chunk_index"] = idx + 1
            chunk["source"] = source
            chunk["metadata"]["total_chunks"] = total_chunks
            
        return final_chunks