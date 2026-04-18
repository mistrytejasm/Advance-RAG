"""Metadata builder for document chunks."""
import uuid
from datetime import datetime, timezone

class MetadataBuilder:

    def build_metadata(
        self,
        document_id,
        page,
        section
    ):
        return {
            "chunk_id": str(
                uuid.uuid4()
            ),
            "document_id": document_id,
            "page": page,
            "section": section,
            "created_at": datetime.now(timezone.utc)
        }