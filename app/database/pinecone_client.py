"""Pinecone Vector Database Client (Deprecated Shim).
Use app.services.vector_store.pinecone_client instead.
"""

from app.services.vector_store.pinecone_client import pinecone_store, PineconeVectorStore

# Aliases for backwards compatibility
pinecone_client = pinecone_store
PineconeClient = PineconeVectorStore

