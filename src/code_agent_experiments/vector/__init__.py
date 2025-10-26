"""Vector backends for dense retrieval."""

from .faiss_backend import FaissVectorStore, VectorRecord, VectorSearchResult

__all__ = ["FaissVectorStore", "VectorRecord", "VectorSearchResult"]
