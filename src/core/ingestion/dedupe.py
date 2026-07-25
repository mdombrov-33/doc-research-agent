from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue


def find_duplicate(vector_store: QdrantVectorStore, file_sha256: str) -> str | None:
    """Return the document_id of an already-indexed file with these exact bytes, or None."""
    points, _ = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="metadata.file_sha256", match=MatchValue(value=file_sha256))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return (points[0].payload or {}).get("metadata", {}).get("document_id")
