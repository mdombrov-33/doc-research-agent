from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


def index_chunks(
    vector_store: QdrantVectorStore,
    document_id: str,
    filename: str,
    enriched_chunks: list[dict],
) -> None:
    documents = [
        Document(
            page_content=chunk_data["text"],
            metadata={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk_data["chunk_index"],
                "chunk_length": chunk_data["chunk_length"],
                "entities": chunk_data["entities"],
                "entity_types": chunk_data["entity_types"],
                "keywords": chunk_data["keywords"],
                "file_extension": chunk_data["file_extension"],
            },
        )
        for chunk_data in enriched_chunks
    ]

    vector_store.add_documents(documents)
