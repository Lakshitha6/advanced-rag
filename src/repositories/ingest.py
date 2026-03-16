"""
Document ingestion script.
Run this to load documents into the vector store.

Usage:
    python -m src.repositories.ingest
"""

import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.repositories.vector_store import VectorStoreRepository
from src.repositories.document_loader import DocumentLoader
from src.utils.logger import setup_logger

logger = setup_logger("ingest")

def save_documents_for_retrieval(
    documents: List[Document], 
    output_path: str = "data/documents_cache.pkl"
):
    """
    Save processed documents for retrieval service to use.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(documents, f)
    
    logger.info(f"Saved {len(documents)} documents to {output_path}")

def load_documents_for_retrieval(
    input_path: str = "data/documents_cache.pkl"
) -> List[Document]:
    """
    Load cached documents for retrieval service.
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Document cache not found: {input_path}")
    
    with open(path, 'rb') as f:
        documents = pickle.load(f)
    
    logger.info(f"Loaded {len(documents)} documents from {input_path}")
    return documents

def main():
    """Main ingestion pipeline."""
    logger.info("=" * 50)
    logger.info("Starting Document Ingestion Pipeline")
    logger.info("=" * 50)

    # 1. Load Settings
    settings = Settings.load()

    # 2. Initialize embeddings service
    embeddings_service = EmbeddingsService(settings=settings)
    embeddings = embeddings_service.get_embeddings()

    # 3. Initialize vector store (receives embeddings from vector store)
    vector_store = VectorStoreRepository(settings=settings, embeddings=embeddings)

    # Optional: Clear existing collection for fresh ingestion
    # Uncomment if want to reset on each run
    # vector_store.clear_collection()
    
    # 4. Load and split documents
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    
    documents_path = "data/documents"
    logger.info(f"Loading documents from: {documents_path}")
    
    chunks = loader.load_and_split(
        source=documents_path,
        is_directory=True,
        glob_pattern="*.pdf"
    )
    
    if not chunks:
        logger.warning("No documents found to ingest!")
        return
    
    # 5. Ingest into vector store
    count = vector_store.ingest_documents(chunks)
    
    # 6. Verify ingestion
    total_docs = vector_store.get_document_count()
    logger.info(f"Total documents in store: {total_docs}")
    
    # 7. Test retrieval
    logger.info("=" * 50)
    logger.info("Testing Retrieval")
    logger.info("=" * 50)
    
    test_queries = [
        "What is a llm",
        "what is prompt",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        docs = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(docs, 1):
            logger.info(f"  Result {i}: {doc.page_content[:100]}...")
    
    logger.info("=" * 50)
    logger.info("Ingestion Complete!")
    logger.info("=" * 50)

    save_documents_for_retrieval(chunks)

if __name__ == "__main__":
    main()