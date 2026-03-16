"""
Test script for retrieval service.
Run after ingestion to verify retrieval quality.

Usage:
    python -m src.services.test_retrieval
"""
from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.services.llm_service import LLMService
from src.services.retrieval_service import RetrievalService
from src.repositories.vector_store import VectorStoreRepository
from src.repositories.ingest import load_documents_for_retrieval
from src.utils.logger import setup_logger

logger = setup_logger("test_retrieval")


def main():
    """Test retrieval strategies."""
    logger.info("=" * 60)
    logger.info("Testing Retrieval Strategies")
    logger.info("=" * 60)
    
    # 1. Load settings
    settings = Settings.load()
    
    # 2. Initialize services
    embeddings_service = EmbeddingsService(settings=settings)
    llm_service = LLMService(settings=settings)
    
    vector_store = VectorStoreRepository(
        settings=settings,
        embeddings=embeddings_service.get_embeddings()
    )
    
    retrieval_service = RetrievalService(
        settings=settings,
        embeddings_service=embeddings_service,
        vector_store=vector_store,
        llm=llm_service.get_llm()
    )
    
    # 3. Load cached documents
    documents = load_documents_for_retrieval()
    
    # 4. Test queries
    test_queries = [
        "What is a prompt",
        "How many prompt strategies ?",
        "What is  few shot prompt",
        "What is chain of thought?",
    ]
    
    for query in test_queries:
        logger.info("\n" + "=" * 60)
        logger.info(f"Query: {query}")
        logger.info("=" * 60)
        
        # Test Vector Only
        logger.info("\n[Vector Search Only]")
        docs = retrieval_service.retrieve(query, documents, use_hybrid=False, use_rerank=False)
        for i, doc in enumerate(docs[:2], 1):
            logger.info(f"  {i}. {doc.page_content[:100]}...")
        
        # Test Hybrid
        logger.info("\n[Hybrid Search]")
        docs = retrieval_service.retrieve(query, documents, use_hybrid=True, use_rerank=False)
        for i, doc in enumerate(docs[:2], 1):
            logger.info(f"  {i}. {doc.page_content[:100]}...")
        
        # Test Hybrid + Rerank
        logger.info("\n[Hybrid + Cross-Encoder Rerank]")
        docs = retrieval_service.retrieve(query, documents, use_hybrid=True, use_rerank=True)
        for i, doc in enumerate(docs[:2], 1):
            logger.info(f"  {i}. {doc.page_content[:100]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("Retrieval Testing Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()