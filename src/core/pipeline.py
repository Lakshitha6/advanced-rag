from typing import List

from langchain_core.documents import Document

from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.services.llm_service import LLMService
from src.services.retrieval_service import RetrievalService
from src.services.evaluation_service import EvaluationService
from src.repositories.vector_store import VectorStoreRepository
from src.repositories.ingest import load_documents_for_retrieval
from src.utils.logger import setup_logger

logger = setup_logger("pipeline")


class RAGPipeline:
    """
    Main RAG pipeline orchestrating retrieval, generation, and evaluation.
    Implements the guardrail: "Answer not in the content."
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize RAG pipeline with all services.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize services (Dependency Injection)
        self.embeddings_service = EmbeddingsService(settings=settings)
        self.llm_service = LLMService(settings=settings)
        
        # Initialize vector store
        self.vector_store = VectorStoreRepository(
            settings=settings,
            embeddings=self.embeddings_service.get_embeddings()
        )
        
        # Initialize retrieval service
        self.retrieval_service = RetrievalService(
            settings=settings,
            embeddings_service=self.embeddings_service,
            vector_store=self.vector_store,
            llm=self.llm_service.get_llm()
        )
        
        # Initialize evaluation service
        self.evaluation_service = EvaluationService(
            settings=settings,
            llm=self.llm_service.get_llm()
        )
        
        # Load cached documents for retrieval
        self.documents = self._load_documents()
        
        logger.info("RAGPipeline initialized")
    
    def _load_documents(self) -> List[Document]:
        """Load cached documents for retrieval service."""
        try:
            docs = load_documents_for_retrieval()
            logger.info(f"Loaded {len(docs)} documents for retrieval")
            return docs
        except FileNotFoundError:
            logger.warning("Document cache not found. Retrieval may be limited.")
            return []
    
    def run(self, query: str) -> str:
        """
        Run the full RAG pipeline for a query.
        
        Flow:
        1. Retrieve relevant documents
        2. Generate answer from context
        3. Evaluate if answer is grounded
        4. Return answer OR fallback message
        
        Args:
            query: User question
            
        Returns:
            Answer string or fallback message
        """
        logger.info("=" * 60)
        logger.info(f"Processing query: '{query}'")
        logger.info("=" * 60)
        
        try:
            # Step 1: Retrieve documents
            logger.info("Step 1: Retrieving documents...")
            documents = self.retrieval_service.retrieve(
                query=query,
                documents=self.documents
            )
            
            if not documents:
                logger.warning("No documents retrieved")
                return self.settings.evaluation.fallback_message
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in documents])
            logger.info(f"Retrieved {len(documents)} documents. Context length: {len(context)} chars")
            
            # Step 2: Generate answer
            logger.info("Step 2: Generating answer...")
            answer = self.llm_service.generate_answer(query=query, context=context)
            logger.info(f"Generated answer: {answer[:100]}...")
            
            # Step 3: Evaluate groundedness
            logger.info("Step 3: Evaluating groundedness...")
            is_groundeds, reason = self.evaluation_service.evaluate_grounding(
                question=query,
                context=context,
                answer=answer
            )
            
            # Step 4: Return answer or fallback
            if self.evaluation_service.should_use_fallback(not is_groundeds):
                logger.warning(f"  Guardrail triggered: {reason}")
                logger.info(f"Returning fallback: '{self.settings.evaluation.fallback_message}'")
                return self.settings.evaluation.fallback_message
            
            logger.info(" Answer passed groundedness check")
            logger.info("=" * 60)
            return answer
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            # Fail safe: return fallback on any error
            return self.settings.evaluation.fallback_message