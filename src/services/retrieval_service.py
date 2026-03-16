from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from typing import List, Optional

from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.repositories.vector_store import VectorStoreRepository
from src.utils.logger import setup_logger

logger = setup_logger("retrieval_service")

class RetrievalService:
    """
    Service for advanced retrieval strategies.
    Implements Hybrid Search, Cross-Encoder Reranking, and MultiQuery.
    """

    def __init__(self,
                settings: Settings,
                embeddings_service: EmbeddingsService,
                vector_store: VectorStoreRepository,
                llm: Optional[BaseChatModel] = None
                ):
        """
        Initialize retrieval service.
        
        Args:
            settings: Application settings
            embeddings_service: Embeddings service for vector retrieval
            vector_store: Vector store repository
            llm: For MultiQuery generation
        """

        self.settings = settings
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.llm = llm

        # Initialize cross encoder for reranking

        self.cross_encoder = None
        if settings.retrieval.use_cross_encoder:
            try:
                self.cross_encoder = HuggingFaceCrossEncoder(
                    model_name = settings.reranker.model
                )
                logger.info(f"Cross-encoder initialized: {settings.reranker.model}")

            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Continuing without reranking.")

        self._bm25_retriever = None
        self._documents_cache: List[Document] = []

        logger.info("RetrievalService initialized")


    def _get_bm25_retriever(self, documents: List[Document]) -> BM25Retriever:
        """
        Get or create BM25 retriever.
        Uses caching to avoid rebuilding on every request.
        
        Args:
            documents: Documents to build BM25 index from
            
        Returns:
            BM25Retriever instance
        """

        # Rebuild if documents changed or not initialized
        if self._bm25_retriever is None or documents != self._documents_cache:
            logger.info("Building BM25 retriever...")

            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = self.settings.retrieval.rerank_top_k
            self._documents_cache = documents

            logger.info(f"BM25 retriever built with {len(documents)} documents")
        
        return self._bm25_retriever
    

    def get_hybrid_retriever(self, documents: List[Document]) -> EnsembleRetriever:
        """
        Create hybrid retriever combining BM25 and Vector search.
        
        Args:
            documents: Documents for BM25 indexing
            
        Returns:
            EnsembleRetriever with BM25 + Vector
        """
        # BM25 retriever
        bm25 = self._get_bm25_retriever(documents)

        # Vector retriever
        vector = self.vector_store.vector_store.as_retriever(
            search_kwargs={"k": self.settings.retrieval.rerank_top_k}
        )

        # combine with equal weights (can be tuned)
        hybrid = EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.5,0.5]
        )

        logger.info("Hybrid retriever created (BM25 + Vector)")
        return hybrid


    def rerank_documents(self, query: str, documents: List[Document], top_k: Optional[int]=None)-> List[Document]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """

        if not documents:
            return[]
        
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available. Returning original order.")
            return documents[:top_k or self.settings.retrieval.top_k]
        
        top_k = top_k or self.settings.retrieval.top_k
        logger.info(f"Reranking {len(documents)} documents with cross-encoder...")

        # create query documents pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance score
        scores = self.cross_encoder.predict(pairs)

        # Sort by score (descending - from top score to lower score)
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log top scores for debugging
        logger.debug(f"Top 3 rerank scores: {[s for _, s in scored_docs[:3]]}")

        # Return top documents
        reranked = [doc for doc, score in scored_docs[:top_k]]

        logger.info(f"Reranking complete. Returning {len(reranked)} documents")
        return reranked
    

    def retrieve(self, 
        query: str, 
        documents: List[Document],
        use_hybrid: Optional[bool] = None,
        use_rerank: Optional[bool] = None
    ) -> List[Document]:
        """
        Main retrieval method with configurable strategies.
        
        Args:
            query: Search query
            documents: All documents for BM25 indexing
            use_hybrid: Use hybrid search (default from settings)
            use_rerank: Use cross-encoder reranking (default from settings)
            
        Returns:
            List of relevant documents
        """

        use_hybrid = use_hybrid if use_hybrid is not None else self.settings.retrieval.use_hybrid
        use_rerank = use_rerank if use_rerank is not None else self.settings.retrieval.use_cross_encoder

        logger.info(f"Retrieving for query: '{query}'")
        logger.debug(f"Strategies: hybrid={use_hybrid}, rerank={use_rerank}")

        # Step 1: Retrieve initial candidates
        if use_hybrid and self.settings.retrieval.use_hybrid:
            logger.info("Using Hybrid Search (BM25 + Vector)")
            retriever = self.get_hybrid_retriever(documents)
            initial_docs = retriever.invoke(query)

        else:
            logger.info("Using Vector Search Only")
            initial_docs = self.vector_store.similarity_search(
                query,
                k = self.settings.retrieval.rerank_top_k
            )

        logger.info(f"Initial retrieval: {len(initial_docs)} documents")

        # Step 2: Rerank if enabled
        if use_rerank and self.settings.retrieval.use_cross_encoder:
            final_docs = self.rerank_documents(
                query,
                initial_docs,
                top_k= self.settings.retrieval.top_k
            )

        else:
            final_docs = initial_docs[:self.settings.retrieval.top_k]

        logger.info(f"Final retrieval: {len(final_docs)} documents")
        return final_docs
    

    def multi_query_retrieve(
            self,
            query: str,
            documents: List[Document],
            num_queries: int = 5
    ) -> List[Document]:
        
        """
        Generate multiple query variations and retrieve for each.
        Combines and deduplicates results.
        
        Args:
            query: Original query
            documents: All documents for retrieval
            num_queries: Number of query variations to generate
            
        Returns:
            Deduplicated list of relevant documents
        """

        if not self.llm:
            logger.warning("LLM not available. Falling back to standard retrieval.")
            return self.retrieve(query, documents)
        
        logger.info(f"MultiQuery retrieval: generating {num_queries} query variations")

            # Prompt to generate query variations
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant helping with document retrieval.
        Generate {num_queries} different versions of the following query that would help find relevant information.
        Each version should be semantically different but aim to retrieve the same information.
        
        Original Query: {query}
        
        Generate {num_queries} variations, one per line. Only output the variations, nothing else.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Generate variations
        response = chain.invoke({"query": query, "num_queries": num_queries})
        variations = [v.strip() for v in response.split("\n") if v.strip()]
        
        logger.info(f"Generated {len(variations)} query variations")

        # Retrieve for each variation
        all_docs =[]
        seen_content = set()

        for variation in variations[:num_queries]:
            docs = self.retrieve(variation, documents, use_rerank=False)
            for doc in docs:
                # Duplicate by content
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)

        logger.info(f"MultiQuery retrieval: {len(all_docs)} unique documents")

        # Final reranking on combined results
        if self.settings.retrieval.use_cross_encoder:
            all_docs = self.rerank_documents(query, all_docs, top_k=self.settings.retrieval.top_k)

        return all_docs[:self.settings.retrieval.top_k]