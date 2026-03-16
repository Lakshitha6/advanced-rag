import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Optional


from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger("vector_store")

class VectorStoreRepository:
    """
    Repository for Chromadb vector store operations
    """

    def __init__(self, settings: Settings, embeddings: Embeddings):
        """
        Initialize ChromaDB vector store.
        
        Args:
            settings: Application settings
            embeddings: Embedding function (from EmbeddingsService)
        """

        self.settings = settings
        self.embeddings = embeddings
        self.persist_directory = Path(settings.vector_store.persist_directory)
        self.collection_name = settings.vector_store.collection_name

        # Ensure persist dir is exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize langchain chroma wrapper
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embeddings
        )

        logger.info(f"VectorStore initialized at: {self.persist_directory}")
        logger.info(f"Collection: {self.collection_name}")

    
    def ingest_documents(self, documents: List[Document]) -> int:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Number of documents ingested
        """

        if not documents:
            logger.warning("No documents to ingest")
            return 0
        
        logger.info(f"Ingesting {len(documents)} documents...")
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        logger.info(f"Successfully ingested {len(documents)} documents")
        return len(documents)
    
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results (defaults to settings.retrieval.top_k)
            
        Returns:
            List of relevant documents
        """

        k = k or self.settings.retrieval.top_k
        logger.debug(f"Searching for: '{query}' (k={k})")

        docs = self.vector_store.similarity_search(query=query, k=k)

        logger.debug(f"Found {len(docs)} documents")
        return docs
    
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        Conveniance for testing or re-ingestion.
        
        Returns:
            True if successful
        """
        logger.warning(f"Clearing collection: {self.collection_name}")

        try:
            self.client.delete_collection(self.collection_name)
            # Reinitialize to recreate collection
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            logger.info("Collection cleared successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
        
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Document count
        """

        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0