from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from torch import device

from src.config.settings import Settings
from src.utils.logger import setup_logger

from dotenv import load_dotenv
import os

load_dotenv()

hf_key = os.getenv("HF_TOKEN")

logger = setup_logger("embeddings_service")

class EmbeddingsService:
    """
    Service for managing embedding model initialization.
    Centralize embedding configuration and creation

    """

    def __init__(self, settings: Settings):
        """
        Initialize embedding service.

        Args:
            settings: Application settings with embedding config
        
        """

        self.settings = settings
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        """
        Get or create embedding model instance.
        Uses singleton pattern to avoid re-initialization.

        Returns:
            Langchain Embedding Object
        """

        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
            logger.info(f"Embeddings model initialized: {self.settings.embedding.model}")

        return self._embeddings
        
    def _create_embeddings(self) -> Embeddings:
        """
        Create embeddings model based on configuration.

        Returns:
            Langchain Embeddings object

        Raises:
            ValueError: If provider is not supported

        """

        provider = self.settings.embedding.provider.lower()
        model = self.settings.embedding.model
        device = self.settings.reranker.device
        
        logger.info(f"Creating embeddings: provider = {provider}, model = {model}")

        if provider == "huggingface_inference":
            return HuggingFaceEndpointEmbeddings(
                huggingfacehub_api_token=hf_key,
                model=model
            )
        
        elif provider == "huggingface_local":
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported: 'huggingface_local', 'huggingface_inference'"
            )
        
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """

        embeddings= self.get_embeddings()
        return embeddings.embed_query(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """

        embeddings= self.get_embeddings()
        return embeddings.embed_documents(texts)