# tests/test_embeddings_service.py
import pytest
from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def embeddings_service(settings):
    return EmbeddingsService(settings=settings)


class TestEmbeddingsService:
    """Test embeddings service functionality."""
    
    def test_get_embeddings(self, embeddings_service):
        """Test that embeddings model is created."""
        embeddings = embeddings_service.get_embeddings()
        assert embeddings is not None
    
    def test_singleton_pattern(self, embeddings_service):
        """Test that same instance is returned."""
        emb1 = embeddings_service.get_embeddings()
        emb2 = embeddings_service.get_embeddings()
        assert emb1 is emb2  # Same object
    
    def test_embed_query(self, embeddings_service):
        """Test embedding a single query."""
        embeddings = embeddings_service.get_embeddings()
        vector = embeddings.embed_query("Hello world")
        
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)