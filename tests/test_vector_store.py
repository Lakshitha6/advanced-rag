import pytest
from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.repositories.vector_store import VectorStoreRepository


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def embeddings_service(settings):
    return EmbeddingsService(settings=settings)


@pytest.fixture
def embeddings(embeddings_service):
    return embeddings_service.get_embeddings()


@pytest.fixture
def vector_store(settings, embeddings):
    store = VectorStoreRepository(settings=settings, embeddings=embeddings)
    store.clear_collection()
    yield store
    store.clear_collection()