# tests/test_retrieval_service.py
import pytest
from langchain_core.documents import Document

from src.config.settings import Settings
from src.services.embeddings_service import EmbeddingsService
from src.services.retrieval_service import RetrievalService
from src.repositories.vector_store import VectorStoreRepository


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def embeddings_service(settings):
    return EmbeddingsService(settings=settings)


@pytest.fixture
def vector_store(settings, embeddings_service):
    store = VectorStoreRepository(
        settings=settings,
        embeddings=embeddings_service.get_embeddings()
    )
    store.clear_collection()
    yield store
    store.clear_collection()


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The capital of France is Paris.", metadata={"source": "geo"}),
        Document(page_content="Python is a programming language created by Guido van Rossum.", metadata={"source": "tech"}),
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "tech"}),
        Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source": "geo"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "tech"}),
    ]


@pytest.fixture
def retrieval_service(settings, embeddings_service, vector_store):
    return RetrievalService(
        settings=settings,
        embeddings_service=embeddings_service,
        vector_store=vector_store,
        llm=None  # No LLM for basic tests
    )


class TestRetrievalService:
    """Test retrieval service functionality."""
    
    def test_vector_retrieval(self, retrieval_service, sample_documents, vector_store):
        """Test basic vector retrieval."""
        vector_store.ingest_documents(sample_documents)
        
        docs = retrieval_service.retrieve(
            query="What is the capital of France?",
            documents=sample_documents,
            use_hybrid=False,
            use_rerank=False
        )
        
        assert len(docs) > 0
        assert "Paris" in docs[0].page_content
    
    def test_hybrid_retrieval(self, retrieval_service, sample_documents, vector_store):
        """Test hybrid retrieval."""
        vector_store.ingest_documents(sample_documents)
        
        docs = retrieval_service.retrieve(
            query="Python programming",
            documents=sample_documents,
            use_hybrid=True,
            use_rerank=False
        )
        
        assert len(docs) > 0
        assert any("Python" in doc.page_content for doc in docs)
    
    def test_rerank_documents(self, retrieval_service, sample_documents):
        """Test cross-encoder reranking."""
        docs = retrieval_service.rerank_documents(
            query="artificial intelligence",
            documents=sample_documents,
            top_k=2
        )
        
        assert len(docs) == 2
        # ML/AI docs should rank higher
        assert any("machine learning" in doc.page_content.lower() for doc in docs)