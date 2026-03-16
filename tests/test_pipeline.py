# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock

from src.config.settings import Settings
from src.core.pipeline import RAGPipeline


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def pipeline(settings):
    # Mock document loading to avoid file I/O in tests
    with patch('src.core.pipeline.load_documents_for_retrieval') as mock_load:
        mock_load.return_value = []
        return RAGPipeline(settings=settings)


class TestRAGPipeline:
    """Test end-to-end RAG pipeline."""
    
    @patch('src.core.pipeline.RAGPipeline._load_documents')
    @patch('src.services.retrieval_service.RetrievalService.retrieve')
    @patch('src.services.llm_service.LLMService.generate_answer')
    @patch('src.services.evaluation_service.EvaluationService.evaluate_grounding')
    def test_happy_path(
        self, 
        mock_eval, 
        mock_generate, 
        mock_retrieve, 
        mock_load,
        pipeline
    ):
        """Test successful answer generation with grounded answer."""
        # Setup mocks
        mock_load.return_value = []
        mock_retrieve.return_value = [
            MagicMock(page_content="The capital of France is Paris.")
        ]
        mock_generate.return_value = "The capital of France is Paris."
        mock_eval.return_value = (True, "Answer supported")
        
        # Run pipeline
        result = pipeline.run("What is the capital of France?")
        
        # Assertions
        assert result == "The capital of France is Paris."
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()
        mock_eval.assert_called_once()
    
    @patch('src.core.pipeline.RAGPipeline._load_documents')
    @patch('src.services.retrieval_service.RetrievalService.retrieve')
    @patch('src.services.llm_service.LLMService.generate_answer')
    @patch('src.services.evaluation_service.EvaluationService.evaluate_grounding')
    def test_fallback_triggered(
        self, 
        mock_eval, 
        mock_generate, 
        mock_retrieve, 
        mock_load,
        pipeline
    ):
        """Test fallback message when answer is not grounded."""
        # Setup mocks
        mock_load.return_value = []
        mock_retrieve.return_value = [
            MagicMock(page_content="The sky is blue.")
        ]
        mock_generate.return_value = "The capital of France is Paris."  # Hallucination!
        mock_eval.return_value = (False, "Answer not supported")
        
        # Run pipeline
        result = pipeline.run("What is the capital of France?")
        
        # Assertions - should return fallback message
        assert result == pipeline.settings.evaluation.fallback_message
        assert "not in the content" in result.lower()
    
    @patch('src.core.pipeline.RAGPipeline._load_documents')
    @patch('src.services.retrieval_service.RetrievalService.retrieve')
    def test_no_documents_retrieved(
        self, 
        mock_retrieve, 
        mock_load,
        pipeline
    ):
        """Test fallback when no documents are retrieved."""
        # Setup mocks
        mock_load.return_value = []
        mock_retrieve.return_value = []  # No docs
        
        # Run pipeline
        result = pipeline.run("Any question")
        
        # Assertions
        assert result == pipeline.settings.evaluation.fallback_message