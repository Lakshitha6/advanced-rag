# test_phase1.py
from src.config.settings import Settings
from src.utils.logger import setup_logger


def test_config():
    """Test that config loads successfully."""
    try:
        logger = setup_logger("test_phase1")
        logger.info("Starting Phase 1 verification...")
        
        settings = Settings.load()
        
        logger.info(f"✅ Config Loaded Successfully")
        logger.info(f"   LLM Model: {settings.llm.model}")
        logger.info(f"   Embedding Model: {settings.embedding.model}")
        logger.info(f"   Top K: {settings.retrieval.top_k}")
        logger.info(f"   Fallback Message: {settings.evaluation.fallback_message}")
        
        return True
    except FileNotFoundError as e:
        print(f"❌ Config File Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Config Validation Error: {e}")
        return False


if __name__ == "__main__":
    success = test_config()
    exit(0 if success else 1)