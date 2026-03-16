import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int = 1024

class EmbeddingConfig(BaseModel):
    provider: str
    model: str
    dimensions: int = 768

class RetrievalConfig(BaseModel):
    top_k: int = 5
    rerank_top_k: int = 10
    use_hybrid: bool = True
    use_cross_encoder: bool = True

class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-large"
    device: str = "cpu"

class EvaluationConfig(BaseModel):
    groundedness_threshold: float = 0.7
    fallback_message: str = "Answer not in the content."

class VectorStoreConfig(BaseModel):
    type: str = "chroma"
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "rag_collection"

class Settings(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    evaluation: EvaluationConfig
    vector_store: VectorStoreConfig

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file, searching from project root."""
        
        # Strategy 1: Try relative to current working directory first
        path = Path(config_path)
        if path.is_absolute() and path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        
        # Strategy 2: Search upward from this file to find project root
        # (looks for pyproject.toml as project root marker)
        current_file = Path(__file__).resolve()
        for parent in [current_file] + list(current_file.parents):
            candidate = parent / config_path
            if candidate.exists():
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                return cls(**data)
            if (parent / "pyproject.toml").exists():
                break
        
        # Strategy 3: Fallback to current directory
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            f"Searched from: {Path.cwd()} and project root candidates."
        )