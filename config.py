"""Configuration settings for the embeddings project."""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    model_name: str = "all-MiniLM-L6-v2"  # Default model from sentence-transformers
    batch_size: int = 32
    device: str = "cpu"  # 'cuda' for GPU acceleration

@dataclass
class SearchConfig:
    """Configuration for semantic search."""
    top_k: int = 5  # Number of results to return
    similarity_threshold: float = 0.6  # Minimum similarity score (0-1)

@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 500  # Characters per chunk
    chunk_overlap: int = 50  # Overlap between chunks
