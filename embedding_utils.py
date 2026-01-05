"""Utilities for generating and working with text embeddings."""
from typing import List, Dict, Any, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch

class EmbeddingGenerator:
    """Handles generation and management of text embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self._get_embedding_dim()
    
    def _get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        # Get embedding dimension by encoding a dummy string
        dummy_embedding = self.model.encode("test", convert_to_numpy=True)
        return dummy_embedding.shape[0]
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32, 
                      normalize_embeddings: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )
        
        return embeddings
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        if len(embeddings) == 0:
            return embeddings
        return normalize(embeddings)
    
    @staticmethod
    def cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n1, dim)
            embeddings2: Second set of embeddings (n2, dim)
            
        Returns:
            numpy.ndarray: Similarity matrix of shape (n1, n2)
        """
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return np.array([])
            
        # Ensure embeddings are normalized
        if not np.allclose(np.linalg.norm(embeddings1, axis=1), 1.0):
            embeddings1 = EmbeddingGenerator.normalize_embeddings(embeddings1)
        if not np.allclose(np.linalg.norm(embeddings2, axis=1), 1.0):
            embeddings2 = EmbeddingGenerator.normalize_embeddings(embeddings2)
            
        return np.dot(embeddings1, embeddings2.T)
    
    @staticmethod
    def vector_arithmetic(
        positive: List[np.ndarray] = None,
        negative: List[np.ndarray] = None,
        normalize_result: bool = True
    ) -> np.ndarray:
        """Perform vector arithmetic on embeddings.
        
        Example: vector_arithmetic(positive=[king, woman], negative=[man]) ~= queen
        
        Args:
            positive: List of vectors to add
            negative: List of vectors to subtract
            normalize_result: Whether to normalize the result vector
            
        Returns:
            numpy.ndarray: Resulting vector
        """
        positive = positive or []
        negative = negative or []
        
        if not positive and not negative:
            raise ValueError("At least one positive or negative vector must be provided")
            
        result = np.zeros_like(positive[0] if positive else negative[0])
        
        for vec in positive:
            result += vec
        for vec in negative:
            result -= vec
            
        if normalize_result and np.linalg.norm(result) > 0:
            result = result / np.linalg.norm(result)
            
        return result
