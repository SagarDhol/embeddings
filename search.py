"""Semantic search functionality using vector embeddings."""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from document_processor import DocumentChunk
from embedding_utils import EmbeddingGenerator

@dataclass
class SearchResult:
    """Represents a search result with score and metadata."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str

class SemanticSearch:
    """Handles semantic search over a collection of documents."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """Initialize with an embedding generator."""
        self.embedding_generator = embedding_generator
        self.documents: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.array([])
    
    def add_documents(self, documents: List[DocumentChunk]):
        """Add documents to the search index.
        
        Args:
            documents: List of DocumentChunk objects to add
        """
        if not documents:
            return
            
        # Generate embeddings for new documents
        texts = [doc.text for doc in documents]
        new_embeddings = self.embedding_generator.get_embeddings(texts)
        
        # Update documents and embeddings
        self.documents.extend(documents)
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """Search for documents similar to the query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not self.documents or len(self.embeddings) == 0:
            return []
            
        # Get query embedding
        query_embedding = self.embedding_generator.get_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = self.embedding_generator.cosine_similarity(
            query_embedding, self.embeddings
        )[0]  # Flatten to 1D array
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and create results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < score_threshold:
                continue
                
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc.text,
                score=score,
                metadata=doc.metadata or {},
                chunk_id=doc.chunk_id
            ))
        
        return results
    
    def most_similar_pairs(
        self, 
        texts: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Find the most similar pairs of texts.
        
        Args:
            texts: List of text strings
            top_k: Number of pairs to return
            
        Returns:
            List of (text1, text2, similarity) tuples
        """
        if len(texts) < 2:
            return []
            
        # Get embeddings for all texts
        embeddings = self.embedding_generator.get_embeddings(texts)
        
        # Compute similarity matrix
        similarity_matrix = self.embedding_generator.cosine_similarity(embeddings, embeddings)
        
        # Get upper triangular matrix without diagonal
        upper_tri = np.triu_indices(len(texts), k=1)
        pairs = list(zip(upper_tri[0], upper_tri[1], similarity_matrix[upper_tri]))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k pairs
        return [(texts[i], texts[j], sim) for i, j, sim in pairs[:top_k]]
