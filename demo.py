"""
Demo script showcasing the embeddings functionality including:
1. Document processing and chunking
2. Generating and normalizing embeddings
3. Semantic search
4. Vector arithmetic
"""
import os
import numpy as np
from typing import List, Dict, Any

from config import ModelConfig, SearchConfig, DocumentConfig
from document_processor import DocumentProcessor, DocumentChunk
from embedding_utils import EmbeddingGenerator
from search import SemanticSearch, SearchResult

def print_search_results(results: List[SearchResult], query: str = None):
    """Helper function to print search results."""
    if query:
        print(f"\nSearch results for: '{query}'")
        print("-" * 80)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.3f}")
        print(f"Text: {result.text[:200]}..." if len(result.text) > 200 else f"Text: {result.text}")
        if result.metadata:
            print(f"Metadata: {result.metadata}")

def demo_document_processing():
    """Demo document chunking functionality."""
    print("\n" + "="*80)
    print("DOCUMENT PROCESSING DEMO")
    print("="*80)
    
    # Sample document
    document = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    natural intelligence displayed by animals including humans. AI research has been defined 
    as the field of study of intelligent agents, which refers to any system that perceives 
    its environment and takes actions that maximize its chance of achieving its goals.
    
    The traditional problems (or goals) of AI research include reasoning, knowledge 
    representation, planning, learning, natural language processing, perception, and the 
    ability to move and manipulate objects. General intelligence (the ability to solve an 
    arbitrary problem) is among the field's long-term goals.
    """
    
    # Initialize document processor
    doc_processor = DocumentProcessor(
        chunk_size=150,
        chunk_overlap=30
    )
    
    # Process document
    chunks = doc_processor.chunk_document(
        document,
        metadata={"source": "AI Wikipedia Article"}
    )
    
    # Display chunks
    print(f"\nSplit document into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (length: {len(chunk.text)}):")
        print(f"{chunk.text[:100]}..." if len(chunk.text) > 100 else chunk.text)
        print(f"Metadata: {chunk.metadata}")
    
    return chunks

def demo_semantic_search(documents: List[DocumentChunk]):
    """Demo semantic search functionality."""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH DEMO")
    print("="*80)
    
    # Initialize embedding generator and search
    model_config = ModelConfig()
    search_config = SearchConfig()
    
    embedding_generator = EmbeddingGenerator(
        model_name=model_config.model_name,
        device=model_config.device
    )

    print(f"\nUsing search config: {search_config}")
    print(embedding_generator)

    
    search_engine = SemanticSearch(embedding_generator)
    
    # Add documents to search index
    search_engine.add_documents(documents)
    
    # Perform searches
    queries = [
        "What is artificial intelligence?",
        "What are the goals of AI research?",
        "How do machines demonstrate intelligence?"
    ]
    
    for query in queries:
        results = search_engine.search(
            query=query,
            top_k=search_config.top_k,
            score_threshold=search_config.similarity_threshold
        )
        print_search_results(results, query)
    
    return search_engine

def demo_vector_arithmetic():
    """Demo vector arithmetic with embeddings."""
    print("\n" + "="*80)
    print("VECTOR ARITHMETIC DEMO")
    print("="*80)
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Example words for vector arithmetic
    word_pairs = [
        ("king", "man", "queen", "woman"),  # king - man + woman ≈ queen
        ("paris", "france", "berlin", "germany"),  # paris - france + germany ≈ berlin
        ("big", "bigger", "small", "smaller")  # big - bigger + smaller ≈ small
    ]
    
    for word1, word2, word3, expected in word_pairs:
        # Get embeddings for all words
        words = [word1, word2, word3, expected]
        embeddings = embedding_generator.get_embeddings(words)
        
        # Perform vector arithmetic: word1 - word2 + word3
        result = embedding_generator.vector_arithmetic(
            positive=[embeddings[0], embeddings[2]],  # word1 + word3
            negative=[embeddings[1]],                # - word2
            normalize_result=True
        )
        
        # Get all word vectors to compare against (including the expected word)
        all_words = [word1, word2, word3, expected]
        all_embeddings = embedding_generator.get_embeddings(all_words)
        
        # Find most similar word to the result
        similarities = embedding_generator.cosine_similarity(
            result.reshape(1, -1), 
            all_embeddings  # Compare with all words including the expected one
        )[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        
        print(f"\n{word1} - {word2} + {word3}:")
        for i, idx in enumerate(top_indices[:3], 1):  # Show top 3 matches
            print(f"  {i}. {all_words[idx]} (similarity: {similarities[idx]:.2f})")
        print(f"  Expected: {expected}")

def main():
    """Run all demos."""
    # Demo 1: Document Processing
    chunks = demo_document_processing()
    
    # Demo 2: Semantic Search
    search_engine = demo_semantic_search(chunks)
    
    # Demo 3: Vector Arithmetic
    demo_vector_arithmetic()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
