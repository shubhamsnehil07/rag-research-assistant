"""
Cross-Encoder Reranker for improving retrieval quality
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import numpy as np

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using cross-encoder for better relevance
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder
        
        Args:
            model_name: Cross-encoder model to use
        """
        print(f"🔄 Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"✅ Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank chunks based on query relevance
        
        Args:
            query: User query
            chunks: List of retrieved chunks
            top_k: Number of top results to return
            
        Returns:
            Reranked list of (chunk, score) tuples
        """
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return scored_chunks[:top_k]


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Test reranking effect
    """
    print("\n" + "="*70)
    print("🧪 TESTING CROSS-ENCODER RERANKING")
    print("="*70 + "\n")
    
    from vector_store import VectorStore
    from embeddings import EmbeddingGenerator
    
    # Load components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(embedding_dimension=384)
    vector_store.load("vector_db")
    
    # Initialize reranker
    reranker = CrossEncoderReranker()
    
    # Test query
    query = "What is described in the abstract?"
    
    print(f"❓ Query: {query}\n")
    
    # Get initial results
    print("📊 BEFORE RERANKING (Vector Search):")
    query_emb = embedder.embed_text(query)
    initial_results = vector_store.search(query_emb, top_k=8)
    
    initial_chunks = [chunk for chunk, _ in initial_results]
    
    for i, (chunk, dist) in enumerate(initial_results[:5], 1):
        print(f"\n{i}. Distance: {dist:.2f}")
        print(f"   Section: {chunk.get('section', 'Unknown')}")
        print(f"   Text: {chunk['text'][:100]}...")
    
    print("\n" + "="*70)
    print("🎯 AFTER RERANKING:")
    print("="*70)
    
    # Rerank
    reranked = reranker.rerank(query, initial_chunks, top_k=5)
    
    for i, (chunk, score) in enumerate(reranked, 1):
        print(f"\n{i}. Relevance Score: {score:.4f}")
        print(f"   Section: {chunk.get('section', 'Unknown')}")
        print(f"   Text: {chunk['text'][:100]}...")
    
    print("\n✅ Reranking test complete!")
    print("\n💡 Notice: Reranking may change the order to put more relevant chunks first!")