"""
Hybrid Retriever combining BM25 (sparse) and Dense (embedding) search
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

class HybridRetriever:
    """
    Combines BM25 (keyword) and Dense (semantic) retrieval
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: Dense vector store (FAISS)
            embedder: Embedding generator
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense scores (0-1)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Build BM25 index
        print("🔧 Building BM25 index...")
        self._build_bm25_index()
        print("✅ BM25 index ready")
    
    def _build_bm25_index(self):
        """Build BM25 index from vector store chunks"""
        # Get all chunks
        self.chunks = self.vector_store.chunks
        
        # Tokenize for BM25
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 8) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining BM25 and dense retrieval
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, combined_score) tuples
        """
        # 1. BM25 search (keyword-based)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # 2. Dense search (semantic)
        query_embedding = self.embedder.embed_text(query)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Convert distances to similarity scores (inverse)
        dense_scores = {}
        for chunk, distance in dense_results:
            chunk_id = chunk['chunk_id']
            # Lower distance = higher similarity
            similarity = 1.0 / (1.0 + distance)
            dense_scores[chunk_id] = similarity
        
        # Normalize dense scores
        if dense_scores:
            max_dense = max(dense_scores.values())
            if max_dense > 0:
                dense_scores = {k: v/max_dense for k, v in dense_scores.items()}
        
        # 3. Combine scores
        combined_scores = {}
        
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk['chunk_id']
            
            # Get BM25 score
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            
            # Get dense score
            dense_score = dense_scores.get(chunk_id, 0)
            
            # Weighted combination
            combined_score = (
                self.bm25_weight * bm25_score +
                self.dense_weight * dense_score
            )
            
            combined_scores[chunk_id] = combined_score
        
        # 4. Sort by combined score
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda k: combined_scores[k],
            reverse=True
        )
        
        # 5. Return top-k results
        results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            # Find the chunk
            chunk = next(c for c in self.chunks if c['chunk_id'] == chunk_id)
            score = combined_scores[chunk_id]
            results.append((chunk, score))
        
        return results


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Test hybrid retrieval vs pure dense
    """
    print("\n" + "="*70)
    print("🧪 TESTING HYBRID RETRIEVAL")
    print("="*70 + "\n")
    
    from vector_store import VectorStore
    from embeddings import EmbeddingGenerator
    
    # Load components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(embedding_dimension=384)
    vector_store.load("vector_db")
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_weight=0.5,
        dense_weight=0.5
    )
    
    # Test queries
    test_queries = [
        "What is described in the abstract?",
        "What datasets were used?",
        "What are the main objectives?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"❓ Query: {query}")
        print(f"{'='*70}\n")
        
        # Hybrid search
        print("🔀 HYBRID RETRIEVAL:")
        hybrid_results = hybrid.search(query, top_k=3)
        
        for i, (chunk, score) in enumerate(hybrid_results, 1):
            print(f"\n{i}. Score: {score:.3f} | Section: {chunk.get('section', 'Unknown')} | Page: {chunk['page_number']}")
            print(f"   Text: {chunk['text'][:150]}...")
        
        print("\n" + "─"*70)
        
        # Pure dense for comparison
        print("\n🎯 PURE DENSE (for comparison):")
        query_emb = embedder.embed_text(query)
        dense_results = vector_store.search(query_emb, top_k=3)
        
        for i, (chunk, distance) in enumerate(dense_results, 1):
            print(f"\n{i}. Distance: {distance:.2f} | Section: {chunk.get('section', 'Unknown')} | Page: {chunk['page_number']}")
            print(f"   Text: {chunk['text'][:150]}...")
        
        input("\nPress Enter for next query...")
    
    print("\n✅ Hybrid retrieval test complete!")