"""
Vector Store Module
Manages FAISS vector database for similarity search
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from embeddings import EmbeddingGenerator

class VectorStore:
    """
    FAISS-based vector store for chunk embeddings
    """
    
    def __init__(self, embedding_dimension: int = 384):
        """
        Initialize vector store
        
        Args:
            embedding_dimension: Size of embedding vectors (384 for MiniLM)
        """
        self.embedding_dimension = embedding_dimension
        
        # Create FAISS index (L2 distance, good for sentence embeddings)
        self.index = faiss.IndexFlatL2(embedding_dimension)
        
        # Store original chunks (for retrieval)
        self.chunks = []
        
        print(f"✅ Vector store initialized (dimension={embedding_dimension})")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, any]]):
        """
        Add embeddings and their chunks to the database
        
        Args:
            embeddings: Numpy array of shape (n_chunks, embedding_dimension)
            chunks: List of chunk dictionaries
        """
        # Convert to float32 (FAISS requirement)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks
        self.chunks.extend(chunks)
        
        print(f"✅ Added {len(embeddings)} embeddings to vector store")
        print(f"📊 Total vectors in database: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for most similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples, sorted by similarity
        """
        # Ensure correct shape and type
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve chunks
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                results.append((chunk, float(dist)))
        
        return results
    
    def save(self, save_dir: str = "vector_db"):
        """
        Save vector store to disk
        
        Args:
            save_dir: Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"💾 Vector store saved to: {save_dir}")
    
    def load(self, save_dir: str = "vector_db"):
        """
        Load vector store from disk
        
        Args:
            save_dir: Directory containing saved files
        """
        # Load FAISS index
        index_path = os.path.join(save_dir, "faiss.index")
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"📂 Vector store loaded from: {save_dir}")
        print(f"📊 Total vectors: {self.index.ntotal}")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with stats
        """
        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "unique_sources": len(set(chunk['source'] for chunk in self.chunks)),
        }


# ==============================================================================
# TESTING CODE
# ==============================================================================

if __name__ == "__main__":
    """
    Test the vector store
    """
    
    print("\n" + "="*60)
    print("🧪 TESTING VECTOR STORE")
    print("="*60 + "\n")
    
    # Create sample data
    from embeddings import EmbeddingGenerator
    
    embedder = EmbeddingGenerator()
    
    sample_chunks = [
        {"text": "Neural networks are trained using backpropagation", "page": 1, "source": "test.pdf"},
        {"text": "Deep learning models achieve high accuracy", "page": 2, "source": "test.pdf"},
        {"text": "Transformers use self-attention mechanisms", "page": 3, "source": "test.pdf"},
        {"text": "Computer vision tasks include image classification", "page": 4, "source": "test.pdf"},
    ]
    
    # Generate embeddings
    embeddings = embedder.embed_chunks(sample_chunks, show_progress=False)
    
    # Create vector store
    vector_store = VectorStore(embedding_dimension=384)
    vector_store.add_embeddings(embeddings, sample_chunks)
    
    # Test search
    print("\n--- Test Search ---")
    query = "How are neural networks trained?"
    print(f"Query: {query}")
    
    query_embedding = embedder.embed_text(query)
    results = vector_store.search(query_embedding, top_k=2)
    
    print(f"\nTop {len(results)} results:")
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"\n{i}. Distance: {distance:.4f}")
        print(f"   Page: {chunk['page']}")
        print(f"   Text: {chunk['text']}")
    
    # Test save/load
    print("\n--- Test Save/Load ---")
    vector_store.save("test_vector_db")
    
    # Create new store and load
    new_store = VectorStore(embedding_dimension=384)
    new_store.load("test_vector_db")
    
    print(f"\nStats after loading:")
    print(new_store.get_stats())
    
    print("\n✅ All vector store tests passed!")