"""
Embedding Module
Converts text chunks into vector embeddings using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import os

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformer model
                       'all-MiniLM-L6-v2' → 384 dimensions, fast, good quality
        """
        print(f"🔄 Loading embedding model: {model_name}")
        
        # Load model (downloads ~80MB on first run)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded!")
        print(f"📊 Embedding dimension: {self.embedding_dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text into embedding vector
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of shape (embedding_dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_chunks(self, chunks: List[Dict[str, any]], show_progress: bool = True) -> List[np.ndarray]:
        """
        Convert multiple chunks into embeddings
        
        Args:
            chunks: List of chunk dictionaries from ingestion.py
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract just the text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings (batch processing for speed)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32  # Process 32 chunks at a time
        )
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📊 Each embedding has {embeddings[0].shape[0]} dimensions")
        
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)


# ==============================================================================
# TESTING CODE
# ==============================================================================

if __name__ == "__main__":
    """
    Test the embedding generator
    """
    
    print("\n" + "="*60)
    print("🧪 TESTING EMBEDDING GENERATOR")
    print("="*60 + "\n")
    
    # Initialize
    embedder = EmbeddingGenerator()
    
    # Test 1: Single text embedding
    print("\n--- Test 1: Single Text Embedding ---")
    text = "Machine learning models achieve high accuracy on benchmark datasets"
    embedding = embedder.embed_text(text)
    
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    
    # Test 2: Similarity comparison
    print("\n--- Test 2: Similarity Comparison ---")
    
    text1 = "The neural network achieved 95% accuracy"
    text2 = "Our model reached 95% precision"
    text3 = "I enjoy eating pizza"
    
    sim_1_2 = embedder.compute_similarity(text1, text2)
    sim_1_3 = embedder.compute_similarity(text1, text3)
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {sim_1_2:.4f} (should be HIGH)")
    
    print(f"\nText 1: {text1}")
    print(f"Text 3: {text3}")
    print(f"Similarity: {sim_1_3:.4f} (should be LOW)")
    
    # Test 3: Batch embedding
    print("\n--- Test 3: Batch Embedding ---")
    
    sample_chunks = [
        {"text": "Introduction to machine learning", "page": 1},
        {"text": "Deep learning methods for computer vision", "page": 2},
        {"text": "Natural language processing with transformers", "page": 3},
    ]
    
    embeddings = embedder.embed_chunks(sample_chunks)
    
    print(f"\nProcessed {len(embeddings)} chunks")
    print(f"Embedding matrix shape: {np.array(embeddings).shape}")
    
    print("\n✅ All embedding tests passed!")