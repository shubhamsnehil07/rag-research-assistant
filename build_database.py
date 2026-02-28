"""
Main pipeline to build the RAG vector database
Connects: PDF → Chunks → Embeddings → Vector Store
"""

from ingestion import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
import os
import glob

def build_vector_database(pdf_files: list[str], output_dir: str = "vector_db"):
    """
    Complete pipeline: PDFs → Vector Database
    
    Args:
        pdf_files: List of paths to PDF files
        output_dir: Where to save the vector database
    """
    print("\n" + "="*70)
    print("🚀 BUILDING RAG VECTOR DATABASE")
    print("="*70 + "\n")
    
    # Step 1: Initialize components
    print("📋 Step 1: Initializing components...")
    pdf_processor = PDFProcessor(chunk_size=800, chunk_overlap=150)
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(embedding_dimension=384)
    
    # Step 2: Process all PDFs
    all_chunks = []
    
    print(f"\n📋 Step 2: Processing {len(pdf_files)} PDF(s)...")
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"⚠️  Warning: File not found: {pdf_file}")
            continue
        
        print(f"\n  📄 Processing: {pdf_file}")
        chunks = pdf_processor.process_pdf(pdf_file)
        all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks from all PDFs: {len(all_chunks)}")
    
    # Step 3: Generate embeddings
    print(f"\n📋 Step 3: Generating embeddings...")
    embeddings = embedder.embed_chunks(all_chunks)
    
    # Step 4: Build vector database
    print(f"\n📋 Step 4: Building vector database...")
    vector_store.add_embeddings(embeddings, all_chunks)
    
    # Step 5: Save to disk
    print(f"\n📋 Step 5: Saving database...")
    vector_store.save(output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("✅ DATABASE BUILD COMPLETE!")
    print("="*70)
    
    stats = vector_store.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Unique sources: {stats['unique_sources']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Saved to: {output_dir}/")
    
    return vector_store


def test_search(vector_store: VectorStore, embedder: EmbeddingGenerator):
    """
    Interactive test of the vector database
    """
    print("\n" + "="*70)
    print("🔍 TESTING VECTOR SEARCH")
    print("="*70 + "\n")
    
    test_queries = [
        "What is the main contribution of this paper?",
        "What dataset was used in the experiments?",
        "What accuracy did the model achieve?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"❓ Query: {query}")
        print(f"{'─'*70}\n")
        
        # Search
        query_embedding = embedder.embed_text(query)
        results = vector_store.search(query_embedding, top_k=3)
        
        # Display results
        for i, (chunk, distance) in enumerate(results, 1):
            print(f"Result {i} (distance: {distance:.4f})")
            print(f"  📄 Source: {chunk['source']}, Page {chunk['page_number']}")
            print(f"  📝 Text: {chunk['text'][:200]}...")
            print()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    Build database from PDFs in data/ folder
    """
    
    # Find all PDFs in data folder
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found in 'data/' folder")
        print("\n📝 Instructions:")
        print("1. Download research papers (from arxiv.org or similar)")
        print("2. Save them to the 'data/' folder")
        print("3. Run this script again")
    else:
        print(f"📚 Found {len(pdf_files)} PDF(s):")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        
        # Build database
        vector_store = build_vector_database(pdf_files, output_dir="vector_db")
        
        # Test search
        embedder = EmbeddingGenerator()
        test_search(vector_store, embedder)
        
        print("\n🎉 Ready for Day 4 - Retrieval + LLM Generation!")