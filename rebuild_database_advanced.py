"""
Rebuild database with section-aware chunking
"""

from ingestion_advanced import AdvancedPDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
import glob
import os

def rebuild_database():
    """
    Rebuild vector database with advanced processing
    """
    print("\n" + "="*70)
    print("🔨 REBUILDING DATABASE WITH ADVANCED PROCESSING")
    print("="*70 + "\n")
    
    # Find PDFs
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found in data/ folder")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    # Step 1: Process PDFs with advanced chunking
    print("\n📋 Step 1: Processing PDFs with section awareness...")
    processor = AdvancedPDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = processor.process_pdf(pdf_file)
        all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    
    # Step 2: Generate embeddings
    print(f"\n📋 Step 2: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_chunks(all_chunks)
    
    # Step 3: Build vector database
    print(f"\n📋 Step 3: Building vector database...")
    vector_store = VectorStore(embedding_dimension=384)
    vector_store.add_embeddings(embeddings, all_chunks)
    
    # Step 4: Save
    print(f"\n📋 Step 4: Saving...")
    
    # Backup old database if exists
    if os.path.exists("vector_db"):
        import shutil
        if os.path.exists("vector_db_backup"):
            shutil.rmtree("vector_db_backup")
        shutil.move("vector_db", "vector_db_backup")
        print("💾 Old database backed up to: vector_db_backup/")
    
    vector_store.save("vector_db")
    
    # Stats
    stats = vector_store.get_stats()
    
    print("\n" + "="*70)
    print("✅ DATABASE REBUILD COMPLETE!")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   • Total chunks: {stats['total_chunks']}")
    print(f"   • Unique sources: {stats['unique_sources']}")
    print(f"   • Saved to: vector_db/")
    print(f"   • Backup: vector_db_backup/")
    print()

if __name__ == "__main__":
    rebuild_database()