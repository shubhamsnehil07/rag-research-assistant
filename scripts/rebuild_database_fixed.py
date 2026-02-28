"""
Rebuild database with abstract-aware processing
"""

from ingestion.ingestion_with_abstract import PDFProcessorWithAbstract
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
import glob
import os
import shutil

def rebuild_with_abstract():
    """
    Rebuild database ensuring abstracts are captured
    """
    print("\n" + "="*70)
    print("🔨 REBUILDING DATABASE (ABSTRACT-AWARE)")
    print("="*70 + "\n")
    
    # Find PDFs
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF(s)\n")
    
    # Process all PDFs
    processor = PDFProcessorWithAbstract(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    for pdf in pdf_files:
        chunks = processor.process_pdf(pdf)
        all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks from all PDFs: {len(all_chunks)}")
    
    # Check if we got abstracts
    abstract_chunks = [c for c in all_chunks if c['section'] == 'Abstract']
    print(f"✅ Abstract chunks: {len(abstract_chunks)}")
    
    # Generate embeddings
    print(f"\n🔢 Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_chunks(all_chunks)
    
    # Build vector store
    print(f"\n💾 Building vector database...")
    vector_store = VectorStore(embedding_dimension=384)
    vector_store.add_embeddings(embeddings, all_chunks)
    
    # Backup and save
    if os.path.exists("vector_db"):
        if os.path.exists("vector_db_backup"):
            shutil.rmtree("vector_db_backup")
        shutil.move("vector_db", "vector_db_backup")
        print("📦 Old database backed up")
    
    vector_store.save("vector_db")
    
    print("\n" + "="*70)
    print("✅ DATABASE REBUILD COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Stats:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Abstract chunks: {len(abstract_chunks)}")
    print(f"   Other chunks: {len(all_chunks) - len(abstract_chunks)}")
    print()

if __name__ == "__main__":
    rebuild_with_abstract()