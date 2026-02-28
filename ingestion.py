"""
PDF Ingestion Module
Handles loading PDFs, extracting text, and chunking for RAG system
"""

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import os

class PDFProcessor:
    """
    Processes PDF files and converts them into chunks suitable for RAG
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Maximum size of each text chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters (last resort)
            ]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Load a PDF file and extract text with metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Loading PDF: {pdf_path}")
        
        # Extract filename without extension
        filename = os.path.basename(pdf_path)
        
        # Read PDF
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"📊 Total pages: {total_pages}")
        
        # Extract text from each page
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            if text.strip():  # Only add if page has content
                pages_data.append({
                    "text": text,
                    "page_number": page_num,
                    "source": filename
                })
        
        print(f"✅ Extracted text from {len(pages_data)} pages")
        
        return pages_data
    
    def chunk_documents(self, pages_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split page texts into smaller chunks with metadata
        
        Args:
            pages_data: List of page data from load_pdf()
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        
        print(f"✂️  Chunking documents (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        for page_data in pages_data:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(page_data["text"])
            
            # Add metadata to each chunk
            for i, chunk in enumerate(text_chunks):
                all_chunks.append({
                    "text": chunk,
                    "page_number": page_data["page_number"],
                    "source": page_data["source"],
                    "chunk_id": f"{page_data['source']}_page{page_data['page_number']}_chunk{i}"
                })
        
        print(f"✅ Created {len(all_chunks)} chunks")
        
        return all_chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete pipeline: Load PDF → Extract text → Create chunks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks ready for embedding
        """
        print("\n" + "="*60)
        print("🚀 Starting PDF Processing Pipeline")
        print("="*60 + "\n")
        
        # Step 1: Load PDF
        pages_data = self.load_pdf(pdf_path)
        
        # Step 2: Create chunks
        chunks = self.chunk_documents(pages_data)
        
        print("\n" + "="*60)
        print("✅ PDF Processing Complete!")
        print("="*60 + "\n")
        
        return chunks
    
    def display_chunk_sample(self, chunks: List[Dict[str, any]], num_samples: int = 3):
        """
        Display sample chunks for verification
        
        Args:
            chunks: List of chunks
            num_samples: Number of samples to display
        """
        print("\n" + "="*60)
        print(f"📋 Sample Chunks (showing {min(num_samples, len(chunks))} of {len(chunks)})")
        print("="*60 + "\n")
        
        for i, chunk in enumerate(chunks[:num_samples], 1):
            print(f"--- Chunk {i} ---")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Text length: {len(chunk['text'])} characters")
            print(f"Preview: {chunk['text'][:200]}...")
            print("\n")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def save_chunks_to_file(chunks: List[Dict[str, any]], output_path: str = "chunks_output.txt"):
    """
    Save chunks to a text file for inspection
    
    Args:
        chunks: List of chunks
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("="*80 + "\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            f.write(f"CHUNK {i}\n")
            f.write(f"Source: {chunk['source']}\n")
            f.write(f"Page: {chunk['page_number']}\n")
            f.write(f"Chunk ID: {chunk['chunk_id']}\n")
            f.write("-"*80 + "\n")
            f.write(chunk['text'])
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"💾 Chunks saved to: {output_path}")


# ==============================================================================
# TESTING CODE (Run this file directly to test)
# ==============================================================================

if __name__ == "__main__":
    """
    Test the PDF processor with a sample PDF
    """
    
    print("\n🧪 TESTING PDF PROCESSOR\n")
    
    # Initialize processor
    processor = PDFProcessor(
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Example: Process a PDF
    # Replace 'sample.pdf' with your actual PDF path
    pdf_file = "data/sample.pdf"  # PUT YOUR PDF HERE
    
    if os.path.exists(pdf_file):
        # Process the PDF
        chunks = processor.process_pdf(pdf_file)
        
        # Display samples
        processor.display_chunk_sample(chunks, num_samples=3)
        
        # Save to file for inspection
        save_chunks_to_file(chunks, "chunks_output.txt")
        
        print(f"\n✅ SUCCESS! Processed {len(chunks)} chunks from PDF")
        
    else:
        print(f"❌ Error: PDF file not found at '{pdf_file}'")
        print("\n📝 To test:")
        print("1. Download a research paper PDF")
        print("2. Save it to 'data/sample.pdf'")
        print("3. Run this script again: python ingestion.py")


from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

processor = PDFProcessor(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)