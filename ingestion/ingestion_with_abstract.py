"""
PDF processor with guaranteed abstract extraction
"""

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import re
import os
from ingestion.extract_abstract import extract_abstract_smart

class PDFProcessorWithAbstract:
    """
    PDF processor that guarantees abstract is properly labeled
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Process PDF with guaranteed abstract extraction
        """
        print("\n" + "="*60)
        print("🚀 PROCESSING PDF WITH ABSTRACT DETECTION")
        print("="*60 + "\n")
        
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        
        print(f"📄 File: {filename}")
        print(f"📊 Pages: {len(reader.pages)}\n")
        
        all_chunks = []
        
        # Step 1: Extract abstract specifically
        print("🔍 Step 1: Extracting abstract...")
        abstract_data = extract_abstract_smart(pdf_path)
        
        if abstract_data:
            print(f"✅ Abstract found ({len(abstract_data['text'])} chars)")
            
            # Create abstract chunk
            all_chunks.append({
                'text': abstract_data['text'],
                'page_number': 1,
                'source': filename,
                'section': 'Abstract',
                'chunk_id': f"{filename}_abstract_0"
            })
        else:
            print("⚠️  No abstract detected")
        
        # Step 2: Process remaining content
        print("\n📄 Step 2: Processing full document...")
        
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            
            # Skip if this is page 1 and we already got abstract
            # (to avoid duplication)
            if page_num == 1 and abstract_data:
                # Remove abstract from page text
                page_text = re.sub(
                    r'Abstract.+?(?=Index Terms|Introduction|I\.)',
                    '',
                    page_text,
                    flags=re.DOTALL | re.IGNORECASE
                )
            
            if not page_text.strip():
                continue
            
            # Detect section for this page (simplified)
            section = self._detect_section_simple(page_text, page_num)
            
            # Chunk the text
            text_chunks = self.text_splitter.split_text(page_text)
            
            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 50:  # Skip tiny chunks
                    continue
                
                all_chunks.append({
                    'text': chunk_text,
                    'page_number': page_num,
                    'source': filename,
                    'section': section,
                    'chunk_id': f"{filename}_page{page_num}_chunk{i}"
                })
        
        print(f"\n✅ Created {len(all_chunks)} total chunks")
        
        # Show section distribution
        from collections import Counter
        section_counts = Counter(c['section'] for c in all_chunks)
        
        print(f"\n📊 Chunks per section:")
        for section, count in section_counts.most_common():
            print(f"   {section}: {count} chunks")
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60 + "\n")
        
        return all_chunks
    
    def _detect_section_simple(self, text: str, page_num: int) -> str:
        """Simple section detection based on keywords and page number"""
        text_lower = text.lower()
        
        # Check for keywords
        if 'introduction' in text_lower and page_num <= 3:
            return 'Introduction'
        elif any(word in text_lower for word in ['method', 'approach', 'algorithm']):
            return 'Methodology'
        elif 'experiment' in text_lower or 'result' in text_lower:
            return 'Results'
        elif 'conclusion' in text_lower:
            return 'Conclusion'
        elif 'reference' in text_lower:
            return 'References'
        else:
            return 'Body'


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    import glob
    
    processor = PDFProcessorWithAbstract()
    
    pdfs = glob.glob("data/*.pdf")
    
    if pdfs:
        # Process first PDF
        chunks = processor.process_pdf(pdfs[0])
        
        # Show abstract chunks
        print("\n" + "="*60)
        print("📋 ABSTRACT CHUNKS:")
        print("="*60 + "\n")
        
        for chunk in chunks:
            if chunk['section'] == 'Abstract':
                print(f"✅ Found Abstract chunk:")
                print(f"   Length: {len(chunk['text'])} chars")
                print(f"   Text: {chunk['text'][:200]}...")
                print()