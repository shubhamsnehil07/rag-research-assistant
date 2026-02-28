"""
Smart abstract extractor for IEEE-style PDFs
"""

from pypdf import PdfReader
import re
from typing import Optional, Dict

def extract_abstract_smart(pdf_path: str) -> Optional[Dict[str, str]]:
    """
    Extract abstract from IEEE/academic papers
    
    Returns:
        Dictionary with abstract text and metadata
    """
    reader = PdfReader(pdf_path)
    
    # Check first page
    if len(reader.pages) == 0:
        return None
    
    first_page_text = reader.pages[0].extract_text()
    
    # Pattern 1: "Abstract—" or "Abstract -" (IEEE style)
    pattern1 = r'Abstract\s*[—\-–]\s*(.+?)(?=\n\s*\n|\bIndex Terms\b|\bI\.\s+INTRODUCTION\b)'
    
    # Pattern 2: "Abstract\n" followed by text
    pattern2 = r'Abstract\s*\n\s*(.+?)(?=\n\s*\n|\bIndex Terms\b|\bI\.\s+INTRODUCTION\b)'
    
    # Pattern 3: Just find text after "Abstract" keyword
    pattern3 = r'(?i)abstract[:\-—\s]+(.{100,2000}?)(?=\n\s*\n\s*(?:Index Terms|Introduction|I\.))'
    
    for pattern in [pattern1, pattern2, pattern3]:
        match = re.search(pattern, first_page_text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract_text = match.group(1).strip()
            
            # Clean up
            abstract_text = re.sub(r'\s+', ' ', abstract_text)  # Normalize whitespace
            abstract_text = re.sub(r'\s*\n\s*', ' ', abstract_text)  # Remove newlines
            
            if len(abstract_text) > 50:  # Minimum length check
                return {
                    'text': abstract_text,
                    'page': 1,
                    'section': 'Abstract',
                    'confidence': 'high'
                }
    
    # Fallback: Take first substantial paragraph after "abstract"
    if 'abstract' in first_page_text.lower():
        lines = first_page_text.split('\n')
        found_abstract = False
        abstract_lines = []
        
        for line in lines:
            if 'abstract' in line.lower():
                found_abstract = True
                # Get text after "Abstract" on same line
                parts = re.split(r'abstract\s*[—\-–:]?\s*', line, flags=re.IGNORECASE)
                if len(parts) > 1:
                    abstract_lines.append(parts[1])
                continue
            
            if found_abstract:
                if line.strip():
                    abstract_lines.append(line.strip())
                else:
                    break  # Stop at first empty line
                
                # Stop at introduction or index terms
                if any(marker in line.lower() for marker in ['introduction', 'index terms', 'keywords']):
                    break
        
        if abstract_lines:
            abstract_text = ' '.join(abstract_lines)
            abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
            
            if len(abstract_text) > 50:
                return {
                    'text': abstract_text,
                    'page': 1,
                    'section': 'Abstract',
                    'confidence': 'medium'
                }
    
    return None


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    import glob
    
    pdfs = glob.glob("data/*.pdf")
    
    if pdfs:
        print("="*70)
        print("🔍 TESTING SMART ABSTRACT EXTRACTOR")
        print("="*70 + "\n")
        
        for pdf in pdfs:
            print(f"📄 Processing: {pdf}\n")
            
            result = extract_abstract_smart(pdf)
            
            if result:
                print(f"✅ Abstract found (confidence: {result['confidence']})")
                print(f"📖 Page: {result['page']}")
                print(f"📝 Section: {result['section']}")
                print(f"\n{'='*70}")
                print("EXTRACTED ABSTRACT:")
                print('='*70)
                print(result['text'])
                print('='*70)
            else:
                print("❌ No abstract found")
            
            print("\n")
    else:
        print("❌ No PDFs found")