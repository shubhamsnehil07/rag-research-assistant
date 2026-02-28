"""
Directly find and extract abstract from PDF
"""

from pypdf import PdfReader
import re
import glob

def find_abstract(pdf_path: str):
    """
    Find abstract text directly
    """
    print(f"🔍 Searching for abstract in: {pdf_path}\n")
    
    reader = PdfReader(pdf_path)
    
    # Check first 3 pages
    for page_num in range(min(3, len(reader.pages))):
        page = reader.pages[page_num]
        text = page.extract_text()
        
        # Look for "Abstract" keyword
        if re.search(r'\babstract\b', text, re.IGNORECASE):
            print(f"✅ Found 'abstract' on page {page_num + 1}\n")
            
            # Extract text after "Abstract" heading
            # Try to find abstract section
            match = re.search(
                r'\babstract\b(.{20,2000})(?:\n\s*\n|\bintroduction\b|\b1\.)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            
            if match:
                abstract_text = match.group(1).strip()
                print("="*60)
                print("���� EXTRACTED ABSTRACT:")
                print("="*60)
                print(abstract_text)
                print("="*60)
                return abstract_text
            else:
                # Just show context around "abstract"
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if re.search(r'\babstract\b', line, re.IGNORECASE):
                        print("="*60)
                        print("📄 TEXT AROUND 'ABSTRACT':")
                        print("="*60)
                        context = '\n'.join(lines[i:min(i+20, len(lines))])
                        print(context)
                        print("="*60)
                        return context
    
    print("❌ No abstract found in first 3 pages")
    print("\n💡 Your PDF might:")
    print("  1. Not have an abstract section")
    print("  2. Use a different language/term")
    print("  3. Have the abstract in an image (scanned PDF)")
    
    return None

# Test
pdfs = glob.glob("data/*.pdf")
if pdfs:
    abstract = find_abstract(pdfs[0])
else:
    print("❌ No PDFs found in data/")