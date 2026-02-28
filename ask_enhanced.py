"""
Simple enhanced Q&A script with citations
"""

import sys
from rag_system_enhanced import EnhancedRAGSystem

def main():
    """Main entry point"""
    
    # Initialize system
    print("🔄 Loading Enhanced RAG System...\n")
    
    rag = EnhancedRAGSystem(
        vector_db_path="vector_db",
        model_name="llama3.2",
        top_k=3,
        citation_style="numeric"
    )
    
    # Check if question provided via command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        
        print(f"\n❓ Question: {question}\n")
        print("🔍 Generating answer with citations...\n")
        
        response = rag.ask(question)
        print(rag.format_response(response))
        
    else:
        # Interactive mode
        rag.interactive_mode()

if __name__ == "__main__":
    main()