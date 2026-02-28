"""
Simple script to ask questions to your research papers
Usage: python ask_question.py "Your question here"
"""

import sys
from rag_system import RAGSystem

def main():
    # Check if question provided
    if len(sys.argv) < 2:
        print("Usage: python ask_question.py \"Your question here\"")
        print("\nOr run without arguments for interactive mode")
        
        # Start interactive mode
        rag = RAGSystem()
        rag.interactive_mode()
        return
    
    # Get question from command line
    question = " ".join(sys.argv[1:])
    
    # Initialize RAG
    print("🔄 Loading RAG system...\n")
    rag = RAGSystem()
    
    # Get answer
    print(f"❓ Question: {question}\n")
    print("🔍 Searching and generating answer...\n")
    
    response = rag.ask(question)
    
    # Display result
    print("="*70)
    print("💬 ANSWER:")
    print("="*70)
    print(response['answer'])
    print()
    
    print("="*70)
    print("📚 SOURCES:")
    print("="*70)
    for i, source in enumerate(response['sources'], 1):
        print(f"{i}. {source['source']} (Page {source['page']}) - Relevance: {source['relevance_score']:.1f}%")
    print()

if __name__ == "__main__":
    main()