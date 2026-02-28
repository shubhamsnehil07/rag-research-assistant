"""
Production-Grade RAG System with All Upgrades
Combines: Section-aware chunking + Hybrid retrieval + Reranking + Query rewriting
"""

from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
from core.llm_handler import OllamaLLM
from core.hybrid_retriever import HybridRetriever
from core.reranker import CrossEncoderReranker
from core.query_rewriter import QueryRewriter
from core.citation_manager import CitationManager, AnswerQualityChecker
from typing import List, Dict
import os

class ProductionRAGSystem:
    """
    Production-grade RAG with all recommended upgrades
    """
    
    def __init__(
        self,
        vector_db_path: str = "vector_db",
        model_name: str = "llama3.2",
        top_k: int = 8,  # Increased from 3
        temperature: float = 0.1,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_query_rewriting: bool = True
    ):
        """
        Initialize production RAG system
        """
        print("\n" + "="*70)
        print("🚀 INITIALIZING PRODUCTION RAG SYSTEM")
        print("="*70 + "\n")
        
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_query_rewriting = use_query_rewriting
        
        # Load core components
        print("📋 Loading core components...")
        self.embedder = EmbeddingGenerator()
        
        self.vector_store = VectorStore(embedding_dimension=384)
        if not os.path.exists(f"{vector_db_path}/faiss.index"):
            raise FileNotFoundError(f"Vector database not found at '{vector_db_path}'")
        self.vector_store.load(vector_db_path)
        
        self.llm = OllamaLLM(model_name=model_name, temperature=temperature)
        
        # Initialize advanced components
        if use_hybrid:
            print("\n🔀 Initializing hybrid retrieval...")
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                bm25_weight=0.4,
                dense_weight=0.6
            )
        
        if use_reranking:
            print("\n🎯 Initializing reranker...")
            self.reranker = CrossEncoderReranker()
        
        if use_query_rewriting:
            print("\n✍️  Initializing query rewriter...")
            self.query_rewriter = QueryRewriter(llm=self.llm)
        
        self.quality_checker = AnswerQualityChecker()
        
        # Stats
        stats = self.vector_store.get_stats()
        
        print("\n" + "="*70)
        print("✅ PRODUCTION RAG SYSTEM READY!")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   • Knowledge Base: {stats['total_chunks']} chunks from {stats['unique_sources']} source(s)")
        print(f"   • LLM: {model_name}")
        print(f"   • Top-K: {top_k}")
        print(f"   • Hybrid Retrieval: {'✅' if use_hybrid else '❌'}")
        print(f"   • Reranking: {'✅' if use_reranking else '❌'}")
        print(f"   • Query Rewriting: {'✅' if use_query_rewriting else '❌'}")
        print()
    
    def ask(
        self,
        question: str,
        include_citations: bool = True,
        include_confidence: bool = True,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Ask question with full production pipeline and abstract-specific handling
        """
        if verbose:
            print(f"\n🔍 Processing: '{question}'")
            
        # ==========================================
        # SPECIAL HANDLING FOR ABSTRACT QUESTIONS
        # ==========================================
        if 'abstract' in question.lower():
            if verbose:
                print("🎯 Detected abstract question - using direct retrieval")
            
            # Get abstract chunks directly
            abstract_chunks = [
                c for c in self.vector_store.chunks 
                if c.get('section') == 'Abstract'
            ]
            
            if abstract_chunks:
                if verbose:
                    print(f"✅ Found {len(abstract_chunks)} abstract chunk(s)")
                
                # Use abstract chunks directly
                sources = []
                for chunk in abstract_chunks:
                    sources.append({
                        "source": chunk['source'],
                        "page": chunk['page_number'],
                        "section": chunk.get('section', 'Abstract'),
                        "chunk_id": chunk['chunk_id'],
                        "text": chunk['text'],
                        "relevance_score": 100.0  # Perfect match
                    })
                
                # Generate answer
                context_chunks = [s['text'] for s in sources]
                raw_answer = self.llm.generate_with_context(question, context_chunks)
                
                # Build response
                citation_manager = CitationManager(citation_style="numeric")
                answer_with_citations = raw_answer
                
                if include_citations:
                    for i, source in enumerate(sources, 1):
                        citation_manager.add_citation(source)
                    
                    cite_markers = " " + " ".join([
                        citation_manager.format_inline_citation(i)
                        for i in range(1, len(sources) + 1)
                    ])
                    answer_with_citations = raw_answer.rstrip('.') + cite_markers + "."
                
                return {
                    "question": question,
                    "answer": answer_with_citations,
                    "raw_answer": raw_answer,
                    "search_query": question,
                    "confidence_label": "HIGH",
                    "confidence_score": 0.95,
                    "sources": sources,
                    "references": citation_manager.format_references(detailed=True) if include_citations else "",
                    "citation_count": len(sources)
                }
            else:
                if verbose:
                    print("⚠️  No abstract chunks found in database. Falling back to normal retrieval...")

        # ==========================================
        # NORMAL RAG PIPELINE
        # ==========================================
        # Step 1: Query rewriting (optional)
        search_query = question
        if self.use_query_rewriting:
            if verbose:
                print("✍️  Rewriting query...")
            search_query = self.query_rewriter.rewrite_query(question)
            if verbose:
                print(f"   → Rewritten: '{search_query}'")
        
        # Step 2: Retrieval (hybrid or dense)
        if self.use_hybrid:
            if verbose:
                print(f"🔀 Hybrid retrieval (top-{self.top_k})...")
            results = self.hybrid_retriever.search(search_query, top_k=self.top_k * 2)
            chunks = [chunk for chunk, _ in results]
        else:
            if verbose:
                print(f"🎯 Dense retrieval (top-{self.top_k})...")
            query_embedding = self.embedder.embed_text(search_query)
            results = self.vector_store.search(query_embedding, top_k=self.top_k * 2)
            chunks = [chunk for chunk, _ in results]
        
        # Step 3: Reranking (optional)
        if self.use_reranking and chunks:
            if verbose:
                print(f"🎯 Reranking with cross-encoder...")
            reranked = self.reranker.rerank(question, chunks, top_k=self.top_k)
            chunks = [chunk for chunk, _ in reranked]
        else:
            chunks = chunks[:self.top_k]
        
        if verbose:
            print(f"✅ Using {len(chunks)} chunks for generation")
            for i, chunk in enumerate(chunks[:3], 1):
                section = chunk.get('section', 'Unknown')
                print(f"   {i}. Section: {section}, Page: {chunk['page_number']}")
        
        # Step 4: Prepare sources
        sources = []
        for chunk in chunks:
            sources.append({
                "source": chunk['source'],
                "page": chunk['page_number'],
                "section": chunk.get('section', 'Unknown'),
                "chunk_id": chunk['chunk_id'],
                "text": chunk['text'],
                "relevance_score": 85.0  # Placeholder (reranker doesn't return scores)
            })
        
        # Step 5: Generate answer
        context_chunks = [s['text'] for s in sources]
        raw_answer = self.llm.generate_with_context(question, context_chunks)
        
        # Step 6: Add citations
        citation_manager = CitationManager(citation_style="numeric")
        answer_with_citations = raw_answer
        
        if include_citations:
            for i, source in enumerate(sources, 1):
                citation_manager.add_citation(source)
            
            if not self.quality_checker.detect_uncertainty(raw_answer):
                cite_markers = " " + " ".join([
                    citation_manager.format_inline_citation(i)
                    for i in range(1, min(4, len(sources) + 1))  # First 3 citations
                ])
                answer_with_citations = raw_answer.rstrip('.') + cite_markers + "."
        
        # Step 7: Calculate confidence
        confidence_label, confidence_score = None, None
        if include_confidence:
            confidence_label, confidence_score = self.quality_checker.calculate_confidence(
                raw_answer, sources
            )
        
        # Step 8: Build response
        response = {
            "question": question,
            "answer": answer_with_citations,
            "raw_answer": raw_answer,
            "search_query": search_query if self.use_query_rewriting else question,
            "confidence_label": confidence_label,
            "confidence_score": confidence_score,
            "sources": sources,
        }
        
        if include_citations:
            response["references"] = citation_manager.format_references(detailed=True)
            response["citation_count"] = citation_manager.get_citation_count()
        
        return response
    
    def interactive_mode(self):
        """Interactive Q&A"""
        print("\n" + "="*70)
        print("💬 PRODUCTION RAG - INTERACTIVE MODE")
        print("="*70)
        print("\n💡 Commands: 'quit' to exit, 'verbose' to toggle debug info")
        print("="*70 + "\n")
        
        verbose = False
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'verbose':
                verbose = not verbose
                print(f"✅ Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            if not question:
                continue
            
            # Get answer
            response = self.ask(question, verbose=verbose)
            
            # Display
            print("\n" + "─"*70)
            print("💬 ANSWER:")
            print("─"*70)
            print(response['answer'])
            
            if response['confidence_label']:
                print(f"\n📊 Confidence: {response['confidence_label']} ({response['confidence_score']:.0%})")
            
            if 'references' in response:
                print("\n" + "─"*70)
                print("📚 REFERENCES:")
                print("─"*70)
                print(response['references'])
            
            print("\n" + "="*70)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    Test production RAG system
    """
    
    try:
        # Initialize production system
        rag = ProductionRAGSystem(
            vector_db_path="vector_db",
            model_name="llama3.2",
            top_k=6,
            use_hybrid=True,
            use_reranking=True,
            use_query_rewriting=True
        )
        
        # Test with problematic questions
        print("\n" + "="*70)
        print("🧪 TESTING PRODUCTION RAG")
        print("="*70)
        
        test_questions = [
            "What is this paper about?",
            "What is described in the abstract?",
            "What are the main objectives?",
            "What datasets were used?",
        ]
        
        for question in test_questions:
            print(f"\n{'═'*70}")
            response = rag.ask(question, verbose=True)
            print(f"\n💬 Answer: {response['answer']}")
            print(f"📊 Confidence: {response['confidence_label']} ({response['confidence_score']:.0%})")
            print(f"{'═'*70}")
            input("\nPress Enter for next question...")
        
        # Interactive mode
        print("\n💡 Start interactive mode? (y/n)")
        if input("> ").strip().lower() == 'y':
            rag.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()