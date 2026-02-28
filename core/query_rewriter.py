"""
Query Rewriter to improve retrieval
"""

from core.llm_handler import OllamaLLM
from typing import List

class QueryRewriter:
    """
    Rewrites user queries to improve retrieval
    """
    
    def __init__(self, llm: OllamaLLM = None):
        """
        Initialize query rewriter
        """
        self.llm = llm or OllamaLLM(model_name="llama3.2", temperature=0.3)
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query
        """
        prompt = f"""You are a query expansion expert. Rewrite the following question to be more specific and retrievable from a research paper.

Original query: "{query}"

Rewrite the query to:
1. Be more specific
2. Include related technical terms
3. Match how research papers are written
4. Keep it concise (one sentence)

Rewritten query:"""
        
        rewritten = self.llm.generate(prompt, max_tokens=100)
        
        # Clean up
        rewritten = rewritten.strip().strip('"')
        
        return rewritten
    
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of the query
        
        Args:
            query: Original query
            num_queries: Number of variations
            
        Returns:
            List of query variations
        """
        prompt = f"""Generate {num_queries} different ways to ask the following question about a research paper:

Original: "{query}"

Generate {num_queries} variations that ask the same thing but with different wording.
Format: One per line, no numbering.

Variations:"""
        
        variations_text = self.llm.generate(prompt, max_tokens=150)
        
        # Parse variations
        variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
        variations = variations[:num_queries]
        
        # Always include original
        if query not in variations:
            variations.insert(0, query)
        
        return variations


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Test query rewriting
    """
    print("\n" + "="*70)
    print("🧪 TESTING QUERY REWRITING")
    print("="*70 + "\n")
    
    rewriter = QueryRewriter()
    
    test_queries = [
        "What is this paper about?",
        "What datasets?",
        "Main contribution?",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"📝 Original: {query}")
        print(f"{'─'*70}")
        
        # Single rewrite
        rewritten = rewriter.rewrite_query(query)
        print(f"✨ Rewritten: {rewritten}")
        
        # Multiple variations
        print(f"\n🔀 Variations:")
        variations = rewriter.generate_multi_queries(query, num_queries=3)
        for i, var in enumerate(variations, 1):
            print(f"   {i}. {var}")
        
        print()
    
    print("✅ Query rewriting test complete!")