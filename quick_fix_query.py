"""
Quick fix: Test without query rewriting
"""

from rag_system_pro import ProductionRAGSystem

# Initialize with query rewriting DISABLED
rag = ProductionRAGSystem(
    vector_db_path="vector_db",
    model_name="llama3.2",
    top_k=6,
    use_hybrid=True,
    use_reranking=True,
    use_query_rewriting=False  # ← DISABLE THIS
)

# Test
response = rag.ask("What is described in the abstract?", verbose=True)

print(f"\n💬 Answer: {response['answer']}")
print(f"📊 Confidence: {response['confidence_label']}")