"""
Configuration file for RAG system
Adjust these parameters to tune performance
"""

# PDF Processing Settings
CHUNK_SIZE = 800         # Characters per chunk
CHUNK_OVERLAP = 150      # Overlap between chunks

# File Paths
DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"
OUTPUT_DIR = "outputs"

# Model Settings (for Day 3)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Settings (Ollama)
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.1    # Lower = more focused answers

# Retrieval Settings (for Day 4)
TOP_K_RESULTS = 3        # Number of chunks to retrieve