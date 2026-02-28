"""
LLM Handler Module
Manages interaction with Ollama (Llama) for answer generation
"""

import subprocess
import json
from typing import List, Dict, Optional
import sys
import io

class OllamaLLM:
    """
    Handler for Ollama local LLM
    """
    
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.1):
        """
        Initialize Ollama LLM
        
        Args:
            model_name: Name of Ollama model to use
            temperature: Sampling temperature (0.0 = focused, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Verify Ollama is available
        if not self._check_ollama_available():
            raise RuntimeError(
                "❌ Ollama not available!\n"
                "Make sure:\n"
                "1. Ollama is installed\n"
                "2. Ollama service is running (ollama serve)\n"
                "3. Model is downloaded (ollama pull llama3.2)"
            )
        
        print(f"✅ Ollama LLM initialized (model={model_name}, temp={temperature})")
    
    def _check_ollama_available(self) -> bool:
        """
        Check if Ollama is running and model is available
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',  # Force UTF-8 encoding
                errors='replace'   # Replace problematic characters
            )
            
            if result.returncode != 0:
                return False
            
            # Check if our model is in the list
            if self.model_name.split(':')[0] in result.stdout:
                return True
            else:
                print(f"⚠️  Model '{self.model_name}' not found")
                print(f"Available models:\n{result.stdout}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate response from Ollama
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            # Call Ollama via subprocess with proper encoding
            result = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',      # Force UTF-8 encoding
                errors='replace'       # Replace problematic characters
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Clean up any replacement characters
                response = response.replace('\ufffd', '')  # Remove replacement chars
                return response
            else:
                error_msg = result.stderr.replace('\ufffd', '')
                return f"Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "Error: LLM response timeout (>60s)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_with_context(
        self, 
        question: str, 
        context_chunks: List[str],
        max_tokens: int = 500
    ) -> str:
        """
        Generate answer based on retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved text chunks
            max_tokens: Maximum response length
            
        Returns:
            Generated answer
        """
        # Build the prompt
        prompt = self._build_rag_prompt(question, context_chunks)
        
        # Generate response
        response = self.generate(prompt, max_tokens)
        
        return response
    
    def _build_rag_prompt(self, question: str, context_chunks: List[str]) -> str:
        """
        Build RAG prompt template with improved handling for broad questions
        """
        # Combine context chunks
        context = "\n\n".join([
            f"[Context {i+1}]\n{chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Detect if it's a broad/summary question
        broad_question_indicators = [
            "what is this paper about",
            "what is the paper about",
            "summarize",
            "overview",
            "main topic",
            "main focus",
        ]
        
        is_broad = any(indicator in question.lower() for indicator in broad_question_indicators)
        
        if is_broad:
            # Use summary-focused prompt
            prompt = f"""You are a helpful research assistant. Answer the question by synthesizing information from the provided context.

    CONTEXT FROM RESEARCH PAPER:
    {context}

    QUESTION:
    {question}

    INSTRUCTIONS:
    - Synthesize the main ideas from the context
    - Focus on the abstract, introduction, or conclusion sections if present
    - Provide a brief, coherent summary
    - If the context doesn't contain overview information, describe what IS in the context

    ANSWER:"""
        else:
            # Use fact-finding prompt (original)
            prompt = f"""You are a helpful research assistant. Your task is to answer questions based ONLY on the provided context from research papers.

    IMPORTANT RULES:
    1. Answer ONLY using information from the context below
    2. If the answer is not in the context, say "This information is not mentioned in the provided context"
    3. Be concise and direct
    4. Do not make up information
    5. If the context is unclear, say so

    CONTEXT FROM RESEARCH PAPERS:
    {context}

    QUESTION:
    {question}

    ANSWER:"""
        
        return prompt


# ==============================================================================
# TESTING CODE
# ==============================================================================

if __name__ == "__main__":
    """
    Test Ollama integration
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING OLLAMA LLM")
    print("="*70 + "\n")
    
    try:
        # Initialize
        llm = OllamaLLM(model_name="llama3.2", temperature=0.1)
        
        # Test 1: Simple generation
        print("--- Test 1: Simple Generation ---\n")
        response = llm.generate("Say hello in one sentence.")
        print(f"Response: {response}\n")
        
        # Test 2: RAG-style generation
        print("--- Test 2: RAG Generation ---\n")
        
        context_chunks = [
            "We trained our model on the ImageNet dataset, which contains 1.2 million images across 1000 categories.",
            "The model achieved 95.3% accuracy on the test set after 100 epochs of training.",
            "We used a ResNet-50 architecture with batch normalization and dropout layers."
        ]
        
        question = "What dataset was used for training?"
        
        print(f"Question: {question}\n")
        print("Generating answer...")
        
        answer = llm.generate_with_context(question, context_chunks)
        
        print(f"\nAnswer: {answer}\n")
        
        # Test 3: Multiple questions
        print("--- Test 3: Multiple Questions ---\n")
        
        test_questions = [
            "What accuracy was achieved?",
            "What architecture was used?",
            "How many epochs were used for training?"
        ]
        
        for q in test_questions:
            print(f"Q: {q}")
            a = llm.generate_with_context(q, context_chunks)
            print(f"A: {a}\n")
        
        print("✅ All Ollama LLM tests passed!")
        
    except RuntimeError as e:
        print(f"\n{e}")
        print("\n💡 Troubleshooting:")
        print("1. Open a new terminal")
        print("2. Run: ollama serve")
        print("3. Keep that terminal open")
        print("4. Run this test again")