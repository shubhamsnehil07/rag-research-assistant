"""
Process multiple questions from a file or list
"""

from rag_system import RAGSystem
import json

def process_questions_from_file(questions_file: str, output_file: str = "answers.json"):
    """
    Read questions from file and save answers
    
    Format of questions.txt:
    What is the main contribution?
    What dataset was used?
    What accuracy was achieved?
    """
    # Load questions
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Loaded {len(questions)} questions from {questions_file}\n")
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Process all questions
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing: {question}")
        response = rag.ask(question)
        results.append(response)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved {len(results)} answers to {output_file}")
    
    # Display summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70 + "\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Q: {result['question']}")
        print(f"   A: {result['answer'][:100]}...")
        print()

if __name__ == "__main__":
    # Example: Create a questions file
    sample_questions = [
        "What is the main contribution of this paper?",
        "What dataset was used in the experiments?",
        "What model architecture is proposed?",
        "What are the key results?",
        "What are the limitations mentioned?"
    ]
    
    # Save sample questions
    with open("sample_questions.txt", 'w') as f:
        f.write("\n".join(sample_questions))
    
    print("📝 Created sample_questions.txt")
    print("Processing questions...\n")
    
    # Process them
    process_questions_from_file("sample_questions.txt")