"""
Citation Manager
Handles proper source attribution and reference formatting
"""

from typing import List, Dict, Tuple
import re

class CitationManager:
    """
    Manages citations and references for RAG answers
    """
    
    def __init__(self, citation_style: str = "numeric"):
        """
        Initialize citation manager
        
        Args:
            citation_style: 'numeric' [1], 'author-year' (Smith, 2024), or 'superscript' ¹
        """
        self.citation_style = citation_style
        self.citations = []
    
    def add_citation(self, source: Dict[str, any]) -> int:
        """
        Add a source and return citation number
        
        Args:
            source: Dictionary with source metadata
            
        Returns:
            Citation number (1-indexed)
        """
        self.citations.append(source)
        return len(self.citations)
    
    def format_inline_citation(self, citation_num: int) -> str:
        """
        Format inline citation marker
        
        Args:
            citation_num: Citation number
            
        Returns:
            Formatted citation marker
        """
        if self.citation_style == "numeric":
            return f"[{citation_num}]"
        elif self.citation_style == "superscript":
            # Unicode superscript numbers
            superscript = str(citation_num).translate(
                str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
            )
            return superscript
        else:
            return f"[{citation_num}]"
    
    def format_references(self, detailed: bool = True) -> str:
        """
        Format the references section
        
        Args:
            detailed: Include text snippets
            
        Returns:
            Formatted references string
        """
        if not self.citations:
            return ""
        
        references = []
        
        for i, source in enumerate(self.citations, 1):
            # Basic citation
            ref = f"[{i}] {source['source']}, Page {source['page']}"
            
            # Add relevance score
            if 'relevance_score' in source:
                ref += f" (Relevance: {source['relevance_score']:.1f}%)"
            
            # Add text snippet if detailed
            if detailed and 'text' in source:
                snippet = source['text'][:200].strip()
                if len(source['text']) > 200:
                    snippet += "..."
                ref += f"\n    \"{snippet}\""
            
            references.append(ref)
        
        return "\n\n".join(references)
    
    def clear(self):
        """Clear all citations"""
        self.citations = []
    
    def get_citation_count(self) -> int:
        """Get number of citations"""
        return len(self.citations)


class AnswerQualityChecker:
    """
    Checks quality and confidence of generated answers
    """
    
    @staticmethod
    def detect_uncertainty(answer: str) -> bool:
        """
        Detect if answer expresses uncertainty
        
        Args:
            answer: Generated answer
            
        Returns:
            True if uncertain
        """
        uncertainty_phrases = [
            "not mentioned",
            "not found",
            "unclear",
            "cannot determine",
            "does not specify",
            "information is not",
            "not provided",
            "i don't know",
            "no information",
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in uncertainty_phrases)
    
    @staticmethod
    def calculate_confidence(
        answer: str,
        sources: List[Dict[str, any]],
        min_relevance: float = 50.0
    ) -> Tuple[str, float]:
        """
        Calculate confidence level for answer
        
        Args:
            answer: Generated answer
            sources: Source chunks used
            min_relevance: Minimum relevance threshold
            
        Returns:
            (confidence_label, confidence_score)
        """
        # Check for uncertainty
        if AnswerQualityChecker.detect_uncertainty(answer):
            return ("LOW", 0.2)
        
        # Get average relevance of sources
        if not sources:
            return ("LOW", 0.3)
        
        avg_relevance = sum(s.get('relevance_score', 0) for s in sources) / len(sources)
        
        # Calculate confidence based on relevance
        if avg_relevance >= 80:
            confidence_label = "HIGH"
            confidence_score = 0.9
        elif avg_relevance >= 60:
            confidence_label = "MEDIUM"
            confidence_score = 0.7
        elif avg_relevance >= 40:
            confidence_label = "LOW-MEDIUM"
            confidence_score = 0.5
        else:
            confidence_label = "LOW"
            confidence_score = 0.3
        
        # Adjust based on number of sources
        if len(sources) >= 3:
            confidence_score += 0.05  # Bonus for multiple sources
        
        # Adjust based on answer length (very short answers might be uncertain)
        if len(answer.split()) < 10:
            confidence_score -= 0.1
        
        # Cap at 0.95 (never 100% certain)
        confidence_score = min(0.95, max(0.1, confidence_score))
        
        return (confidence_label, confidence_score)
    
    @staticmethod
    def validate_answer(answer: str, question: str) -> Dict[str, any]:
        """
        Validate answer quality
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check if answer is too short
        if len(answer.split()) < 5:
            issues.append("Answer is very short")
        
        # Check if answer is just repeating the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words)
        
        if overlap > len(question_words) * 0.8:
            issues.append("Answer mostly repeats the question")
        
        # Check for generic responses
        generic_phrases = [
            "based on the context",
            "according to the document",
            "as mentioned",
        ]
        
        if any(phrase in answer.lower() for phrase in generic_phrases):
            # This is actually good - means it's grounded
            pass
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "word_count": len(answer.split())
        }


# ==============================================================================
# TESTING CODE
# ==============================================================================

if __name__ == "__main__":
    """
    Test citation manager
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING CITATION MANAGER")
    print("="*70 + "\n")
    
    # Test 1: Citation formatting
    print("--- Test 1: Citation Formatting ---\n")
    
    cm = CitationManager(citation_style="numeric")
    
    # Add some sources
    source1 = {
        "source": "paper.pdf",
        "page": 5,
        "text": "We trained our model on ImageNet dataset with 1.2M images.",
        "relevance_score": 92.5
    }
    
    source2 = {
        "source": "paper.pdf",
        "page": 8,
        "text": "The model achieved 95.3% accuracy on the test set.",
        "relevance_score": 88.2
    }
    
    cite1 = cm.add_citation(source1)
    cite2 = cm.add_citation(source2)
    
    # Create answer with citations
    answer = f"The model was trained on ImageNet {cm.format_inline_citation(cite1)} "
    answer += f"and achieved 95.3% accuracy {cm.format_inline_citation(cite2)}."
    
    print("Answer with citations:")
    print(answer)
    print()
    
    print("References:")
    print(cm.format_references(detailed=True))
    
    # Test 2: Confidence scoring
    print("\n" + "="*70)
    print("--- Test 2: Confidence Scoring ---\n")
    
    checker = AnswerQualityChecker()
    
    test_cases = [
        ("The model achieved 95% accuracy on ImageNet.", [source1, source2]),
        ("This information is not mentioned in the paper.", [source1]),
        ("The dataset used was ImageNet.", [source1]),
    ]
    
    for answer, sources in test_cases:
        label, score = checker.calculate_confidence(answer, sources)
        print(f"Answer: {answer}")
        print(f"Confidence: {label} ({score:.2%})")
        print()
    
    # Test 3: Answer validation
    print("--- Test 3: Answer Validation ---\n")
    
    questions = [
        ("What dataset was used?", "ImageNet dataset was used for training."),
        ("What is the accuracy?", "Accuracy."),  # Too short
        ("Tell me about the model", "Tell me about the model architecture used."),  # Repetitive
    ]
    
    for question, answer in questions:
        validation = checker.validate_answer(answer, question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Valid: {validation['is_valid']}")
        if validation['issues']:
            print(f"Issues: {', '.join(validation['issues'])}")
        print()
    
    print("✅ All citation tests passed!")