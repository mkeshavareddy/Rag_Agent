"""
Metrics module: Evaluation metrics for answer quality (BLEU, ROUGE, etc.).
"""
from typing import Dict, Any, List, Optional
import numpy as np
from src.pipeline.utils import setup_logger
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logger("metrics", LOG_FILE, LOG_LEVEL)


def calculate_bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """
    Calculate BLEU score between reference and candidate text.
    Simplified implementation using n-gram overlap.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        n: Maximum n-gram order (default: 4)
        
    Returns:
        BLEU score between 0 and 1
    """
    def get_ngrams(text: str, n: int) -> List[tuple]:
        """Get n-grams from text."""
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return ngrams
    
    ref_ngrams = []
    cand_ngrams = []
    
    for i in range(1, n + 1):
        ref_ngrams.append(set(get_ngrams(reference, i)))
        cand_ngrams.append(set(get_ngrams(candidate, i)))
    
    # Calculate precision for each n-gram order
    precisions = []
    for ref_ngram, cand_ngram in zip(ref_ngrams, cand_ngrams):
        if len(cand_ngram) == 0:
            precisions.append(0.0)
        else:
            matches = len(ref_ngram.intersection(cand_ngram))
            precisions.append(matches / len(cand_ngram))
    
    # Brevity penalty
    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Calculate BLEU
    if all(p > 0 for p in precisions):
        bleu = brevity_penalty * np.exp(np.mean(np.log(precisions)))
    else:
        bleu = 0.0
    
    return float(bleu)


def calculate_rouge_l_score(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence based).
    Simplified implementation.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        ROUGE-L score between 0 and 1
    """
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if len(ref_words) == 0 or len(cand_words) == 0:
        return 0.0
    
    lcs_len = lcs_length(ref_words, cand_words)
    
    # ROUGE-L = LCS length / reference length
    rouge_l = lcs_len / len(ref_words)
    
    return rouge_l


def calculate_rouge_n_score(reference: str, candidate: str, n: int = 2) -> float:
    """
    Calculate ROUGE-N score (n-gram recall).
    
    Args:
        reference: Reference text
        candidate: Candidate text
        n: N-gram order (default: 2 for bigrams)
        
    Returns:
        ROUGE-N score between 0 and 1
    """
    def get_ngrams(text: str, n: int) -> List[tuple]:
        """Get n-grams from text."""
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return ngrams
    
    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)
    
    if len(ref_ngrams) == 0:
        return 0.0
    
    ref_ngram_set = set(ref_ngrams)
    cand_ngram_set = set(cand_ngrams)
    
    matches = len(ref_ngram_set.intersection(cand_ngram_set))
    rouge_n = matches / len(ref_ngram_set)
    
    return rouge_n


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using simple word overlap.
    For production, consider using sentence transformers or other embeddings.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    return jaccard_similarity


def evaluate_answer_quality(
    question: str,
    answer: str,
    reference_answer: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of answer quality.
    
    Args:
        question: Original question
        answer: Generated answer
        reference_answer: Reference answer (optional, for comparison)
        context: Retrieved context (optional)
        
    Returns:
        Dictionary with various metrics
    """
    metrics = {
        "answer_length": len(answer),
        "answer_word_count": len(answer.split()),
    }
    
    # If reference answer is provided, calculate comparison metrics
    if reference_answer:
        metrics["bleu_score"] = calculate_bleu_score(reference_answer, answer)
        metrics["rouge_l"] = calculate_rouge_l_score(reference_answer, answer)
        metrics["rouge_2"] = calculate_rouge_n_score(reference_answer, answer, n=2)
        metrics["semantic_similarity"] = calculate_semantic_similarity(reference_answer, answer)
    
    # Check answer completeness
    if context:
        context_similarity = calculate_semantic_similarity(answer, context)
        metrics["context_relevance"] = context_similarity
    else:
        metrics["context_relevance"] = 0.0
    
    # Check for "I don't know" indicators
    answer_lower = answer.lower()
    insufficient_indicators = [
        "i don't have",
        "i do not have",
        "not enough information",
        "insufficient",
        "cannot answer",
        "unable to",
        "no information"
    ]
    metrics["has_insufficient_info"] = any(
        indicator in answer_lower for indicator in insufficient_indicators
    )
    
    return metrics
