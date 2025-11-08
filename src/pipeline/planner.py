"""
Planner module: Decides if retrieval is needed based on question analysis.
"""
from typing import Dict, Any
from src.config import RETRIEVAL_KEYWORDS
from src.pipeline.utils import setup_logger
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logger("planner", LOG_FILE, LOG_LEVEL)


def plan_retrieval(question: str) -> Dict[str, Any]:
    """
    Decide whether retrieval is needed based on the question.
    
    Args:
        question: User's question string
        
    Returns:
        Dictionary with 'retrieval_needed' boolean and 'reason' string
    """
    question_lower = question.lower().strip()
    
    # Check if question contains retrieval keywords
    contains_keyword = any(keyword in question_lower for keyword in RETRIEVAL_KEYWORDS)
    
    # Check if question is too short (likely doesn't need retrieval)
    is_too_short = len(question_lower.split()) < 3
    
    # Decision logic
    if is_too_short:
        retrieval_needed = False
        reason = "Question is too short, likely a simple query"
    elif contains_keyword:
        retrieval_needed = True
        reason = "Question contains retrieval keywords (what, how, explain, etc.)"
    else:
        # Default to retrieval for most questions to be safe
        retrieval_needed = True
        reason = "Default: Retrieval enabled for comprehensive answers"
    
    result = {
        "retrieval_needed": retrieval_needed,
        "reason": reason,
        "question": question
    }
    
    logger.info(f"Planner decision: retrieval_needed={retrieval_needed}, reason={reason}")
    
    return result
