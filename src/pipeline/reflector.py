"""
Reflector module: Evaluates answer quality and relevance using Groq LLM.
"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE
)
from src.pipeline.utils import setup_logger, calculate_similarity_score
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logger("reflector", LOG_FILE, LOG_LEVEL)


class AnswerReflector:
    """Evaluates answer quality and relevance using Groq LLM."""
    
    def __init__(self):
        """Initialize the reflector with Groq LLM."""
        self.use_llm_judge = False
        self.llm = None
        
        # Try to initialize Groq LLM judge
        if GROQ_API_KEY:
            try:
                self.llm = ChatGroq(
                    model=LLM_MODEL,
                    temperature=TEMPERATURE,
                    groq_api_key=GROQ_API_KEY
                )
                self.use_llm_judge = True
                logger.info("Initialized Groq LLM-as-judge for reflection")
            except Exception as e:
                logger.warning(f"Could not initialize Groq LLM judge: {e}, using keyword-based evaluation")
                self.use_llm_judge = False
        else:
            logger.info("No Groq API key found, using keyword-based evaluation")
    
    def reflect(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        """
        Evaluate the quality and relevance of an answer.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context (optional)
            
        Returns:
            Dictionary with 'verdict', 'score', and 'feedback'
        """
        if self.use_llm_judge:
            return self._llm_reflect(question, answer, context)
        else:
            return self._keyword_reflect(question, answer, context)
    
    def _llm_reflect(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Use LLM-as-judge to evaluate answer quality."""
        try:
            system_message = SystemMessage(content="""You are an evaluator that assesses answer quality.
Evaluate if the answer is relevant, complete, and accurate based on the question and context.
Return one of these verdicts:
- "relevant": Answer is relevant, complete, and accurate
- "maybe_irrelevant": Answer is partially relevant but may be incomplete or inaccurate
- "not_enough_context": Answer indicates insufficient context or information
- "irrelevant": Answer is clearly not relevant to the question

Provide a score from 0.0 to 1.0 and brief feedback.""")
            
            prompt_content = f"""QUESTION: {question}

ANSWER: {answer}

CONTEXT AVAILABLE: {'Yes' if context else 'No'}

Evaluate the answer and provide:
1. Verdict (one of: relevant, maybe_irrelevant, not_enough_context, irrelevant)
2. Score (0.0 to 1.0)
3. Brief feedback (1-2 sentences)

Format your response as:
VERDICT: <verdict>
SCORE: <score>
FEEDBACK: <feedback>"""
            
            human_message = HumanMessage(content=prompt_content)
            response = self.llm.invoke([system_message, human_message])
            
            # Parse response
            response_text = response.content if hasattr(response, 'content') else str(response)
            verdict, score, feedback = self._parse_llm_response(response_text)
            
            result = {
                "verdict": verdict,
                "score": score,
                "feedback": feedback,
                "method": "llm_judge"
            }
            
            logger.info(f"Reflection verdict: {verdict}, score: {score}")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM reflection: {e}, falling back to keyword-based")
            return self._keyword_reflect(question, answer, context)
    
    def _parse_llm_response(self, response_text: str) -> tuple:
        """Parse LLM response to extract verdict, score, and feedback."""
        verdict = "maybe_irrelevant"
        score = 0.5
        feedback = "Could not parse LLM response"
        
        lines = response_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('verdict:'):
                verdict_part = line.split(':', 1)[1].strip().lower()
                if 'relevant' in verdict_part and 'not' not in verdict_part and 'maybe' not in verdict_part:
                    verdict = "relevant"
                elif 'maybe' in verdict_part or 'irrelevant' in verdict_part:
                    if 'not_enough' in verdict_part:
                        verdict = "not_enough_context"
                    elif 'irrelevant' in verdict_part:
                        verdict = "irrelevant"
                    else:
                        verdict = "maybe_irrelevant"
                elif 'not_enough' in verdict_part:
                    verdict = "not_enough_context"
            elif line_lower.startswith('score:'):
                try:
                    score_part = line.split(':', 1)[1].strip()
                    score = float(score_part)
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except:
                    pass
            elif line_lower.startswith('feedback:'):
                feedback = line.split(':', 1)[1].strip()
        
        return verdict, score, feedback
    
    def _keyword_reflect(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Use keyword-based similarity to evaluate answer quality."""
        # Calculate similarity score
        similarity_score = calculate_similarity_score(question, answer)
        
        # Check for indicators of insufficient context
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
        
        has_insufficient = any(indicator in answer_lower for indicator in insufficient_indicators)
        
        # Determine verdict
        if has_insufficient:
            verdict = "not_enough_context"
            score = 0.2
            feedback = "Answer indicates insufficient context or information"
        elif similarity_score >= 0.7:
            verdict = "relevant"
            score = similarity_score
            feedback = "Answer appears relevant based on keyword overlap"
        elif similarity_score >= 0.4:
            verdict = "maybe_irrelevant"
            score = similarity_score
            feedback = "Answer has moderate relevance based on keyword overlap"
        else:
            verdict = "irrelevant"
            score = similarity_score
            feedback = "Answer has low relevance based on keyword overlap"
        
        result = {
            "verdict": verdict,
            "score": score,
            "feedback": feedback,
            "method": "keyword_based"
        }
        
        logger.info(f"Reflection verdict: {verdict}, score: {score}")
        return result


# Global reflector instance
_reflector = None


def get_reflector() -> AnswerReflector:
    """Get or create the global reflector instance."""
    global _reflector
    if _reflector is None:
        _reflector = AnswerReflector()
    return _reflector


def reflect_answer(question: str, answer: str, context: str = "") -> Dict[str, Any]:
    """Convenience function to reflect on an answer."""
    reflector = get_reflector()
    return reflector.reflect(question, answer, context)