"""
Answerer module: Generates answers using Groq LLM with retrieved context.
"""
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)
from src.pipeline.utils import setup_logger
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logger("answerer", LOG_FILE, LOG_LEVEL)


class AnswerGenerator:
    """Generates answers using Groq LLM with context."""
    
    def __init__(self):
        """Initialize the answer generator with Groq LLM."""
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Groq LLM model."""
        if not GROQ_API_KEY:
            logger.error("Groq API key not found. Please set GROQ_API_KEY in .env file")
            raise ValueError("Groq API key is required. Please set GROQ_API_KEY in .env file")
        
        logger.info(f"Initializing Groq LLM: {LLM_MODEL}")
        return ChatGroq(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            groq_api_key=GROQ_API_KEY
        )
    
    def generate_answer(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Generate an answer using LLM with optional context.
        
        Args:
            question: User's question
            context: Retrieved context from documents (optional)
            
        Returns:
            Dictionary with 'answer' string and metadata
        """
        try:
            # Create prompt
            if context:
                prompt = self._create_rag_prompt(question, context)
            else:
                prompt = self._create_direct_prompt(question)
            
            # Generate answer
            response = self.llm.invoke(prompt)
            
            # Extract answer text
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            return {
                "answer": answer.strip(),
                "question": question,
                "used_context": bool(context),
                "model": LLM_MODEL
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "question": question,
                "error": str(e),
                "used_context": bool(context)
            }
    
    def _create_rag_prompt(self, question: str, context: str) -> list:
        """Create a RAG prompt with context."""
        system_message = SystemMessage(content="""You are an AI assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question accurately."
Be concise, accurate, and cite sources when possible.""")
        
        human_message = HumanMessage(content=f"""CONTEXT:
{context}

QUESTION:
{question}

Please provide a clear and concise answer based on the context above.""")
        
        return [system_message, human_message]
    
    def _create_direct_prompt(self, question: str) -> list:
        """Create a direct prompt without context."""
        system_message = SystemMessage(content="You are a helpful AI assistant. Provide clear and concise answers.")
        human_message = HumanMessage(content=question)
        
        return [system_message, human_message]


# Global answer generator instance
_answerer: Optional[AnswerGenerator] = None


def get_answerer() -> AnswerGenerator:
    """Get or create the global answer generator instance."""
    global _answerer
    if _answerer is None:
        _answerer = AnswerGenerator()
    return _answerer


def generate_answer(question: str, context: str = "") -> Dict[str, Any]:
    """Convenience function to generate an answer."""
    answerer = get_answerer()
    return answerer.generate_answer(question, context)