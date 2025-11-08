"""
Logger module: Enhanced logging and monitoring for the RAG pipeline.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from src.config import LOG_FILE, LOG_LEVEL
from src.pipeline.utils import setup_logger

# Setup logger
logger = setup_logger("rag_agent", LOG_FILE, LOG_LEVEL)


class PipelineLogger:
    """Enhanced logger for pipeline monitoring."""
    
    def __init__(self):
        """Initialize the pipeline logger."""
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}
    
    def log_question(self, question: str):
        """Log the user's question."""
        logger.info("=" * 80)
        logger.info(f"NEW QUESTION: {question}")
        logger.info("=" * 80)
        self.start_time = time.time()
    
    def log_planner(self, planner_result: Dict[str, Any]):
        """Log planner decision."""
        stage_start = time.time()
        logger.info(f"[PLANNER] Retrieval needed: {planner_result.get('retrieval_needed')}")
        logger.info(f"[PLANNER] Reason: {planner_result.get('reason')}")
        self.stage_times['planner'] = time.time() - stage_start
    
    def log_retriever(self, retrieval_result: Dict[str, Any]):
        """Log retrieval results."""
        stage_start = time.time()
        num_docs = len(retrieval_result.get('docs', []))
        context_length = len(retrieval_result.get('context', ''))
        logger.info(f"[RETRIEVER] Retrieved {num_docs} documents")
        logger.info(f"[RETRIEVER] Context length: {context_length} characters")
        if retrieval_result.get('error'):
            logger.error(f"[RETRIEVER] Error: {retrieval_result.get('error')}")
        self.stage_times['retriever'] = time.time() - stage_start
    
    def log_answerer(self, answer_result: Dict[str, Any]):
        """Log answer generation."""
        stage_start = time.time()
        answer = answer_result.get('answer', '')
        logger.info(f"[ANSWERER] Generated answer ({len(answer)} characters)")
        logger.info(f"[ANSWERER] Used context: {answer_result.get('used_context')}")
        logger.info(f"[ANSWERER] Model: {answer_result.get('model', 'unknown')}")
        if answer_result.get('error'):
            logger.error(f"[ANSWERER] Error: {answer_result.get('error')}")
        self.stage_times['answerer'] = time.time() - stage_start
    
    def log_reflector(self, reflection_result: Dict[str, Any]):
        """Log reflection results."""
        stage_start = time.time()
        verdict = reflection_result.get('verdict', 'unknown')
        score = reflection_result.get('score', 0.0)
        feedback = reflection_result.get('feedback', '')
        logger.info(f"[REFLECTOR] Verdict: {verdict}")
        logger.info(f"[REFLECTOR] Score: {score:.2f}")
        logger.info(f"[REFLECTOR] Feedback: {feedback}")
        logger.info(f"[REFLECTOR] Method: {reflection_result.get('method', 'unknown')}")
        self.stage_times['reflector'] = time.time() - stage_start
    
    def log_final_result(self, final_result: Dict[str, Any]):
        """Log final pipeline result."""
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info("Stage times:")
        for stage, stage_time in self.stage_times.items():
            logger.info(f"  - {stage}: {stage_time:.3f} seconds")
        logger.info(f"Final verdict: {final_result.get('reflection', {}).get('verdict', 'unknown')}")
        logger.info("=" * 80)
    
    def log_error(self, error: Exception, stage: str = "unknown"):
        """Log an error."""
        logger.error(f"[ERROR] Stage: {stage}, Error: {str(error)}", exc_info=True)


# Global logger instance
_pipeline_logger: Optional[PipelineLogger] = None


def get_pipeline_logger() -> PipelineLogger:
    """Get or create the global pipeline logger."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger()
    return _pipeline_logger
