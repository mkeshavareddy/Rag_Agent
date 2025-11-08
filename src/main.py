"""
Main entry point for the RAG AI Agent pipeline.
Orchestrates the plan → retrieve → answer → reflect flow.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    CHROMA_DB_PATH,
    LOG_FILE,
    LOG_LEVEL
)
from src.pipeline.planner import plan_retrieval
from src.pipeline.retriever import get_retriever
from src.pipeline.answerer import get_answerer
from src.pipeline.reflector import get_reflector
from src.eval.logger import get_pipeline_logger
from src.pipeline.utils import setup_logger

# Setup logger
logger = setup_logger("main", LOG_FILE, LOG_LEVEL)


class RAGPipeline:
    """Main pipeline orchestrator for RAG agent."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        logger.info("Initializing RAG Pipeline...")
        self.retriever = get_retriever()
        self.answerer = get_answerer()
        self.reflector = get_reflector()
        self.pipeline_logger = get_pipeline_logger()
        logger.info("RAG Pipeline initialized successfully")
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the complete pipeline: plan → retrieve → answer → reflect.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with complete pipeline results
        """
        try:
            # Log question
            self.pipeline_logger.log_question(question)
            
            # Step 1: Plan
            planner_result = plan_retrieval(question)
            self.pipeline_logger.log_planner(planner_result)
            
            # Step 2: Retrieve (if needed)
            context = ""
            retrieval_result = {"context": "", "docs": []}
            
            if planner_result.get("retrieval_needed", True):
                retrieval_result = self.retriever.retrieve(question)
                context = retrieval_result.get("context", "")
                self.pipeline_logger.log_retriever(retrieval_result)
            else:
                logger.info("[RETRIEVER] Skipped (not needed based on planner decision)")
            
            # Step 3: Answer
            answer_result = self.answerer.generate_answer(question, context)
            self.pipeline_logger.log_answerer(answer_result)
            
            # Step 4: Reflect
            reflection_result = self.reflector.reflect(
                question,
                answer_result.get("answer", ""),
                context
            )
            self.pipeline_logger.log_reflector(reflection_result)
            
            # Compile final result
            final_result = {
                "question": question,
                "planner": planner_result,
                "retrieval": retrieval_result,
                "answer": answer_result,
                "reflection": reflection_result,
                "success": True
            }
            
            # Log final result
            self.pipeline_logger.log_final_result(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}", exc_info=True)
            self.pipeline_logger.log_error(e, "pipeline")
            return {
                "question": question,
                "success": False,
                "error": str(e)
            }
    
    def answer(self, question: str) -> str:
        """
        Simple interface: just return the answer string.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        result = self.run(question)
        if result.get("success"):
            return result.get("answer", {}).get("answer", "No answer generated")
        else:
            return f"Error: {result.get('error', 'Unknown error')}"


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG AI Agent - Q&A System")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the agent"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    if args.interactive or not args.question:
        # Interactive mode
        print("=" * 80)
        print("RAG AI Agent - Interactive Mode")
        print("Type 'exit' or 'quit' to exit")
        print("=" * 80)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = pipeline.run(question)
                
                if result.get("success"):
                    print("\n" + "=" * 80)
                    print("ANSWER:")
                    print("=" * 80)
                    print(result.get("answer", {}).get("answer", ""))
                    print("\n" + "=" * 80)
                    print("REFLECTION:")
                    print(f"Verdict: {result.get('reflection', {}).get('verdict', 'unknown')}")
                    print(f"Score: {result.get('reflection', {}).get('score', 0.0):.2f}")
                    print(f"Feedback: {result.get('reflection', {}).get('feedback', '')}")
                    print("=" * 80)
                else:
                    print(f"\nError: {result.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    else:
        # Single question mode
        result = pipeline.run(args.question)
        
        if result.get("success"):
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(result.get("answer", {}).get("answer", ""))
            print("\n" + "=" * 80)
            print("REFLECTION:")
            print(f"Verdict: {result.get('reflection', {}).get('verdict', 'unknown')}")
            print(f"Score: {result.get('reflection', {}).get('score', 0.0):.2f}")
            print(f"Feedback: {result.get('reflection', {}).get('feedback', '')}")
            print("=" * 80)
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
