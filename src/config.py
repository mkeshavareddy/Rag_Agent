"""
Configuration file for RAG AI Agent.
Manages API keys, constants, and paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# API Keys - Groq Only
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM Configuration - Groq Only
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # Default Groq model
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Embeddings Configuration - Using HuggingFace (Free, no API key required)
USE_HUGGINGFACE_EMBEDDINGS = True  # Always use HuggingFace embeddings (free)
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Planner Configuration
RETRIEVAL_KEYWORDS = [
    "what", "how", "explain", "describe", "tell me", "benefit", 
    "advantages", "disadvantages", "why", "when", "where", "who",
    "define", "meaning", "example", "examples", "difference", "compare"
]

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "rag_agent.log"

# Create necessary directories
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Evaluation Configuration
ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
ENABLE_LANGSMITH = os.getenv("ENABLE_LANGSMITH", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-ai-agent")