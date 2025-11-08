"""
Common helper functions for the pipeline.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_logger(name: str, log_file: Path = None, level: str = "INFO") -> logging.Logger:
    """Setup and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def load_text_file(file_path: Path) -> str:
    """Load text content from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading text file {file_path}: {str(e)}")


def load_pdf_file(file_path: Path) -> str:
    """Load text content from a .pdf file."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error loading PDF file {file_path}: {str(e)}")


def load_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all documents from the data directory.
    Returns a list of dictionaries with 'content' and 'metadata' keys.
    """
    documents = []
    
    # Load .txt files
    for txt_file in data_dir.rglob("*.txt"):
        try:
            content = load_text_file(txt_file)
            documents.append({
                "content": content,
                "metadata": {
                    "source": str(txt_file.relative_to(data_dir)),
                    "type": "text"
                }
            })
        except Exception as e:
            print(f"Warning: Could not load {txt_file}: {e}")
    
    # Load .pdf files
    for pdf_file in data_dir.rglob("*.pdf"):
        try:
            content = load_pdf_file(pdf_file)
            documents.append({
                "content": content,
                "metadata": {
                    "source": str(pdf_file.relative_to(data_dir)),
                    "type": "pdf"
                }
            })
        except Exception as e:
            print(f"Warning: Could not load {pdf_file}: {e}")
    
    return documents


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split text into chunks with overlap.
    Returns a list of dictionaries with 'content' and 'metadata' keys.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    chunk_docs = []
    
    for i, chunk in enumerate(chunks):
        chunk_meta = metadata.copy() if metadata else {}
        chunk_meta["chunk_index"] = i
        chunk_docs.append({
            "content": chunk,
            "metadata": chunk_meta
        })
    
    return chunk_docs


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a context string."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("metadata", {}).get("source", "Unknown")
        content = doc.get("content", doc.get("page_content", ""))
        context_parts.append(f"[Document {i} - Source: {source}]\n{content}\n")
    return "\n---\n".join(context_parts)


def calculate_similarity_score(query: str, answer: str) -> float:
    """
    Simple keyword overlap-based similarity score.
    Returns a score between 0 and 1.
    """
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    if not query_words:
        return 0.0
    
    intersection = query_words.intersection(answer_words)
    similarity = len(intersection) / len(query_words)
    return min(similarity, 1.0)