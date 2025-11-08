"""
Retriever module: Handles document loading, chunking, and vector search using ChromaDB.
Uses HuggingFace embeddings (free, no API key required).
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace embeddings (free, no API key required)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    raise ImportError(
        "HuggingFace embeddings are required. Please install: pip install sentence-transformers"
    )

from src.config import (
    DATA_DIR,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    HF_EMBEDDING_MODEL
)
from src.pipeline.utils import (
    setup_logger,
    load_documents,
    chunk_text,
    format_context
)
from src.config import LOG_FILE, LOG_LEVEL

logger = setup_logger("retriever", LOG_FILE, LOG_LEVEL)


class DocumentRetriever:
    """Handles document retrieval using ChromaDB with HuggingFace embeddings."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize the retriever with ChromaDB and HuggingFace embeddings.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = self._initialize_embeddings()
        self._initialize_vectorstore()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model (free, no API key required)."""
        if not HF_AVAILABLE:
            raise ValueError(
                "HuggingFace embeddings are required but not available. "
                "Please install: pip install sentence-transformers"
            )
        
        logger.info(f"Using HuggingFace embeddings: {HF_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
    
    def _initialize_vectorstore(self):
        """Initialize or load ChromaDB vector store."""
        try:
            # Try to load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            # Check if collection exists and has documents
            if self.vectorstore._collection.count() > 0:
                logger.info(f"Loaded existing vectorstore with {self.vectorstore._collection.count()} documents")
                return
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore: {e}")
        
        # Create new vectorstore and index documents
        logger.info("Creating new vectorstore and indexing documents...")
        self._index_documents()
    
    def _index_documents(self):
        """Load documents from data directory and index them in ChromaDB."""
        # Load all documents
        documents = load_documents(DATA_DIR)
        
        if not documents:
            logger.warning(f"No documents found in {DATA_DIR}")
            # Create empty vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            return
        
        # Chunk documents
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                })
        
        logger.info(f"Indexing {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create vectorstore with documents
        texts = [chunk["page_content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_PATH),
            collection_name=self.collection_name
        )
        
        logger.info(f"Successfully indexed {len(all_chunks)} chunks in vectorstore")
    
    def retrieve(self, query: str, k: int = TOP_K) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with 'context' (formatted string) and 'docs' (list of documents)
        """
        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return {
                "context": "",
                "docs": [],
                "error": "Vectorstore not initialized"
            }
        
        try:
            # Retrieve similar documents
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Format context
            context = format_context([
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ])
            
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            
            return {
                "context": context,
                "docs": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            }
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {
                "context": "",
                "docs": [],
                "error": str(e)
            }
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add new documents to the vectorstore."""
        if not self.vectorstore:
            self._initialize_vectorstore()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": i
                    }
                })
        
        texts = [chunk["page_content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f"Added {len(all_chunks)} new chunks to vectorstore")


# Global retriever instance
_retriever: Optional[DocumentRetriever] = None


def get_retriever() -> DocumentRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever