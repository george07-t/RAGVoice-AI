"""
RAG (Retrieval-Augmented Generation) System for Document Q&A
==============================================================

This module implements a complete RAG pipeline using:
- Document loading and chunking
- Vector embeddings (OpenAI or Sentence Transformers)
- FAISS vector database for similarity search
- Retrieval tool integrated with LangChain
"""

import os
import time
import logging
from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env.local")

# Configuration
DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3  # Number of documents to retrieve


class RAGSystem:
    """
    RAG System for document ingestion, embedding, and retrieval.
    
    Attributes:
        embeddings: Embedding model for converting text to vectors
        vector_store: FAISS vector database
        text_splitter: Text chunking utility
    """
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            use_openai_embeddings: If True, use OpenAI embeddings; 
                                  if False, use sentence-transformers (local)
        """
        self.embeddings = self._initialize_embeddings(use_openai_embeddings)
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _initialize_embeddings(self, use_openai: bool):
        """Initialize embedding model based on configuration."""
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            logger.info("Using OpenAI embeddings (text-embedding-3-small)")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key
            )
        else:
            # Use sentence-transformers as fallback (local, no API key needed)
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def load_documents(self, directory: str = DOCUMENTS_DIR) -> List[Document]:
        """
        Load all text documents from the specified directory.
        
        Args:
            directory: Path to documents directory
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from {directory}")
        start_time = time.time()
        
        # Load all .txt files
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        load_time = time.time() - start_time
        
        logger.info(f"Loaded {len(documents)} documents in {load_time:.2f}s")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info("Chunking documents...")
        start_time = time.time()
        
        chunks = self.text_splitter.split_documents(documents)
        chunk_time = time.time() - start_time
        
        logger.info(f"Created {len(chunks)} chunks in {chunk_time:.2f}s")
        return chunks
    
    def create_vector_store(
        self, 
        documents: List[Document],
        save_local: bool = True
    ) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of document chunks
            save_local: Whether to save the vector store locally
            
        Returns:
            FAISS vector store
        """
        logger.info("Creating vector store and generating embeddings...")
        start_time = time.time()
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        embedding_time = time.time() - start_time
        logger.info(f"Created vector store in {embedding_time:.2f}s")
        
        # Save locally for persistence
        if save_local:
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            vector_store.save_local(VECTOR_STORE_DIR)
            logger.info(f"Saved vector store to {VECTOR_STORE_DIR}")
        
        self.vector_store = vector_store
        return vector_store
    
    def load_vector_store(self, directory: str = VECTOR_STORE_DIR) -> FAISS:
        """
        Load existing vector store from disk.
        
        Args:
            directory: Path to vector store directory
            
        Returns:
            Loaded FAISS vector store
        """
        logger.info(f"Loading vector store from {directory}")
        
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Vector store not found at {directory}. "
                "Please run ingest_documents() first."
            )
        
        vector_store = FAISS.load_local(
            directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.vector_store = vector_store
        logger.info("Vector store loaded successfully")
        return vector_store
    
    def ingest_documents(
        self,
        force_recreate: bool = False
    ) -> FAISS:
        """
        Complete document ingestion pipeline: load, chunk, embed, and store.
        
        Args:
            force_recreate: If True, recreate vector store even if it exists
            
        Returns:
            FAISS vector store
        """
        # Check if vector store already exists
        if not force_recreate and os.path.exists(VECTOR_STORE_DIR):
            logger.info("Vector store already exists. Loading from disk...")
            return self.load_vector_store()
        
        # Complete ingestion pipeline
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        vector_store = self.create_vector_store(chunks)
        
        logger.info("✅ Document ingestion complete!")
        return vector_store
    
    def retrieve(self, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Please run ingest_documents() first."
            )
        
        start_time = time.time()
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        
        retrieval_time = time.time() - start_time
        logger.info(
            f"Retrieved {len(results)} documents in {retrieval_time:.3f}s"
        )
        
        return results
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = RETRIEVAL_K
    ) -> List[tuple[Document, float]]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Please run ingest_documents() first."
            )
        
        start_time = time.time()
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        retrieval_time = time.time() - start_time
        logger.info(
            f"Retrieved {len(results)} documents with scores in {retrieval_time:.3f}s"
        )
        
        return results


# Initialize global RAG system
_rag_system = None


def get_rag_system() -> RAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(use_openai_embeddings=True)
        try:
            _rag_system.ingest_documents()
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    return _rag_system


@tool
def search_documents(query: str, num_results: int = 3) -> str:
    """
    Search through knowledge base documents to find relevant information.
    
    Use this tool to answer questions about:
    - Artificial Intelligence and Machine Learning
    - Climate Change and Environmental Science
    - Modern World History (1900-present)
    - Blockchain Technology and Cryptocurrencies
    - Health, Wellness, and Medicine
    
    Args:
        query: The search query or question to find information about
        num_results: Number of relevant passages to retrieve (default: 3)
    """
    try:
        start_time = time.time()
        
        # Get RAG system
        rag = get_rag_system()
        
        # Retrieve relevant documents
        results = rag.retrieve_with_scores(query, k=num_results)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Format results
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_name = Path(source).stem.replace('_', ' ').title()
            
            # Include score for transparency (lower is better for FAISS)
            formatted_results.append(
                f"**Source {i}: {source_name}** (relevance: {1/score:.2f})\n"
                f"{doc.page_content}\n"
            )
        
        retrieval_time = time.time() - start_time
        logger.info(f"Document search completed in {retrieval_time:.3f}s")
        
        result_text = "\n---\n".join(formatted_results)
        return result_text
        
    except Exception as e:
        logger.error(f"Error in search_documents: {e}")
        return f"Error searching documents: {str(e)}"


# Standalone script for testing and initialization
if __name__ == "__main__":
    print("=" * 60)
    print("RAG System Initialization and Testing")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RAGSystem(use_openai_embeddings=True)
    
    # Ingest documents
    print("\n[INFO] Ingesting documents...")
    vector_store = rag.ingest_documents(force_recreate=False)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "What causes climate change?",
        "What was World War II?",
        "What is blockchain technology?",
        "How much sleep do adults need?"
    ]
    
    print("\n[INFO] Testing retrieval with sample queries:")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, k=2)
        
        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get('source', 'Unknown')).stem
            preview = doc.page_content[:150] + "..."
            print(f"  [{i}] {source}: {preview}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] RAG System Ready!")
    print("=" * 60)
