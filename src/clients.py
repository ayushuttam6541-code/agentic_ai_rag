"""
API Clients - Simplified and Easy to Understand

This file initializes and manages all the API clients we need:
- OpenAI: For embeddings and LLM (paid, but high quality)
- Groq: For free LLM alternative
- HuggingFace: For free embeddings (runs locally)
- Pinecone: For vector database storage
"""

from openai import OpenAI
from pinecone import Pinecone

from .config import get_settings

# Load settings (API keys, model names, etc.)
settings = get_settings()


# ============================================================================
# OPENAI CLIENT (Paid, but high quality)
# ============================================================================

# Initialize OpenAI client
# Used for:
# - Text embeddings (converting text to vectors)
# - LLM (generating answers) if not using Groq
openai_client = OpenAI(api_key=settings.openai_api_key)


# ============================================================================
# GROQ CLIENT (Free LLM alternative)
# ============================================================================

# Try to initialize Groq client
# Groq provides free LLM API access
# Used for: LLM (generating answers) as a free alternative to OpenAI
try:
    from groq import Groq
    
    # Only initialize if API key is provided
    if settings.groq_api_key:
        groq_client = Groq(api_key=settings.groq_api_key)
    else:
        groq_client = None
        
except ImportError:
    # Groq package not installed
    groq_client = None


# ============================================================================
# HUGGINGFACE CLIENT (Free embeddings, runs locally)
# ============================================================================

# HuggingFace sentence transformers
# This runs locally on your machine and is completely free
# Used for: Text embeddings (converting text to vectors)
# Note: We import this lazily (only when needed) to avoid memory issues
huggingface_model = None


# ============================================================================
# PINECONE CLIENT (Vector database)
# ============================================================================

# Initialize Pinecone client
# Used for: Storing and searching vector embeddings
pinecone_client = Pinecone(api_key=settings.pinecone_api_key)


# ============================================================================
# HELPER FUNCTIONS TO GET THE RIGHT CLIENT
# ============================================================================

def get_llm_client():
    """
    Get the appropriate LLM client based on settings.
    
    Returns:
        - Groq client if LLM_PROVIDER is set to "groq" and key is available
        - OpenAI client otherwise (default)
    """
    # Check if we should use Groq (free)
    if settings.llm_provider == "groq" and groq_client:
        return groq_client
    
    # Default to OpenAI
    return openai_client


def get_embedding_model():
    """
    Get the appropriate embedding model based on settings.
    
    Returns:
        - HuggingFace model if EMBEDDING_PROVIDER is "huggingface"
        - OpenAI client otherwise (default)
        
    Note: HuggingFace model is loaded lazily (only when first needed)
    """
    global huggingface_model
    
    # Check if we should use HuggingFace (free, local)
    if settings.embedding_provider == "huggingface":
        # Import and load the model if not already loaded
        if huggingface_model is None:
            from sentence_transformers import SentenceTransformer
            huggingface_model = SentenceTransformer(settings.huggingface_embedding_model)
        return huggingface_model
    
    # Default to OpenAI
    return openai_client
