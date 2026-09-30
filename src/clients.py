"""
API Clients - Simplified and Easy to Understand

This file initializes and manages all the API clients we need:
- OpenAI: For embeddings and LLM (paid, but high quality)
- Groq: For free LLM alternative
- HuggingFace: For free embeddings (runs locally)
- Pinecone: For vector database storage
"""

import os
from .config import get_settings

# Load settings (API keys, model names, etc.)
settings = get_settings()


# ============================================================================
# OPENAI CLIENT (Optional, only initialized if key is provided)
# ============================================================================

openai_client = None
if settings.openai_api_key:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)
    except Exception:
        openai_client = None


# ============================================================================
# GROQ CLIENT (Free LLM alternative)
# ============================================================================

groq_client = None
if settings.groq_api_key:
    try:
        from groq import Groq
        groq_client = Groq(api_key=settings.groq_api_key)
    except Exception:
        groq_client = None


# ============================================================================
# HUGGINGFACE CLIENT (Free embeddings, runs locally)
# ============================================================================

huggingface_model = None


# ============================================================================
# PINECONE CLIENT (Vector database)
# ============================================================================

pinecone_client = None
if settings.pinecone_api_key:
    try:
        from pinecone import Pinecone
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    except Exception:
        pinecone_client = None


# ============================================================================
# HELPER FUNCTIONS TO GET THE RIGHT CLIENT
# ============================================================================

def get_llm_client():
    """
    Get the appropriate LLM client based on settings.
    """
    if settings.llm_provider == "groq" and groq_client is not None:
        return groq_client
    
    if openai_client is not None:
        return openai_client

    if groq_client is not None:
        return groq_client

    raise RuntimeError("No LLM client available. Please set GROQ_API_KEY or OPENAI_API_KEY in environment variables.")


def get_embedding_model():
    """
    Get the appropriate embedding model based on settings.
    """
    global huggingface_model
    
    if settings.embedding_provider == "huggingface":
        if huggingface_model is None:
            from sentence_transformers import SentenceTransformer
            huggingface_model = SentenceTransformer(settings.huggingface_embedding_model)
        return huggingface_model
    
    if openai_client is not None:
        return openai_client

    # Fallback to local HuggingFace if OpenAI not available
    if huggingface_model is None:
        from sentence_transformers import SentenceTransformer
        huggingface_model = SentenceTransformer(settings.huggingface_embedding_model)
    return huggingface_model
