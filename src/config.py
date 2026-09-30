from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str = ""
    pinecone_api_key: str = ""
    groq_api_key: str = ""
    huggingface_api_key: str = ""

    pinecone_index_name: str = "agentic-ai-rag"
    pinecone_namespace: str = "agentic-ai"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Provider selection: "openai" or "groq"
    llm_provider: str = "openai"
    embedding_provider: str = "openai"  # "openai" or "huggingface"

    # OpenAI models
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Groq models (free alternative)
    groq_embedding_model: str = "text-embedding-3-small"  # Groq uses OpenAI embeddings
    groq_chat_model: str = "qwen/qwen3.8-27b"

    # Hugging Face models (free alternative)
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    pdf_url: str = "https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing"

    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_retrieval_score: float = 0.20
    min_groundedness_score: float = 0.70

    pdf_path: str = "data/Ebook-Agentic-AI.pdf"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_embedding_model(self) -> str:
        if self.embedding_provider == "huggingface":
            return self.huggingface_embedding_model
        if self.llm_provider == "groq":
            return self.groq_embedding_model
        return self.openai_embedding_model

    def get_chat_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_chat_model
        return self.openai_chat_model

@lru_cache
def get_settings() -> Settings:
    return Settings()
