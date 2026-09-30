from .clients import get_embedding_model
from .config import get_settings
from .ingestion import ensure_index
from .models import ContextChunk

settings = get_settings()

def retrieve(question: str, top_k: int | None = None) -> list[ContextChunk]:
    index = ensure_index()

    embedding_client = get_embedding_model()

    if settings.embedding_provider == "huggingface":
        # Use Hugging Face sentence transformers (free)
        embedding = embedding_client.encode(question, convert_to_numpy=True).tolist()
    else:
        # Use OpenAI embeddings
        embedding = embedding_client.embeddings.create(
            model=settings.get_embedding_model(),
            input=question,
        ).data[0].embedding

    result = index.query(
        namespace=settings.pinecone_namespace,
        vector=embedding,
        top_k=top_k or settings.top_k,
        include_metadata=True,
    )

    chunks = []

    for match in result.matches:
        metadata = match.metadata or {}
        score = float(match.score or 0.0)

        if score < settings.min_retrieval_score:
            continue

        chunks.append(
            ContextChunk(
                text=str(metadata.get("text", "")),
                page=int(metadata.get("page", 0)),
                score=score,
                chunk_id=str(match.id),
            )
        )

    return chunks
