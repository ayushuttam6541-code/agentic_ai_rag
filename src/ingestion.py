import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec

from .clients import get_embedding_model, pinecone_client
from .config import get_settings

settings = get_settings()

def google_drive_download_url(url: str) -> str:
    match = re.search(r"/file/d/([^/]+)", url)
    if not match:
        return url
    file_id = match.group(1)
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

def download_pdf() -> Path:
    path = Path(settings.pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        return path

    url = google_drive_download_url(settings.pdf_url)
    response = requests.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise RuntimeError(
            "Downloaded content is not a PDF. Check PDF_URL and Google Drive sharing permissions."
        )

    path.write_bytes(response.content)
    return path

def extract_and_chunk(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            raw_id = f"{pdf_path.name}:{page_number}:{chunk_index}:{chunk_text}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "page": page_number,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                },
            })

    return chunks

def ensure_index():
    existing = {item["name"] for item in pinecone_client.list_indexes()}

    # Determine embedding dimension based on provider
    if settings.embedding_provider == "huggingface":
        dimension = 384  # sentence-transformers/all-MiniLM-L6-v2 produces 384 dimensions
    else:
        dimension = 1536  # OpenAI text-embedding-3-small produces 1536 dimensions

    if settings.pinecone_index_name not in existing:
        pinecone_client.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    description = pinecone_client.describe_index(settings.pinecone_index_name)
    return pinecone_client.Index(host=description.host)

def embed_texts(texts: list[str]) -> list[list[float]]:
    embedding_client = get_embedding_model()

    if settings.embedding_provider == "huggingface":
        # Use Hugging Face sentence transformers (free)
        embeddings = embedding_client.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    else:
        # Use OpenAI embeddings
        response = embedding_client.embeddings.create(
            model=settings.get_embedding_model(),
            input=texts,
        )
        return [item.embedding for item in response.data]

def ingest(batch_size: int = 64) -> dict:
    pdf_path = download_pdf()
    chunks = extract_and_chunk(pdf_path)
    index = ensure_index()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = embed_texts([item["text"] for item in batch])

        records = [
            {
                "id": item["id"],
                "values": vector,
                "metadata": item["metadata"],
            }
            for item, vector in zip(batch, embeddings)
        ]

        index.upsert(
            vectors=records,
            namespace=settings.pinecone_namespace,
        )

    return {
        "pdf": str(pdf_path),
        "pages": len(PdfReader(str(pdf_path)).pages),
        "chunks": len(chunks),
        "index": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
    }

if __name__ == "__main__":
    print(ingest())
