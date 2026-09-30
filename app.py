from fastapi import FastAPI, HTTPException

from src.graph import chat
from src.models import ChatRequest, ChatResponse

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return chat(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
