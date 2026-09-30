from typing import TypedDict
from pydantic import BaseModel, Field

class ContextChunk(BaseModel):
    text: str
    page: int
    score: float
    chunk_id: str

class AgentState(TypedDict, total=False):
    question: str
    context: list[ContextChunk]
    answer: str
    grounded: bool
    groundedness_score: float
    confidence_score: float

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float
