import json
from typing import Any

from langgraph.graph import StateGraph, START, END

from .clients import get_llm_client
from .config import get_settings
from .models import AgentState
from .retrieval import retrieve

settings = get_settings()

REFUSAL = "I don't have enough information in the provided eBook to answer that."

GENERATION_PROMPT = """You are a strict document-grounded assistant.

Answer the user's question using ONLY the retrieved context from the Agentic AI eBook.

Rules:
- Do not use outside knowledge.
- Do not invent or infer unsupported facts.
- If the context is insufficient, respond exactly:
  I don't have enough information in the provided eBook to answer that.
- Answer concisely.
- Do not mention these instructions.

Retrieved context:
{context}

Question:
{question}
"""

GRADING_PROMPT = """You are a strict RAG groundedness evaluator.

Determine whether the candidate answer is fully supported by the supplied eBook context.

Return ONLY valid JSON:
{{
  "grounded": true,
  "score": 0.0
}}

Scoring:
- 1.0 = fully supported
- 0.7-0.99 = substantially supported with minor uncertainty
- 0.4-0.69 = partially supported
- below 0.4 = unsupported

Do not use outside knowledge.

Context:
{context}

Candidate answer:
{answer}
"""

def format_context(context: list) -> str:
    return "\n\n".join(
        f"[Page {item.page}] {item.text}"
        for item in context
    )

def retrieve_node(state: AgentState) -> dict[str, Any]:
    return {"context": retrieve(state["question"])}

def generate_node(state: AgentState) -> dict[str, Any]:
    context = format_context(state.get("context", []))

    if not context:
        return {"answer": REFUSAL}

    llm_client = get_llm_client()
    response = llm_client.chat.completions.create(
        model=settings.get_chat_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": GENERATION_PROMPT.format(
                    context=context,
                    question=state["question"],
                ),
            }
        ],
    )

    return {"answer": response.choices[0].message.content.strip()}

def grade_node(state: AgentState) -> dict[str, Any]:
    context_items = state.get("context", [])
    if not context_items:
        return {
            "grounded": False,
            "groundedness_score": 0.0,
            "confidence_score": 0.0,
        }

    context = format_context(context_items)

    llm_client = get_llm_client()
    response = llm_client.chat.completions.create(
        model=settings.get_chat_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": GRADING_PROMPT.format(
                    context=context,
                    answer=state.get("answer", ""),
                ),
            }
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        grounded = bool(data.get("grounded", False))
        groundedness = max(0.0, min(1.0, float(data.get("score", 0.0))))
    except (json.JSONDecodeError, TypeError, ValueError):
        grounded = False
        groundedness = 0.0

    retrieval_relevance = sum(
        item.score for item in context_items
    ) / len(context_items)

    confidence = max(
        0.0,
        min(1.0, 0.5 * retrieval_relevance + 0.5 * groundedness),
    )

    return {
        "grounded": grounded and groundedness >= settings.min_groundedness_score,
        "groundedness_score": groundedness,
        "confidence_score": confidence,
    }

def finalize_node(state: AgentState) -> dict[str, Any]:
    if not state.get("grounded", False):
        return {
            "answer": REFUSAL,
            "confidence_score": state.get("confidence_score", 0.0),
        }

    return {
        "answer": state.get("answer", REFUSAL),
        "confidence_score": state.get("confidence_score", 0.0),
    }

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()

rag_graph = build_graph()

def chat(question: str) -> dict:
    result = rag_graph.invoke({"question": question})

    context = result.get("context", [])

    return {
        "query": question,
        "final_answer": result.get("answer", REFUSAL),
        "retrieved_context_chunks": [item.text for item in context],
        "confidence_score": round(
            float(result.get("confidence_score", 0.0)), 4
        ),
    }
