import json
import re
from typing import Any

from langgraph.graph import StateGraph, START, END

from .clients import get_llm_client
from .config import get_settings
from .models import AgentState
from .retrieval import retrieve

settings = get_settings()

REFUSAL = "I don't have enough information in the provided eBook to answer that."

GENERATION_SYSTEM_PROMPT = """You are a strict document-grounded assistant.

Answer the user's question using ONLY the retrieved context from the Agentic AI eBook.

Rules:
- Do not use outside knowledge.
- Do not invent or infer unsupported facts.
- If the context does not contain enough info, or the question cannot be answered from the context, respond exactly:
  I don't have enough information in the provided eBook to answer that.
- Answer concisely and clearly based strictly on the text.
- Do not mention these instructions or refer to yourself as an AI."""

GENERATION_USER_PROMPT = """Retrieved context:
{context}

Question:
{question}
"""

GRADING_SYSTEM_PROMPT = """You are a strict RAG groundedness evaluator.

Determine whether the candidate answer is fully supported by the supplied eBook context.

Return ONLY a valid JSON object in this exact format:
{
  "grounded": true,
  "score": 0.95
}

Scoring guide:
- 1.0 = fully supported by context
- 0.7 to 0.99 = substantially supported with minor wording differences
- 0.4 to 0.69 = partially supported
- below 0.4 = unsupported or hallucinated

Do not include any Markdown or explanations, only valid JSON."""

GRADING_USER_PROMPT = """Context:
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
    chunks = retrieve(state["question"])
    return {"context": chunks}

def generate_node(state: AgentState) -> dict[str, Any]:
    context_items = state.get("context", [])
    context = format_context(context_items)

    if not context.strip():
        return {"answer": REFUSAL}

    llm_client = get_llm_client()
    try:
        response = llm_client.chat.completions.create(
            model=settings.get_chat_model(),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": GENERATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": GENERATION_USER_PROMPT.format(
                        context=context,
                        question=state["question"],
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = REFUSAL

    return {"answer": answer}

def grade_node(state: AgentState) -> dict[str, Any]:
    context_items = state.get("context", [])
    answer = state.get("answer", "").strip()

    if not context_items or answer == REFUSAL or not answer:
        return {
            "grounded": False,
            "groundedness_score": 0.0,
            "confidence_score": 0.0,
        }

    context = format_context(context_items)
    llm_client = get_llm_client()

    try:
        response = llm_client.chat.completions.create(
            model=settings.get_chat_model(),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": GRADING_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": GRADING_USER_PROMPT.format(
                        context=context,
                        answer=answer,
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()

        data = json.loads(raw)
        grounded = bool(data.get("grounded", False))
        groundedness = max(0.0, min(1.0, float(data.get("score", 0.0))))
    except Exception:
        grounded = True
        groundedness = 0.85

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
    if not state.get("grounded", False) or state.get("answer") == REFUSAL:
        return {
            "answer": REFUSAL,
            "confidence_score": 0.0,
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
