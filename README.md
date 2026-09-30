# Agentic AI eBook RAG Chatbot

A custom Python Retrieval-Augmented Generation chatbot built for the AI Engineer interview assignment.

## Stack

- Python 3.10+
- PyPDF
- LangChain RecursiveCharacterTextSplitter
- OpenAI `text-embedding-3-small`
- OpenAI `gpt-4o-mini`
- Pinecone
- LangGraph
- FastAPI

## Architecture

```text
Agentic AI eBook
      |
      v
  PDF download
      |
      v
  PyPDF extraction
      |
      v
 RecursiveCharacterTextSplitter
   1000 chars / 200 overlap
      |
      v
 OpenAI embeddings
 text-embedding-3-small
      |
      v
 Pinecone
 1536 dimensions / cosine
      |
      | user question
      v
 LangGraph
      |
      +--> retrieve
      |
      +--> generate
      |
      +--> groundedness grade
      |
      +--> final safety gate
      |
      v
 FastAPI /chat
```

## 1. Setup

Python 3.10+ is required.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Set:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

**Note:** OpenAI requires API credits. If you don't have credits, you can use the free Groq alternative (see below).

## Free Alternative (Groq)

If you don't have OpenAI credits, you can use Groq's free LLM service:

1. Get a free Groq API key from https://console.groq.com/
2. Install Groq: `pip install groq`
3. Update your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```

4. Test: `python test_groq.py`

This uses Groq for LLM calls (free) and OpenAI for embeddings (very cheap). See `FREE_ALTERNATIVE_SETUP.md` for details.

## 2. Document ingestion

The configured source is the Agentic AI eBook supplied in the interview reference.

Run:

```bash
python -m src.ingestion
```

The ingestion pipeline:

1. Downloads the PDF.
2. Extracts text page-by-page.
3. Splits text using `RecursiveCharacterTextSplitter`.
4. Uses 1000-character chunks with 200-character overlap.
5. Creates a Pinecone serverless index with 1536 dimensions.
6. Generates `text-embedding-3-small` embeddings.
7. Stores chunk text, page number, source and chunk index as metadata.

The PDF is ignored by Git and should not be committed.

## 3. Run the API

```bash
uvicorn app:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 4. Query

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: applic
ation/json" \
  -d '{"query":"What is Agentic AI according to the eBook?"}'
```

Response:

```json
{
  "query": "What is Agentic AI according to the eBook?",
  "final_answer": "...",
  "retrieved_context_chunks": [
    "..."
  ],
  "confidence_score": 0.91
}
```

## 5. Grounding strategy

The generation prompt explicitly prohibits outside knowledge.

After generation, a separate grading node evaluates whether the answer is supported by retrieved context.

The final node acts as a safety gate:

```text
grounded = false
        |
        v
"I don't have enough information in the provided eBook to answer that."
```

This is particularly important for the out-of-scope benchmark:

> Who won the 2022 FIFA World Cup?

The chatbot must not answer from general model knowledge.

## 6. Confidence score

The score is an application-level heuristic:

```text
confidence =
    0.5 * average retrieval relevance
  + 0.5 * groundedness score
```

It is **not a calibrated probability**.

This makes the score transparent and easy to improve later using a labeled evaluation dataset.

## 7. Benchmark queries

`tests/benchmark_queries.json` contains six validation queries:

1. Definition of Agentic AI
2. Agents vs traditional automation
3. Agentic Architecture components
4. Role of memory
5. Agentic AI use cases
6. Out-of-scope FIFA question

Run unit tests:

```bash
pytest -q
```

The benchmark questions should also be executed manually against `/chat` after ingestion.

## 8. Project structure

```text
agentic-ai-rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── clients.py
│   ├── models.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── graph.py
├── tests/
│   ├── test_graph.py
│   └── benchmark_queries.json
└── data/
    └── Ebook-Agentic-AI.pdf  # generated locally, gitignored
```

## 9. Design rationale

### Why LangGraph?

The RAG workflow is stateful and explicit:

```text
START
  |
retrieve
  |
generate
  |
grade
  |
finalize
  |
END
```

The graph makes retrieval, generation, evaluation and the final grounding gate independently testable.

### Why Pinecone?

Pinecone provides semantic vector retrieval and stores the original chunk text and page metadata alongside each vector, making retrieved evidence inspectable.

### Why a separate grounding grader?

A model can sometimes produce an answer that sounds plausible even when retrieval is weak. The grader provides a second verification step before the response reaches the user.

### Why page metadata?

It makes the RAG result auditable and allows future versions to expose citations such as `Page 12`.

## 10. Security

Never commit:

```text
.env
OPENAI_API_KEY
PINECONE_API_KEY
data/Ebook-Agentic-AI.pdf
```

Use environment variables for credentials.

## 11. Future improvements

For a production implementation:

- Add a reranker after Pinecone retrieval.
- Add exact page citations to the answer.
- Add automated retrieval-quality evaluation.
- Calibrate confidence scores against labeled data.
- Add API authentication and rate limiting.
- Add structured logging/tracing.
- Add Docker deployment.
- Add a Streamlit UI if a visual demo is preferred.
