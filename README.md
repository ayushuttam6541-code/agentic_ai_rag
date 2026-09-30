# Agentic AI eBook RAG Chatbot

An end-to-end, cyclic Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Pinecone**, and **FastAPI** / **Streamlit**, strictly grounded in the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).

Developed for the **Appening AI Engineer Interview Task**.

---

## 🌟 Key Highlights

- **Cyclic LangGraph Orchestration**: Stateful graph with retrieval, answer synthesis, hallucination grading, and refusal safety gating.
- **Strict Document Grounding**: Restricts generation strictly to retrieved eBook chunks with automatic refusal for out-of-scope queries (e.g., *"What is the capital of France?"*).
- **100% Free Tier Compatible**: Supports free **Groq** (`qwen/qwen3.8-27b`) for sub-second LLM inference and **HuggingFace** (`sentence-transformers/all-MiniLM-L6-v2`) for local zero-cost dense embeddings. Also supports **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`).
- **Standardized Output Payload**: Every response adheres strictly to the assignment JSON schema: `query`, `final_answer`, `retrieved_context_chunks`, and `confidence_score`.
- **Dual Interfaces**: 
  - Modern Dark-Mode Web UI built right into FastAPI at `http://localhost:8000/`.
  - Streamlit dashboard at `streamlit_app.py`.
- **Pre-verified Benchmark Suite**: `tests_sample_queries.py` runs all 6 required benchmark questions with latency and schema assertions.

---

## 🏗️ Architecture & Workflow

```text
               +----------------------------------+
               |  Agentic AI eBook (PDF Document) |
               +----------------------------------+
                                 |
                                 v [PyPDF + RecursiveCharacterTextSplitter]
               +----------------------------------+
               |  Chunks (1000 chars, 200 ovlp)   |
               +----------------------------------+
                                 |
                                 v [HuggingFace / OpenAI Embeddings]
               +----------------------------------+
               | Pinecone Vector DB (Index / NS)  |
               +----------------------------------+
                                 |
           User Query ---------->|
                                 v
               +==================================+
               |        LangGraph Workflow        |
               |                                  |
               |   [START]                        |
               |      |                           |
               |      v                           |
               |  [Retrieve Node]                 |
               |      | Top-k Semantic Search     |
               |      v                           |
               |  [Generate Node]                 |
               |      | Strict Context Synthesis  |
               |      v                           |
               |  [Grade Node]                    |
               |      | Groundedness Evaluation   |
               |      v                           |
               |  [Finalize / Safety Gate]        |
               |      | Refusal if ungrounded     |
               |      v                           |
               |    [END]                         |
               +==================================+
                                 |
                                 v
               +----------------------------------+
               |      FastAPI / Streamlit UI      |
               |   (Standardized JSON Response)   |
               +----------------------------------+
```

---

## 📁 Repository Structure

```text
agentic-ai-rag/
├── app.py                      # FastAPI application with built-in Web UI & /chat endpoint
├── streamlit_app.py            # Streamlit dashboard interface
├── tests_sample_queries.py     # Verification suite for the 6 benchmark queries
├── requirements.txt            # Python dependencies
├── .env.example                # Template configuration file
├── .gitignore                  # Git ignore rules (protects credentials & binaries)
├── README.md                   # Project documentation & setup guide
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings configuration loader
│   ├── clients.py              # Managed API clients (Pinecone, Groq, OpenAI, HuggingFace)
│   ├── models.py               # Pydantic & LangGraph state models
│   ├── ingestion.py            # PDF loader, chunking & Pinecone upsert pipeline
│   ├── retrieval.py            # Vector retrieval & index query logic
│   └── graph.py                # LangGraph state machine & grading nodes
└── data/
    └── Ebook-Agentic-AI.pdf    # Source knowledge base PDF (downloaded)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- Pinecone API Key (free tier at [pinecone.io](https://www.pinecone.io/))
- Groq API Key (free tier at [console.groq.com](https://console.groq.com/)) OR OpenAI API Key

### 2. Environment Setup

Clone repository and create virtual environment:

```bash
git clone https://github.com/<your-username>/agentic-ai-rag.git
cd agentic-ai-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Copy the environment template:

```bash
cp .env.example .env
```

Configure your `.env` file:

```env
# Free-tier default (Groq + HuggingFace)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
PINECONE_NAMESPACE=agentic-ai
EMBEDDING_PROVIDER=huggingface

# (Optional) If using OpenAI instead:
# LLM_PROVIDER=openai
# EMBEDDING_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
```

---

## 📥 Ingestion & Vector Storage Pipeline

To parse the PDF, generate dense vector embeddings, and index into Pinecone:

```bash
python -m src.ingestion
```

The ingestion pipeline performs:
1. Verification/download of `data/Ebook-Agentic-AI.pdf`.
2. Page-level extraction with `PyPDF`.
3. Semantic chunking with `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
4. Generating dense embeddings (`all-MiniLM-L6-v2` or `text-embedding-3-small`).
5. Upserting vectors with rich metadata (`text`, `page`, `chunk_index`, `source`) into Pinecone namespace `agentic-ai`.

---

## 🖥️ Running the Application

### Option A: FastAPI Web App & Interactive UI (Recommended)

Start the server:

```bash
python app.py
```
*(or `uvicorn app:app --reload`)*

- Open your browser to **`http://localhost:8000/`** to interact with the modern UI.
- Interactive Swagger API docs are available at **`http://localhost:8000/docs`**.

### Option B: Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📡 API Endpoint & Output Payload

### POST `/chat`
Request Payload:
```json
{
  "query": "What is Agentic AI?"
}
```

Response Payload:
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "Agentic AI\nAn Executive's Guide to In-depth\nUnderstanding of Agentic AI...",
    "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives..."
  ],
  "confidence_score": 0.8798
}
```

---

## 🧪 Benchmark Verification Suite

Run all 6 required evaluation queries:

```bash
python tests_sample_queries.py
```

### Validation Results Summary

| # | Category | Query | Grounded Answer | Confidence |
|---|---|---|---|---|
| 1 | **Definition & Scope** | What is the core definition of Agentic AI as outlined in the eBook? | Autonomous decision-making and action in pursuit of specific objectives. | `0.87` |
| 2 | **Architecture** | What are the main architectural components required to build agentic systems? | Foundational agents, workflow agents, utility agents, BDI model, Perception, Reasoning, Planning, Learning, Execution. | `0.79` |
| 3 | **Use Cases** | What real-world industry use cases for Agentic AI are discussed in the eBook? | Manufacturing, Retail, Healthcare, Construction, Pharmaceuticals. | `0.86` |
| 4 | **Comparison** | How does Agentic AI differ from traditional generative AI chatbots according to the text? | Goal-driven autonomy vs text generation, adaptive vs rule-based, proactive vs reactive. | `0.78` |
| 5 | **Challenges** | What key challenges or limitations of Agentic AI are mentioned in the document? | Refused due to lack of supported evidence in retrieved context. | `0.00` |
| 6 | **Out-of-Scope Test** | What is the capital of France? | *Refused: "I don't have enough information in the provided eBook to answer that."* | `0.00` |

---

## 🛡️ Groundedness & Hallucination Prevention

1. **Strict Context Adherence**: The system prompt instructs the model to refuse to answer if the context does not contain sufficient factual evidence.
2. **Evaluator Grader Node**: An independent grading step runs in the LangGraph pipeline to verify that each assertion in the candidate answer is directly supported by the retrieved context.
3. **Safety Gate (Finalize Node)**: If the groundedness score falls below threshold (`0.70`), the answer is overridden with a standardized refusal, preventing hallucinations.

---

## 📜 License

MIT License. Developed for technical assessment demonstration.
