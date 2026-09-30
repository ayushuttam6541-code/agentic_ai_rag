import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.graph import chat
from src.models import ChatRequest, ChatResponse

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="A robust LangGraph & Pinecone RAG system grounded strictly on the Agentic AI eBook.",
    version="1.0.0",
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic AI RAG Chatbot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-primary: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      background: radial-gradient(circle at 50% 0%, #171e38 0%, var(--bg) 70%);
      color: var(--text-main);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 1.5rem 2rem;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(9, 13, 22, 0.85);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .badge {
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .badge-rag {
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-score {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.8rem;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-weight: 600;
    }

    .score-high {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .score-zero {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    main {
      flex: 1;
      max-width: 1000px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
    }

    .hero-text {
      text-align: center;
      margin-bottom: 0.5rem;
    }

    .hero-text h1 {
      font-size: 2.25rem;
      font-weight: 700;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }

    .hero-text p {
      color: var(--text-muted);
      font-size: 0.95rem;
    }

    .presets-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
      backdrop-filter: blur(8px);
    }

    .presets-title {
      font-size: 0.825rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
      font-weight: 600;
    }

    .chips-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .chip-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: #cbd5e1;
      font-size: 0.825rem;
      padding: 0.5rem 0.85rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: left;
    }

    .chip-btn:hover {
      background: rgba(99, 102, 241, 0.12);
      border-color: rgba(99, 102, 241, 0.4);
      color: #ffffff;
      transform: translateY(-1px);
    }

    .input-section {
      display: flex;
      gap: 0.75rem;
      position: relative;
    }

    .query-input {
      flex: 1;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.9rem 1.25rem;
      color: var(--text-main);
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .query-input:focus {
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .submit-btn {
      background: var(--accent-primary);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 0 1.5rem;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: opacity 0.2s ease, transform 0.1s ease;
    }

    .submit-btn:hover {
      opacity: 0.9;
    }

    .submit-btn:active {
      transform: scale(0.98);
    }

    .submit-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .result-container {
      display: none;
      flex-direction: column;
      gap: 1.25rem;
      animation: fadeIn 0.3s ease-out forwards;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.5rem;
      backdrop-filter: blur(8px);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .answer-text {
      font-size: 1.05rem;
      line-height: 1.65;
      color: #ffffff;
      white-space: pre-wrap;
    }

    .chunks-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .chunk-item {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 0.875rem;
      line-height: 1.5;
      color: #cbd5e1;
    }

    .chunk-tag {
      display: inline-block;
      font-family: var(--mono);
      font-size: 0.75rem;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      margin-bottom: 0.4rem;
    }

    .json-pre {
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 1rem;
      font-family: var(--mono);
      font-size: 0.8rem;
      overflow-x: auto;
      color: #38bdf8;
    }

    .loader {
      display: none;
      align-items: center;
      justify-content: center;
      gap: 0.75rem;
      padding: 2rem;
      color: var(--text-muted);
    }

    .spinner {
      width: 24px;
      height: 24px;
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top-color: var(--accent-primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h2 style="font-size: 1.25rem; font-weight: 700;">Agentic AI RAG</h2>
      <span class="badge badge-rag">LangGraph + Pinecone</span>
    </div>
    <div>
      <a href="/docs" target="_blank" style="color: var(--text-muted); text-decoration: none; font-size: 0.85rem; margin-right: 1rem;">Swagger API</a>
      <span style="font-size: 0.85rem; color: #64748b;">PDF Knowledge Base Grounded</span>
    </div>
  </header>

  <main>
    <div class="hero-text">
      <h1>Agentic AI eBook Assistant</h1>
      <p>Strictly document-grounded answers powered by LangGraph, Pinecone vector search, and confidence grading.</p>
    </div>

    <div class="presets-container">
      <div class="presets-title">Sample Benchmark Queries (Click to test)</div>
      <div class="chips-grid">
        <button class="chip-btn" onclick="setQuery('What is the core definition of Agentic AI as outlined in the eBook?')">1. Definition & Scope</button>
        <button class="chip-btn" onclick="setQuery('What are the main architectural components required to build agentic systems?')">2. Architecture & Components</button>
        <button class="chip-btn" onclick="setQuery('What real-world industry use cases for Agentic AI are discussed in the eBook?')">3. Industry Use Cases</button>
        <button class="chip-btn" onclick="setQuery('How does Agentic AI differ from traditional generative AI chatbots according to the text?')">4. Comparison with GenAI</button>
        <button class="chip-btn" onclick="setQuery('What key challenges or limitations of Agentic AI are mentioned in the document?')">5. Challenges & Limitations</button>
        <button class="chip-btn" onclick="setQuery('What is the capital of France?')">6. Out-of-Scope Grounding Check</button>
      </div>
    </div>

    <div class="input-section">
      <input type="text" id="queryInput" class="query-input" placeholder="Ask any question grounded in the Agentic AI eBook..." onkeydown="if(event.key==='Enter') executeChat()">
      <button id="submitBtn" class="submit-btn" onclick="executeChat()">
        <span>Ask</span>
      </button>
    </div>

    <div id="loading" class="loader">
      <div class="spinner"></div>
      <span>Querying Pinecone & running LangGraph orchestration...</span>
    </div>

    <div id="results" class="result-container">
      <!-- Final Answer Card -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Grounded Answer</span>
          <div id="scoreBadge" class="badge-score"></div>
        </div>
        <div id="answerContent" class="answer-text"></div>
      </div>

      <!-- Retrieved Context Chunks -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Retrieved Context Chunks</span>
          <span id="chunkCount" style="font-size: 0.8rem; color: var(--text-muted);"></span>
        </div>
        <div id="chunksList" class="chunks-list"></div>
      </div>

      <!-- Raw JSON Payload -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Structured Output Payload (JSON)</span>
        </div>
        <pre id="jsonPayload" class="json-pre"></pre>
      </div>
    </div>
  </main>

  <script>
    function setQuery(text) {
      document.getElementById('queryInput').value = text;
      executeChat();
    }

    async function executeChat() {
      const input = document.getElementById('queryInput');
      const query = input.value.trim();
      if (!query) return;

      const submitBtn = document.getElementById('submitBtn');
      const loader = document.getElementById('loading');
      const results = document.getElementById('results');

      submitBtn.disabled = true;
      loader.style.display = 'flex';
      results.style.display = 'none';

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });

        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        loader.style.display = 'none';
      }
    }

    function renderResults(data) {
      const results = document.getElementById('results');
      const answerContent = document.getElementById('answerContent');
      const scoreBadge = document.getElementById('scoreBadge');
      const chunksList = document.getElementById('chunksList');
      const chunkCount = document.getElementById('chunkCount');
      const jsonPayload = document.getElementById('jsonPayload');

      answerContent.textContent = data.final_answer;
      
      const score = data.confidence_score;
      scoreBadge.className = score > 0.5 ? 'badge-score score-high' : 'badge-score score-zero';
      scoreBadge.textContent = 'Confidence Score: ' + score;

      const chunks = data.retrieved_context_chunks || [];
      chunkCount.textContent = chunks.length + ' chunk(s) retrieved';
      chunksList.innerHTML = '';

      if (chunks.length === 0) {
        chunksList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No chunks retrieved (Query out-of-scope or below similarity threshold).</div>';
      } else {
        chunks.forEach((chunk, index) => {
          const item = document.createElement('div');
          item.className = 'chunk-item';
          item.innerHTML = `<span class="chunk-tag">Chunk #${index + 1}</span><br>` + chunk.replace(/\\n/g, '<br>');
          chunksList.appendChild(item);
        });
      }

      jsonPayload.textContent = JSON.stringify(data, null, 2);
      results.style.display = 'flex';
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_INTERFACE)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return chat(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
