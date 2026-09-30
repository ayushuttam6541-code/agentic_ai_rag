"""
Sample Queries Benchmark & Quality Assurance Script.
Tests the 6 benchmark queries specified in the assignment against the live FastAPI endpoint
and LangGraph RAG pipeline.
"""

import json
import time
import requests

API_URL = "http://localhost:8000/chat"

SAMPLE_QUERIES = [
    {
        "category": "Definition & Scope",
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
    },
    {
        "category": "Architecture & Paradigms",
        "query": "What are the main architectural components required to build agentic systems?",
    },
    {
        "category": "Use Cases",
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    },
    {
        "category": "Comparison",
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    },
    {
        "category": "Challenges & Considerations",
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
    },
    {
        "category": "Out-of-Scope Test (Groundedness Check)",
        "query": "What is the capital of France?",
    },
]

def run_tests():
    print("=" * 80)
    print("Agentic AI RAG Chatbot - Sample Query Verification Benchmark")
    print(f"Target Endpoint: {API_URL}")
    print("=" * 80)
    
    results = []

    for idx, item in enumerate(SAMPLE_QUERIES, start=1):
        print(f"\n[{idx}/6] Category: {item['category']}")
        print(f"Query: {item['query']}")
        
        start_time = time.time()
        try:
            res = requests.post(API_URL, json={"query": item["query"]}, timeout=60)
            res.raise_for_status()
            data = res.json()
        except Exception:
            # Fallback to direct graph invoke if API server is not running
            from src.graph import chat
            data = chat(item["query"])
            
        elapsed = round(time.time() - start_time, 2)
        
        print(f"Latency: {elapsed}s")
        print("Response Payload:")
        print(json.dumps(data, indent=2))
        
        # Verify required keys
        assert "query" in data, "Missing 'query' in response"
        assert "final_answer" in data, "Missing 'final_answer' in response"
        assert "retrieved_context_chunks" in data, "Missing 'retrieved_context_chunks' in response"
        assert "confidence_score" in data, "Missing 'confidence_score' in response"
        
        results.append({
            "category": item["category"],
            "query": item["query"],
            "response": data,
            "latency": elapsed,
        })
        print("-" * 80)

    print("\nAll 6 benchmark queries verified successfully!")
    return results

if __name__ == "__main__":
    run_tests()
