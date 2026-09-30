import json
import streamlit as st
from src.graph import chat

st.set_page_config(
    page_title="Agentic AI eBook RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI eBook RAG Chatbot")
st.caption("Powered by LangGraph, Pinecone Vector DB, and Strict Groundedness Evaluation")

# Sidebar with Sample Benchmark Queries
st.sidebar.header("Benchmark Sample Queries")
sample_queries = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]

selected_sample = st.sidebar.selectbox("Select a benchmark query:", ["-- Select or type below --"] + sample_queries)

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if selected_sample != "-- Select or type below --":
    st.session_state.query_input = selected_sample

user_query = st.text_input("Enter your question:", value=st.session_state.query_input)

if st.button("Ask Agent", type="primary") and user_query.strip():
    with st.spinner("Retrieving from Pinecone & evaluating with LangGraph..."):
        response = chat(user_query.strip())

    st.subheader("Grounded Answer")
    score = response.get("confidence_score", 0.0)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(response.get("final_answer", ""))
    with col2:
        if score >= 0.7:
            st.success(f"Confidence: {score:.2f}")
        elif score > 0:
            st.warning(f"Confidence: {score:.2f}")
        else:
            st.error(f"Confidence: {score:.2f} (Refused/Out-of-Scope)")

    st.subheader("Retrieved Context Chunks")
    chunks = response.get("retrieved_context_chunks", [])
    if not chunks:
        st.info("No context chunks retrieved.")
    else:
        for idx, chunk in enumerate(chunks, 1):
            with st.expander(f"Chunk #{idx}"):
                st.write(chunk)

    st.subheader("Output JSON Payload")
    st.json(response)
