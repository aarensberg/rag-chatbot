"""Streamlit RAG chatbot over a corpus of AI/ML research papers.

For each question the app:
  1. retrieves the top-K candidate chunks from the persisted Chroma index,
  2. reranks them with a cross-encoder and keeps only the best TOP_N,
  3. sends those chunks + the question to a Groq-hosted LLM,
  4. streams the grounded answer and lists its sources.

Run (after building the index with ``python build_index.py``):
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# --- Configuration (keep in sync with build_index.py) ---
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")  # load GROQ_API_KEY from a .env next to this file
PERSIST_DIR = str(ROOT / "chroma_db")
MANIFEST = ROOT / "papers" / "manifest.json"
COLLECTION = "papers"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant"
K = 10          # candidates retrieved by similarity search
TOP_N = 4       # chunks kept after reranking and sent to the LLM

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY the "
    "context below, which comes from a corpus of AI/ML research papers. If the "
    "answer is not in the context, say so plainly. Cite the paper titles and "
    "years you rely on."
)


# --- Heavy resources: loaded once and cached across reruns ---
@st.cache_resource(show_spinner="Loading embeddings + Chroma index…")
def get_db() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


@st.cache_resource
def get_groq() -> Groq:
    return Groq()  # reads GROQ_API_KEY from the environment


@st.cache_data
def get_categories() -> list[str]:
    entries = json.loads(MANIFEST.read_text())
    return sorted({entry["category"] for entry in entries})


def retrieve_and_rerank(question: str, categories: list[str] | None):
    """Similarity search -> cross-encoder rerank -> keep the TOP_N chunks."""
    where = {"category": {"$in": categories}} if categories else None
    candidates = get_db().similarity_search(question, k=K, filter=where)
    if not candidates:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_N]]


def format_context(docs) -> str:
    blocks = []
    for d in docs:
        m = d.metadata
        header = f"[{m.get('title', m.get('source'))} ({m.get('year', '?')}), p.{m.get('page', '?')}]"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blocks)


def answer_stream(question: str, context: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    completion = get_groq().chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2, stream=True
    )
    for chunk in completion:
        yield chunk.choices[0].delta.content or ""


# --- UI ---
st.set_page_config(page_title="AI Papers RAG Chatbot", page_icon="📚")
st.title("📚 AI Papers RAG Chatbot")
st.caption("Grounded answers over 15 landmark AI papers, from Turing (1950) to GPT-4 (2023).")

if not os.environ.get("GROQ_API_KEY"):
    st.error("`GROQ_API_KEY` is not set. Add it to your environment or a `.env` file.")
    st.stop()
if not Path(PERSIST_DIR).exists():
    st.error("No Chroma index found. Run `python build_index.py` first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    categories = st.multiselect("Limit to topic families", options=get_categories())
    st.divider()
    st.markdown(f"**LLM:** `{GROQ_MODEL}`")
    st.markdown(f"Retrieve **{K}** → rerank → keep **{TOP_N}**")

question = st.text_input(
    "Your question", placeholder="e.g. Which paper introduced the Transformer architecture?"
)

if st.button("Ask", type="primary") and question:
    docs = retrieve_and_rerank(question, categories or None)
    if not docs:
        st.warning("No relevant chunks found in the corpus.")
        st.stop()

    st.subheader("Answer")
    st.write_stream(answer_stream(question, format_context(docs)))

    with st.expander("Sources used"):
        for d in docs:
            m = d.metadata
            st.markdown(
                f"- **{m.get('title')}** ({m.get('year')}) — page {m.get('page')} "
                f"· `{m.get('source')}`"
            )
