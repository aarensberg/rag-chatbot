# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course assignment ("Introduction to generative AI", Unit 3): build a small **Streamlit RAG chatbot** that answers questions over a corpus of PDF research papers, using local HuggingFace embeddings, a persisted Chroma index, cross-encoder reranking, and a Groq-hosted LLM. The full brief is in `guidelines.pdf`.

**Current state: greenfield.** Only the corpus (`papers/`), its metadata, and an empty `.venv` (Python 3.14) exist. The two pieces of code — an indexing script and `app.py` — and `requirements.txt` still need to be written.

## Mandated stack (do not substitute — these are graded requirements)

- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, via the `langchain-huggingface` package (local, no API key; first run downloads a few hundred MB).
- **Vector store:** Chroma, **persisted to disk**.
- **Reranker:** `sentence-transformers` `CrossEncoder` with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Chat LLM:** Groq — `llama-3.1-8b-instant` or `llama-3.3-70b-versatile` (needs a `GROQ_API_KEY`).
- **UI:** Streamlit (`streamlit run app.py`).

## Required architecture

Two separate stages — keep indexing out of the app's hot path:

1. **Indexing script (run once):** load `papers/*.pdf` → chunk → embed with all-MiniLM-L6-v2 → write to a persistent Chroma directory on disk. Not rebuilt on app start.
2. **`app.py` (Streamlit):** loads the **persisted** Chroma index from disk; on each question, retrieve ~`k=10` chunks by similarity → **rerank with the cross-encoder** → pass only the **top 3–5** reranked chunks + question to Groq → display the answer.

Reranking between retrieval and generation is **mandatory**, not optional. Similarity search is recall-oriented and noisy; the cross-encoder re-scores each (query, chunk) pair jointly for relevance, so only the most on-topic chunks reach the prompt.

Streamlit reruns the entire script on every interaction. Wrap all heavy loads — embeddings model, Chroma index, cross-encoder, Groq client — in `@st.cache_resource` so they load once. Keep `app.py` short (~50 lines is a fine baseline).

## Corpus

`papers/` holds 15 primary-source AI/ML research papers (1950 Turing → 2023 GPT-4), named `YYYY-Title.pdf`. `papers/manifest.json` carries per-paper metadata (`filename`, `year`, `title`, `authors`, `category`, `tags`, `summary`) for use as chunk metadata and optional UI filters; `papers/README.md` describes the corpus and the recommended per-chunk metadata fields.

**Gotchas when joining manifest → files by name:**
- Several filenames contain a non-ASCII non-breaking hyphen (`‑`, U+2011), not a normal `-`. Prefer globbing the directory over hardcoding names.

## Submission constraints (affect `.gitignore` and what code may assume)

The final submission must **exclude the PDFs, the Chroma database, and API keys**. So: read the Groq key from the environment (never hardcode), and ensure the Chroma persist directory and `papers/*.pdf` are git-ignored. The graded README (one page) must state how to run, which models were used, a 2–3 sentence justification of reranking, and 3 tested example questions with their answers.
