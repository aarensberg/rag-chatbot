# AI Papers RAG Chatbot

A small **Streamlit RAG chatbot** that answers questions over a corpus of 15 landmark
AI/ML research papers (Turing 1950 → GPT-4 2023). Each question is answered by
retrieving chunks from a persisted **Chroma** index, refining them with a
**cross-encoder reranker**, and generating a grounded answer with a **Groq**-hosted LLM.

```
question ──▶ Chroma similarity search (k=10) ──▶ cross-encoder rerank ──▶ top 4 chunks ──▶ Groq LLM ──▶ answer
```

## How to run

```bash
# 1. Environment + dependencies  (tested on Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Groq API key  (free key: https://console.groq.com/keys)
cp .env.example .env          # then edit .env and paste your key

# 3. Build the index ONCE  (downloads the embedding model on first run, ~90 MB)
python build_index.py

# 4. Launch the app
streamlit run app.py
```

> If your default `python` is very new (e.g. 3.13/3.14), some ML wheels may not exist yet.
> The easiest fix is `uv venv --python 3.12 .venv` before installing.

## Models used

| Role          | Model |
|---------------|-------|
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` — local, via `langchain-huggingface` |
| Vector store  | **Chroma**, persisted to `./chroma_db` |
| Reranker      | `cross-encoder/ms-marco-MiniLM-L-6-v2` — `sentence-transformers` `CrossEncoder` |
| Chat LLM      | **Groq** `llama-3.3-70b-versatile` (swap to `llama-3.1-8b-instant` for speed) |

## Why reranking helps

Similarity search embeds the query and each chunk *independently*, so it is recall-oriented
and noisy — the single most relevant passage is often **not** ranked first. The cross-encoder
re-scores every `(question, chunk)` pair *jointly*, capturing fine-grained relevance, so only
the most on-topic chunks reach the prompt. Concretely, for *"Which paper introduced the
Transformer?"* plain similarity ranks the 2020 RAG paper #1, while reranking correctly
promotes **Attention Is All You Need (2017)** to the top.

## Example questions (tested)

*Answers below are as produced by `llama-3.3-70b-versatile`, lightly trimmed; the sources are
the papers the model cited.*

**Q1 — Which paper introduced the Transformer architecture, and what was its key innovation?**
The Transformer was introduced in *Attention Is All You Need* (2017). Its key innovation was
replacing recurrent networks (RNNs) with **self-attention** — specifically scaled dot-product
attention and multi-head attention — which removes sequential processing, enabling far more
parallelization and a more efficient model.
*Sources: Attention Is All You Need (2017).*

**Q2 — What is the main idea behind chain-of-thought prompting?**
Chain-of-thought prompting augments each few-shot exemplar with a worked "chain of thought"
leading to its answer. This lets the model decompose a multi-step problem into intermediate
reasoning steps — a step-by-step thought process — which elicits much stronger reasoning across
a range of tasks.
*Sources: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022).*

**Q3 — What problem do LSTMs address compared to standard recurrent networks?**
LSTMs are designed to bridge very long time lags — the long-range temporal dependencies that
standard RNNs struggle to learn. They cope with noise, distributed representations, and
continuous values, and (as shown in *Sequence to Sequence Learning with Neural Networks*, 2014)
can be trained to translate even very long sentences where plain RNNs fail.
*Sources: Long Short-Term Memory (1997); Sequence to Sequence Learning with Neural Networks (2014).*

## Notes

- **Optional extras implemented:** a sidebar filter by topic family (paper `category`) and
  token-by-token **streaming** of the answer.
- **Submission hygiene:** `papers/*.pdf`, `chroma_db/`, and `.env` are git-ignored. The Groq key
  is read from the environment, never hardcoded.
- **Files:** `build_index.py` (one-off indexing) · `app.py` (the chatbot, ~120 lines) ·
  `requirements.txt` · `papers/manifest.json` (per-paper metadata used as chunk metadata + UI filter).
