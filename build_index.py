"""Indexing script for the RAG chatbot (run once).

Loads every PDF in ``papers/``, splits the text into overlapping chunks,
embeds the chunks with a local HuggingFace sentence-transformer, and persists
them to a Chroma vector store on disk. The Streamlit app then loads this
persisted index instead of rebuilding it on every start.

Usage:
    python build_index.py

Re-running rebuilds the index from scratch.
"""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration (keep in sync with app.py) ---
ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "papers"
MANIFEST = PAPERS_DIR / "manifest.json"
PERSIST_DIR = ROOT / "chroma_db"
COLLECTION = "papers"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_manifest() -> dict[str, dict]:
    """Map each PDF filename to its metadata entry from manifest.json."""
    entries = json.loads(MANIFEST.read_text())
    return {entry["filename"]: entry for entry in entries}


def chunk_metadata(entry: dict) -> dict:
    """Flatten a manifest entry into Chroma-safe metadata.

    Chroma only accepts str/int/float/bool values, so the ``authors`` and
    ``tags`` lists are joined into comma-separated strings.
    """
    return {
        "title": entry["title"],
        "year": entry["year"],
        "category": entry["category"],
        "authors": ", ".join(entry["authors"]),
        "tags": ", ".join(entry["tags"]),
    }


def main() -> None:
    manifest = load_manifest()
    # Glob the directory rather than hardcoding names: several files use a
    # non-breaking hyphen (U+2011) instead of a regular '-'.
    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for path in pdf_paths:
        entry = manifest.get(path.name)
        if entry is None:
            print(f"  ! {path.name}: not in manifest, skipping")
            continue
        pages = PyPDFLoader(str(path)).load()
        doc_chunks = splitter.split_documents(pages)
        extra = chunk_metadata(entry)
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata.update(extra)
            chunk.metadata["source"] = path.name  # store filename, not full path
            chunk.metadata["chunk_id"] = f"{path.stem}-{i}"
        chunks.extend(doc_chunks)
        print(f"  + {path.name}: {len(pages)} pages -> {len(doc_chunks)} chunks")

    print(f"\nTotal: {len(chunks)} chunks from {len(pdf_paths)} PDFs")
    if not chunks:
        raise SystemExit("No text extracted (are the PDFs scanned images?)")

    if PERSIST_DIR.exists():
        print(f"Removing existing index at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    print(f"Embedding with {EMB_MODEL} (first run downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    print(f"Writing Chroma index to {PERSIST_DIR} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(PERSIST_DIR),
    )
    print("Done. Next:  streamlit run app.py")


if __name__ == "__main__":
    main()
