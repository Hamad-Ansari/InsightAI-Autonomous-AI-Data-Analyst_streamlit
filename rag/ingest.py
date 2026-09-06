"""
InsightAI - RAG knowledge-base ingestion.

Reads methodology documents from ``rag/knowledge_base``, chunking them and
building a persistent-ish index. Two backends are supported:

* ChromaDB (preferred, local & persistent) when installed.
* A lightweight in-memory + JSON index fallback (always available).

The knowledge base is a curated set of analytical methodology documents; it is
never derived from the uploaded dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag import embeddings as emb

_HAS_CHROMA = False
try:
    import chromadb  # type: ignore
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False


# ---------------------------------------------------------------- Chunking


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
            # trim oversize sentence if longer than chunk
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def load_knowledge_docs(kb_dir: Path) -> List[Dict]:
    """Read all ``.md`` / ``.txt`` files in the knowledge base folder."""
    docs: List[Dict] = []
    if not kb_dir.exists():
        return docs
    for path in sorted(kb_dir.glob("*.md")) + sorted(kb_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
            docs.append({"source": path.stem, "content": text})
        except Exception:
            continue
    return docs


def build_index(kb_dir: Path, model_name: str, persist_path: str) -> Dict:
    """Build the vector index from the knowledge base."""
    docs = load_knowledge_docs(kb_dir)
    chunks: List[Dict] = []
    for doc in docs:
        for i, ch in enumerate(chunk_text(doc["content"])):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": ch})

    texts = [c["text"] for c in chunks]
    vectors = emb.embed_documents(texts, model_name=model_name) if texts else np.zeros((0, 100))

    index: Dict = {
        "chunks": chunks,
        "vectors": vectors,
        "model_name": model_name,
        "backend": emb.backend_name(model_name),
    }

    if _HAS_CHROMA:
        try:
            client = chromadb.PersistentClient(path=persist_path)
            collection_name = "insightai_kb"
            try:
                collection = client.get_or_create_collection(name=collection_name)
                collection.upsert(
                    ids=[f"{c['source']}_{c['chunk_id']}" for c in chunks],
                    documents=[c["text"] for c in chunks],
                    metadatas=[{"source": c["source"]} for c in chunks],
                )
                index["chroma"] = collection
            except Exception:
                pass
        except Exception:
            pass

    # Persist fallback index to JSON.
    persist_path_obj = Path(persist_path)
    persist_path_obj.mkdir(parents=True, exist_ok=True)
    try:
        with open(persist_path_obj / "index.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "model_name": model_name, "backend": index["backend"]}, f)
        np.save(persist_path_obj / "vectors.npy", vectors)
    except Exception:
        pass
    return index
