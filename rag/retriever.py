"""
InsightAI - RAG retriever.

Given a query, retrieves the most relevant chunks of analytical methodology from
the knowledge base and returns them as context for the LLM. Uses Chroma when
available, otherwise an in-memory cosine search over the embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from rag import embeddings as emb
from rag import ingest


class Retriever:
    """A lightweight RAG retriever over the InsightAI knowledge base."""

    def __init__(self, index: Dict):
        self.index = index
        self.chunks = index.get("chunks", [])
        self.vectors = index.get("vectors", np.zeros((0, 100)))
        self.chroma = index.get("chroma")
        self.model_name = index.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top ``top_k`` relevant chunks."""
        if not self.chunks:
            return []

        # Chroma path.
        if self.chroma is not None:
            try:
                qv = emb.embed_query(query, model_name=self.model_name).astype(float).tolist()
                res = self.chroma.query(query_embeddings=[qv], n_results=top_k)
                metadatas = res.get("metadatas", [[]])[0]
                documents = res.get("documents", [[]])[0]
                return [{"text": d, "source": m.get("source", "knowledge_base"), "score": None}
                        for d, m in zip(documents, metadatas)]
            except Exception:
                pass

        # Cosine-similarity fallback.
        qv = emb.embed_query(query, model_name=self.model_name)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        vecs = self.vectors
        if vecs.shape[0] == 0:
            return []
        sims = vecs @ qv
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            if float(sims[idx]) <= 0.0:
                continue
            results.append({
                "text": self.chunks[int(idx)]["text"],
                "source": self.chunks[int(idx)]["source"],
                "score": float(sims[idx]),
            })
        return results

    def build_prompt_context(self, query: str, top_k: int = 5) -> str:
        """Format retrieved chunks into a compact context string."""
        chunks = self.search(query, top_k=top_k)
        if not chunks:
            return "No relevant context retrieved."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] ({c['source']}) {c['text'].strip()}")
        return "\n\n".join(parts)
