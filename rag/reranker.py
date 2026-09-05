from __future__ import annotations

from typing import Any, Dict, List


class DocumentReranker:
    """Cross-encoder reranker for retrieved financial chunks.

    A deterministic token-overlap fallback is provided so the application can
    still start when the cross-encoder model cannot be loaded. In normal use,
    sentence-transformers CrossEncoder performs the actual semantic reranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re
        return set(re.findall(r"[a-zA-Z0-9$%.-]+", text.lower()))

    def _fallback_score(self, query: str, text: str) -> float:
        q = self._tokens(query)
        d = self._tokens(text)
        return len(q & d) / max(1, len(q))

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []

        if self.model is not None:
            pairs = [(query, r.get("text", "")) for r in results]
            scores = self.model.predict(pairs)
        else:
            scores = [self._fallback_score(query, r.get("text", "")) for r in results]

        ranked = []
        for original, score in zip(results, scores):
            item = dict(original)
            item["rerank_score"] = float(score)
            item["retrieval_method"] = "reranked_hybrid"
            ranked.append(item)

        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        for rank, item in enumerate(ranked[:top_k], 1):
            item["rank"] = rank
        return ranked[:top_k]
