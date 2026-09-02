from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate semantic embeddings for FinInsight-AI documents."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings for document chunks."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a user query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()
