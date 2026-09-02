from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore


class VectorRetriever:
    """Semantic vector retriever for financial document chunks."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if persist_directory is None:
            persist_directory = (
                PROJECT_ROOT / "data" / "vector_db"
            )

        self.embedding_model = EmbeddingModel(model_name)

        self.vector_store = VectorStore(
            persist_directory=str(persist_directory)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve the most semantically similar chunks."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.embed_query(
            query.strip()
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        retrieved = []

        for rank, result in enumerate(results, start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": result.get("id"),
                    "text": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "distance": result.get("distance"),
                }
            )

        return retrieved


if __name__ == "__main__":
    retriever = VectorRetriever()

    query = "What was Apple's total net sales in fiscal year 2025?"

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print("=" * 70)
    print("FinInsight-AI — VECTOR RETRIEVAL TEST")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print()

    for result in results:
        print("-" * 70)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']}")
        print(
            f"Text: {result['text'][:500].replace(chr(10), ' ')}"
        )
