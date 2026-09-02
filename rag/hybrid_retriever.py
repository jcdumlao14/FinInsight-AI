from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.bm25_retriever import BM25Retriever
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore


class HybridRetriever:
    """Hybrid financial retriever using Vector Search + BM25 + RRF."""

    def __init__(
        self,
        chunks_file: str | Path | None = None,
        vector_db: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        if chunks_file is None:
            chunks_file = (
                PROJECT_ROOT
                / "data"
                / "chunks"
                / "chunks.jsonl"
            )

        if vector_db is None:
            vector_db = (
                PROJECT_ROOT
                / "data"
                / "vector_db"
            )

        self.rrf_k = rrf_k

        self.embedding_model = EmbeddingModel(
            model_name=model_name
        )

        self.vector_store = VectorStore(
            persist_directory=str(vector_db)
        )

        self.bm25_retriever = BM25Retriever(
            chunks_file=str(chunks_file)
        )

    # ---------------------------------------------------------
    # Vector retrieval
    # ---------------------------------------------------------

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using semantic vector search."""

        if not query.strip():
            return []

        query_embedding = (
            self.embedding_model.embed_query(
                query.strip()
            )
        )

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results = []

        for rank, chunk_id in enumerate(ids, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": documents[rank - 1],
                    "metadata": metadatas[rank - 1],
                    "distance": distances[rank - 1],
                    "retrieval_method": "vector",
                }
            )

        return results

    # ---------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25 lexical search."""

        return self.bm25_retriever.search(
            query,
            top_k=top_k,
        )

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Combine ranked results using Reciprocal Rank Fusion."""

        scores: Dict[str, float] = {}
        documents: Dict[str, Dict[str, Any]] = {}

        # Vector contribution
        for rank, result in enumerate(
            vector_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            scores[chunk_id] = scores.get(
                chunk_id,
                0.0,
            ) + 1.0 / (
                self.rrf_k + rank
            )

            documents[chunk_id] = result

        # BM25 contribution
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = result.get("chunk_id")

            if chunk_id is None:
                chunk_id = result.get("id")

            scores[chunk_id] = scores.get(
                chunk_id,
                0.0,
            ) + 1.0 / (
                self.rrf_k + rank
            )

            if chunk_id not in documents:
                documents[chunk_id] = result

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        results = []

        for rank, chunk_id in enumerate(
            ranked_ids[:top_k],
            start=1,
        ):
            result = dict(
                documents[chunk_id]
            )

            result["rank"] = rank
            result["chunk_id"] = chunk_id
            result["rrf_score"] = scores[chunk_id]
            result["retrieval_method"] = "hybrid_rrf"

            results.append(result)

        return results

    # ---------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run Vector + BM25 + RRF hybrid retrieval."""

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        vector_results = self.vector_search(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_search(
            query,
            top_k=candidate_k,
        )

        return self.reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k=top_k,
        )


if __name__ == "__main__":

    retriever = HybridRetriever()

    query = (
        "What was Apple's total net sales "
        "in fiscal year 2025?"
    )

    print("=" * 70)
    print("FinInsight-AI — HYBRID RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print()

    results = retriever.retrieve(
        query,
        top_k=5,
        candidate_k=10,
    )

    print(f"Hybrid results: {len(results)}")
    print()

    for result in results:
        print("-" * 70)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Text: "
            f"{result.get('text', '')[:500]}"
            .replace("\n", " ")
        )
