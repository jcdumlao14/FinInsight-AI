from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.hybrid_retriever import HybridRetriever


MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "hybrid_retrieval_manifest.json"
)


def main() -> None:

    print("=" * 70)
    print("FinInsight-AI — PHASE 7 HYBRID RETRIEVAL")
    print("=" * 70)

    retriever = HybridRetriever(
        rrf_k=60
    )

    print("[OK] Hybrid retriever initialized")
    print(
        f"[OK] BM25 chunks: "
        f"{retriever.bm25_retriever.count()}"
    )
    print(
        f"[OK] ChromaDB vectors: "
        f"{retriever.vector_store.count()}"
    )
    print(
        f"[OK] RRF k: "
        f"{retriever.rrf_k}"
    )

    queries = [
        "What was Apple's total net sales in fiscal year 2025?",
        "What was Apple's net income in fiscal year 2025?",
        "What were NVIDIA's research and development expenses?",
        "What was NVIDIA's revenue?",
    ]

    successful = 0

    print()
    print("Running hybrid retrieval tests...")
    print("-" * 70)

    for query in queries:

        results = retriever.retrieve(
            query,
            top_k=5,
            candidate_k=10,
        )

        if results:
            successful += 1

            print(
                f"[OK] {query}"
            )

            print(
                f"     Results: "
                f"{len(results)}"
            )

            print(
                f"     Top chunk: "
                f"{results[0]['chunk_id']}"
            )

            print(
                f"     Top RRF score: "
                f"{results[0]['rrf_score']:.6f}"
            )

        else:
            print(
                f"[FAIL] No results: {query}"
            )

    status = (
        "passed"
        if successful == len(queries)
        else "failed"
    )

    manifest = {
        "phase": 7,
        "component": "hybrid retrieval",
        "retrieval_methods": [
            "vector",
            "bm25",
            "reciprocal_rank_fusion",
        ],
        "rrf_k": retriever.rrf_k,
        "vector_count": retriever.vector_store.count(),
        "bm25_count": retriever.bm25_retriever.count(),
        "test_queries": len(queries),
        "successful_queries": successful,
        "status": status,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("PHASE 7 HYBRID RETRIEVAL SUMMARY")
    print("=" * 70)

    print(
        f"Vector documents : "
        f"{retriever.vector_store.count()}"
    )

    print(
        f"BM25 documents   : "
        f"{retriever.bm25_retriever.count()}"
    )

    print(
        f"Queries tested   : "
        f"{len(queries)}"
    )

    print(
        f"Successful       : "
        f"{successful}"
    )

    print(
        f"RRF k            : "
        f"{retriever.rrf_k}"
    )

    print(
        f"Manifest         : "
        f"{MANIFEST_FILE}"
    )

    if status != "passed":
        print()
        print(
            "[FAIL] Phase 7 validation failed."
        )
        raise SystemExit(1)

    print()
    print(
        "Phase 7 hybrid retrieval completed successfully."
    )


if __name__ == "__main__":
    main()
