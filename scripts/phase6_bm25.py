from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.bm25_retriever import BM25Retriever


CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
MANIFEST_FILE = PROJECT_ROOT / "data" / "metadata" / "bm25_manifest.json"


def main() -> None:
    print("=" * 70)
    print("FinInsight-AI — PHASE 6 BM25 LEXICAL RETRIEVAL")
    print("=" * 70)

    print(f"Chunks file: {CHUNKS_FILE}")

    if not CHUNKS_FILE.exists():
        print("[FAIL] chunks.jsonl not found.")
        raise SystemExit(1)

    retriever = BM25Retriever(
        chunks_file=str(CHUNKS_FILE)
    )

    count = retriever.count()

    print(f"[OK] BM25 index created")
    print(f"[OK] Indexed chunks: {count}")

    test_queries = [
        "Apple total net sales fiscal year 2025",
        "Microsoft operating income fiscal year 2025",
        "NVIDIA research and development expenses",
    ]

    print()
    print("Running retrieval tests...")
    print("-" * 70)

    successful_queries = 0

    for query in test_queries:
        results = retriever.search(
            query,
            top_k=5,
        )

        if results:
            successful_queries += 1

            print(f"[OK] Query: {query}")
            print(f"     Results: {len(results)}")
            print(
                f"     Top BM25 score: "
                f"{results[0]['score']:.4f}"
            )
        else:
            print(f"[FAIL] Query returned no results: {query}")

    manifest = {
        "phase": 6,
        "component": "BM25 lexical retrieval",
        "chunks_file": str(CHUNKS_FILE),
        "indexed_chunks": count,
        "test_queries": len(test_queries),
        "successful_queries": successful_queries,
        "status": "passed"
        if successful_queries == len(test_queries)
        else "failed",
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
    print("PHASE 6 BM25 SUMMARY")
    print("=" * 70)
    print(f"Chunks indexed      : {count}")
    print(f"Test queries        : {len(test_queries)}")
    print(f"Successful queries  : {successful_queries}")
    print(f"Manifest            : {MANIFEST_FILE}")

    if successful_queries != len(test_queries):
        print()
        print("[FAIL] BM25 retrieval validation failed.")
        raise SystemExit(1)

    print()
    print("Phase 6 BM25 lexical retrieval completed successfully.")


if __name__ == "__main__":
    main()
