from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.hybrid_retriever import HybridRetriever


QUESTIONS = ROOT / "evaluation" / "questions.json"
RESULTS = ROOT / "evaluation" / "retrieval_results.json"


def reciprocal_rank(results, expected_document):
    for rank, result in enumerate(results, start=1):
        filename = (
            result.get("filename")
            or result.get("metadata", {}).get("filename")
        )

        if filename == expected_document:
            return 1.0 / rank

    return 0.0


def main():
    questions = json.loads(
        QUESTIONS.read_text(encoding="utf-8")
    )

    retriever = HybridRetriever(rrf_k=60)

    records = []

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    mrr_total = 0.0
    latencies = []

    print("=" * 70)
    print("FinInsight-AI — PHASE 8 RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in questions:
        start = time.perf_counter()

        results = retriever.retrieve(
            item["question"],
            top_k=5,
            candidate_k=20,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency_ms)

        filenames = [
            (
                result.get("filename")
                or result.get("metadata", {}).get("filename")
            )
            for result in results
        ]

        expected = item["expected_document"]

        h1 = expected in filenames[:1]
        h3 = expected in filenames[:3]
        h5 = expected in filenames[:5]

        hit_at_1 += int(h1)
        hit_at_3 += int(h3)
        hit_at_5 += int(h5)

        rr = reciprocal_rank(
            results,
            expected,
        )

        mrr_total += rr

        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_document": expected,
                "retrieved_documents": filenames,
                "hit_at_1": h1,
                "hit_at_3": h3,
                "hit_at_5": h5,
                "reciprocal_rank": rr,
                "latency_ms": latency_ms,
            }
        )

        status = "PASS" if h5 else "FAIL"

        print(
            f"[{status}] {item['id']} "
            f"| latency={latency_ms:.2f} ms"
        )

    n = len(questions)

    summary = {
        "questions": n,
        "hit_rate_at_1": hit_at_1 / n,
        "hit_rate_at_3": hit_at_3 / n,
        "hit_rate_at_5": hit_at_5 / n,
        "mrr": mrr_total / n,
        "mean_latency_ms": sum(latencies) / n,
        "results": records,
    }

    RESULTS.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("PHASE 8 SUMMARY")
    print("=" * 70)
    print(f"Questions       : {n}")
    print(f"Hit Rate@1      : {summary['hit_rate_at_1']:.2%}")
    print(f"Hit Rate@3      : {summary['hit_rate_at_3']:.2%}")
    print(f"Hit Rate@5      : {summary['hit_rate_at_5']:.2%}")
    print(f"MRR             : {summary['mrr']:.4f}")
    print(
        f"Mean latency    : "
        f"{summary['mean_latency_ms']:.2f} ms"
    )
    print(f"Results         : {RESULTS}")

    if summary["hit_rate_at_5"] < 0.75:
        raise SystemExit(
            "Phase 8 FAILED: Hit Rate@5 below 75%."
        )

    print()
    print("Phase 8 retrieval evaluation PASSED.")


if __name__ == "__main__":
    main()
