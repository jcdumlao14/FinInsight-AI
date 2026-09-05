from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.hybrid_retriever import HybridRetriever
from rag.query_rewriter import FinancialQueryRewriter
from rag.reranker import DocumentReranker

QUESTIONS = ROOT / "evaluation" / "questions.json"
RESULTS = ROOT / "evaluation" / "retrieval_comparison.json"


def filename_of(result):
    return result.get("filename") or result.get("metadata", {}).get("filename")


def reciprocal_rank(results, expected):
    for rank, result in enumerate(results, 1):
        if filename_of(result) == expected:
            return 1.0 / rank
    return 0.0


def score(results, expected):
    names = [filename_of(r) for r in results]
    return {
        "hit_at_1": expected in names[:1],
        "hit_at_3": expected in names[:3],
        "hit_at_5": expected in names[:5],
        "reciprocal_rank": reciprocal_rank(results, expected),
    }


def main():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = HybridRetriever(rrf_k=60)
    rewriter = FinancialQueryRewriter()
    reranker = DocumentReranker()
    methods = {"bm25": [], "vector": [], "hybrid_rrf": [], "reranked_hybrid": []}

    for item in questions:
        original = item["question"]
        query = rewriter.rewrite(original)
        expected = item["expected_document"]
        candidates = retriever.retrieve(query, top_k=20, candidate_k=20)
        raw = {
            "bm25": retriever.bm25_search(query, top_k=5),
            "vector": retriever.vector_search(query, top_k=5),
            "hybrid_rrf": candidates[:5],
            "reranked_hybrid": reranker.rerank(query, candidates, top_k=5),
        }
        for method, results in raw.items():
            t0 = time.perf_counter()
            metrics = score(results, expected)
            metrics["latency_ms"] = (time.perf_counter() - t0) * 1000
            methods[method].append(metrics)

    summary = {}
    for method, rows in methods.items():
        n = len(rows)
        summary[method] = {
            "hit_rate_at_1": sum(r["hit_at_1"] for r in rows) / n,
            "hit_rate_at_3": sum(r["hit_at_3"] for r in rows) / n,
            "hit_rate_at_5": sum(r["hit_at_5"] for r in rows) / n,
            "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
            "mean_latency_ms": sum(r["latency_ms"] for r in rows) / n,
        }

    winner = max(summary, key=lambda m: (summary[m]["mrr"], summary[m]["hit_rate_at_5"], -summary[m]["mean_latency_ms"]))
    output = {
        "evaluation": "BM25 vs Vector vs Hybrid RRF vs Reranked Hybrid",
        "questions": len(questions),
        "methods": summary,
        "best_method": winner,
        "selection_rule": "highest MRR, then Hit@5, then lowest measured evaluation overhead",
    }
    RESULTS.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (ROOT / "evaluation" / "best_retrieval.json").write_text(json.dumps({"best_method": winner}, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))



if __name__ == "__main__":
    main()
