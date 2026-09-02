from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import FinInsightPipeline


QUESTIONS = ROOT / "evaluation" / "questions.json"
RESULTS = ROOT / "evaluation" / "llm_results.json"


def main():
    questions = json.loads(
        QUESTIONS.read_text(encoding="utf-8")
    )

    pipeline = FinInsightPipeline()

    passed = 0
    records = []

    print("=" * 70)
    print("FinInsight-AI — LLM EVALUATION")
    print("=" * 70)

    for item in questions:
        result = pipeline.answer(
            item["question"],
            top_k=5,
        )

        answer_lower = result[
            "answer"
        ].lower()

        expected_terms = [
            term.lower()
            for term in item.get(
                "expected_terms",
                [],
            )
        ]

        term_hits = [
            term
            for term in expected_terms
            if term in answer_lower
        ]

        source_files = [
            source.get("filename")
            for source in result["sources"]
        ]

        source_grounded = (
            item["expected_document"]
            in source_files
        )

        answer_match = (
            bool(term_hits)
            if expected_terms
            else True
        )

        success = (
            source_grounded
            and answer_match
        )

        passed += int(success)

        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result["answer"],
                "source_grounded": source_grounded,
                "answer_match": answer_match,
                "success": success,
                "sources": result["sources"],
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {item['id']}"
        )

    score = (
        passed / len(questions)
    )

    output = {
        "questions": len(questions),
        "passed": passed,
        "score": score,
        "results": records,
    }

    RESULTS.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("LLM EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"Passed     : "
        f"{passed}/{len(questions)}"
    )
    print(
        f"Score      : "
        f"{score:.2%}"
    )
    print(
        f"Results    : "
        f"{RESULTS}"
    )


if __name__ == "__main__":
    main()
