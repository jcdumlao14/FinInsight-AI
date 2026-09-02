from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED = [
    "rag/embedding.py",
    "rag/vector_store.py",
    "rag/bm25_retriever.py",
    "rag/hybrid_retriever.py",
    "rag/pipeline.py",
    "llm/generator.py",
    "evaluation/questions.json",
    "evaluation/retrieval_evaluation.py",
    "evaluation/llm_evaluation.py",
    "app/streamlit_app.py",
    "monitoring/feedback.py",
    "monitoring/dashboard.py",
    "Dockerfile",
    "Dockerfile.monitoring",
    "docker-compose.yml",
    "requirements.txt",
]


def main():
    print("=" * 70)
    print("FinInsight-AI — RELEASE VALIDATION")
    print("=" * 70)

    errors = 0

    for name in REQUIRED:
        path = ROOT / name

        if path.exists():
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            errors += 1

    print()
    print("Checking Python syntax...")

    for path in ROOT.rglob("*.py"):
        if (
            ".venv" in path.parts
            or "__pycache__" in path.parts
        ):
            continue

        try:
            ast.parse(
                path.read_text(
                    encoding="utf-8-sig"
                )
            )

        except Exception as exc:
            print(
                f"[FAIL] {path.relative_to(ROOT)} "
                f"— {exc}"
            )

            errors += 1

    manifest = (
        ROOT
        / "data"
        / "metadata"
        / "hybrid_retrieval_manifest.json"
    )

    if manifest.exists():
        data = json.loads(
            manifest.read_text(
                encoding="utf-8"
            )
        )

        if data.get("status") == "passed":
            print(
                "[PASS] Hybrid retrieval manifest"
            )
        else:
            print(
                "[FAIL] Hybrid retrieval manifest"
            )

            errors += 1

    print()
    print("=" * 70)

    if errors == 0:
        print("RESULT: READY FOR RELEASE")
        print("=" * 70)
        return 0

    print(
        f"RESULT: NOT READY — "
        f"{errors} issue(s)"
    )
    print("=" * 70)

    return 1


if __name__ == "__main__":
    sys.exit(main())
