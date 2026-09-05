from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "requirements.txt", ".env.example", "Dockerfile", "Dockerfile.monitoring", "docker-compose.yml",
    "rag/query_rewriter.py", "rag/reranker.py", "llm/generator.py", "rag/pipeline.py",
    "evaluation/retrieval_evaluation.py", "evaluation/llm_evaluation.py", "monitoring/dashboard.py",
    "scripts/prefect_ingestion.py", "data/README.md", "data/SOURCES.md",
]

CRITERIA = {
    "problem_description": "README.md",
    "knowledge_base_and_llm": "rag/pipeline.py",
    "retrieval_comparison": "evaluation/retrieval_evaluation.py",
    "llm_comparison": "evaluation/llm_evaluation.py",
    "interface": "app/streamlit_app.py",
    "automated_ingestion": "scripts/prefect_ingestion.py",
    "monitoring_5_plus_charts": "monitoring/dashboard.py",
    "containerization": "docker-compose.yml",
    "reproducibility": "requirements.txt",
    "hybrid_search": "rag/hybrid_retriever.py",
    "document_reranking": "rag/reranker.py",
    "query_rewriting": "rag/query_rewriter.py",
}


def main():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    report = {"project": "FinInsight-AI", "required_files": len(REQUIRED), "missing_files": missing, "criteria": {k: (ROOT / v).exists() for k, v in CRITERIA.items()}}
    report["ready_for_rubric_review"] = not missing
    out = ROOT / "evaluation" / "rubric_release_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if missing else 0)



if __name__ == "__main__":
    main()
