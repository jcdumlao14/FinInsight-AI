from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import FinInsightPipeline

QUESTIONS = ROOT / "evaluation" / "questions.json"
RESULTS = ROOT / "evaluation" / "llm_comparison.json"
STYLES = ["basic", "grounded", "structured"]


def main():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    pipeline = FinInsightPipeline()
    rows = {style: [] for style in STYLES}

    for item in questions:
        rewritten, retrieved = pipeline.retrieve(item["question"], top_k=5, candidate_k=20)
        context = pipeline.build_context(retrieved)
        expected_doc = item["expected_document"]
        grounded = any((r.get("filename") or r.get("metadata", {}).get("filename")) == expected_doc for r in retrieved)
        terms = [t.lower() for t in item.get("expected_terms", [])]
        for style in STYLES:
            answer = pipeline.llm.generate(item["question"], context, style=style)
            low = answer.lower()
            term_hits = sum(t in low for t in terms)
            term_score = term_hits / len(terms) if terms else 1.0
            score = 0.6 * term_score + 0.4 * float(grounded)
            rows[style].append({"id": item["id"], "style": style, "grounded": grounded, "term_score": term_score, "score": score})

    summary = {}
    for style, values in rows.items():
        summary[style] = {
            "mean_score": sum(x["score"] for x in values) / len(values),
            "mean_term_score": sum(x["term_score"] for x in values) / len(values),
            "grounded_rate": sum(x["grounded"] for x in values) / len(values),
        }
    winner = max(summary, key=lambda s: summary[s]["mean_score"])
    output = {
        "evaluation": "Basic vs Grounded vs Structured prompts",
        "questions": len(questions),
        "styles": summary,
        "best_prompt_style": winner,
        "selection_rule": "highest mean evaluation score",
    }
    RESULTS.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (ROOT / "evaluation" / "best_llm_prompt.json").write_text(json.dumps({"best_prompt_style": winner}, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))



if __name__ == "__main__":
    main()
