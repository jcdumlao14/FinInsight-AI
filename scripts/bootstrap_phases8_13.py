from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FILES = {
    # ============================================================
    # PHASE 8 — RETRIEVAL EVALUATION DATA
    # ============================================================

    "evaluation/questions.json": r'''
[
  {
    "id": "apple_sales_2025",
    "question": "What was Apple's total net sales in fiscal year 2025?",
    "expected_document": "apple_10k_2025.pdf",
    "expected_terms": ["net sales", "416,161"]
  },
  {
    "id": "apple_net_income_2025",
    "question": "What was Apple's net income in fiscal year 2025?",
    "expected_document": "apple_10k_2025.pdf",
    "expected_terms": ["net income", "112,010"]
  },
  {
    "id": "apple_operating_income_2025",
    "question": "What was Apple's operating income in fiscal year 2025?",
    "expected_document": "apple_10k_2025.pdf",
    "expected_terms": ["operating income"]
  },
  {
    "id": "apple_gross_margin_2025",
    "question": "What was Apple's gross margin in fiscal year 2025?",
    "expected_document": "apple_10k_2025.pdf",
    "expected_terms": ["gross margin"]
  },
  {
    "id": "nvidia_revenue_2026",
    "question": "What was NVIDIA's revenue in fiscal year 2026?",
    "expected_document": "nvidia_10k_2026.pdf",
    "expected_terms": ["revenue", "215,938"]
  },
  {
    "id": "nvidia_rd_2026",
    "question": "What were NVIDIA's research and development expenses?",
    "expected_document": "nvidia_10k_2026.pdf",
    "expected_terms": ["research and development"]
  },
  {
    "id": "nvidia_operating_expenses_2026",
    "question": "What were NVIDIA's total operating expenses?",
    "expected_document": "nvidia_10k_2026.pdf",
    "expected_terms": ["operating expenses"]
  },
  {
    "id": "nvidia_business_2026",
    "question": "Describe NVIDIA's revenue performance in fiscal year 2026.",
    "expected_document": "nvidia_10k_2026.pdf",
    "expected_terms": ["revenue"]
  }
]
''',

    # ============================================================
    # PHASE 8 — RETRIEVAL EVALUATION
    # ============================================================

    "evaluation/retrieval_evaluation.py": r'''
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
''',

    # ============================================================
    # PHASE 9 — GEMINI GENERATOR
    # ============================================================

    "llm/generator.py": r'''
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()


class FinancialLLM:
    """Grounded financial answer generator."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
    ):
        self.model_name = model_name

        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        question: str,
        context: str,
        retries: int = 3,
    ) -> str:
        prompt = f"""
You are FinInsight AI, a financial research assistant.

Answer the user's question using ONLY the retrieved
financial context below.

Rules:
1. Do not invent financial figures.
2. Do not use unsupported information.
3. Preserve units such as millions, billions, and percentages.
4. If the answer is unavailable, explicitly say so.
5. Give a concise answer followed by a short explanation.
6. Refer to the evidence as Source 1, Source 2, etc.

QUESTION:
{question}

RETRIEVED FINANCIAL CONTEXT:
{context}

ANSWER:
""".strip()

        last_error = None

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if text:
                    return text.strip()

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            except Exception as exc:
                last_error = exc

                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM generation failed: {last_error}"
        )
''',

    # ============================================================
    # PHASE 10 — END-TO-END PIPELINE
    # ============================================================

    "rag/pipeline.py": r'''
from __future__ import annotations

from rag.hybrid_retriever import HybridRetriever
from llm.generator import FinancialLLM


class FinInsightPipeline:

    def __init__(self):
        self.retriever = HybridRetriever(
            rrf_k=60
        )

        self.llm = FinancialLLM()

    @staticmethod
    def build_context(results):
        sections = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            metadata = result.get(
                "metadata",
                {},
            )

            filename = (
                result.get("filename")
                or metadata.get("filename")
                or "Unknown"
            )

            document_id = (
                result.get("document_id")
                or metadata.get("document_id")
                or "Unknown"
            )

            chunk_id = result.get(
                "chunk_id",
                "Unknown",
            )

            text = result.get(
                "text",
                "",
            )

            sections.append(
                f"""
SOURCE {index}
Document: {filename}
Document ID: {document_id}
Chunk ID: {chunk_id}

{text}
""".strip()
            )

        return "\n\n".join(sections)

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            candidate_k=20,
        )

        context = self.build_context(
            results
        )

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        sources = []

        for result in results:
            metadata = result.get(
                "metadata",
                {},
            )

            sources.append(
                {
                    "filename": (
                        result.get("filename")
                        or metadata.get("filename")
                    ),
                    "document_id": (
                        result.get("document_id")
                        or metadata.get("document_id")
                    ),
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "rrf_score": result.get(
                        "rrf_score"
                    ),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
''',

    # ============================================================
    # PHASE 9/10 — LLM EVALUATION
    # ============================================================

    "evaluation/llm_evaluation.py": r'''
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
''',

    # ============================================================
    # PHASE 11 — STREAMLIT
    # ============================================================

    "app/streamlit_app.py": r'''
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import FinInsightPipeline
from monitoring.feedback import save_feedback


st.set_page_config(
    page_title="FinInsight AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FinInsight AI")
st.subheader(
    "Intelligent Financial Research with Hybrid RAG"
)

st.markdown(
    """
Ask questions about the indexed financial reports.

**Retrieval:** Vector Search + BM25 + Reciprocal Rank Fusion  
**Generation:** Gemini 2.5 Flash  
**Evidence:** Source-grounded financial document chunks
"""
)

st.divider()


@st.cache_resource
def load_pipeline():
    return FinInsightPipeline()


try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error(
        "Unable to initialize FinInsight AI."
    )
    st.exception(exc)
    st.stop()


if "result" not in st.session_state:
    st.session_state.result = None

if "question" not in st.session_state:
    st.session_state.question = ""


question = st.text_area(
    "Financial question",
    placeholder=(
        "Example: What was Apple's total "
        "net sales in fiscal year 2025?"
    ),
)

ask = st.button(
    "🔎 Ask FinInsight AI",
    type="primary",
)


if ask:
    if not question.strip():
        st.warning(
            "Please enter a question."
        )
    else:
        with st.spinner(
            "Retrieving evidence and generating answer..."
        ):
            try:
                result = pipeline.answer(
                    question.strip(),
                    top_k=5,
                )

                st.session_state.result = result
                st.session_state.question = question.strip()

            except Exception as exc:
                st.error(
                    "Unable to process the question."
                )
                st.exception(exc)


result = st.session_state.result


if result:
    st.subheader("💡 Answer")

    st.success(
        result["answer"]
    )

    st.subheader(
        "📚 Retrieved Sources"
    )

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        with st.expander(
            f"Source {index}: "
            f"{source.get('filename', 'Unknown')}"
        ):
            st.write(
                "**Document:**",
                source.get("filename"),
            )

            st.write(
                "**Chunk:**",
                source.get("chunk_id"),
            )

            st.write(
                "**RRF Score:**",
                source.get("rrf_score"),
            )

    st.divider()

    st.subheader(
        "📝 Was this answer helpful?"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful"):
            save_feedback(
                question=st.session_state.question,
                answer=result["answer"],
                feedback="positive",
                sources=result["sources"],
            )

            st.success(
                "Thank you for your feedback!"
            )

    with col2:
        if st.button("👎 Not Helpful"):
            save_feedback(
                question=st.session_state.question,
                answer=result["answer"],
                feedback="negative",
                sources=result["sources"],
            )

            st.success(
                "Thank you for your feedback!"
            )
''',

    # ============================================================
    # PHASE 12 — FEEDBACK
    # ============================================================

    "monitoring/feedback.py": r'''
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_FILE = (
    ROOT
    / "monitoring"
    / "feedback.json"
)


def load_feedback():
    if not FEEDBACK_FILE.exists():
        return []

    try:
        return json.loads(
            FEEDBACK_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return []


def save_feedback(
    question,
    answer,
    feedback,
    sources=None,
):
    records = load_feedback()

    records.append(
        {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "sources": sources or [],
        }
    )

    FEEDBACK_FILE.write_text(
        json.dumps(
            records,
            indent=2,
        ),
        encoding="utf-8",
    )
''',

    # ============================================================
    # PHASE 12 — MONITORING
    # ============================================================

    "monitoring/dashboard.py": r'''
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.feedback import load_feedback


st.set_page_config(
    page_title="FinInsight AI Monitoring",
    page_icon="📈",
    layout="wide",
)

st.title(
    "📈 FinInsight AI Monitoring Dashboard"
)

records = load_feedback()

if not records:
    st.info(
        "No feedback records are available yet."
    )
    st.stop()


df = pd.DataFrame(records)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df["date"] = df["timestamp"].dt.date

df["feedback"] = (
    df["feedback"]
    .astype(str)
    .str.lower()
)


total = len(df)

positive = (
    df["feedback"]
    .eq("positive")
    .sum()
)

negative = (
    df["feedback"]
    .eq("negative")
    .sum()
)

rate = (
    positive / total * 100
    if total
    else 0
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Feedback",
    total,
)

c2.metric(
    "👍 Positive",
    positive,
)

c3.metric(
    "👎 Negative",
    negative,
)

c4.metric(
    "Positive Rate",
    f"{rate:.1f}%",
)


st.divider()

st.subheader(
    "Feedback Distribution"
)

st.bar_chart(
    df["feedback"]
    .value_counts()
)


st.subheader(
    "Feedback Over Time"
)

over_time = (
    df.groupby(
        ["date", "feedback"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

st.line_chart(
    over_time
)


st.subheader(
    "Recent Questions"
)

columns = [
    "timestamp",
    "question",
    "feedback",
]

st.dataframe(
    df[columns]
    .sort_values(
        "timestamp",
        ascending=False,
    )
    .head(20),
    use_container_width=True,
    hide_index=True,
)
''',

    # ============================================================
    # PHASE 13 — DOCKER
    # ============================================================

    "Dockerfile": r'''
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
''',

    "Dockerfile.monitoring": r'''
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "monitoring/dashboard.py", "--server.address=0.0.0.0", "--server.port=8502"]
''',

    "docker-compose.yml": r'''
services:

  fininsight:
    build:
      context: .
      dockerfile: Dockerfile

    ports:
      - "8501:8501"

    env_file:
      - .env

    volumes:
      - ./data:/app/data
      - ./monitoring:/app/monitoring

    restart: unless-stopped

  monitoring:
    build:
      context: .
      dockerfile: Dockerfile.monitoring

    ports:
      - "8502:8502"

    volumes:
      - ./monitoring:/app/monitoring

    restart: unless-stopped
''',

    ".env.example": r'''
GEMINI_API_KEY=your_gemini_api_key_here
''',

    # ============================================================
    # PHASE 13 — TESTS
    # ============================================================

    "tests/test_retrieval.py": r'''
from rag.hybrid_retriever import HybridRetriever


def test_hybrid_retrieval():
    retriever = HybridRetriever()

    results = retriever.retrieve(
        "Apple total net sales 2025",
        top_k=5,
        candidate_k=10,
    )

    assert len(results) == 5
    assert all(
        "chunk_id" in result
        for result in results
    )
''',

    "tests/test_feedback.py": r'''
from monitoring.feedback import load_feedback


def test_feedback_loader():
    records = load_feedback()

    assert isinstance(
        records,
        list,
    )
''',

    # ============================================================
    # PHASE 13 — RELEASE CHECK
    # ============================================================

    "scripts/release_check.py": r'''
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
''',
}


def main():
    print("=" * 70)
    print("FinInsight-AI — AUTOMATING PHASES 8–13")
    print("=" * 70)

    created = 0
    updated = 0

    for name, content in FILES.items():
        path = ROOT / name

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existed = path.exists()

        path.write_text(
            content.strip() + "\n",
            encoding="utf-8",
        )

        if existed:
            print(
                f"[UPDATED] {name}"
            )

            updated += 1
        else:
            print(
                f"[CREATED] {name}"
            )

            created += 1

    print()
    print(
        f"Created: {created}"
    )

    print(
        f"Updated: {updated}"
    )

    print()
    print(
        "Phase 8–13 project files generated successfully."
    )


if __name__ == "__main__":
    main()