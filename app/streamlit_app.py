from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import FinInsightPipeline
from monitoring.feedback import save_feedback


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FinInsight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 8% 4%, rgba(50, 115, 220, .08), transparent 26%),
            radial-gradient(circle at 92% 18%, rgba(111, 78, 255, .07), transparent 24%),
            #f7f9fc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        padding: 42px 44px 34px 44px;
        color: white;
        min-height: 360px;
        background:
            radial-gradient(circle at 78% 32%, rgba(44, 151, 255, .42), transparent 22%),
            radial-gradient(circle at 90% 75%, rgba(0, 213, 190, .18), transparent 25%),
            linear-gradient(120deg, #061b38 0%, #0a315c 55%, #0d4d7d 100%);
        box-shadow: 0 16px 45px rgba(15, 48, 85, .20);
    }

    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        opacity: .20;
        background:
            repeating-linear-gradient(90deg, transparent 0 42px, rgba(255,255,255,.16) 43px 44px),
            repeating-linear-gradient(0deg, transparent 0 42px, rgba(255,255,255,.12) 43px 44px);
        transform: perspective(500px) rotateY(-7deg);
        transform-origin: right center;
        pointer-events: none;
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 720px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 16px;
    }

    .brand-icon {
        width: 54px;
        height: 54px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.28);
        box-shadow: 0 8px 22px rgba(0,0,0,.16);
    }

    .brand-name {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -.8px;
    }

    .hero h2 {
        font-size: 28px;
        margin: 0 0 10px 0;
        color: white;
    }

    .hero p {
        font-size: 16px;
        line-height: 1.65;
        color: rgba(255,255,255,.86);
        margin: 0;
    }

    .feature-row {
        position: relative;
        z-index: 3;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 28px;
    }

    .feature {
        padding: 15px 16px;
        border: 1px solid rgba(255,255,255,.20);
        border-radius: 14px;
        background: rgba(5, 22, 45, .40);
        backdrop-filter: blur(8px);
    }

    .feature-title {
        font-weight: 750;
        font-size: 15px;
        margin-bottom: 4px;
    }

    .feature-text {
        font-size: 12px;
        color: rgba(255,255,255,.72);
        line-height: 1.45;
    }

    .section-card {
        background: white;
        border: 1px solid #e3eaf3;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 8px 26px rgba(20, 48, 80, .06);
        margin-top: 18px;
    }

    .section-title {
        color: #12345b;
        font-size: 21px;
        font-weight: 780;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #63748a;
        font-size: 14px;
        margin-bottom: 16px;
    }

    .metric {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        border: 1px solid #e5edf7;
        border-radius: 14px;
        padding: 17px 12px;
        text-align: center;
        min-height: 105px;
    }

    .metric-value {
        color: #153e70;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-label {
        color: #53667d;
        font-size: 12px;
        margin-top: 7px;
    }

    .info-box {
        background: linear-gradient(135deg, #f2f8ff, #ffffff);
        border: 1px solid #dceafa;
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }

    .info-box.green {
        background: linear-gradient(135deg, #f2fcf7, #ffffff);
        border-color: #d8f0e1;
    }

    .info-box.purple {
        background: linear-gradient(135deg, #f8f4ff, #ffffff);
        border-color: #e6dcfb;
    }

    .info-heading {
        font-size: 17px;
        font-weight: 760;
        color: #173b68;
        margin-bottom: 10px;
    }

    .info-item {
        margin: 9px 0;
        color: #52657b;
        font-size: 13px;
        line-height: 1.45;
    }

    .info-item b {
        color: #243e5c;
    }

    .answer-card {
        background: linear-gradient(135deg, #f6fbff, #ffffff);
        border: 1px solid #d9e8f8;
        border-left: 5px solid #2384df;
        border-radius: 15px;
        padding: 22px;
        margin-top: 18px;
    }

    .source-card {
        background: #fbfcfe;
        border: 1px solid #e3eaf3;
        border-radius: 12px;
        padding: 13px 16px;
        margin: 8px 0;
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #e9f9ef;
        color: #168447;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #ccefd9;
    }

    .footer-note {
        margin-top: 20px;
        padding: 14px 18px;
        border-radius: 12px;
        background: #f1f7ff;
        border: 1px solid #d9e8fa;
        color: #536a84;
        font-size: 12px;
    }

    @media (max-width: 800px) {
        .hero { padding: 30px 24px; }
        .brand-name { font-size: 32px; }
        .hero h2 { font-size: 23px; }
        .feature-row { grid-template-columns: 1fr; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================
def load_count(path: Path, default: int) -> int:
    try:
        if not path.exists():
            return default
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("documents", "total_documents", "chunk_count", "total_chunks", "count"):
                value = data.get(key)
                if isinstance(value, int):
                    return value
        return default
    except Exception:
        return default


def get_document_count() -> int:
    manifest = ROOT / "data" / "metadata" / "documents.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                docs = data.get("documents")
                if isinstance(docs, list):
                    return len(docs)
        except Exception:
            pass
    return 2


def get_chunk_count() -> int:
    chunks = ROOT / "data" / "chunks" / "chunks.jsonl"
    if chunks.exists():
        try:
            return sum(1 for line in chunks.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:
            pass
    return 1104


# ============================================================
# HEADER / HERO
# ============================================================
document_count = get_document_count()
chunk_count = get_chunk_count()

# ============================================================
# FININSIGHT AI â€” HERO SECTION
# ============================================================

st.html("""
<style>
.hero {
    position: relative;
    overflow: hidden;
    padding: 38px 40px 34px 40px;
    margin: 0 0 26px 0;
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 88% 12%,
            rgba(43, 183, 229, 0.30),
            transparent 34%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(30, 120, 210, 0.18),
            transparent 38%
        ),
        linear-gradient(
            135deg,
            #071a38 0%,
            #0a315a 52%,
            #07547c 100%
        );
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 20px 50px rgba(5,27,55,0.20);
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: 0.12;
    background-image:
        linear-gradient(
            rgba(255,255,255,0.18) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.18) 1px,
            transparent 1px
        );
    background-size: 34px 34px;
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero h2 {
    margin: 0 0 12px 0;
    color: #ffffff;
    font-size: 31px;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.hero-description {
    max-width: 820px;
    margin: 0 0 28px 0;
    color: #d9ecff;
    font-size: 16px;
    line-height: 1.65;
}

.feature-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}

.feature {
    padding: 19px 20px;
    border-radius: 16px;
    background: rgba(255,255,255,0.085);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.feature-title {
    margin-bottom: 8px;
    color: #ffffff;
    font-size: 15px;
    font-weight: 750;
}

.feature-text {
    color: #cce4f8;
    font-size: 13px;
    line-height: 1.5;
}

@media (max-width: 800px) {
    .feature-row {
        grid-template-columns: 1fr;
    }

    .hero {
        padding: 28px 24px;
    }

    .hero h2 {
        font-size: 25px;
    }
}
</style>

<div class="hero">
    <div class="hero-content">

        <h2>Intelligent Financial Research with Hybrid RAG</h2>

        <p class="hero-description">
            Your AI-powered research assistant for accurate,
            source-grounded insights from indexed financial reports
            and filings.
        </p>

        <div class="feature-row">

            <div class="feature">
                <div class="feature-title">
                    🔎 Hybrid Retrieval
                </div>
                <div class="feature-text">
                    Vector Search + BM25 + Reciprocal Rank Fusion
                </div>
            </div>

            <div class="feature">
                <div class="feature-title">
                    ✨ Smart Generation
                </div>
                <div class="feature-text">
                    Powered by Gemini 2.5 Flash
                </div>
            </div>

            <div class="feature">
                <div class="feature-title">
                    🛡️ Source-Grounded
                </div>
                <div class="feature-text">
                    Answers backed by indexed financial evidence
                </div>
            </div>

        </div>

    </div>
</div>
""")




# ============================================================
# INTRO + METRICS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
left, m1, m2, m3, m4 = st.columns([2.6, 1, 1, 1, 1])

with left:
    st.markdown(
        """
        <div class="info-box">
          <div class="info-heading">ℹ️ What is FinInsight AI?</div>
          <div style="color:#53677e;font-size:14px;line-height:1.6;">
            FinInsight AI helps analysts, researchers, and finance professionals
            quickly find reliable answers from indexed financial reports using
            Hybrid Retrieval-Augmented Generation (RAG).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m1:
    st.markdown(
        f'<div class="metric"><div class="metric-value">{document_count}</div>'
        '<div class="metric-label">Financial Reports<br>Indexed</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f'<div class="metric"><div class="metric-value">{chunk_count:,}</div>'
        '<div class="metric-label">Chunks<br>Embedded</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        '<div class="metric"><div class="metric-value">Hybrid</div>'
        '<div class="metric-label">RRF<br>Retrieval</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        '<div class="metric"><div class="metric-value">Gemini 2.5</div>'
        '<div class="metric-label">Flash<br>Generation</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PIPELINE
# ============================================================
@st.cache_resource
def load_pipeline():
    return FinInsightPipeline()


try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error("Unable to initialize FinInsight AI.")
    st.exception(exc)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "result": None,
    "question": "",
    "question_input": "",
    "example_question": "",
    "feedback_given": None,
    "example_selector": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def apply_example():
    """Load the selected example into the question input."""
    example = st.session_state["example_selector"]

    if example:
        st.session_state["question_input"] = example
        st.session_state["example_question"] = example
        st.session_state["question"] = example


def clear_state():
    """Reset the question, answer, example, and feedback state."""
    st.session_state["result"] = None
    st.session_state["question"] = ""
    st.session_state["question_input"] = ""
    st.session_state["example_question"] = ""
    st.session_state["feedback_given"] = None
    st.session_state["example_selector"] = ""



# ============================================================
# QUESTION CARD
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)

title_col, badge_col = st.columns([5, 1])

with title_col:
    st.markdown(
        '<div class="section-title">💬 Ask a Financial Question</div>'
        '<div class="section-subtitle">'
        'Ask about the indexed financial reports and receive a source-grounded answer.'
        '</div>',
        unsafe_allow_html=True,
    )

with badge_col:
    st.markdown('<div class="badge">🛡️ Source-Grounded</div>', unsafe_allow_html=True)

examples = [
    "",
    "What was Apple's total net sales in fiscal year 2025?",
    "What was Apple's net income in fiscal year 2025?",
    "What was Apple's operating income in fiscal year 2025?",
    "What were NVIDIA's research and development expenses?",
    "What was NVIDIA's revenue in fiscal year 2026?",
]

selected_example = st.selectbox(
    "Example Questions",
    examples,
    index=0,
    key="example_selector",
    on_change=apply_example,
)

question = st.text_area(
    "Financial question",
    key="question_input",
    height=125,
    placeholder=(
        "Example: What was Apple's total net sales in fiscal year 2025?\n\n"
        "Try revenue, net income, operating income, margins, expenses, and more."
    ),
)

ask_col, clear_col, monitor_col = st.columns([1.3, 1, 1.2])

with ask_col:
    ask = st.button(
        "🔎 Ask FinInsight AI",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    st.button(
        "🧹 Clear",
        key="clear_button",
        use_container_width=True,
        on_click=clear_state,
    )

with monitor_col:
    st.link_button(
        "📊 Monitoring Dashboard",
        "http://localhost:8504",
        use_container_width=True,
    )

# Clear is handled by the button callback above.

if ask:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Please enter a financial question.")
    else:
        with st.spinner("🔎 Retrieving evidence and generating a grounded answer..."):
            try:
                result = pipeline.answer(clean_question, top_k=5)
                st.session_state.result = result
                st.session_state.question = clean_question
                st.session_state.feedback_given = None
            except Exception as exc:
                error_text = str(exc)

                if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                    st.error(
                        "Gemini API quota/rate limit was reached. "
                        "Please try again after the quota resets."
                    )
                else:
                    st.error("Unable to process the question.")
                    st.exception(exc)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANSWER + DOWNLOAD + FEEDBACK
# ============================================================
result = st.session_state.result

if result:
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">💡 Answer</div>'
        '<div class="section-subtitle">Generated from retrieved financial evidence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="answer-card">{html.escape(str(answer)).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    download_text = (
        "FinInsight AI â€” Financial Research Answer\n"
        "=" * 48
        + "\n\n"
        f"Question:\n{st.session_state.question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Retrieved Sources:\n"
    )

    for i, source in enumerate(sources, start=1):
        download_text += (
            f"{i}. {source.get('filename', 'Unknown')} | "
            f"Chunk: {source.get('chunk_id', 'Unknown')} | "
            f"RRF Score: {source.get('rrf_score', 'N/A')}\n"
        )

    dcol, spacer = st.columns([1.4, 4])
    with dcol:
        st.download_button(
            "📥 Download Answer",
            data=download_text,
            file_name="fininsight_answer.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("### 📚 Retrieved Sources")

    if sources:
        for index, source in enumerate(sources, start=1):
            filename = source.get("filename") or "Unknown document"
            chunk_id = source.get("chunk_id") or "Unknown"
            score = source.get("rrf_score")

            with st.expander(f"Source {index} · {filename}", expanded=index == 1):
                st.markdown(
                    f"""
                    <div class="source-card">
                      <b>📄 Document:</b> {filename}<br>
                      <b>🔖 Chunk:</b> {chunk_id}<br>
                      <b>🔀 RRF Score:</b> {score if score is not None else "N/A"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No source metadata was returned.")

    st.markdown("### 📝 Was this answer helpful?")

    f1, f2, f3 = st.columns([1, 1, 3])

    with f1:
        positive = st.button(
            "👍 Helpful",
            use_container_width=True,
            disabled=st.session_state.feedback_given is not None,
        )

    with f2:
        negative = st.button(
            "👎 Not Helpful",
            use_container_width=True,
            disabled=st.session_state.feedback_given is not None,
        )

    if positive or negative:
        feedback_value = "positive" if positive else "negative"

        try:
            save_feedback(
                question=st.session_state.question,
                answer=answer,
                feedback=feedback_value,
                sources=sources,
                latency_ms=result.get("latency_ms"),
                rewritten_query=result.get("rewritten_query"),
            )
            st.session_state.feedback_given = feedback_value
            st.success("Thank you! Your feedback has been recorded.")
        except Exception as exc:
            st.error("The answer was generated, but feedback could not be saved.")
            st.exception(exc)

    if st.session_state.feedback_given:
        st.info("Feedback recorded. Thank you for helping improve FinInsight AI.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INFORMATION CARDS
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="section-card" style="margin-top:18px;">
          <div class="info-box green">
            <div class="info-heading">📈 Supported Topics</div>
            <div class="info-item">✓ <b>Financial Performance</b><br>
              Revenue, net income, margins</div>
            <div class="info-item">✓ <b>Financial Position</b><br>
              Assets, liabilities, equity</div>
            <div class="info-item">✓ <b>Cash Flow Analysis</b><br>
              Operating, investing, financing</div>
            <div class="info-item">✓ <b>Ratios & Metrics</b><br>
              Profitability, efficiency, earnings</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="section-card" style="margin-top:18px;">
          <div class="info-box">
            <div class="info-heading">⚙️ How It Works</div>
            <div class="info-item">1. <b>You ask a question</b><br>
              Enter a financial research question.</div>
            <div class="info-item">2. <b>We retrieve evidence</b><br>
              Vector + BM25 retrieval are fused with RRF.</div>
            <div class="info-item">3. <b>AI generates an answer</b><br>
              Gemini 2.5 Flash uses retrieved context.</div>
            <div class="info-item">4. <b>You review and provide feedback</b><br>
              Feedback is recorded for monitoring.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="section-card" style="margin-top:18px;">
          <div class="info-box purple">
            <div class="info-heading">✨ Why FinInsight AI?</div>
            <div class="info-item">✓ <b>Source-Grounded</b><br>
              Answers use retrieved financial evidence.</div>
            <div class="info-item">✓ <b>Hybrid Retrieval</b><br>
              Combines semantic and lexical signals.</div>
            <div class="info-item">✓ <b>Transparent Sources</b><br>
              Inspect the documents and chunks used.</div>
            <div class="info-item">✓ <b>Continuous Improvement</b><br>
              User feedback supports monitoring.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div class="footer-note">
  🛡️ <b>Important:</b> FinInsight AI is a financial research assistant
  for the indexed documents. It is not financial, investment, tax, or legal advice.
  Always verify critical figures against the original financial reports.
</div>
""",
    unsafe_allow_html=True,
)

