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
