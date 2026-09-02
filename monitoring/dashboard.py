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
