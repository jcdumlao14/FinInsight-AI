from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.feedback import load_feedback

st.set_page_config(page_title="FinInsight AI Monitoring", page_icon="📈", layout="wide")
st.title("📈 FinInsight AI Monitoring Dashboard")
st.caption("Operational monitoring for user feedback, query activity, latency, retrieval and answer quality signals.")

records = load_feedback()
if not records:
    st.info("No feedback records are available yet. Use the application and submit feedback.")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["feedback"] = df["feedback"].astype(str).str.lower()
df["question_length"] = df["question"].astype(str).str.len()
df["source_count"] = df.get("sources", pd.Series([[]] * len(df))).apply(lambda x: len(x) if isinstance(x, list) else 0)
if "latency_ms" not in df:
    df["latency_ms"] = pd.NA

positive = int(df["feedback"].eq("positive").sum())
negative = int(df["feedback"].eq("negative").sum())
total = len(df)
rate = positive / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Feedback", total)
c2.metric("👍 Positive", positive)
c3.metric("👎 Negative", negative)
c4.metric("Positive Rate", f"{rate:.1f}%")

st.divider()

# 1
st.subheader("1. Feedback Distribution")
st.bar_chart(df["feedback"].value_counts())

# 2
st.subheader("2. Feedback Over Time")
over_time = df.groupby(["date", "feedback"]).size().unstack(fill_value=0)
st.line_chart(over_time)

# 3
st.subheader("3. Daily Positive Rate")
daily = df.groupby("date").agg(total=("feedback", "size"), positive=("feedback", lambda s: (s == "positive").sum()))
daily["positive_rate_pct"] = daily["positive"] / daily["total"] * 100
st.line_chart(daily["positive_rate_pct"])

# 4
st.subheader("4. Query Volume by Hour")
st.bar_chart(df.groupby("hour").size())

# 5
st.subheader("5. Question Length Distribution")
st.bar_chart(df["question_length"].value_counts().sort_index())

# 6
st.subheader("6. Retrieved Sources per Answer")
st.bar_chart(df["source_count"].value_counts().sort_index())

# 7 when latency is available
if df["latency_ms"].notna().any():
    st.subheader("7. Answer Latency")
    latency = df.dropna(subset=["latency_ms"]).groupby("date")["latency_ms"].mean()
    st.line_chart(latency)

st.subheader("Recent Feedback")
cols = [c for c in ["timestamp", "question", "feedback", "latency_ms"] if c in df.columns]
st.dataframe(df[cols].sort_values("timestamp", ascending=False).head(30), use_container_width=True, hide_index=True)
