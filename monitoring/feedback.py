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
