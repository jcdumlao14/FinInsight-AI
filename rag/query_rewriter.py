from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()


class FinancialQueryRewriter:
    """Rewrite financial questions into retrieval-friendly search queries.

    Gemini is used when a key is available. A deterministic fallback keeps the
    pipeline usable for offline/unit-test environments.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = None
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception:
                self.client = None

    @staticmethod
    def _fallback(question: str) -> str:
        q = re.sub(r"\s+", " ", question.strip())
        replacements = {
            "how much did": "what was",
            "how much was": "what was",
            "what did": "what was",
        }
        low = q.lower()
        for old, new in replacements.items():
            if low.startswith(old):
                q = new + q[len(old):]
                break
        return q

    def rewrite(self, question: str) -> str:
        if not question or not question.strip():
            return question
        if self.client is None:
            return self._fallback(question)

        prompt = f"""Rewrite this financial research question into one precise search query.
Preserve company names, fiscal years, metrics, periods, units, and accounting terms.
Do not answer the question. Return only the rewritten query.

Question: {question.strip()}"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            return text.strip() if text else self._fallback(question)
        except Exception:
            return self._fallback(question)
