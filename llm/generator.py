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
