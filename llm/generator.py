from __future__ import annotations

import os
import time
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


PROMPTS: Dict[str, str] = {
    "basic": """Answer the question using only the retrieved financial context. Be concise and do not invent facts.
Question: {question}
Context:
{context}
Answer:""",
    "grounded": """You are FinInsight AI, a financial research assistant.
Answer using ONLY the retrieved financial context.
Rules: do not invent figures; preserve units and periods; state when evidence is unavailable;
refer to evidence as Source 1, Source 2, etc.; give a concise answer followed by a short explanation.
Question: {question}
Retrieved financial context:
{context}
Answer:""",
    "structured": """You are a meticulous financial analyst. Use ONLY the retrieved context.
Return a short response with: Direct Answer, Evidence, and Interpretation.
Never infer a number that is not supported by the context. Preserve fiscal year, currency, units, and percentages.
Question: {question}
Retrieved financial context:
{context}
Response:""",
}


class FinancialLLM:
    """Gemini generator with three evaluable prompt strategies."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def build_prompt(self, question: str, context: str, style: str = "grounded") -> str:
        if style not in PROMPTS:
            raise ValueError(f"Unknown prompt style: {style}")
        return PROMPTS[style].format(question=question, context=context)

    def generate(self, question: str, context: str, retries: int = 3, style: str = "grounded") -> str:
        prompt = self.build_prompt(question, context, style)
        last_error = None
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=prompt)
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                raise RuntimeError("Gemini returned an empty response.")
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM generation failed: {last_error}")
