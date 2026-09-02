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
