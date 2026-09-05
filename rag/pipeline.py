from __future__ import annotations

import time

from rag.hybrid_retriever import HybridRetriever
from rag.query_rewriter import FinancialQueryRewriter
from rag.reranker import DocumentReranker
from llm.generator import FinancialLLM


class FinInsightPipeline:
    """Query rewrite -> hybrid retrieval -> cross-encoder rerank -> grounded LLM."""

    def __init__(self, use_query_rewrite: bool = True, use_reranker: bool = True):
        self.retriever = HybridRetriever(rrf_k=60)
        self.query_rewriter = FinancialQueryRewriter() if use_query_rewrite else None
        self.reranker = DocumentReranker() if use_reranker else None
        self.llm = FinancialLLM()

    @staticmethod
    def build_context(results):
        sections = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            filename = result.get("filename") or metadata.get("filename") or "Unknown"
            document_id = result.get("document_id") or metadata.get("document_id") or "Unknown"
            chunk_id = result.get("chunk_id", "Unknown")
            text = result.get("text", "")
            sections.append(
                f"SOURCE {index}\nDocument: {filename}\nDocument ID: {document_id}\nChunk ID: {chunk_id}\n\n{text}".strip()
            )
        return "\n\n".join(sections)

    def retrieve(self, question: str, top_k: int = 5, candidate_k: int = 20):
        rewritten = self.query_rewriter.rewrite(question) if self.query_rewriter else question
        results = self.retriever.retrieve(rewritten, top_k=top_k, candidate_k=candidate_k)
        if self.reranker:
            rerank_candidates = self.retriever.retrieve(rewritten, top_k=candidate_k, candidate_k=candidate_k)
            results = self.reranker.rerank(rewritten, rerank_candidates, top_k=top_k)
        return rewritten, results

    def answer(self, question: str, top_k: int = 5):
        start = time.perf_counter()
        rewritten, results = self.retrieve(question, top_k=top_k, candidate_k=max(20, top_k * 4))
        context = self.build_context(results)
        answer = self.llm.generate(question=question, context=context, style="grounded")
        latency_ms = (time.perf_counter() - start) * 1000

        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append({
                "filename": result.get("filename") or metadata.get("filename"),
                "document_id": result.get("document_id") or metadata.get("document_id"),
                "chunk_id": result.get("chunk_id"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
            })

        return {
            "question": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
        }
