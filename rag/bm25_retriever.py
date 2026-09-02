from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 lexical retriever for FinInsight-AI."""

    def __init__(
        self,
        chunks_file: str = "data/chunks/chunks.jsonl",
    ):
        self.chunks_file = Path(chunks_file)

        self.records: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self.tokenized_documents: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

        self._load_chunks()
        self._build_index()

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Convert text into simple normalized lexical tokens."""
        text = text.lower()

        # Preserve useful financial characters such as numbers,
        # decimals, percentages, and common financial terminology.
        tokens = re.findall(
            r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?",
            text,
        )

        return tokens

    # ---------------------------------------------------------
    # Load chunks
    # ---------------------------------------------------------

    def _load_chunks(self) -> None:
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}"
            )

        with self.chunks_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if "text" not in record:
                    raise ValueError(
                        "Chunk record is missing required 'text' field."
                    )

                self.records.append(record)
                self.documents.append(record["text"])

        if not self.records:
            raise ValueError(
                "No chunk records were found in the chunks file."
            )

    # ---------------------------------------------------------
    # Build BM25 index
    # ---------------------------------------------------------

    def _build_index(self) -> None:
        self.tokenized_documents = [
            self.tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top BM25 lexical matches."""

        if not query.strip():
            return []

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been initialized.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []

        for index in ranked_indices[:top_k]:
            record = dict(self.records[index])

            record["score"] = float(scores[index])
            record["retrieval_method"] = "bm25"

            results.append(record)

        return results

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.records)
