from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb


DEFAULT_COLLECTION = "fininsight_documents"


class VectorStore:
    """Persistent ChromaDB vector store for FinInsight-AI."""

    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "FinInsight-AI financial document chunks"},
        )

    def count(self) -> int:
        """Return the number of stored documents."""
        return self.collection.count()

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add embedded chunks to ChromaDB."""
        if not ids:
            return

        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError("IDs, documents, embeddings, and metadata must have equal lengths.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Search the vector database using a query embedding."""
        if self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
