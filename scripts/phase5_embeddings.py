from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python import path BEFORE importing rag.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore


CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
VECTOR_DB = PROJECT_ROOT / "data" / "vector_db"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks():
    records = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def main():
    print("=" * 70)
    print("FinInsight-AI — PHASE 5 EMBEDDINGS & VECTOR DATABASE")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    records = load_chunks()

    print(f"Chunks discovered: {len(records):,}")
    print(f"Embedding model:  {MODEL_NAME}")
    print()

    if not records:
        raise RuntimeError("No chunks were found.")

    required_fields = {"chunk_id", "text"}

    for index, record in enumerate(records):
        missing = required_fields - set(record)

        if missing:
            raise ValueError(
                f"Chunk {index} is missing fields: {missing}"
            )

    print("[OK] Chunk structure validated")

    embedding_model = EmbeddingModel(MODEL_NAME)

    vector_store = VectorStore(
        persist_directory=str(VECTOR_DB)
    )

    print("[OK] ChromaDB collection ready")
    print(f"[INFO] Existing vectors: {vector_store.count():,}")
    print()

    texts = [record["text"] for record in records]
    ids = [str(record["chunk_id"]) for record in records]

    metadatas = []

    for record in records:
        metadata = {}

        for key, value in record.items():
            if key in {"chunk_id", "text"}:
                continue

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        metadatas.append(metadata)

    print("Generating embeddings...")
    print("-" * 70)

    embeddings = embedding_model.embed_documents(
        texts,
        batch_size=BATCH_SIZE,
    )

    if len(embeddings) != len(records):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    print()
    print("[OK] Embeddings generated")
    print(f"[OK] Embedding dimension: {len(embeddings[0])}")
    print()

    print("Indexing into ChromaDB...")
    print("-" * 70)

    for start in range(0, len(records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(records))

        vector_store.add_documents(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"[OK] Indexed {end:,}/{len(records):,}")

    final_count = vector_store.count()

    if final_count != len(records):
        raise RuntimeError(
            f"Expected {len(records):,} vectors but found "
            f"{final_count:,}."
        )

    manifest = {
        "phase": 5,
        "status": "completed",
        "embedding_model": MODEL_NAME,
        "embedding_dimension": len(embeddings[0]),
        "source_chunks": len(records),
        "indexed_vectors": final_count,
        "collection": "fininsight_documents",
        "vector_database": str(VECTOR_DB),
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        METADATA_DIR / "embedding_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("PHASE 5 EMBEDDING SUMMARY")
    print("=" * 70)
    print(f"Chunks discovered : {len(records):,}")
    print(f"Vectors indexed   : {final_count:,}")
    print(f"Embedding model   : {MODEL_NAME}")
    print(f"Dimensions        : {len(embeddings[0])}")
    print(f"Manifest          : {manifest_path}")
    print()
    print("Phase 5 embedding and vector indexing completed successfully.")


if __name__ == "__main__":
    main()
