from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import VectorStore


CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
MANIFEST_FILE = PROJECT_ROOT / "data" / "metadata" / "embedding_manifest.json"
VECTOR_DB = PROJECT_ROOT / "data" / "vector_db"

EXPECTED_CHUNKS = 1104
EXPECTED_DIMENSION = 384


def main():
    print("=" * 70)
    print("FinInsight-AI — PHASE 5 VALIDATION")
    print("=" * 70)

    errors = 0

    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------

    if VECTOR_DB.exists():
        print("[OK] Directory: data/vector_db")
    else:
        print("[FAIL] Missing directory: data/vector_db")
        errors += 1

    # ------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------

    if CHUNKS_FILE.exists():
        print("[OK] File:      data/chunks/chunks.jsonl")
    else:
        print("[FAIL] Missing file: data/chunks/chunks.jsonl")
        errors += 1

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    if MANIFEST_FILE.exists():
        print("[OK] File:      data/metadata/embedding_manifest.json")
    else:
        print("[FAIL] Missing file: data/metadata/embedding_manifest.json")
        errors += 1

    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(
                MANIFEST_FILE.read_text(encoding="utf-8")
            )

            print("[OK] Valid JSON: embedding_manifest.json")

            indexed = manifest.get("indexed_vectors")
            dimension = manifest.get("embedding_dimension")
            model = manifest.get("embedding_model")

            if indexed == EXPECTED_CHUNKS:
                print(f"[OK] Manifest vectors: {indexed:,}")
            else:
                print(
                    f"[FAIL] Manifest vectors: "
                    f"{indexed:,} (expected {EXPECTED_CHUNKS:,})"
                )
                errors += 1

            if dimension == EXPECTED_DIMENSION:
                print(f"[OK] Embedding dimension: {dimension}")
            else:
                print(
                    f"[FAIL] Embedding dimension: "
                    f"{dimension} (expected {EXPECTED_DIMENSION})"
                )
                errors += 1

            if model == "all-MiniLM-L6-v2":
                print(f"[OK] Embedding model: {model}")
            else:
                print(f"[FAIL] Unexpected embedding model: {model}")
                errors += 1

        except Exception as exc:
            print(f"[FAIL] Invalid embedding manifest: {exc}")
            errors += 1

    # ------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------

    try:
        vector_store = VectorStore(
            persist_directory=str(VECTOR_DB)
        )

        count = vector_store.count()

        if count == EXPECTED_CHUNKS:
            print(f"[OK] ChromaDB vectors: {count:,}")
        else:
            print(
                f"[FAIL] ChromaDB vectors: "
                f"{count:,} (expected {EXPECTED_CHUNKS:,})"
            )
            errors += 1

    except Exception as exc:
        print(f"[FAIL] ChromaDB validation error: {exc}")
        errors += 1

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print()
    print("-" * 70)

    if errors == 0:
        print("VALIDATION PASSED")
        print("Phase 5 embeddings and vector database are ready.")
    else:
        print(f"VALIDATION FAILED — {errors} issue(s)")

    raise SystemExit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
