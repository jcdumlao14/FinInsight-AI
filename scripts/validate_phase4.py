from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

CHUNKS = ROOT / "data" / "chunks"
METADATA = ROOT / "data" / "metadata"

chunks_file = CHUNKS / "chunks.jsonl"
manifest_file = METADATA / "chunking_manifest.json"

print("=" * 70)
print("FinInsight-AI — PHASE 4 VALIDATION")
print("=" * 70)

errors = 0

# Directories
for path in [
    ROOT / "data",
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    CHUNKS,
    METADATA,
]:
    if path.exists():
        print(f"[OK] Directory: {path.relative_to(ROOT)}")
    else:
        print(f"[FAIL] Missing directory: {path.relative_to(ROOT)}")
        errors += 1

# Chunks file
if chunks_file.exists():
    print(f"[OK] File:      {chunks_file.relative_to(ROOT)}")

    chunk_count = 0
    invalid_chunks = 0

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                chunk_count += 1

                required = [
                    "chunk_id",
                    "document_id",
                    "filename",
                    "chunk_index",
                    "text",
                    "character_count",
                ]

                if not all(key in record for key in required):
                    invalid_chunks += 1

                if not record.get("text", "").strip():
                    invalid_chunks += 1

            except json.JSONDecodeError:
                invalid_chunks += 1

    print(f"[OK] Chunk records: {chunk_count:,}")

    if chunk_count == 0:
        print("[FAIL] No chunks found.")
        errors += 1

    if invalid_chunks:
        print(f"[FAIL] Invalid chunks: {invalid_chunks}")
        errors += 1
    else:
        print("[OK] Chunk structure: valid")

else:
    print(f"[FAIL] Missing file: {chunks_file.relative_to(ROOT)}")
    errors += 1

# Manifest
if manifest_file.exists():
    print(f"[OK] File:      {manifest_file.relative_to(ROOT)}")

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print("[OK] Valid JSON: chunking_manifest.json")

        manifest_chunks = manifest.get("total_chunks", 0)

        if chunks_file.exists() and manifest_chunks != chunk_count:
            print(
                f"[FAIL] Manifest mismatch: "
                f"{manifest_chunks:,} vs {chunk_count:,}"
            )
            errors += 1
        else:
            print(
                f"[OK] Manifest chunks: {manifest_chunks:,}"
            )

    except Exception as exc:
        print(f"[FAIL] Invalid manifest: {exc}")
        errors += 1

else:
    print(
        f"[FAIL] Missing file: "
        f"{manifest_file.relative_to(ROOT)}"
    )
    errors += 1

print()
print("-" * 70)

if errors == 0:
    print("VALIDATION PASSED")
    print("Phase 4 chunking foundation is ready.")
    sys.exit(0)

print(f"VALIDATION FAILED — {errors} issue(s)")
sys.exit(1)
