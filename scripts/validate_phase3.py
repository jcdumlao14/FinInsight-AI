from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "scripts/phase3_ingestion.py",
]

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/metadata",
]


errors = 0

print("=" * 70)
print("FinInsight-AI — PHASE 3 VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# Validate directories
# ------------------------------------------------------------

for directory in REQUIRED_DIRECTORIES:

    path = ROOT / directory

    if path.is_dir():
        print(f"[OK] Directory: {directory}")
    else:
        print(
            f"[FAIL] Missing directory: "
            f"{directory}"
        )
        errors += 1


# ------------------------------------------------------------
# Validate ingestion script
# ------------------------------------------------------------

for filename in REQUIRED_FILES:

    path = ROOT / filename

    if path.is_file():
        print(f"[OK] File:      {filename}")
    else:
        print(
            f"[FAIL] Missing file: "
            f"{filename}"
        )
        errors += 1


# ------------------------------------------------------------
# Validate ingestion outputs when available
# ------------------------------------------------------------

metadata_path = (
    ROOT / "data/metadata/documents.json"
)

manifest_path = (
    ROOT / "data/metadata/ingestion_manifest.json"
)


if manifest_path.exists():

    print(
        "[OK] File:      "
        "data/metadata/ingestion_manifest.json"
    )

    try:

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            manifest = json.load(f)

        print(
            "[OK] Valid JSON: "
            "ingestion_manifest.json"
        )

        print(
            f"[INFO] Documents processed: "
            f"{manifest.get('documents_processed', 0)}"
        )

    except Exception as exc:

        print(
            f"[FAIL] Invalid ingestion manifest: "
            f"{exc}"
        )

        errors += 1


if metadata_path.exists():

    print(
        "[OK] File:      "
        "data/metadata/documents.json"
    )

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            documents = json.load(f)

        if isinstance(documents, list):

            print(
                "[OK] Valid JSON: "
                "documents.json"
            )

        else:

            print(
                "[FAIL] documents.json "
                "must contain a list"
            )

            errors += 1

    except Exception as exc:

        print(
            f"[FAIL] Invalid documents.json: "
            f"{exc}"
        )

        errors += 1


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("-" * 70)

if errors == 0:

    print("VALIDATION PASSED")
    print(
        "Phase 3 ingestion foundation is ready."
    )

    sys.exit(0)


print(
    f"VALIDATION FAILED — "
    f"{errors} issue(s)"
)

sys.exit(1)