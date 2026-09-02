from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "data/chunks",
    "data/metadata",
]

REQUIRED_FILES = [
    "data/metadata/data_inventory.json",
    "data/metadata/data_inventory.csv",
    "data/metadata/data_manifest.json",
]

errors = 0

print("=" * 70)
print("FinInsight-AI — PHASE 2 VALIDATION")
print("=" * 70)

for directory in REQUIRED_DIRECTORIES:
    path = ROOT / directory

    if path.is_dir():
        print(f"[OK] Directory: {directory}")
    else:
        print(f"[FAIL] Missing directory: {directory}")
        errors += 1


for filename in REQUIRED_FILES:
    path = ROOT / filename

    if path.is_file():
        print(f"[OK] File:      {filename}")
    else:
        print(f"[FAIL] Missing file: {filename}")
        errors += 1


# Validate JSON files

for filename in [
    "data/metadata/data_inventory.json",
    "data/metadata/data_manifest.json",
]:

    path = ROOT / filename

    if path.exists():

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                json.load(f)

            print(f"[OK] Valid JSON: {filename}")

        except Exception as exc:
            print(
                f"[FAIL] Invalid JSON: "
                f"{filename} — {exc}"
            )
            errors += 1


print()
print("-" * 70)

if errors == 0:
    print("VALIDATION PASSED")
    print("Phase 2 data foundation is ready.")
    sys.exit(0)

print(
    f"VALIDATION FAILED — "
    f"{errors} issue(s)"
)

sys.exit(1)