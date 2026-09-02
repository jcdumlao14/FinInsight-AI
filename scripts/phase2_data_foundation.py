from pathlib import Path
import json
import csv
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
METADATA_DIR = DATA_DIR / "metadata"

for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    CHUNKS_DIR,
    METADATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Create data inventory
# ------------------------------------------------------------

files = []

for path in RAW_DIR.rglob("*"):
    if path.is_file():
        files.append(
            {
                "filename": path.name,
                "relative_path": str(path.relative_to(ROOT)),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )


inventory = {
    "project": "FinInsight-AI",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "raw_documents": files,
    "document_count": len(files),
}


# ------------------------------------------------------------
# Save JSON inventory
# ------------------------------------------------------------

inventory_json = METADATA_DIR / "data_inventory.json"

with inventory_json.open("w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# Save CSV inventory
# ------------------------------------------------------------

inventory_csv = METADATA_DIR / "data_inventory.csv"

with inventory_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "relative_path",
            "extension",
            "size_bytes",
        ],
    )

    writer.writeheader()
    writer.writerows(files)


# ------------------------------------------------------------
# Create data manifest
# ------------------------------------------------------------

manifest = {
    "project": "FinInsight-AI",
    "data_layer": "financial_documents",
    "raw_directory": str(RAW_DIR.relative_to(ROOT)),
    "processed_directory": str(PROCESSED_DIR.relative_to(ROOT)),
    "chunks_directory": str(CHUNKS_DIR.relative_to(ROOT)),
    "metadata_directory": str(METADATA_DIR.relative_to(ROOT)),
    "document_count": len(files),
    "status": "initialized",
}

manifest_path = METADATA_DIR / "data_manifest.json"

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)


# ------------------------------------------------------------
# Phase 2 report
# ------------------------------------------------------------

print("=" * 70)
print("FinInsight-AI — PHASE 2 DATA FOUNDATION")
print("=" * 70)

print(f"[OK] Raw directory:       {RAW_DIR}")
print(f"[OK] Processed directory: {PROCESSED_DIR}")
print(f"[OK] Chunks directory:    {CHUNKS_DIR}")
print(f"[OK] Metadata directory:  {METADATA_DIR}")
print(f"[OK] Documents detected:  {len(files)}")
print(f"[OK] Inventory:            {inventory_json}")
print(f"[OK] CSV inventory:        {inventory_csv}")
print(f"[OK] Manifest:             {manifest_path}")

if files:
    print()
    print("Documents:")
    for item in files:
        print(
            f"  - {item['filename']} "
            f"({item['size_bytes']} bytes)"
        )
else:
    print()
    print("[INFO] No financial documents have been added yet.")

print()
print("Phase 2 data foundation initialized.")