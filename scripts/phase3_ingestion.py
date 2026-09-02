from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"


PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize extracted PDF text while preserving readability."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces while preserving line breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash for document identification."""

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def create_document_id(filename: str, sha256: str) -> str:
    """Create a stable document identifier."""

    stem = Path(filename).stem.lower()

    safe_stem = re.sub(
        r"[^a-z0-9]+",
        "_",
        stem,
    ).strip("_")

    return f"{safe_stem}_{sha256[:12]}"


# ------------------------------------------------------------
# Discover source documents
# ------------------------------------------------------------

pdf_files = sorted(
    RAW_DIR.rglob("*.pdf")
)

print("=" * 70)
print("FinInsight-AI — PHASE 3 DOCUMENT INGESTION")
print("=" * 70)

print(f"Source directory: {RAW_DIR}")
print(f"PDF documents discovered: {len(pdf_files)}")
print()


if not pdf_files:
    print("[WARNING] No PDF documents found.")
    print()
    print(
        "Add financial PDF documents to:"
    )
    print(
        f"  {RAW_DIR}"
    )
    print()
    print(
        "Phase 3 ingestion cannot process documents until "
        "PDF files are available."
    )

    sys.exit(0)


documents = []
failures = []


# ------------------------------------------------------------
# Process every PDF
# ------------------------------------------------------------

for pdf_path in pdf_files:

    print("-" * 70)
    print(f"Processing: {pdf_path.name}")

    try:

        sha256 = calculate_sha256(pdf_path)

        document_id = create_document_id(
            pdf_path.name,
            sha256,
        )

        reader = PdfReader(str(pdf_path))

        page_records = []
        total_characters = 0
        empty_pages = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            text = normalize_text(raw_text)

            if not text:
                empty_pages += 1

            total_characters += len(text)

            page_records.append(
                {
                    "page": page_number,
                    "text": text,
                    "characters": len(text),
                }
            )


        # ----------------------------------------------------
        # Create processed text file
        # ----------------------------------------------------

        output_text = PROCESSED_DIR / (
            f"{document_id}.txt"
        )

        with output_text.open(
            "w",
            encoding="utf-8",
        ) as f:

            for record in page_records:

                f.write(
                    f"\n===== PAGE "
                    f"{record['page']} =====\n\n"
                )

                f.write(
                    record["text"]
                )

                f.write("\n")


        # ----------------------------------------------------
        # Create page-level JSON
        # ----------------------------------------------------

        output_pages = PROCESSED_DIR / (
            f"{document_id}_pages.json"
        )

        with output_pages.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                page_records,
                f,
                indent=2,
                ensure_ascii=False,
            )


        # ----------------------------------------------------
        # Document metadata
        # ----------------------------------------------------

        metadata = {
            "document_id": document_id,
            "filename": pdf_path.name,
            "source_path": str(
                pdf_path.relative_to(ROOT)
            ),
            "file_size_bytes": pdf_path.stat().st_size,
            "sha256": sha256,
            "page_count": len(reader.pages),
            "total_characters": total_characters,
            "empty_pages": empty_pages,
            "processed_text": str(
                output_text.relative_to(ROOT)
            ),
            "processed_pages": str(
                output_pages.relative_to(ROOT)
            ),
            "ingested_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": (
                "success"
                if total_characters > 0
                else "empty"
            ),
        }


        documents.append(metadata)

        print(
            f"[OK] Pages: {metadata['page_count']}"
        )

        print(
            f"[OK] Characters: "
            f"{metadata['total_characters']:,}"
        )

        print(
            f"[OK] Empty pages: "
            f"{metadata['empty_pages']}"
        )

        print(
            f"[OK] Document ID: "
            f"{metadata['document_id']}"
        )

    except Exception as exc:

        failure = {
            "filename": pdf_path.name,
            "error": str(exc),
        }

        failures.append(failure)

        print(
            f"[FAIL] {pdf_path.name}: {exc}"
        )


# ------------------------------------------------------------
# Save document metadata
# ------------------------------------------------------------

metadata_path = (
    METADATA_DIR / "documents.json"
)

with metadata_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        documents,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# Save ingestion manifest
# ------------------------------------------------------------

manifest = {
    "project": "FinInsight-AI",
    "phase": "Phase 3",
    "pipeline": "PDF document ingestion",
    "generated_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_directory": str(
        RAW_DIR.relative_to(ROOT)
    ),
    "processed_directory": str(
        PROCESSED_DIR.relative_to(ROOT)
    ),
    "documents_discovered": len(pdf_files),
    "documents_processed": len(documents),
    "documents_failed": len(failures),
    "total_pages": sum(
        d["page_count"]
        for d in documents
    ),
    "total_characters": sum(
        d["total_characters"]
        for d in documents
    ),
    "failures": failures,
    "status": (
        "success"
        if not failures
        else "completed_with_errors"
    ),
}


manifest_path = (
    METADATA_DIR / "ingestion_manifest.json"
)

with manifest_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 3 INGESTION SUMMARY")
print("=" * 70)

print(
    f"Documents discovered : "
    f"{len(pdf_files)}"
)

print(
    f"Documents processed  : "
    f"{len(documents)}"
)

print(
    f"Documents failed     : "
    f"{len(failures)}"
)

print(
    f"Total pages          : "
    f"{manifest['total_pages']:,}"
)

print(
    f"Total characters     : "
    f"{manifest['total_characters']:,}"
)

print()
print(
    f"[OK] Metadata: {metadata_path}"
)

print(
    f"[OK] Manifest: {manifest_path}"
)

if failures:

    print()
    print("Failures:")

    for failure in failures:
        print(
            f"  - {failure['filename']}: "
            f"{failure['error']}"
        )

    print()
    print(
        "Phase 3 completed with ingestion errors."
    )

    sys.exit(1)


print()
print(
    "Phase 3 document ingestion completed successfully."
)