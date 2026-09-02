from pathlib import Path
import json
import re
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHUNKS = ROOT / "data" / "chunks"
METADATA = ROOT / "data" / "metadata"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_chunks(text: str):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)

        if end < text_length:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary > start + 400:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def load_documents():
    documents_file = METADATA / "documents.json"

    if not documents_file.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {documents_file}"
        )

    with open(documents_file, "r", encoding="utf-8") as f:
        return json.load(f)


def locate_text_file(document):
    possible_names = [
        document.get("text_file"),
        document.get("processed_file"),
        document.get("text_path"),
    ]

    for name in possible_names:
        if not name:
            continue

        path = Path(name)

        if not path.is_absolute():
            path = ROOT / path

        if path.exists():
            return path

    document_id = document.get("document_id", "")

    matches = list(PROCESSED.glob(f"*{document_id}*"))

    if matches:
        return matches[0]

    return None


def main():
    print("=" * 70)
    print("FinInsight-AI — PHASE 4 DOCUMENT CHUNKING")
    print("=" * 70)

    CHUNKS.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    if isinstance(documents, dict):
        if "documents" in documents:
            documents = documents["documents"]
        else:
            documents = list(documents.values())

    print(f"Documents available: {len(documents)}")
    print(f"Chunk size:          {CHUNK_SIZE}")
    print(f"Chunk overlap:       {CHUNK_OVERLAP}")
    print()

    all_chunks = []
    document_summary = []

    for document in documents:
        document_id = document.get("document_id", "unknown")
        filename = document.get("filename", "unknown")

        text_file = locate_text_file(document)

        print("-" * 70)
        print(f"Processing: {filename}")

        if text_file is None:
            print("[WARNING] Processed text file not found.")
            print(f"Document ID: {document_id}")
            continue

        text = text_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        chunks = make_chunks(text)

        print(f"[OK] Characters: {len(text):,}")
        print(f"[OK] Chunks:     {len(chunks):,}")

        document_chunk_count = 0

        for index, chunk_text in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{document_id}:{index}:{chunk_text}".encode("utf-8")
            ).hexdigest()[:16]

            record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "text": chunk_text,
                "character_count": len(chunk_text),
            }

            all_chunks.append(record)
            document_chunk_count += 1

        document_summary.append({
            "document_id": document_id,
            "filename": filename,
            "chunks": document_chunk_count,
            "source_text_file": str(text_file.relative_to(ROOT)),
        })

    output_file = CHUNKS / "chunks.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_file = METADATA / "chunking_manifest.json"

    summary = {
        "phase": "Phase 4 — Document Chunking",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents_processed": len(document_summary),
        "total_chunks": len(all_chunks),
        "documents": document_summary,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("PHASE 4 CHUNKING SUMMARY")
    print("=" * 70)
    print(f"Documents processed : {len(document_summary)}")
    print(f"Total chunks        : {len(all_chunks):,}")
    print()
    print(f"[OK] Chunks:   {output_file}")
    print(f"[OK] Manifest: {summary_file}")
    print()
    print("Phase 4 document chunking completed successfully.")


if __name__ == "__main__":
    main()
