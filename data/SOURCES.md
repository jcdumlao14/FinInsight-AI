# FinInsight-AI Dataset Sources

This file documents the financial documents currently used by the FinInsight-AI
knowledge base.

The project uses public corporate filings as the source material for
retrieval-augmented financial question answering.

## Active Knowledge Base

The current ingestion manifest reports:

- Documents discovered by the PDF ingestion pipeline: 2
- Documents successfully processed: 2
- Documents failed: 0
- Total pages processed: 255
- Total extracted characters: 905,609

### 1. Apple Inc. — Fiscal 2025 Form 10-K

| Field | Value |
|---|---|
| Company / issuer | Apple Inc. |
| Filing | Annual Report on Form 10-K |
| Fiscal year | Fiscal year ended September 27, 2025 |
| Local filename | `apple_10k_2025.pdf` |
| Local path | `data/raw/apple_10k_2025.pdf` |
| SHA-256 | `108590052c3ba5400c63660d787fe7ed4e43868292946d7a7facebe9ab7d1aab` |
| Pages | 80 |
| Processing status | Successfully processed |
| Source | U.S. Securities and Exchange Commission (SEC) |

Official SEC filing:

https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm

SEC filing accession number: `0000320193-25-000079`

### 2. NVIDIA Corporation — Fiscal 2026 Form 10-K

| Field | Value |
|---|---|
| Company / issuer | NVIDIA Corporation |
| Filing | Annual Report on Form 10-K |
| Fiscal year | Fiscal year ended January 25, 2026 |
| Local filename | `nvidia_10k_2026.pdf` |
| Local path | `data/raw/nvidia_10k_2026.pdf` |
| SHA-256 | `0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c` |
| Pages | 175 |
| Processing status | Successfully processed |
| Source | U.S. Securities and Exchange Commission (SEC) |

Official SEC filing:

https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm

SEC filing accession number: `0001045810-26-000021`

## Present but Not Currently Indexed

The raw-data inventory also contains:

`microsoft_10K_2025.pdf.docx`

This document is currently present in `data/raw/`, but the current PDF ingestion pipeline does not process it. Therefore, it is **not counted as part of the active processed knowledge base**.

It should not be treated as an indexed source until DOCX ingestion support is intentionally added.

## Reproducibility

The raw source documents are intentionally excluded from Git when appropriate
because financial filings may be large and should be reconstructed from their
public sources.

For reproducibility:

1. Obtain the public filings from the official SEC sources above.
2. Place the permitted source files in `data/raw/`.
3. Verify filenames and SHA-256 hashes against the metadata in
   `data/metadata/documents.json`.
4. Run the ingestion and indexing pipeline.
5. Validate the generated manifests under `data/metadata/`.

## Important Note

The source URLs above identify the official SEC filings corresponding to the
processed Apple and NVIDIA documents. The local SHA-256 hashes identify the
specific files used by this project.

Access dates and local metadata are recorded in the generated project metadata.