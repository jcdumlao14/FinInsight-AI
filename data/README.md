# FinInsight-AI data and reproducibility

The project knowledge base is built from financial PDF documents placed in `data/raw/`.

## Rebuild the knowledge base

1. Put the permitted financial PDF source documents in `data/raw/`.
2. Run:

```bash
python scripts/phase2_data_foundation.py
python scripts/phase3_ingestion.py
python scripts/phase4_chunking.py
python scripts/phase5_embeddings.py
python scripts/phase6_bm25.py
python scripts/phase7_hybrid_retrieval.py
```

Or run the orchestrated pipeline:

```bash
python scripts/prefect_ingestion.py
```

Do not commit confidential or licensed documents. Record the exact public source URLs, document names, and access dates in `data/SOURCES.md` so another evaluator can reconstruct the dataset.
