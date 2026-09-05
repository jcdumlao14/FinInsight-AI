\# 📊 FinInsight-AI



\## Intelligent Financial Research and Insight Platform



FinInsight-AI is an end-to-end financial AI project designed to retrieve, analyze, and explain information from financial documents using modern Data Science, Machine Learning, and Retrieval-Augmented Generation techniques.



\### Project Status



🚧 \*\*Phase 1 — Project Foundation\*\*



The project is being developed from the ground up with a focus on:



\- Financial document ingestion

\- Data processing and chunking

\- Vector search

\- Lexical retrieval

\- Hybrid retrieval

\- LLM-based financial analysis

\- Evaluation

\- Monitoring

\- Streamlit user interface

\- Docker deployment

\- Reproducible project structure



\### Project Structure



```text

FinInsight-AI/

│

├── app/

├── data/

│   ├── raw/

│   ├── processed/

│   ├── chunks/

│   └── metadata/

├── evaluation/

├── llm/

├── monitoring/

├── rag/

├── scripts/

├── tests/

├── docs/

├── config/

├── .gitignore

├── requirements.txt

└── README.md

# Problem statement

Financial analysts and researchers often need to answer precise questions from lengthy annual reports and financial filings. Traditional keyword search can miss semantically relevant passages, while direct LLM queries can produce unsupported or hallucinated financial figures. FinInsight-AI addresses this problem by combining a financial-document knowledge base, lexical and semantic retrieval, query rewriting, document reranking, and source-grounded LLM generation so users can obtain concise answers with traceable evidence.


## Rubric-ready architecture

```text
User question
    |
    v
Query rewriting
    |
    v
+-----------------------+
| BM25 + Vector Search  |
+-----------------------+
    |
    v
Reciprocal Rank Fusion
    |
    v
Cross-Encoder Reranking
    |
    v
Top evidence chunks
    |
    v
Gemini grounded generation
    |
    v
Answer + source metadata + feedback
```

### Evaluation

- Retrieval comparison evaluates BM25, Vector, Hybrid RRF, and Reranked Hybrid.
- LLM comparison evaluates Basic, Grounded, and Structured prompts.
- The evaluation scripts write the selected winner to `evaluation/best_retrieval.json` and `evaluation/best_llm_prompt.json`.
- Monitoring provides more than five charts and collects user feedback.
- Prefect orchestrates the ingestion pipeline.
- Docker Compose runs the main application and monitoring dashboard.

### Reproducible run

```bash
python -m pip install -r requirements.txt
copy .env.example .env
# add GEMINI_API_KEY to .env
python scripts/prefect_ingestion.py
python evaluation/retrieval_evaluation.py
python evaluation/llm_evaluation.py
python scripts/release_check.py
streamlit run app/streamlit_app.py
```

For Docker:

```bash
docker compose build
docker compose up -d
```

