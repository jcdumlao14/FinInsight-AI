# FinInsight-AI

### Intelligent Financial Research & Insight Platform using Retrieval-Augmented Generation

> **FinInsight-AI** is a source-grounded financial research assistant that combines hybrid information retrieval, query rewriting, document reranking, and LLM generation to answer questions from corporate financial filings.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60.0-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.9-5A67D8)](https://www.trychroma.com/)
[![Prefect](https://img.shields.io/badge/Prefect-3.4.14-070E10)](https://www.prefect.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google\&logoColor=white)](https://ai.google.dev/)

---

## 🎯 Project Overview

Financial analysts and researchers often need to locate precise figures, trends, and disclosures inside lengthy annual reports and financial filings.

Traditional keyword search can miss semantically related information, while asking an LLM directly can produce unsupported or hallucinated financial figures.

**FinInsight-AI addresses this problem with a complete Retrieval-Augmented Generation (RAG) pipeline.**

The system:

1. Ingests financial filings.
2. Extracts and cleans document text.
3. Splits documents into retrieval-friendly chunks.
4. Generates semantic embeddings.
5. Performs vector similarity search.
6. Performs BM25 lexical search.
7. Combines retrieval results using Reciprocal Rank Fusion (RRF).
8. Rewrites user queries to improve retrieval.
9. Reranks retrieved documents using a cross-encoder.
10. Sends only relevant financial context to the LLM.
11. Generates source-grounded answers.
12. Displays retrieved evidence to the user.
13. Collects user feedback.
14. Provides operational monitoring and evaluation.

The goal is simple:

> **Find the right financial evidence first, then generate an answer grounded in that evidence.**

---
## Problem Description

Financial research often requires analysts and researchers to search through long annual reports and regulatory filings to find specific information such as revenue, net sales, earnings, business performance, and year-over-year changes. Manually locating and interpreting these figures can be time-consuming, especially when the relevant information is distributed across many pages and sections of a document.

FinInsight-AI addresses this problem by providing a **source-grounded financial research assistant** that allows users to ask natural-language questions about financial documents and receive concise answers supported by retrieved evidence.

The system combines **query rewriting, vector search, BM25 keyword search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking** to improve the retrieval of relevant financial passages. The retrieved context is then provided to an LLM for answer generation, with prompts designed to reduce unsupported claims and preserve important financial details such as fiscal periods, currencies, units, and percentages.

The goal is not to replace financial analysts, but to make document-based financial research **faster, more searchable, and easier to verify** by keeping answers grounded in the underlying source documents.

---

## 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │  Financial Research  │
                         │      Assistant       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Query Rewriting    │
                         │  Retrieval-Friendly  │
                         │       Query          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │        Hybrid Retrieval         │
                    │                                 │
                    │   ┌──────────┐   ┌──────────┐  │
                    │   │  Vector  │   │   BM25   │  │
                    │   │  Search  │   │  Search  │  │
                    │   └────┬─────┘   └────┬─────┘  │
                    │        └────────┬──────┘        │
                    │                 ▼               │
                    │       Reciprocal Rank           │
                    │          Fusion (RRF)            │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │  Cross-Encoder       │
                         │     Reranking        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Relevant Financial   │
                         │      Context         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Gemini 2.5 Flash  │
                         │   Grounded Generation│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Source-Grounded     │
                         │       Answer         │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                  User Feedback           Source Evidence
                         │
                         ▼
                  Monitoring Dashboard
```

---

# 📚 Knowledge Base & Data Pipeline

The current knowledge base is built from public corporate filings.

### Active processed documents

| Company | Filing    | Fiscal Year | Pages | Status      |
| ------- | --------- | ----------: | ----: | ----------- |
| Apple   | Form 10-K |      FY2025 |    80 | ✅ Processed |
| NVIDIA  | Form 10-K |      FY2026 |   175 | ✅ Processed |

The raw-data inventory also contains a Microsoft document, but the current ingestion workflow processes PDF files only. Therefore, the Microsoft document is **not currently counted as part of the indexed knowledge base**.

Detailed source information, SHA-256 hashes, and official SEC filing references are documented in:

`data/SOURCES.md`

---

# 🔄 Ingestion Pipeline

FinInsight-AI supports automated document ingestion through **Prefect**.

```text
Financial PDF
     │
     ▼
Data Foundation
     │
     ▼
PDF Text Extraction
     │
     ▼
Document Chunking
     │
     ▼
Embedding Generation
     │
     ▼
ChromaDB Vector Store
     │
     ├──────────────► BM25 Index
     │
     ▼
Hybrid Retrieval
     │
     ▼
Evaluation / Validation
```

The orchestrated workflow is implemented in:

```text
scripts/prefect_ingestion.py
```

Run the automated pipeline with:

```bash
python scripts/prefect_ingestion.py
```

The underlying pipeline consists of:

```bash
python scripts/phase2_data_foundation.py
python scripts/phase3_ingestion.py
python scripts/phase4_chunking.py
python scripts/phase5_embeddings.py
python scripts/phase6_bm25.py
python scripts/phase7_hybrid_retrieval.py
```

---

# 🔎 Retrieval Strategy

FinInsight-AI uses a multi-stage retrieval architecture.

## 1. Semantic Vector Search

Vector embeddings are used to retrieve passages based on semantic similarity.

This helps with questions where the wording of the query differs from the wording used in the financial filing.

## 2. BM25 Lexical Search

BM25 provides keyword-based retrieval that is particularly useful for:

* financial terminology
* company names
* accounting terms
* fiscal years
* exact figures
* percentages
* product names

## 3. Reciprocal Rank Fusion

The vector and BM25 rankings are combined using **Reciprocal Rank Fusion (RRF)**.

```text
Vector Results
      +
 BM25 Results
      │
      ▼
   RRF Fusion
      │
      ▼
Combined Ranking
```

## 4. Query Rewriting

`rag/query_rewriter.py` transforms the original question into a more retrieval-friendly financial search query.

The implementation uses Gemini when available and provides a deterministic fallback for offline/test environments.

## 5. Document Reranking

`rag/reranker.py` applies a cross-encoder reranking stage after initial retrieval.

This provides a second relevance assessment before context is passed to the LLM.

---

# 🤖 LLM Generation

The project uses **Gemini 2.5 Flash** for final answer generation.

Three prompt strategies are implemented:

### Basic

A concise answer using only retrieved financial context.

### Grounded

A source-focused financial research prompt designed to:

* avoid unsupported figures
* preserve financial units
* preserve fiscal periods
* acknowledge missing evidence
* reference retrieved sources

### Structured

Produces a concise response organized into:

```text
Direct Answer
Evidence
Interpretation
```

The default generation strategy is designed around source-grounded financial answers rather than unconstrained generation.

> **Important:** Actual Gemini evaluation runs depend on available API quota. The repository contains the multi-strategy evaluation framework, while live API experiments should be run when sufficient quota is available.

---

# 📊 Evaluation

FinInsight-AI includes dedicated retrieval and LLM evaluation components.

## Retrieval Evaluation

The project evaluates multiple retrieval strategies, including:

* Vector Search
* BM25
* Hybrid RRF retrieval

The evaluation pipeline is designed to compare retrieval quality and identify the strongest retrieval strategy for the final system.

## LLM Evaluation

The project provides a comparison framework for multiple generation strategies:

```text
Basic
  │
  ├── Evaluation
  │
Grounded
  │
  ├── Evaluation
  │
Structured
  │
  └── Evaluation
```

Evaluation artifacts are stored under:

```text
evaluation/
```

---

# 🖥️ User Interface

The application uses **Streamlit** and provides:

* Financial question input
* Example questions
* Query processing
* Source-grounded answers
* Retrieved document evidence
* Retrieval scores
* Document/chunk information
* Feedback collection
* Clear/reset functionality
* Answer download functionality

Run locally:

```bash
streamlit run app/streamlit_app.py
```
---
# 📷 Application Screenshots

## 🏠 Streamlit Home Page

![](https://github.com/jcdumlao14/FinInsight-AI/blob/main/docs/streamlit%20home.png)

---

## 🤖 AI Generated Answer

![](https://github.com/jcdumlao14/FinInsight-AI/blob/main/docs/Gen%20Answer.png)

---

## 📚 Retrieved Sources

![](https://github.com/jcdumlao14/FinInsight-AI/blob/main/docs/R%20source.png)

---

# 📈 Monitoring

FinInsight-AI includes a dedicated monitoring application.

The monitoring system collects user feedback and provides operational views of application activity.

The dashboard includes visualizations for signals such as:

* Feedback distribution
* Query activity over time
* Feedback trends
* Retrieval activity
* Answer-quality-related signals
* Application usage

Launch the monitoring dashboard with:

```bash
streamlit run monitoring/dashboard.py
```

---

# 🐳 Docker

The project is fully containerized with Docker Compose.

The architecture includes:

```text
┌─────────────────────────────┐
│       fininsight-app        │
│       Streamlit :8501       │
└──────────────┬──────────────┘
               │
               │ Host :8503
               ▼
        FinInsight-AI UI


┌─────────────────────────────┐
│    fininsight-monitoring    │
│       Streamlit :8502       │
└──────────────┬──────────────┘
               │
               │ Host :8504
               ▼
        Monitoring Dashboard
```

Start the complete application:

```bash
docker compose up --build
```

Application:

```text
http://localhost:8503
```

Monitoring:

```text
http://localhost:8504
```

Stop the services:

```bash
docker compose down
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/jcdumlao14/FinInsight-AI.git
cd FinInsight-AI
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Pinned direct dependencies are also documented in:

```text
requirements.lock.txt
```

> `requirements.lock.txt` is a project-specific pinned direct-dependency file; it is not intended to represent every transitive package dependency.

---

# 🔐 Environment Variables

Create a local `.env` file based on `.env.example`.

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit API keys, credentials, or private data.

---

# 🗂️ Project Structure

```text
FinInsight-AI/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   ├── metadata/
│   ├── vector_db/
│   ├── README.md
│   └── SOURCES.md
│
├── evaluation/
│   ├── questions.json
│   ├── llm_evaluation.py
│   └── retrieval_evaluation.py
│
├── llm/
│   └── generator.py
│
├── monitoring/
│   ├── dashboard.py
│   └── feedback.py
│
├── rag/
│   ├── pipeline.py
│   ├── query_rewriter.py
│   ├── reranker.py
│   └── ...
│
├── scripts/
│   ├── phase2_data_foundation.py
│   ├── phase3_ingestion.py
│   ├── phase4_chunking.py
│   ├── phase5_embeddings.py
│   ├── phase6_bm25.py
│   ├── phase7_hybrid_retrieval.py
│   ├── prefect_ingestion.py
│   └── release_check.py
│
├── .streamlit/
│   └── config.toml
│
├── Dockerfile
├── Dockerfile.monitoring
├── docker-compose.yml
├── requirements.txt
├── requirements.lock.txt
├── .env.example
└── README.md
```

---

# 🧪 Testing & Validation

Run Python compilation checks:

```bash
python -m compileall app evaluation llm monitoring rag scripts
```

Run the test suite:

```bash
python -m pytest -q
```

Run the project release/rubric validation:

```bash
python scripts/release_check.py
```

The release validation checks the presence of the major project capabilities, including:

* Problem description
* Knowledge base and LLM
* Retrieval comparison
* LLM comparison
* User interface
* Automated ingestion
* Monitoring
* Containerization
* Reproducibility
* Hybrid search
* Document reranking
* Query rewriting

---

# 📋 Reproducibility

The project is designed so that the knowledge base can be reconstructed from public financial filings.

Reproducibility information is maintained in:

```text
data/README.md
data/SOURCES.md
data/metadata/
```

The source documentation records:

* Document names
* Filing information
* Official SEC sources
* Local filenames
* SHA-256 hashes
* Page counts
* Processing status
* Ingestion metadata

Generated artifacts such as the local ChromaDB vector database are intentionally excluded from Git.

To rebuild the knowledge base:

```bash
python scripts/prefect_ingestion.py
```

---

# 🛡️ Source-Grounded Design

FinInsight-AI is intentionally designed to reduce unsupported financial claims.

The generation layer instructs the LLM to:

* use only retrieved financial context
* avoid inventing financial figures
* preserve units and periods
* acknowledge when evidence is unavailable
* reference retrieved evidence
* distinguish retrieved information from generated explanation

This makes the system more appropriate for **financial-document research** than a general-purpose conversational chatbot.

---

# ⚠️ Limitations

### Limited document corpus

The currently processed knowledge base contains a limited number of corporate filings.

Questions about companies or filings outside the indexed corpus may not be answerable.

### Limited evaluation dataset

The current evaluation set is relatively small. Larger and more diverse evaluation datasets would provide stronger evidence of general retrieval and answer quality.

### Document scope

The system is designed around the financial documents available in its knowledge base.

It should **not** be treated as a general financial advisor.

### External knowledge

The system intentionally prioritizes retrieved evidence instead of silently supplementing missing information with unrelated external knowledge.

### API quota

Live Gemini evaluations depend on API availability and quota limits.

---

# 🚀 Future Improvements

Potential future enhancements include:

* Expand the financial filing corpus
* Add DOCX ingestion support
* Increase the retrieval evaluation dataset
* Add more embedding models
* Tune RRF parameters
* Add retrieval and generation latency tracking
* Improve citation tracking
* Add more advanced financial entity extraction
* Add multi-document reasoning
* Add automated evaluation reporting
* Add cloud deployment
* Add authentication and access control
* Add production observability
* Add continuous evaluation pipelines

---

# 🎓 LLM Zoomcamp 2026

FinInsight-AI was developed as part of **LLM Zoomcamp 2026** learning and project work.

The project applies core LLM engineering concepts to a practical financial research use case, including:

* Retrieval-Augmented Generation
* Vector Search
* BM25
* Hybrid Retrieval
* Reciprocal Rank Fusion
* Query Rewriting
* Document Reranking
* LLM Prompt Engineering
* Evaluation
* Monitoring
* Prefect orchestration
* Docker containerization
* Reproducible data pipelines

---

# 🏆 Evaluation Rubric Coverage

| Criterion            | Implementation                                           |
| -------------------- | -------------------------------------------------------- |
| Problem description  | Financial research and filing QA problem clearly defined |
| Retrieval flow       | Knowledge base + retrieval + Gemini generation           |
| Retrieval evaluation | Vector, BM25, and hybrid retrieval evaluation            |
| LLM evaluation       | Multiple generation/prompt strategies                    |
| Interface            | Streamlit application                                    |
| Ingestion pipeline   | Prefect-orchestrated ingestion                           |
| Monitoring           | User feedback + monitoring dashboard                     |
| Containerization     | Docker + Docker Compose                                  |
| Reproducibility      | Pinned dependencies + source documentation + hashes      |
| Hybrid search        | Vector + BM25 + RRF                                      |
| Document reranking   | Cross-encoder reranker                                   |
| Query rewriting      | Financial query rewriting component                      |

---

# 👤 Author

## Jocelyn C. Dumlao

**Independent Data Scientist | Machine Learning Engineer**

GitHub: [@jcdumlao14](https://github.com/jcdumlao14)

FinInsight-AI is an independent project focused on applying modern information retrieval and generative AI techniques to practical financial research workflows.

---

# 📄 License

This project is provided for educational, research, and portfolio purposes.

Please respect the licensing and usage requirements of all third-party datasets, documents, APIs, and models used with the project.

---

## ⭐ Project Status

**Status: Completed and ready for rubric review**

FinInsight-AI currently provides an end-to-end financial RAG workflow covering:

```text
Financial Filings
       ↓
Automated Ingestion
       ↓
Document Processing
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector Search + BM25
       ↓
RRF Hybrid Retrieval
       ↓
Query Rewriting
       ↓
Cross-Encoder Reranking
       ↓
Gemini Generation
       ↓
Source-Grounded Answer
       ↓
User Feedback
       ↓
Monitoring & Evaluation
```

**Built with Python, Streamlit, ChromaDB, BM25, Sentence Transformers, Gemini, Prefect, and Docker.**
