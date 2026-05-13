# HireFlow 🚀

An AI-powered recruitment platform that matches candidates to job descriptions using semantic search, hybrid indexing, and RAG-based evaluation. Built with LangChain, Pinecone, Google Gemini, and RAGAS.

---

## What it does

HireFlow automates resume screening by:

- Ingesting resumes and job descriptions from PDF files
- Embedding and indexing them in a Pinecone vector database
- Routing search queries intelligently (shallow vs deep) using a LangChain-powered router
- Ranking candidates using hybrid search + LLM reranking
- Evaluating search quality using RAGAS metrics (answer relevancy, faithfulness, context precision, answer correctness)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Google Gemini (`gemini-embedding-001`, 3072 dims) |
| Vector DB | Pinecone (cosine similarity) |
| LLM | Google Gemini |
| Search | Hybrid (vector + BM25) with LLM reranking |
| Evaluation | RAGAS |
| Orchestration | LangChain |
| UI | Streamlit |

---

## Project Structure & File Explanations

```
HireFlow/
├── core/
│   ├── ingestion.py
│   ├── vector_store.py
│   ├── hybrid_indexer.py
│   ├── search_router.py
│   ├── evaluator.py
│   ├── parsing.py
│   ├── re_ranker.py
│   └── memory_rag.py
├── utils/
│   ├── config.py
│   ├── utils.py
│   └── schemas.py
├── data/
│   ├── resumes/
│   └── jds/
├── streamlit/
├── main.py
├── reindex.py
├── check.py
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

### `main.py`
A CLI-based system demonstration and testing script. Initializes all HireFlow components, checks prerequisites (API keys, data directories), loads and indexes documents, then runs automatic demos of all search modes, AI evaluation, and memory tracking. Drops into an interactive text menu (options 1–7) for manually exploring each feature. Run with:

```bash
uv run python main.py
```

### `reindex.py`
A standalone script that initializes the `VectorStore`, loads resumes from `data/resumes/` and job descriptions from `data/jds/`, upserts them into Pinecone, and prints final index stats. Run this once before starting the app, and again whenever you add new documents.

```bash
uv run python reindex.py
```

### `check.py`
A debugging script that initializes `VectorStore` and `SearchRouter`, runs a hardcoded shallow search for `"Python developer"`, and prints the `name`, `type`, metadata keys, and a content preview for each result. Useful for quickly verifying that the router, vector store, and metadata fields are all working correctly after indexing.

### `pyproject.toml`
Project configuration file for `uv` (the package manager used in this project). Defines the project name, Python version requirements, and dependencies. Used by `uv sync` to install all packages into the virtual environment.

### `requirements.txt`
Alternative dependency list for projects not using `uv`. Install with `pip install -r requirements.txt`.

---

### `core/ingestion.py`
Handles loading PDF files from the `data/resumes/` and `data/jds/` directories and converting them into LangChain `Document` objects with metadata. Each document gets metadata like `candidate_id`, `name`, `filename`, and `source` (for resumes) or `jd_id` and `title` (for job descriptions). These Document objects are what gets passed to the vector store for embedding.

### `core/vector_store.py`
Manages the Pinecone vector database. Handles three main responsibilities: creating/verifying the Pinecone index on startup, upserting resume and job description embeddings in batches (with rate limit handling), and running semantic similarity searches filtered by document type (`resume` or `job_description`). Includes a type filter so resume searches never return job descriptions and vice versa.

### `core/hybrid_indexer.py`
Implements a hybrid search system that combines BM25 keyword search with Pinecone vector search. Uses Reciprocal Rank Fusion (RRF) to merge results from both methods into a single ranked list. This gives better recall than pure vector search — especially for specific skill names or job titles that might not embed well semantically.

### `core/search_router.py`
The intelligent query router built with LangChain's `RunnableBranch`. Analyzes each search query and decides whether to use shallow search (fast vector-only for simple queries like "Python developer") or deep search (hybrid BM25 + vector + LLM reranking for complex queries like "senior accountant with QuickBooks and tax experience"). The LLM makes the routing decision; simple word count is the fallback if the LLM is unavailable.

### `core/evaluator.py`
The RAGAS-based evaluation system. After a search is run, this module evaluates response quality across four metrics: Answer Relevancy, Context Precision, Faithfulness, and Answer Correctness. Builds an `EvaluationDataset` from the search results and runs them through RAGAS using Gemini as both the LLM and embedding model. Stores evaluation history and supports CSV export.

### `core/parsing.py`
Contains `ResumeParser` and `JobParser` — AI-powered parsers that use Gemini to extract structured information from raw resume and job description text. Used during candidate evaluation to convert unstructured PDF content into structured objects for comparison.

### `core/re_ranker.py`
Contains the `ReRanker` component that uses Gemini to evaluate and score candidate–job fit. Given a candidate and a job description, it produces a fit score, a list of strengths, identified gaps, and a summary. Used in deep search mode after initial retrieval.

### `core/memory_rag.py`
The `MemoryRAG` component that tracks conversation and search history across a session. Records each search query and candidate view, and exposes stats and history for display in the UI or CLI.

---

### `utils/config.py`
Loads all environment variables from the `.env` file using `python-dotenv`. Provides a single source of truth for API keys (`GOOGLE_API_KEY`, `PINECONE_API_KEY`), Pinecone settings (`PINECONE_INDEX_NAME`, `PINECONE_DIMENSION`, `PINECONE_METRIC`), and model names (`LLM_MODEL`, `EMBEDDING_MODEL`). All other files import from here instead of reading env vars directly.

### `utils/utils.py`
Shared utility functions used across the project. Includes `get_logger()` for consistent logging, `load_pdf()` for extracting text from PDF files, `split_text()` for chunking documents, `get_embeddings()` for initializing the Google Gemini embedding model, and `is_quota_error()` for detecting API rate limit errors so they can be handled gracefully.

### `utils/schemas.py`
Pydantic schema definitions used across the project. Includes `JobDescription` and other structured data models used by the parsers, re-ranker, and evaluator.

---

### `data/resumes/`
Place all candidate resume PDFs here. These are not committed to the repo — add your own. Each PDF filename becomes the candidate's name (e.g., `john_doe.pdf` → "John Doe").

### `data/jds/`
Place job description PDFs here. Not committed to the repo. Each PDF is indexed with its filename as the job title.

### `streamlit/`
Contains Streamlit configuration files (e.g., `config.toml`) for customizing the app's theme, layout, and server settings.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dumpty-420/Hireflow.git
cd Hireflow
```

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hireflow
PINECONE_DIMENSION=3072
PINECONE_METRIC=cosine
LLM_MODEL=gemini-pro
```

### 4. Add your documents

Place resume PDFs in `data/resumes/` and job description PDFs in `data/jds/`.

### 5. Index documents into Pinecone

```bash
uv run python reindex.py
```

### 6. Run the demo / test CLI

```bash
uv run python main.py
```

---

## How Search Works

HireFlow uses a `RunnableBranch` router to decide between two search strategies:

**Shallow search** — fast vector-only search via Pinecone. Best for simple, specific queries like "Python developer" or "accountant".

**Deep search** — hybrid BM25 + vector search merged with RRF, followed by LLM reranking. Best for complex queries like "senior accountant with QuickBooks and 5 years of tax experience". The LLM reranker re-orders results based on semantic fit, not just similarity scores.

The routing decision is made automatically by the LLM on each query, or you can force a mode manually.

---

## Evaluation

HireFlow includes a RAGAS-based evaluation system:

| Metric | Description |
|---|---|
| Answer Relevancy | How relevant the retrieved candidates are to the query |
| Context Precision | Precision of the retrieved resume context |
| Faithfulness | Factual consistency of the response with context |
| Answer Correctness | Overall quality of the candidate match |

Run evaluation from the interactive menu in `main.py` (option 3), or trigger it programmatically via `core/evaluator.py`. Results are tracked across sessions and can be exported to CSV.

---

## Notes

- Pinecone free tier supports one index — make sure `PINECONE_DIMENSION=3072` matches your embedding model
- Resume and job description PDFs are gitignored — add your own to `data/resumes/` and `data/jds/`
- API quota limits are handled automatically with batched indexing in `core/vector_store.py`
- Run `check.py` to debug search results if candidates aren't showing up correctly

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
