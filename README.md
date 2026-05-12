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
│   └── evaluator.py
├── utils/
│   ├── config.py
│   └── utils.py
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
The Streamlit application entry point. Runs the full recruitment UI — including the candidate search interface, job description input, search mode selector, results display, and the RAGAS evaluation tab. This is the file you run to launch the app.

### `reindex.py`
A standalone script that loads all resume and job description PDFs from the `data/` folder, embeds them using Google Gemini, and upserts them into Pinecone in batches. Run this once before starting the app, and again whenever you add new documents. Includes batching with delays to avoid hitting API quota limits.

### `check.py`
A debugging script used during development to verify that the vector store is working correctly — checks if search results are returning properly, confirms metadata fields (like `page_content`, `name`, `type`) are present, and prints result previews. Useful for diagnosing search issues without running the full app.

### `pyproject.toml`
Project configuration file for `uv` (the package manager used in this project). Defines the project name, Python version requirements, and dependencies. Used by `uv sync` to install all packages into the virtual environment.

### `requirements.txt`
Alternative dependency list for projects not using `uv`. Install with `pip install -r requirements.txt`.

---

### `core/ingestion.py`
Handles loading PDF files from the `data/resumes/` and `data/job_descriptions/` directories and converting them into LangChain `Document` objects with metadata. Each document gets metadata like `candidate_id`, `name`, `filename`, and `source` (for resumes) or `jd_id` and `title` (for job descriptions). These Document objects are what gets passed to the vector store for embedding.

### `core/vector_store.py`
Manages the Pinecone vector database. Handles three main responsibilities: creating/verifying the Pinecone index on startup, upserting resume and job description embeddings in batches (with rate limit handling), and running semantic similarity searches filtered by document type (`resume` or `job_description`). Includes a type filter so resume searches never return job descriptions and vice versa.

### `core/hybrid_indexer.py`
Implements a hybrid search system that combines BM25 keyword search with Pinecone vector search. Uses Reciprocal Rank Fusion (RRF) to merge results from both methods into a single ranked list. This gives better recall than pure vector search — especially for specific skill names or job titles that might not embed well semantically.

### `core/search_router.py`
The intelligent query router built with LangChain's `RunnableBranch`. Analyzes each search query and decides whether to use shallow search (fast vector-only for simple queries like "Python developer") or deep search (hybrid BM25 + vector + LLM reranking for complex queries like "senior accountant with QuickBooks and tax experience"). The LLM makes the routing decision; simple word count is the fallback if the LLM is unavailable.

### `core/evaluator.py`
The RAGAS-based evaluation system. After a search is run, this module evaluates response quality across four metrics: Answer Relevancy, Context Precision, Faithfulness, and Answer Correctness. Builds an `EvaluationDataset` from the search results and runs them through RAGAS using Gemini as both the LLM and embedding model. Stores evaluation history and supports CSV export.

---

### `utils/config.py`
Loads all environment variables from the `.env` file using `python-dotenv`. Provides a single source of truth for API keys (`GOOGLE_API_KEY`, `PINECONE_API_KEY`), Pinecone settings (`PINECONE_INDEX_NAME`, `PINECONE_DIMENSION`, `PINECONE_METRIC`), and model names (`LLM_MODEL`, `EMBEDDING_MODEL`). All other files import from here instead of reading env vars directly.

### `utils/utils.py`
Shared utility functions used across the project. Includes `get_logger()` for consistent logging, `load_pdf()` for extracting text from PDF files, `split_text()` for chunking documents, `get_embeddings()` for initializing the Google Gemini embedding model, and `is_quota_error()` for detecting API rate limit errors so they can be handled gracefully.

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
LLM_MODEL=gemini-2.0-flash
```

### 4. Add your documents

Place resume PDFs in `data/resumes/` and job description PDFs in `data/job_descriptions/`.

### 5. Index documents into Pinecone

```bash
uv run python reindex.py
```

### 6. Run the app

```bash
uv run streamlit run main.py
```

---

## How Search Works

HireFlow uses a `RunnableBranch` router to decide between two search strategies:

**Shallow search** — fast vector-only search via Pinecone. Best for simple, specific queries like "Python developer" or "accountant".

**Deep search** — hybrid BM25 + vector search merged with RRF, followed by LLM reranking. Best for complex queries like "senior accountant with QuickBooks and 5 years of tax experience". The LLM reranker re-orders results based on semantic fit, not just similarity scores.

The routing decision is made automatically by the LLM on each query, or you can force a mode manually from the UI.

---

## Evaluation

HireFlow includes a RAGAS-based evaluation system accessible from the **Evaluate** tab in the UI:

| Metric | Description |
|---|---|
| Answer Relevancy | How relevant the retrieved candidates are to the query |
| Context Precision | Precision of the retrieved resume context |
| Faithfulness | Factual consistency of the response with context |
| Answer Correctness | Overall quality of the candidate match |

Results are tracked across sessions and can be exported to CSV.

---

## Notes

- Pinecone free tier supports one index — make sure `PINECONE_DIMENSION=3072` matches your embedding model
- Resume and job description PDFs are gitignored — add your own to `data/`
- API quota limits are handled automatically with batched indexing and delays in `reindex.py`
- Run `check.py` to debug search results if candidates aren't showing up correctly

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
