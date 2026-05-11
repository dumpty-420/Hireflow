# HireFlow 🚀

An AI-powered recruitment platform that matches candidates to job descriptions using semantic search, hybrid indexing, and RAG-based evaluation.

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
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector DB | Pinecone (cosine similarity, 3072 dimensions) |
| LLM | Google Gemini Pro |
| Search | Hybrid (vector + BM25) with LLM reranking |
| Evaluation | RAGAS |
| Orchestration | LangChain |
| UI | Streamlit |

---

## Project Structure

```
HireFlow/
├── core/
│   ├── ingestion.py        # PDF loading and Document creation
│   ├── vector_store.py     # Pinecone embedding and search
│   ├── hybrid_indexer.py   # BM25 + vector hybrid search
│   ├── search_router.py    # LangChain RunnableBranch query router
│   └── evaluator.py        # RAGAS evaluation system
├── utils/
│   ├── config.py           # API keys and configuration
│   └── utils.py            # Shared utilities (logging, PDF loading, embeddings)
├── data/
│   ├── resumes/            # Resume PDFs (not tracked in git)
│   └── job_descriptions/   # Job description PDFs (not tracked in git)
├── main.py                 # Streamlit app entry point
├── reindex.py              # Script to embed and index all documents
└── README.md
```

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

Place resume PDFs in `data/resumes/` and job description PDFs in `data/job_descriptions/`.

### 5. Index documents

```bash
uv run python reindex.py
```

### 6. Run the app

```bash
uv run streamlit run main.py
```

---

## How search works

HireFlow uses a **RunnableBranch** router to decide between two search strategies:

- **Shallow search** — fast vector-only search for simple queries (e.g. "Python developer")
- **Deep search** — hybrid BM25 + vector search with LLM reranking for complex queries (e.g. "senior accountant with QuickBooks and tax experience")

The LLM analyzes the query and picks the right strategy automatically, or you can force a mode manually.

---

## Evaluation

HireFlow includes a RAGAS-based evaluation system that measures:

| Metric | Description |
|---|---|
| Answer Relevancy | How relevant the retrieved candidates are to the query |
| Context Precision | Precision of the retrieved resume context |
| Faithfulness | Factual consistency of the response with context |
| Answer Correctness | Overall quality of the candidate match |

Run evaluation from the Streamlit UI under the **Evaluate** tab.

---

## Notes

- Pinecone free tier supports one index — make sure your index dimension matches your embedding model (3072 for `gemini-embedding-001`)
- Resume PDFs are not committed to the repo — add your own to `data/resumes/`
- API quota limits may require batched indexing (handled automatically in `reindex.py`)

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
