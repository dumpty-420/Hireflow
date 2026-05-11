# reindex.py
from core.vector_store import VectorStore
from core.ingestion import load_resumes, load_job_descriptions
import os

vs = VectorStore()
vs.initialize()


jd_path = "data/jds"
print(f"JD directory exists: {os.path.exists(jd_path)}")
print(f"Files in JD dir: {os.listdir(jd_path) if os.path.exists(jd_path) else 'N/A'}")

resumes = load_resumes("data/resumes")

jobs = load_job_descriptions("data/jds")

print(f"Indexing {len(resumes)} resumes...")
vs.add_resumes(resumes)

print(f"Indexing {len(jobs)} job descriptions...")
vs.add_job_descriptions(jobs)

print("Done! Stats:", vs.get_stats())
