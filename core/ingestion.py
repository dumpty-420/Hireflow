"""
Document loading and processing for resumes and job descriptions.
Converts PDF files to LangChain Document objects with metadata.
"""

import os
from pathlib import Path
from langchain_core.documents import Document
from utils.utils import get_logger, load_pdf, split_text
import re

logger = get_logger(__name__)


def load_resumes(directory: str) -> list:
    """Load all resume PDFs from directory and convert to Document objects"""
    resumes = []

    if not os.path.exists(directory):
        return resumes

    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        text = load_pdf(file_path)

        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": pdf_file,
                    "candidate_id": f"c_{Path(pdf_file).stem}",
                    "name": re.sub(
                        r"\s*resume\s*\d*",
                        "",
                        Path(pdf_file).stem.replace("_", " "),
                        flags=re.IGNORECASE,
                    )
                    .strip()
                    .title(),
                },
            )
            resumes.append(doc)

    return resumes


def load_job_descriptions(directory: str) -> list:
    """Load all job description PDFs from directory and convert to Document objects"""
    jobs = []

    if not os.path.exists(directory):
        return jobs

    pdf_files = [
        f for f in os.listdir(directory) if f.endswith(".pdf")
    ]  # [senior_software_engineer.pdf, data_scientist.pdf,GenAI_engineer.pdf]

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)  # [GenAI_engineer.pdf]
        text = load_pdf(file_path)

        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": pdf_file,
                    "jd_id": f"jd_{Path(pdf_file).stem}",
                    "title": Path(pdf_file).stem.replace("_", " ").title(),
                },
            )
            jobs.append(doc)

    return jobs  # [All text_content forsenior_software_engineer.pdf,data_scientist]


class DocumentProcessor:
    """Legacy document processor class - kept for backward compatibility"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize with text chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, file_path: str):
        """Load PDF using utility function"""
        return load_pdf(file_path)

    def split_text(self, text: str):
        """Split text using utility function"""
        return split_text(text, self.chunk_size, self.chunk_overlap)

    def process_resume_pdf(self, file_path: str):
        """Process single resume PDF into Document object"""
        text = self.load_pdf(file_path)
        if text:
            return Document(page_content=text, metadata={"source": file_path})
        return None
