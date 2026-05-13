"""HireFlow - Clean Architecture Implementation"""

import streamlit as st
from pathlib import Path
import sys
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from core.re_ranker import ReRanker
from core.ingestion import load_resumes, DocumentProcessor
from core.parsing import ResumeParser, JobParser
from utils.schemas import JobDescription, Resume
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from core.memory_rag import MemoryRAG

# Data directory
DATA_RESUMES_DIR = project_root / "data" / "resumes"

# Page config
st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================


class SystemManager:
    """Manages system components without business logic"""

    def __init__(self):
        self._components = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize core system components"""
        if self._initialized:
            return True

        try:
            # Initialize VectorStore
            vector_store = VectorStore()
            vector_store_ready = vector_store.initialize()

            # Create LLM
            llm = None
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                st.warning(f"LLM initialization failed: {e}")

            # Create core components
            hybrid_indexer = HybridIndexer() if vector_store_ready else None
            document_processor = DocumentProcessor()
            resume_parser = ResumeParser()
            job_parser = JobParser()
            reranker = ReRanker()
            memory_rag = MemoryRAG()

            # Auto-index existing resumes using core module
            if hybrid_indexer and vector_store_ready:
                try:
                    existing_resumes = load_resumes(str(DATA_RESUMES_DIR))
                    if existing_resumes:
                        hybrid_indexer.index_resumes(existing_resumes)
                except Exception as e:
                    st.warning(f"Auto-indexing failed: {e}")

            # Store components
            self._components = {
                "vector_store": vector_store,
                "hybrid_indexer": hybrid_indexer,
                "llm": llm,
                "document_processor": document_processor,
                "resume_parser": resume_parser,
                "job_parser": job_parser,
                "reranker": reranker,
                "memory_rag": memory_rag,
                "vector_store_ready": vector_store_ready,
            }

            self._initialized = True
            return True

        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return False

    def get_component(self, name: str) -> Any:
        """Get component by name"""
        if not self._initialized:
            raise RuntimeError("System not initialized")
        return self._components.get(name)

    def is_ready(self) -> bool:
        """Check if system is ready"""
        return self._initialized and self._components.get("vector_store_ready", False)


# ============================================================================
# UI LAYER
# ============================================================================


class HireFlowUI:
    """Clean UI layer that delegates to core modules"""

    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager

    def render_upload_section(self):
        """Render resume upload section"""
        st.header("Add Resumes")
        st.info("Upload PDF resumes to search through")

        resume_files = st.file_uploader(
            "Select PDF Resumes", type="pdf", accept_multiple_files=True
        )

        if resume_files:
            st.success(f"Selected {len(resume_files)} resume(s)")
            if st.button("Process & Index Resumes", type="primary"):
                self.handle_resume_upload(resume_files)

    def render_search_section(self):
        """Render candidate search section"""
        st.header("Search Candidates")
        st.info("Enter job details to find matching candidates")

        with st.form("search_form"):
            job_title = st.text_input(
                "Job Title", placeholder="e.g., Senior Accountant"
            )
            job_description = st.text_area(
                "Job Description",
                placeholder="Enter detailed job requirements...",
                height=100,
            )
            required_skills = st.text_area(
                "Required Skills (one per line)",
                placeholder="Python\nJavaScript\nReact",
            )
            top_k = st.slider("Number of Results", 3, 10, 5)

            submitted = st.form_submit_button("Find Candidates", type="primary")

        if submitted and (job_title or job_description):
            self.handle_search(job_title, job_description, required_skills, top_k)

    def render_status_sidebar(self):
        """Render system status in sidebar"""
        st.sidebar.header("System Status")

        if self.system_manager.is_ready():
            hybrid_indexer = self.system_manager.get_component("hybrid_indexer")
            if hybrid_indexer:
                st.sidebar.write(
                    f"**Resumes Indexed:** {len(hybrid_indexer.resume_texts)}"
                )
            st.sidebar.write("**System:** Ready")
        else:
            st.sidebar.write("**System:** Initializing...")

        st.sidebar.markdown("---")
        st.sidebar.write("**Memory & Evaluation:**")

        memory_rag = self.system_manager.get_component("memory_rag")
        if memory_rag:
            stats = memory_rag.get_memory_stats()
            st.sidebar.write(f"• Total Interactions: {stats['total_messages']}")
            st.sidebar.write(f"• Searches: {stats['search_count']}")
            st.sidebar.write(f"• Candidate Views: {stats['candidate_views']}")

        st.sidebar.markdown("---")
        st.sidebar.write("**Navigation:**")
        if st.sidebar.button("📊 Memory & Evaluation", use_container_width=True):
            st.session_state.page = "memory_eval"
        if st.sidebar.button("🏠 Main Page", use_container_width=True):
            st.session_state.page = "main"

    def render_memory_evaluation_page(self):
        """Render Memory & Evaluation page"""
        st.header("🧠 Memory & Evaluation Dashboard")

        st.subheader("📝 Search Memory")
        memory_rag = self.system_manager.get_component("memory_rag")

        if memory_rag:
            col1, col2, col3 = st.columns(3)
            stats = memory_rag.get_memory_stats()

            with col1:
                st.metric("Total Interactions", stats["total_messages"])
            with col2:
                st.metric("Searches", stats["search_count"])
            with col3:
                st.metric("Candidate Views", stats["candidate_views"])

            st.subheader("🔍 Recent Search History")
            search_history = memory_rag.get_search_history()
            if search_history:
                for i, query in enumerate(search_history, 1):
                    st.write(f"{i}. **{query}**")
            else:
                st.info(
                    "No search history yet. Perform some searches to see them here!"
                )

            st.subheader("👥 Recent Interactions")
            messages = memory_rag.memory.chat_memory.messages[-10:]
            for msg in messages:
                if hasattr(msg, "content"):
                    if "Search:" in msg.content:
                        st.write(f"🔍 **{msg.content}**")
                    elif "Viewed candidate:" in msg.content:
                        st.write(f"👀 **{msg.content}**")
                    else:
                        st.write(f"💬 {msg.content}")
        else:
            st.warning("Memory RAG not available")

        st.subheader("📊 RAG Quality Evaluation")
        st.info("Evaluate the quality of your search results using RAGAS metrics")

        with st.form("evaluation_form"):
            eval_query = st.text_input(
                "Query to Evaluate",
                placeholder="e.g., Senior Accountant with QuickBooks",
            )
            expected_skills = st.text_area(
                "Expected Skills (one per line)",
                placeholder="QuickBooks\nAccounting\nExcel",
            )
            eval_top_k = st.slider("Number of Results to Evaluate", 3, 10, 5)

            if st.form_submit_button("Evaluate Search Quality"):
                if eval_query and expected_skills:
                    self.run_evaluation(
                        eval_query, expected_skills.split("\n"), eval_top_k
                    )
                else:
                    st.warning("Please provide both query and expected skills")

    def run_evaluation(self, query: str, expected_skills: list, top_k: int):
        """Run RAG evaluation"""
        try:
            from core.evaluator import RAGEvaluator

            evaluator = RAGEvaluator()
            st.info("Running evaluation... This may take a moment.")

            skills = [s.strip() for s in expected_skills if s.strip()]
            metrics = evaluator.evaluate_search_quality(query, skills, "deep", top_k)

            if metrics:
                st.success("Evaluation completed!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Answer Relevancy", f"{metrics.answer_relevancy:.2f}")
                    st.metric("Context Precision", f"{metrics.context_precision:.2f}")
                with col2:
                    st.metric("Faithfulness", f"{metrics.faithfulness:.2f}")
                    st.metric("Answer Correctness", f"{metrics.answer_correctness:.2f}")

                st.metric("Overall Score", f"{metrics.overall_score:.2f}")

                if metrics.overall_score >= 8:
                    st.success("Excellent search quality! 🎉")
                elif metrics.overall_score >= 6:
                    st.info("Good search quality. Room for improvement.")
                else:
                    st.warning("Search quality needs improvement.")
            else:
                st.warning(
                    "Evaluation returned no results. Check your query and available data."
                )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.info(
                "This might be due to missing dependencies or configuration issues."
            )

    def handle_resume_upload(self, resume_files):
        """Handle resume upload using core modules"""
        try:
            document_processor = self.system_manager.get_component("document_processor")
            resume_parser = self.system_manager.get_component("resume_parser")
            hybrid_indexer = self.system_manager.get_component("hybrid_indexer")

            if not all([document_processor, resume_parser, hybrid_indexer]):
                st.error("System not ready for resume processing")
                return

            st.info(f"Processing {len(resume_files)} resume(s)...")

            processed = 0
            for file in resume_files:
                try:
                    import os

                    os.makedirs(DATA_RESUMES_DIR, exist_ok=True)
                    temp_path = str(DATA_RESUMES_DIR / file.name)

                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    text = document_processor.load_pdf(temp_path)
                    if text:
                        from langchain_core.documents import Document

                        resume_doc = Document(
                            page_content=text,
                            metadata={
                                "source": temp_path,
                                "filename": file.name,
                                "candidate_id": f"c_{file.name.replace('.pdf', '')}",
                                "name": file.name.replace(".pdf", "")
                                .replace("_", " ")
                                .title(),
                            },
                        )

                        candidate_id = f"c_{file.name.replace('.pdf', '')}"
                        resume_parser.parse_resume(text, candidate_id)

                        indexing_result = hybrid_indexer.index_resumes([resume_doc])

                        if indexing_result:
                            processed += 1
                            st.success(f"{file.name} - Indexed successfully")
                        else:
                            st.warning(f"{file.name} - Indexing failed")
                    else:
                        st.error(f"{file.name} - Could not extract text")

                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")

            if processed > 0:
                st.success(f"Successfully processed and indexed {processed} resumes!")
                st.rerun()

        except Exception as e:
            st.error(f"Resume upload failed: {e}")

    def handle_search(
        self, job_title: str, job_description: str, required_skills: str, top_k: int
    ):
        """Handle candidate search using vector store + hybrid indexer"""
        try:
            vector_store = self.system_manager.get_component("vector_store")
            hybrid_indexer = self.system_manager.get_component("hybrid_indexer")
            reranker = self.system_manager.get_component("reranker")
            memory_rag = self.system_manager.get_component("memory_rag")

            if not vector_store:
                st.error("Search service not available")
                return

            # Build a clean, focused search query — avoid dumping full JD text
            skills_list = (
                [s.strip() for s in required_skills.split("\n") if s.strip()]
                if required_skills
                else []
            )
            search_query = (
                f"{job_title} {' '.join(skills_list)}".strip()
                if skills_list
                else job_title
            )

            with st.spinner("Searching candidates..."):
                # Primary: use Pinecone vector store for semantic search
                candidates_data = vector_store.search_resumes(search_query, top_k=top_k)

                # Fallback: use hybrid indexer if vector store returns nothing
                if not candidates_data and hybrid_indexer:
                    candidates_data = hybrid_indexer.search_resumes(
                        search_query, top_k=top_k
                    )

                # Record search in memory
                if memory_rag:
                    memory_rag.record_search(
                        search_query, len(candidates_data) if candidates_data else 0
                    )

                if candidates_data:
                    st.success(f"Found {len(candidates_data)} candidates!")

                    # Record top 3 candidate views in memory
                    if memory_rag:
                        for candidate in candidates_data[:3]:
                            name = candidate.get("name") or candidate.get(
                                "metadata", {}
                            ).get("name", "Unknown")
                            memory_rag.record_candidate_view(name)

                    self.display_search_results(
                        candidates_data,
                        job_title or "Position",
                        job_description,
                        reranker,
                    )
                else:
                    st.warning("No matching candidates found")

        except Exception as e:
            st.error(f"Search failed: {e}")

    def display_search_results(
        self,
        candidates_data: list,
        job_title: str,
        job_description: str,
        reranker: ReRanker,
    ):
        """Display search results — handles both vector store and hybrid indexer result shapes"""
        st.header(f"Top Matches for: {job_title}")

        for i, candidate in enumerate(candidates_data):
            # Normalise fields — vector store and hybrid indexer return different shapes
            metadata = candidate.get("metadata", {})
            name = candidate.get("name") or metadata.get("name") or f"Candidate {i + 1}"
            score = candidate.get("score") or candidate.get("combined_score", 0)
            skills = candidate.get("skills") or metadata.get("skills", [])
            experience = candidate.get("experience") or metadata.get(
                "experience", "N/A"
            )
            location = candidate.get("location") or metadata.get("location", "N/A")
            text = (
                candidate.get("text")
                or candidate.get("page_content")
                or metadata.get("page_content", "")
            )

            with st.expander(f"{name}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Score:** {score:.3f}")
                    if skills:
                        skills_str = (
                            ", ".join(skills) if isinstance(skills, list) else skills
                        )
                        st.write(f"**Skills:** {skills_str}")
                    st.write(f"**Experience:** {experience}")
                    st.write(f"**Location:** {location}")

                    if text:
                        st.markdown("**Resume Preview:**")
                        st.text(text[:300] + "..." if len(text) > 300 else text)

                    # AI evaluation for top 3 candidates
                    if i < 3 and reranker:
                        # Normalise candidate dict before passing to evaluator
                        candidate["name"] = name
                        candidate["text"] = text
                        candidate["skills"] = skills
                        self.display_ai_evaluation(
                            candidate, job_title, job_description, reranker
                        )

                with col2:
                    st.metric("Score", f"{score:.3f}")

    def display_ai_evaluation(
        self, candidate: dict, job_title: str, job_description: str, reranker: ReRanker
    ):
        """Display AI evaluation using core ReRanker"""
        try:
            jd = JobDescription(
                jd_id=f"jd_{job_title.lower().replace(' ', '_')}",
                title=job_title or "Position",
                text=job_description,
            )

            resume = Resume(
                candidate_id=candidate.get("candidate_id", "unknown"),
                name=candidate.get("name", "Unknown"),
                email="candidate@example.com",
                phone="+1-555-0000",
                experience=5,
                skills=candidate.get("skills", []),
                education="Bachelor's Degree",
                text=candidate.get("text", ""),
            )

            evaluation = reranker.evaluate_candidate(resume, jd)
            if evaluation:
                eval_data = evaluation.model_dump()
                fit_score = eval_data.get("fit_score", 0)

                st.markdown("**AI Evaluation:**")
                if fit_score >= 8:
                    st.success(f"**AI Score: {fit_score}/10**")
                elif fit_score >= 6:
                    st.info(f"**AI Score: {fit_score}/10**")
                else:
                    st.warning(f"**AI Score: {fit_score}/10**")

                strengths = eval_data.get("strengths", [])
                if strengths:
                    st.write("**Strengths:**")
                    for s in strengths[:2]:
                        st.write(f"• {s}")

                gaps = eval_data.get("gaps", [])
                if gaps:
                    st.write("**Gaps:**")
                    for g in gaps[:2]:
                        st.write(f"• {g}")

                summary = eval_data.get("summary", "")
                if summary:
                    st.write("**Summary:**")
                    st.write(summary)

        except Exception as e:
            st.info("AI evaluation not available")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


def main():
    st.title("HireFlow - AI Resume Search")

    if "page" not in st.session_state:
        st.session_state.page = "main"

    system_manager = SystemManager()
    if not system_manager.initialize():
        st.error("System initialization failed. Please check configuration.")
        return

    ui = HireFlowUI(system_manager)
    ui.render_status_sidebar()

    if st.session_state.get("page") == "memory_eval":
        ui.render_memory_evaluation_page()
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            ui.render_upload_section()
        with col2:
            ui.render_search_section()


if __name__ == "__main__":
    main()
