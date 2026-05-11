"""
RAGAS-based evaluation system for measuring search quality.
Evaluates retrieval and answer generation performance using standard metrics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass
from core.search_router import SearchRouter
from utils.utils import get_logger

# Lazy import: ragas may not be installed or may have dependency conflicts.
# We gracefully degrade so the rest of the application still works.
_ragas_available = False
try:
    from ragas import evaluate as _ragas_evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
        AnswerCorrectness,
    )
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    _ragas_available = True
except (ImportError, ModuleNotFoundError) as _exc:
    _ragas_evaluate = None

logger = get_logger(__name__)


@dataclass
class RAGEvaluationMetrics:
    """Container for RAGAS evaluation metrics with scoring"""

    answer_relevancy: float  # How relevant the answer is to the question
    context_precision: float  # Precision of retrieved context
    faithfulness: float  # Factual consistency with context
    answer_correctness: float  # Overall answer quality
    overall_score: float  # Weighted average of all metrics

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "overall_score": self.overall_score,
        }


class RAGEvaluator:
    """RAGAS-powered search quality evaluator with history tracking"""

    def __init__(self):
        """Initialize RAGAS metrics and evaluation tracking"""
        self.ragas_llm = None
        self.ragas_embeddings = None
        if _ragas_available:
            try:
                from langchain_google_genai import (
                    ChatGoogleGenerativeAI,
                    GoogleGenerativeAIEmbeddings,
                )
                from utils.config import GOOGLE_API_KEY, LLM_MODEL
                import warnings

                warnings.filterwarnings("ignore", category=DeprecationWarning)

                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1
                )
                self.ragas_llm = LangchainLLMWrapper(llm)

                emb = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY
                )
                self.ragas_embeddings = LangchainEmbeddingsWrapper(emb)

                self.ragas_metrics = [
                    AnswerRelevancy(),
                    ContextPrecision(),
                    Faithfulness(),
                    AnswerCorrectness(),
                ]
                logger.info("RAGAS evaluator initialized with Gemini LLM & embeddings")
            except Exception as e:
                logger.warning(
                    f"RAGAS LLM init failed: {e}. Evaluation will return default metrics."
                )
                self.ragas_metrics = []
        else:
            self.ragas_metrics = []
            logger.warning(
                "RAGAS not available. Evaluation will return default metrics."
            )
        self.evaluation_history = []

    def evaluate_search_quality(
        self,
        query: str,
        expected_skills: List[str],
        search_mode: str = "deep",
        top_k: int = 5,
        search_router: Optional[Any] = None,
    ) -> RAGEvaluationMetrics:
        """Run search and evaluate results using RAGAS quality metrics"""
        try:
            if search_router is None:
                search_router = SearchRouter()
            search_result = search_router.search(query, top_k, search_mode=search_mode)

            if not search_result.get("results"):
                return self.create_default_metrics()

            candidates = search_result["results"]
            return self.evaluate_with_ragas(query, candidates, expected_skills)

        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return self.create_default_metrics()

    def create_default_metrics(self) -> RAGEvaluationMetrics:
        """Return zero metrics when search fails or returns no results"""
        return RAGEvaluationMetrics(
            answer_relevancy=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            answer_correctness=0.0,
            overall_score=0.0,
        )

    def evaluate_with_ragas(
        self, query: str, candidates: List[Dict], expected_skills: List[str]
    ) -> RAGEvaluationMetrics:
        """Run RAGAS evaluation on search results and calculate final metrics"""
        if not _ragas_available:
            logger.warning("RAGAS not available – returning default metrics")
            return self.create_default_metrics()

        evaluation_data = self.prepare_ragas_data(query, candidates, expected_skills)
        results = _ragas_evaluate(
            dataset=evaluation_data,
            metrics=self.ragas_metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        metrics = RAGEvaluationMetrics(
            answer_relevancy=float(results["answer_relevancy"]),
            context_precision=float(results["context_precision"]),
            faithfulness=float(results["faithfulness"]),
            answer_correctness=float(results["answer_correctness"]),
            overall_score=0.0,
        )

        metrics.overall_score = self.calculate_overall_score(metrics)
        self.store_evaluation(query, metrics)

        return metrics

    def prepare_ragas_data(
        self, query: str, candidates: List[Dict], expected_skills: List[str]
    ) -> "EvaluationDataset":
        """Convert search results to ragas EvaluationDataset format"""
        samples = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})

            # Use actual resume text as context
            context = candidate.get("page_content", "")[:2000]
            if not context:
                context = str(metadata)

            # Build response from name + content snippet
            name = metadata.get("name", "Candidate")
            snippet = candidate.get("page_content", "")[:300]
            response = f"{name}: {snippet}"

            # Ground truth based on skill keywords found in page_content
            page_text = candidate.get("page_content", "").lower()
            matched = [s for s in expected_skills if s.lower() in page_text]
            match_pct = len(matched) / len(expected_skills) if expected_skills else 0

            if match_pct >= 0.8:
                reference = f"Excellent match: {len(matched)}/{len(expected_skills)} skills found"
            elif match_pct >= 0.6:
                reference = (
                    f"Good match: {len(matched)}/{len(expected_skills)} skills found"
                )
            elif match_pct >= 0.4:
                reference = f"Moderate match: {len(matched)}/{len(expected_skills)} skills found"
            else:
                reference = (
                    f"Poor match: {len(matched)}/{len(expected_skills)} skills found"
                )

            samples.append(
                SingleTurnSample(
                    user_input=query,
                    retrieved_contexts=[context],
                    response=response,
                    reference=reference,
                )
            )

        return EvaluationDataset(samples=samples)

    def create_ground_truth(
        self, candidate_skills: List[str], expected_skills: List[str]
    ) -> str:
        """Generate ground truth labels based on skill matching percentage"""
        if not expected_skills:
            return "No skills specified"

        matched_skills = [
            skill
            for skill in expected_skills
            if skill.lower() in [s.lower() for s in candidate_skills]
        ]
        match_percentage = len(matched_skills) / len(expected_skills)

        if match_percentage >= 0.8:
            return (
                f"Excellent match: {len(matched_skills)}/{len(expected_skills)} skills"
            )
        elif match_percentage >= 0.6:
            return f"Good match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.4:
            return (
                f"Moderate match: {len(matched_skills)}/{len(expected_skills)} skills"
            )
        else:
            return f"Poor match: {len(matched_skills)}/{len(expected_skills)} skills"

    def calculate_overall_score(self, metrics: RAGEvaluationMetrics) -> float:
        """Compute weighted average of all RAGAS metrics"""
        weights = {
            "answer_relevancy": 0.30,
            "context_precision": 0.30,
            "faithfulness": 0.20,
            "answer_correctness": 0.20,
        }

        overall_score = (
            metrics.answer_relevancy * weights["answer_relevancy"]
            + metrics.context_precision * weights["context_precision"]
            + metrics.faithfulness * weights["faithfulness"]
            + metrics.answer_correctness * weights["answer_correctness"]
        )

        return overall_score

    def store_evaluation(self, query: str, metrics: RAGEvaluationMetrics):
        """Save evaluation results to history for later analysis"""
        evaluation_record = {
            "query": query,
            "timestamp": pd.Timestamp.now(),
            "metrics": metrics.to_dict(),
        }
        self.evaluation_history.append(evaluation_record)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}

        avg_metrics = {}
        for metric in [
            "answer_relevancy",
            "context_precision",
            "faithfulness",
            "answer_correctness",
            "overall_score",
        ]:
            values = [record["metrics"][metric] for record in self.evaluation_history]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values)

        return {
            "total_evaluations": len(self.evaluation_history),
            "average_metrics": avg_metrics,
            "recent_evaluations": self.evaluation_history[-5:],
        }

    def export_evaluations(self, filename: str = "rag_evaluations.csv") -> bool:
        """Export all evaluation history to CSV file for analysis"""
        if not self.evaluation_history:
            return False

        try:
            export_data = []
            for i, record in enumerate(self.evaluation_history):
                row = {
                    "evaluation_id": i + 1,
                    "query": record["query"],
                    "timestamp": record["timestamp"],
                    **record["metrics"],
                }
                export_data.append(row)

            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            return True

        except Exception as e:
            logger.error(f"Failed to export evaluations: {e}")
            return False
