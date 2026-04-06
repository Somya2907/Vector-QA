"""
evaluator.py

Measures the quality of the QA pipeline across multiple dimensions:
  - Retrieval quality  (recall@k, MRR, NDCG)
  - Answer faithfulness (is the answer grounded in the retrieved context?)
  - Answer correctness  (does it match a reference answer?)
  - Permission integrity (are access-control rules being respected?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.pipeline.qa_pipeline import QAPipeline, QARequest, QAResponse


@dataclass
class EvalSample:
    """A single labelled evaluation example."""

    query: str
    user_context: dict[str, Any]
    reference_answer: str
    relevant_doc_ids: list[str]           # Ground-truth relevant documents
    accessible_doc_ids: list[str]         # Documents this user should be able to see


@dataclass
class EvalMetrics:
    """Aggregated evaluation results over a dataset."""

    recall_at_k: float = 0.0
    mrr: float = 0.0                      # Mean Reciprocal Rank
    faithfulness: float = 0.0            # Fraction of answers grounded in context
    correctness: float = 0.0             # Similarity to reference answers
    permission_violation_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Runs an evaluation suite against a QAPipeline and returns EvalMetrics."""

    def __init__(self, pipeline: QAPipeline) -> None:
        """
        Args:
            pipeline: The QAPipeline instance to evaluate.
        """
        self.pipeline = pipeline

    def evaluate(self, samples: list[EvalSample]) -> EvalMetrics:
        """Run all evaluation metrics over the provided samples.

        Args:
            samples: Labelled evaluation examples.

        Returns:
            Aggregated EvalMetrics across all samples.
        """
        raise NotImplementedError

    def compute_retrieval_metrics(
        self,
        samples: list[EvalSample],
        responses: list[QAResponse],
    ) -> dict[str, float]:
        """Compute recall@k, MRR, and NDCG for the retrieved chunks.

        Args:
            samples:   Ground-truth evaluation examples.
            responses: Corresponding pipeline responses.

        Returns:
            Dict with metric names mapped to their scores.
        """
        raise NotImplementedError

    def compute_permission_violation_rate(
        self,
        samples: list[EvalSample],
        responses: list[QAResponse],
    ) -> float:
        """Measure how often the pipeline returns chunks the user shouldn't see.

        Args:
            samples:   Ground-truth evaluation examples (with accessible_doc_ids).
            responses: Corresponding pipeline responses.

        Returns:
            Fraction of responses containing at least one permission violation.
        """
        raise NotImplementedError
