"""
scripts/evaluate.py

CLI script to run the evaluation suite against the QA pipeline.
Loads a labelled evaluation dataset, runs each sample through the pipeline,
and prints a metrics report.

Usage:
    python scripts/evaluate.py --dataset ./data/eval_set.jsonl --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the QA pipeline.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the evaluation dataset (JSONL, one EvalSample per line).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query during evaluation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON metrics report.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    """Load evaluation samples from a JSONL file.

    Args:
        path: Path to the JSONL evaluation dataset.

    Returns:
        List of raw dicts (one per line).
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for the evaluation CLI."""
    args = parse_args()

    # TODO: Build pipeline from settings
    # settings = get_settings()
    # pipeline = build_pipeline(settings)
    # evaluator = Evaluator(pipeline)

    # TODO: Load dataset, run evaluation, print / save metrics
    raise NotImplementedError("Evaluation pipeline not yet implemented.")


if __name__ == "__main__":
    main()
