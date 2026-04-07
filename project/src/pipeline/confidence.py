"""
src/pipeline/confidence.py

Retrieval confidence scoring for the RAG pipeline.

Combines three signals computed from the top-k retrieval scores:
  - avg_score:   mean score across all retrieved chunks
  - consistency: how tightly clustered the scores are (1 / (1 + std))
  - dominance:   gap between the top and second-best score

Final formula:
  confidence = 0.5 * avg_score + 0.3 * consistency + 0.2 * dominance
  clamped to [0, 1]

Thresholds:
  < LOW_CONF  → abstain (no answer)
  < MED_CONF  → answer with uncertainty warning
  >= MED_CONF → normal answer
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LOW_CONF: float = 0.15
MED_CONF: float = 0.30

# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------


def compute_confidence(scores: list[float], debug: bool = False) -> float:
    """Compute a [0, 1] confidence score from a list of retrieval scores.

    Args:
        scores: Retrieval scores for the top-k results (highest first).
        debug:  Print per-signal breakdown to stdout.

    Returns:
        Confidence value clamped to [0, 1].
    """
    if not scores:
        return 0.0

    n = len(scores)

    # ── Signal 1: average score ───────────────────────────────────────────────
    avg_score = sum(scores) / n

    # ── Signal 2: consistency (inverse of score spread) ──────────────────────
    if n > 1:
        mean = avg_score
        variance = sum((s - mean) ** 2 for s in scores) / n
        score_std = math.sqrt(variance)
    else:
        score_std = 0.0
    consistency = 1.0 / (1.0 + score_std)

    # ── Signal 3: top-score dominance ─────────────────────────────────────────
    top_score = scores[0]
    second_score = scores[1] if n > 1 else 0.0
    dominance = top_score - second_score

    # ── Combine ───────────────────────────────────────────────────────────────
    raw = 0.5 * avg_score + 0.3 * consistency + 0.2 * dominance
    confidence = max(0.0, min(1.0, raw))

    if debug:
        print(
            f"\n[confidence] scores={[round(s, 4) for s in scores]}\n"
            f"             avg={avg_score:.4f}  std={score_std:.4f}  "
            f"consistency={consistency:.4f}  dominance={dominance:.4f}\n"
            f"             confidence={confidence:.4f}"
        )

    return confidence


def confidence_label(confidence: float) -> str:
    """Map a confidence value to a human-readable label.

    Returns:
        "low", "medium", or "high"
    """
    if confidence < LOW_CONF:
        return "low"
    if confidence < MED_CONF:
        return "medium"
    return "high"
