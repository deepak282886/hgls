"""
attention.py — Attention Mechanism.

Allocates resources and selects what gets processed.
Novel or high-salience inputs receive more hypothesis budget.
"""

import random
from typing import List, TypeVar, Dict

T = TypeVar('T')


class AttentionMechanism:

    def __init__(self):
        self._salience_counts: Dict[str, int] = {}

    def focus(
        self,
        candidates: List[T],
        scores: List[float] = None,
        top_k: int = None,
    ) -> List[T]:
        """Select candidates to attend to, weighted by scores."""
        if not candidates:
            return []
        if top_k is None:
            top_k = max(1, len(candidates) // 2)
        if scores is None:
            scores = [1.0] * len(candidates)

        paired = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        selected = [paired[0][1]]

        if len(paired) > 1 and top_k > 1:
            rest        = paired[1:]
            rest_scores = [s for s, _ in rest]
            total       = sum(rest_scores) or 1
            probs       = [s / total for s in rest_scores]
            n_sample    = min(top_k - 1, len(rest))
            indices     = random.choices(range(len(rest)), weights=probs, k=n_sample)
            for idx in set(indices):
                selected.append(rest[idx][1])

        return selected

    def compute_salience(self, text: str, level: int) -> float:
        """Novel inputs are more salient; frequently seen inputs less so."""
        count = self._salience_counts.get(text, 0)
        self._salience_counts[text] = count + 1
        return max(0.2, 1.0 - count * 0.08)

    def allocate_hypotheses(self, base_n: int, salience: float) -> int:
        """Scale hypothesis budget by salience."""
        return max(3, int(base_n * salience))

    def stats(self) -> dict:
        return {'unique_inputs_seen': len(self._salience_counts)}