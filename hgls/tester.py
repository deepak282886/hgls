"""
tester.py — Extreme Tester.

Tests hypotheses against target input and produces extreme signals:
  success  (score ≥ 0.95) → disproportionately high reward, stored
  failure  (score ≤ 0.30) → disproportionately high penalty, stored
  mediocre (everything in between) → discarded, leaves no trace

This is the selection pressure that drives generative learning.
"""

from difflib import SequenceMatcher
from typing import Tuple, Literal, Optional, TYPE_CHECKING

from hgls.structures import GenerativeStructure

if TYPE_CHECKING:
    from hgls.library import Library
    from hgls.llm_parent import LLMParentalInterface

Outcome = Literal['success', 'failure', 'mediocre']

SUCCESS_THRESHOLD = 0.99   # Near-exact: single-char mutations on long strings must be mediocre
FAILURE_THRESHOLD = 0.30


class ExtremeTester:
    """
    Evaluates a hypothesis by comparing its generated output to the target.

    LLM parental signal is blended in (weighted by signal_strength) at
    higher levels where pure reconstruction isn't sufficient to judge quality.
    """

    def __init__(self, llm_parent: Optional['LLMParentalInterface'] = None):
        self.llm_parent = llm_parent
        self._n_tested   = 0
        self._n_success  = 0
        self._n_failure  = 0
        self._n_mediocre = 0

    def test(
        self,
        hypothesis: GenerativeStructure,
        target: str,
        library: Optional['Library'] = None,
        use_llm: bool = False,
    ) -> Tuple[Outcome, float]:
        """
        Test hypothesis against target.

        Returns
        -------
        outcome : 'success' | 'failure' | 'mediocre'
        score   : float in [0, 1]
        """
        self._n_tested += 1
        hypothesis.test_count += 1

        generated = hypothesis.generate(library)
        score = self._similarity(generated, target)

        # Blend in LLM parental signal only when reconstruction quality
        # is uncertain — never let LLM override a clearly perfect match.
        # This prevents failed/slow API calls from corrupting clean successes.
        if (
            use_llm
            and self.llm_parent
            and self.llm_parent.signal_strength > 0.01
            and FAILURE_THRESHOLD < score < SUCCESS_THRESHOLD  # uncertain zone only
        ):
            llm_score = self.llm_parent.evaluate_reconstruction(generated, target)
            w = min(self.llm_parent.signal_strength, 0.4)  # cap LLM weight at 40%
            score = score * (1 - w) + llm_score * w

        hypothesis.fitness = score

        if score >= SUCCESS_THRESHOLD:
            self._n_success  += 1
            return 'success', score
        elif score <= FAILURE_THRESHOLD:
            self._n_failure  += 1
            return 'failure', score
        else:
            self._n_mediocre += 1
            return 'mediocre', score

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not b:
            return 1.0 if not a else 0.0
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    def stats(self) -> dict:
        total = max(1, self._n_tested)
        return {
            'total_tests': self._n_tested,
            'successes':   self._n_success,
            'failures':    self._n_failure,
            'mediocre':    self._n_mediocre,
            'success_rate': round(self._n_success  / total, 3),
            'failure_rate': round(self._n_failure  / total, 3),
        }