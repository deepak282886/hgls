"""
tester.py — Extreme Tester with Autoregressive Token-Level Reward.

Simplified: LLM blend removed entirely.
Score is pure SequenceMatcher ratio — a true gradient.
Emotional evaluator handles semantic scoring separately.
"""

from difflib import SequenceMatcher
from typing import Tuple, List, Set, Optional, Literal, TYPE_CHECKING

from hgls.structures import GenerativeStructure

if TYPE_CHECKING:
    from hgls.library import Library

Outcome = Literal['success', 'failure', 'mediocre']

SUCCESS_THRESHOLD = 0.99
FAILURE_THRESHOLD = 0.30


class ExtremeTester:

    def __init__(self, llm_parent=None):
        # llm_parent kept for compatibility but no longer used in testing
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
        self._n_tested += 1
        hypothesis.test_count += 1
        generated     = hypothesis.generate(library)
        score         = _similarity(generated, target)
        hypothesis.fitness = score
        outcome       = _classify(score)
        self._bump(outcome)
        return outcome, score

    def test_autoregressive(
        self,
        hypothesis: GenerativeStructure,
        target: str,
        library: Optional['Library'] = None,
    ) -> Tuple[Outcome, float, List[int], List[Set[str]]]:
        self._n_tested += 1
        hypothesis.test_count += 1

        generated, trace = hypothesis.generate_with_trace(library)

        gen_tokens = generated.split()
        tgt_tokens = target.split()

        token_spans: List[Tuple[str, int, int]] = []
        pos = 0
        remaining = generated
        for tok in gen_tokens:
            idx = remaining.find(tok)
            if idx < 0:
                break
            abs_start = pos + idx
            token_spans.append((tok, abs_start, abs_start + len(tok)))
            pos = abs_start + len(tok)
            remaining = generated[pos:]

        token_mask:    List[int]      = []
        token_structs: List[Set[str]] = []

        for i, (tok, t_start, t_end) in enumerate(token_spans):
            correct = (i < len(tgt_tokens) and tok == tgt_tokens[i])
            token_mask.append(1 if correct else 0)
            contributors: Set[str] = set()
            for sid, s_start, s_end in trace:
                if s_start < t_end and s_end > t_start:
                    contributors.add(sid)
            token_structs.append(contributors)

        token_acc  = sum(token_mask) / max(len(token_mask), 1) if token_mask else 0.0
        char_score = _similarity(generated, target)
        score      = 0.6 * token_acc + 0.4 * char_score

        hypothesis.fitness = score
        outcome = _classify(score)
        self._bump(outcome)
        return outcome, score, token_mask, token_structs

    def stats(self) -> dict:
        total = max(1, self._n_tested)
        return {
            'total_tests':  self._n_tested,
            'successes':    self._n_success,
            'failures':     self._n_failure,
            'mediocre':     self._n_mediocre,
            'success_rate': round(self._n_success / total, 3),
        }

    def _bump(self, outcome: Outcome) -> None:
        if outcome == 'success':   self._n_success  += 1
        elif outcome == 'failure': self._n_failure  += 1
        else:                      self._n_mediocre += 1


def _similarity(a: str, b: str) -> float:
    if not b:
        return 1.0 if not a else 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _classify(score: float) -> Outcome:
    if score >= SUCCESS_THRESHOLD: return 'success'
    elif score <= FAILURE_THRESHOLD: return 'failure'
    return 'mediocre'