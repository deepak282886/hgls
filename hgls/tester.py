"""
tester.py — Extreme Tester with Autoregressive Token-Level Reward.

Two modes:

  test()               — original reconstruction test, returns (outcome, score)
                         Used during hypothesis generation / mutation loops.

  test_autoregressive() — token-level test. Compares generated vs target
                          word by word, returning a binary mask per token
                          AND which structures contributed each token.
                          Used for hierarchical reward propagation.

Storage decisions still use the extreme thresholds:
  success  (score >= SUCCESS_THRESHOLD) — stored, mutated
  failure  (score <= FAILURE_THRESHOLD) — stored as negative example
  mediocre — discarded

But the score itself is now a true gradient (0.0–1.0), not forced binary.
This gradient flows into structure.fitness and into the reward propagation.
"""

from difflib import SequenceMatcher
from typing import Tuple, List, Set, Dict, Optional, Literal, TYPE_CHECKING

from hgls.structures import GenerativeStructure

if TYPE_CHECKING:
    from hgls.library import Library
    from hgls.llm_parent import LLMParentalInterface

Outcome = Literal['success', 'failure', 'mediocre']

SUCCESS_THRESHOLD = 0.99
FAILURE_THRESHOLD = 0.30


class ExtremeTester:
    """
    Evaluates hypotheses. Returns gradient scores, stores extreme outcomes.
    LLM parental signal blended in for phrase/schema level uncertainty only.
    """

    def __init__(self, llm_parent: Optional['LLMParentalInterface'] = None):
        self.llm_parent  = llm_parent
        self._n_tested   = 0
        self._n_success  = 0
        self._n_failure  = 0
        self._n_mediocre = 0

    # ── Standard test ─────────────────────────────────────────────

    def test(
        self,
        hypothesis: GenerativeStructure,
        target: str,
        library: Optional['Library'] = None,
        use_llm: bool = False,
    ) -> Tuple[Outcome, float]:
        """
        Test hypothesis against target. Returns (outcome, score).
        Score is a true gradient in [0, 1] — not forced to 0 or 1.
        Fitness is set to the actual score, not overridden.
        """
        self._n_tested += 1
        hypothesis.test_count += 1

        generated = hypothesis.generate(library)
        score = _similarity(generated, target)

        # Blend LLM signal only in the uncertain zone at higher levels
        if (
            use_llm
            and self.llm_parent
            and self.llm_parent.signal_strength > 0.01
            and FAILURE_THRESHOLD < score < SUCCESS_THRESHOLD
        ):
            llm_score = self.llm_parent.evaluate_reconstruction(generated, target)
            w = min(self.llm_parent.signal_strength, 0.4)
            score = score * (1 - w) + llm_score * w

        # Fitness = actual gradient score (not forced binary)
        hypothesis.fitness = score

        outcome = _classify(score)
        self._bump(outcome)
        return outcome, score

    # ── Autoregressive token-level test ───────────────────────────

    def test_autoregressive(
        self,
        hypothesis: GenerativeStructure,
        target: str,
        library: Optional['Library'] = None,
    ) -> Tuple[Outcome, float, List[int], List[Set[str]]]:
        """
        Token-level test with full hierarchical trace.

        Returns
        -------
        outcome       : 'success' | 'failure' | 'mediocre'
        score         : gradient float in [0, 1]
        token_mask    : list of 1/0 per generated token (1=correct, 0=wrong)
        token_structs : list of sets of struct_ids that contributed each token

        How it works:
          1. Generate text + character-level trace from hypothesis
          2. Tokenise generated and target by whitespace
          3. Compare token by token — autoregressive 1/0 mask
          4. For each token, find all struct_ids whose trace spans overlap
          5. Score = blend of token accuracy + char similarity
        """
        self._n_tested += 1
        hypothesis.test_count += 1

        generated, trace = hypothesis.generate_with_trace(library)

        gen_tokens = generated.split()
        tgt_tokens = target.split()

        # Map each generated token to its character span
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

        # Build token mask and collect contributing structures per token
        token_mask: List[int] = []
        token_structs: List[Set[str]] = []

        for i, (tok, t_start, t_end) in enumerate(token_spans):
            correct = (i < len(tgt_tokens) and tok == tgt_tokens[i])
            token_mask.append(1 if correct else 0)

            # Collect all struct_ids whose trace overlaps this token's char span
            contributors: Set[str] = set()
            for sid, s_start, s_end in trace:
                if s_start < t_end and s_end > t_start:
                    contributors.add(sid)
            token_structs.append(contributors)

        # Score: blend token accuracy with char similarity
        token_acc  = sum(token_mask) / max(len(token_mask), 1) if token_mask else 0.0
        char_score = _similarity(generated, target)
        score      = 0.6 * token_acc + 0.4 * char_score

        hypothesis.fitness = score
        outcome = _classify(score)
        self._bump(outcome)

        return outcome, score, token_mask, token_structs

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = max(1, self._n_tested)
        return {
            'total_tests':  self._n_tested,
            'successes':    self._n_success,
            'failures':     self._n_failure,
            'mediocre':     self._n_mediocre,
            'success_rate': round(self._n_success  / total, 3),
            'failure_rate': round(self._n_failure  / total, 3),
        }

    def _bump(self, outcome: Outcome) -> None:
        if outcome == 'success':
            self._n_success  += 1
        elif outcome == 'failure':
            self._n_failure  += 1
        else:
            self._n_mediocre += 1


# ── Module-level helpers ───────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    if not b:
        return 1.0 if not a else 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _classify(score: float) -> Outcome:
    if score >= SUCCESS_THRESHOLD:
        return 'success'
    elif score <= FAILURE_THRESHOLD:
        return 'failure'
    return 'mediocre'
