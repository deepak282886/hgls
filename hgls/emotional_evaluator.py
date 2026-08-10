"""
emotional_evaluator.py — Graph-Aware Emotional Evaluator.

Starts from two primitive states: positive and negative.
Grows finer gradient through experience.

Fixed in this version:
  - Maturity grows from the very first evaluation: n/(n+100)
    After 1 eval:   0.01 maturity
    After 100 evals: 0.50 maturity
    After 1000 evals: 0.91 maturity
  - Weight adaptation starts immediately (no 1000 eval guard)
  - All thresholds removed from internal logic
"""

import math
from difflib import SequenceMatcher
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.graph    import MemoryGraph
    from hgls.library  import Library
    from hgls.structures import GenerativeStructure


class EmotionalEvaluator:

    def __init__(
        self,
        graph:   Optional['MemoryGraph'] = None,
        library: Optional['Library']     = None,
    ):
        self.graph   = graph
        self.library = library

        self._eval_count          = 0
        self._positive_count      = 0
        self._negative_count      = 0
        self._surprise_count      = 0
        self._contradiction_count = 0

        # Maturity grows from first evaluation — no artificial delay
        self.maturity = 0.0

        # Learned weights — start equal, drift with experience
        self._w_surface   = 0.4
        self._w_coherence = 0.3
        self._w_novelty   = 0.15
        self._w_surprise  = 0.15

    # ── Primary evaluation ────────────────────────────────────────

    def evaluate(
        self,
        generated: str,
        target:    str,
        source_id: str = None,
        target_id: str = None,
    ) -> dict:
        """
        Evaluate a generated string against a target.
        Grows more nuanced with every evaluation from the very first.
        """
        self._eval_count += 1
        self._update_maturity()

        surface = _surface_similarity(generated, target)

        coherence     = 0.5
        novelty       = 0.5
        surprise      = 0.0
        contradiction = False

        if self.graph and source_id and target_id and self.maturity > 0.01:
            coherence     = self._graph_coherence(source_id, target_id)
            novelty       = self._novelty_signal(source_id, target_id)
            surprise      = self._surprise_signal(source_id, target_id)
            contradiction = self._detects_contradiction(source_id, target_id)

        # Blend signals weighted by maturity
        w_s = self._w_surface
        w_c = self._w_coherence * self.maturity
        w_n = self._w_novelty   * self.maturity
        w_r = self._w_surprise  * self.maturity

        total_w = w_s + w_c + w_n + w_r
        if total_w > 0:
            w_s /= total_w; w_c /= total_w
            w_n /= total_w; w_r /= total_w

        score = w_s * surface + w_c * coherence + w_n * novelty + w_r * surprise

        if contradiction:
            score *= 0.5
            self._contradiction_count += 1

        state = self._emotional_state(score, surface, coherence, novelty, surprise, contradiction)

        if score >= 0.5:
            self._positive_count += 1
        else:
            self._negative_count += 1

        if state == 'deeply_surprising':
            self._surprise_count += 1

        # Adapt weights — starts from first evaluation, just slowly
        self._adapt_weights(surface, coherence, score)

        return {
            'score':         score,
            'state':         state,
            'surface':       surface,
            'coherence':     coherence,
            'novelty':       novelty,
            'surprise':      surprise,
            'contradiction': contradiction,
            'maturity':      self.maturity,
        }

    def quick_score(self, generated: str, target: str) -> float:
        surface = _surface_similarity(generated, target)
        if not self.graph or self.maturity < 0.01:
            return surface
        result = self.evaluate(generated, target)
        return result['score']

    # ── Graph signals ─────────────────────────────────────────────

    def _graph_coherence(self, source_id: str, target_id: str) -> float:
        if not self.graph:
            return 0.5
        src_neighbours = {
            (e.source_id if e.target_id == source_id else e.target_id)
            for e in self.graph.get_neighbours(source_id, min_strength=0.05)
        }
        tgt_neighbours = {
            (e.source_id if e.target_id == target_id else e.target_id)
            for e in self.graph.get_neighbours(target_id, min_strength=0.05)
        }
        if not src_neighbours or not tgt_neighbours:
            return 0.3
        shared = len(src_neighbours & tgt_neighbours)
        total  = len(src_neighbours | tgt_neighbours)
        return shared / max(total, 1)

    def _novelty_signal(self, source_id: str, target_id: str) -> float:
        if not self.graph:
            return 0.5
        src_density = self.graph.density(source_id)
        tgt_density = self.graph.density(target_id)
        neighbours  = {
            (e.source_id if e.target_id == source_id else e.target_id)
            for e in self.graph.get_neighbours(source_id)
        }
        if target_id in neighbours:
            return 0.2
        if src_density > 0.3 and tgt_density > 0.3:
            return 0.9
        if src_density > 0.1 or tgt_density > 0.1:
            return 0.6
        return 0.3

    def _surprise_signal(self, source_id: str, target_id: str) -> float:
        if not self.graph:
            return 0.0
        src_density = self.graph.density(source_id)
        tgt_density = self.graph.density(target_id)
        if src_density < 0.2 or tgt_density < 0.2:
            return 0.0
        src_region = set(self.graph.get_region(source_id, depth=2).keys())
        if target_id in src_region:
            return 0.0
        return min(1.0, src_density * tgt_density * 2.0)

    def _detects_contradiction(self, source_id: str, target_id: str) -> bool:
        if not self.graph or not self.library:
            return False
        src = self.library.get(source_id)
        tgt = self.library.get(target_id)
        if not src or not tgt:
            return False
        src_ratio = src.penalty_count / max(src.reward_count + src.penalty_count, 1)
        tgt_ratio = tgt.penalty_count / max(tgt.reward_count + tgt.penalty_count, 1)
        return src_ratio > 0.7 and tgt_ratio > 0.7

    # ── Emotional state ───────────────────────────────────────────

    def _emotional_state(
        self, score, surface, coherence, novelty, surprise, contradiction
    ) -> str:
        # At very low maturity: just positive/negative
        if self.maturity < 0.05:
            return 'positive' if score >= 0.5 else 'negative'

        if contradiction:
            return 'contradicting'
        if surprise > 0.6 and score > 0.5:
            return 'deeply_surprising'
        if score >= 0.85:
            return 'familiar_confirming'
        if surface > 0.85 and score < 0.7:
            return 'almost_right'
        if novelty > 0.7 and coherence > 0.5:
            return 'novel_exciting'
        if novelty > 0.7 and coherence < 0.3:
            return 'novel_incoherent'
        if score >= 0.5:
            return 'curious'
        if score < 0.25:
            return 'negative'
        return 'uncertain'

    # ── Maturity and adaptation ───────────────────────────────────

    def _update_maturity(self) -> None:
        """
        Maturity grows from first evaluation.
        n/(n+100): reaches 0.5 at 100 evals, 0.9 at 900 evals.
        No artificial delay.
        """
        n = self._eval_count
        self.maturity = n / (n + 100)

    def _adapt_weights(self, surface: float, coherence: float, outcome: float) -> None:
        """
        Adapt signal weights based on what correlates with good outcomes.
        Starts from first evaluation — learning rate is small so early
        noise doesn't destabilize, but signal accumulates immediately.
        """
        lr = 0.0005  # small but active from the start

        if outcome > 0.7:
            if surface > coherence:
                self._w_surface   = min(0.8, self._w_surface   + lr)
                self._w_coherence = max(0.1, self._w_coherence - lr)
            else:
                self._w_coherence = min(0.8, self._w_coherence + lr)
                self._w_surface   = max(0.1, self._w_surface   - lr)

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'eval_count':          self._eval_count,
            'maturity':            round(self.maturity, 4),
            'positive_count':      self._positive_count,
            'negative_count':      self._negative_count,
            'surprise_count':      self._surprise_count,
            'contradiction_count': self._contradiction_count,
            'weights': {
                'surface':   round(self._w_surface,   3),
                'coherence': round(self._w_coherence, 3),
                'novelty':   round(self._w_novelty,   3),
                'surprise':  round(self._w_surprise,  3),
            },
        }


def _surface_similarity(a: str, b: str) -> float:
    if not b:
        return 1.0 if not a else 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()