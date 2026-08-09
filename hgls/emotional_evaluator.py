"""
emotional_evaluator.py — Graph-Aware Emotional Evaluator.

Starts from two primitive states: positive and negative.
Grows finer gradient through experience with the graph.

Replaces raw SequenceMatcher as primary evaluation signal.
SequenceMatcher becomes one input among several.

Emotional states that emerge over time:
  familiar_confirming  — dense region, expected connection
  novel_exciting       — sparse region, structurally coherent
  novel_incoherent     — sparse region, no structural support
  almost_right         — high similarity, one element wrong
  contradicting        — conflicts with existing strong edge
  deeply_surprising    — links two previously disconnected dense regions
  curious              — moderate novelty, worth exploring

These are not programmed. They emerge from the gradient growing
through millions of evaluation cycles.
"""

import math
from difflib import SequenceMatcher
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.graph    import MemoryGraph
    from hgls.library  import Library
    from hgls.structures import GenerativeStructure


# Primitive signal weights — start equal, grow through experience
_W_SURFACE    = 0.4   # SequenceMatcher character similarity
_W_COHERENCE  = 0.3   # graph coherence of proposed connection
_W_NOVELTY    = 0.15  # how far from known dense regions
_W_SURPRISE   = 0.15  # discovery bonus for linking distant dense regions

# Emotional state thresholds — these sharpen over time
_FAMILIAR_THRESHOLD   = 0.85
_CURIOUS_THRESHOLD    = 0.50
_INCOHERENT_THRESHOLD = 0.25


class EmotionalEvaluator:
    """
    Evaluates proposed connections using both surface similarity
    and graph topology. Grows finer gradient through experience.
    """

    def __init__(
        self,
        graph:   Optional['MemoryGraph'] = None,
        library: Optional['Library']     = None,
    ):
        self.graph   = graph
        self.library = library

        # Experience accumulation — shapes the gradient over time
        self._eval_count         = 0
        self._positive_count     = 0
        self._negative_count     = 0
        self._surprise_count     = 0
        self._contradiction_count = 0

        # Maturity: 0 = primitive (just pos/neg), 1 = fully differentiated
        self.maturity = 0.0

        # Learned weights (start equal, drift through experience)
        self._w_surface   = _W_SURFACE
        self._w_coherence = _W_COHERENCE
        self._w_novelty   = _W_NOVELTY
        self._w_surprise  = _W_SURPRISE

    # ── Primary evaluation ────────────────────────────────────────

    def evaluate(
        self,
        generated: str,
        target: str,
        source_id: str     = None,
        target_id: str     = None,
    ) -> dict:
        """
        Evaluate a generated string against a target.
        Returns a rich signal dict with score, state, and components.

        At low maturity: mostly surface similarity (familiar + novel detection).
        At high maturity: full graph-aware gradient.
        """
        self._eval_count += 1
        self._update_maturity()

        # Always compute surface similarity
        surface = _surface_similarity(generated, target)

        # Graph signals — only meaningful once graph exists
        coherence  = 0.5
        novelty    = 0.5
        surprise   = 0.0
        contradiction = False

        if self.graph and source_id and target_id and self.maturity > 0.1:
            coherence     = self._graph_coherence(source_id, target_id)
            novelty       = self._novelty_signal(source_id, target_id)
            surprise      = self._surprise_signal(source_id, target_id)
            contradiction = self._detects_contradiction(source_id, target_id)

        # Blend signals — weights drift with maturity
        w_s = self._w_surface
        w_c = self._w_coherence   * self.maturity
        w_n = self._w_novelty     * self.maturity
        w_r = self._w_surprise    * self.maturity

        # Renormalise
        total_w = w_s + w_c + w_n + w_r
        if total_w > 0:
            w_s /= total_w; w_c /= total_w
            w_n /= total_w; w_r /= total_w

        score = (
            w_s * surface   +
            w_c * coherence +
            w_n * novelty   +
            w_r * surprise
        )

        # Contradiction penalty
        if contradiction:
            score *= 0.5
            self._contradiction_count += 1

        # Determine emotional state
        state = self._emotional_state(
            score, surface, coherence, novelty, surprise, contradiction
        )

        # Track positive/negative
        if score >= 0.5:
            self._positive_count += 1
        else:
            self._negative_count += 1

        if state == 'deeply_surprising':
            self._surprise_count += 1

        # Adapt weights based on what's working
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
        """Fast path — just the score, no state. For hypothesis pre-screening."""
        surface = _surface_similarity(generated, target)
        if not self.graph or self.maturity < 0.1:
            return surface
        result = self.evaluate(generated, target)
        return result['score']

    # ── Graph signal components ───────────────────────────────────

    def _graph_coherence(self, source_id: str, target_id: str) -> float:
        """
        How well does a connection between source and target
        fit existing graph topology?

        High if both nodes are in the same dense region.
        Low if they are in completely disconnected regions.
        """
        if not self.graph:
            return 0.5

        src_neighbours = {
            (e.source_id if e.target_id == source_id else e.target_id)
            for e in self.graph.get_neighbours(source_id, min_strength=0.2)
        }
        tgt_neighbours = {
            (e.source_id if e.target_id == target_id else e.target_id)
            for e in self.graph.get_neighbours(target_id, min_strength=0.2)
        }

        if not src_neighbours or not tgt_neighbours:
            return 0.3  # sparse — low coherence but not zero

        # Shared neighbours = topological coherence
        shared = len(src_neighbours & tgt_neighbours)
        total  = len(src_neighbours | tgt_neighbours)
        return shared / max(total, 1)

    def _novelty_signal(self, source_id: str, target_id: str) -> float:
        """
        How novel is this connection?
        Novel connections in coherent regions get positive signal.
        Novel connections in incoherent regions get low signal.
        """
        if not self.graph:
            return 0.5

        src_density = self.graph.density(source_id)
        tgt_density = self.graph.density(target_id)

        # Direct connection already exists → not novel
        neighbours = {
            (e.source_id if e.target_id == source_id else e.target_id)
            for e in self.graph.get_neighbours(source_id)
        }
        if target_id in neighbours:
            return 0.2  # already known, low novelty signal

        # Both in dense regions but not directly connected → exciting novelty
        if src_density > 0.5 and tgt_density > 0.5:
            return 0.9

        # One sparse, one dense → moderate novelty
        if src_density > 0.3 or tgt_density > 0.3:
            return 0.6

        # Both sparse → low novelty (unknown territory)
        return 0.3

    def _surprise_signal(self, source_id: str, target_id: str) -> float:
        """
        Surprise: connection links two previously disconnected dense regions.
        This is the strongest positive signal — genuine discovery.
        """
        if not self.graph:
            return 0.0

        src_density = self.graph.density(source_id)
        tgt_density = self.graph.density(target_id)

        # Both need to be dense for this to be surprising
        if src_density < 0.4 or tgt_density < 0.4:
            return 0.0

        # Check if they are in disconnected regions (no path between them at depth 2)
        src_region = set(self.graph.get_region(source_id, depth=2).keys())
        if target_id in src_region:
            return 0.0  # already reachable — not surprising

        # Dense and disconnected — this is a discovery
        return min(1.0, src_density * tgt_density * 2.0)

    def _detects_contradiction(self, source_id: str, target_id: str) -> bool:
        """
        Does this connection conflict with an existing strong edge?
        Currently: checks if source and target are in each other's
        failure neighbourhood (both rewarded in contexts where the other was penalised).
        Placeholder — grows more sophisticated with experience.
        """
        if not self.graph or not self.library:
            return False

        src = self.library.get(source_id)
        tgt = self.library.get(target_id)
        if not src or not tgt:
            return False

        # Simple heuristic: if both have high penalty_count in similar topic contexts
        src_penalty_ratio = src.penalty_count / max(src.reward_count + src.penalty_count, 1)
        tgt_penalty_ratio = tgt.penalty_count / max(tgt.reward_count + tgt.penalty_count, 1)

        # Both frequently penalised — connecting them likely wrong
        return src_penalty_ratio > 0.7 and tgt_penalty_ratio > 0.7

    # ── Emotional state classification ────────────────────────────

    def _emotional_state(
        self,
        score:         float,
        surface:       float,
        coherence:     float,
        novelty:       float,
        surprise:      float,
        contradiction: bool,
    ) -> str:
        """
        Map signal components to emotional state.
        At low maturity: just 'positive' or 'negative'.
        At high maturity: full differentiated vocabulary.
        """
        if self.maturity < 0.2:
            return 'positive' if score >= 0.5 else 'negative'

        if contradiction:
            return 'contradicting'

        if surprise > 0.6 and score > 0.5:
            return 'deeply_surprising'

        if score >= _FAMILIAR_THRESHOLD:
            return 'familiar_confirming'

        if surface > 0.85 and score < 0.7:
            return 'almost_right'

        if novelty > 0.7 and coherence > 0.5:
            return 'novel_exciting'

        if novelty > 0.7 and coherence < 0.3:
            return 'novel_incoherent'

        if score >= _CURIOUS_THRESHOLD:
            return 'curious'

        if score < _INCOHERENT_THRESHOLD:
            return 'negative'

        return 'uncertain'

    # ── Weight adaptation ─────────────────────────────────────────

    def _adapt_weights(
        self,
        surface:   float,
        coherence: float,
        outcome:   float,
    ) -> None:
        """
        Slowly shift weights toward signals that correlate with good outcomes.
        This is how the evaluator learns what to pay attention to.
        """
        if self._eval_count < 1000:
            return  # too early to adapt

        lr = 0.0001  # very slow adaptation

        if outcome > 0.7:
            # Good outcome — reinforce whichever signal was highest
            if surface > coherence:
                self._w_surface   = min(0.8, self._w_surface   + lr)
                self._w_coherence = max(0.1, self._w_coherence - lr)
            else:
                self._w_coherence = min(0.8, self._w_coherence + lr)
                self._w_surface   = max(0.1, self._w_surface   - lr)

    # ── Maturity ──────────────────────────────────────────────────

    def _update_maturity(self) -> None:
        """Maturity grows with experience. Sigmoid curve."""
        self.maturity = 1.0 / (1.0 + math.exp(-0.00005 * (self._eval_count - 50000)))

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'eval_count':         self._eval_count,
            'maturity':           round(self.maturity, 4),
            'positive_count':     self._positive_count,
            'negative_count':     self._negative_count,
            'surprise_count':     self._surprise_count,
            'contradiction_count': self._contradiction_count,
            'weights': {
                'surface':   round(self._w_surface,   3),
                'coherence': round(self._w_coherence, 3),
                'novelty':   round(self._w_novelty,   3),
                'surprise':  round(self._w_surprise,  3),
            },
        }


# ── Module helper ─────────────────────────────────────────────────

def _surface_similarity(a: str, b: str) -> float:
    if not b:
        return 1.0 if not a else 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()
