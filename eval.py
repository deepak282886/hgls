"""
eval.py — Internal Evaluator.

One job: keep the graph coherent.
Primitive: reward / no reward.

Eval reads the coherence of a path from the graph itself —
edge strengths ARE the accumulated history of past rewards.
A high-coherence path is one that has been right many times before.

Emotional intensity shapes the adjustment:

  high coherence + reward     → strong positive  → large reinforcement
  low  coherence + reward     → discovery        → moderate reinforcement
  high coherence + no reward  → surprise/error   → moderate weakening
  low  coherence + no reward  → strong negative  → large weakening

The intensity is not hardcoded. It emerges from the coherence score
combined with the reward signal. Extreme situations produce extreme
adjustments. Ambiguous situations produce small ones.

Over time eval gets better at scoring paths internally —
because the graph's own structure encodes what has been right before.
No separate model needed. The graph IS the model.
"""

from typing import List
from memory import Graph


class Eval:

    def __init__(self, graph: Graph):
        self.graph = graph

    # ── Primary interface ─────────────────────────────────────────

    def score(self, node_ids: List[str]) -> float:
        """
        Score a path by its coherence.
        Coherence = average edge strength across all edges on this path.

        A path with no edges scores 0 — completely unknown territory.
        A path with strong edges scores high — well learned territory.

        This is eval's internal judgement before external signal arrives.
        It gets better over time as the graph accumulates reward history.
        """
        edges = self.graph.activated_path(node_ids)
        if not edges:
            return 0.0
        # Sum not average — a path with more strong edges is better known
        # than one with fewer edges at the same average strength.
        # More connections = richer, more certain knowledge.
        return sum(e.strength for e in edges)

    def signal(self, node_ids: List[str], reward: bool) -> float:
        """
        External reward has arrived. Adjust the graph.

        Returns the emotional intensity of this interaction —
        how strongly the graph was adjusted.
        Caller can use this to understand the system's state.
        """
        coherence = self.score(node_ids)
        intensity = self._intensity(coherence, reward)

        if reward:
            self._reinforce(node_ids, intensity)
        else:
            self._weaken(node_ids, intensity)

        return intensity

    # ── Emotional intensity ───────────────────────────────────────

    def _intensity(self, coherence: float, reward: bool) -> float:
        """
        Intensity is not hardcoded. It emerges from the situation:

        reward=True:
          high coherence → confirming something already strong
                         → strong positive, large reinforcement
          low coherence  → something new worked
                         → discovery, moderate reinforcement

        reward=False:
          low coherence  → path was already weak, confirmed wrong
                         → strong negative, large weakening
          high coherence → something believed to be right was wrong
                         → surprise, moderate weakening
                         → don't collapse a strong path from one error

        The asymmetry is intentional:
          discoveries are moderate  — don't over-commit to one success
          surprises are moderate    — don't over-punish one failure on strong path
          confirmations and clear errors are the strongest signals
        """
        if reward:
            # high coherence + reward = strong confirmation
            # low coherence  + reward = discovery (moderate)
            return 0.5 + (coherence * 0.5)
        else:
            # low coherence  + no reward = clear error (strong negative)
            # high coherence + no reward = surprise  (moderate negative)
            return 0.5 + ((1.0 - coherence) * 0.5)

    # ── Graph adjustment ──────────────────────────────────────────

    def _reinforce(self, node_ids: List[str], intensity: float) -> None:
        """
        Reward path. Strengthen edges and nodes proportionally.
        Base adjustment scaled by emotional intensity.
        """
        edges = self.graph.activated_path(node_ids)
        for edge in edges:
            amount = intensity / (1.0 + edge.strength)
            edge.reinforce(amount)

        for nid in node_ids:
            node = self.graph.get_node(nid)
            if node:
                amount = intensity / (1.0 + node.strength)
                node.strength += amount

    def _weaken(self, node_ids: List[str], intensity: float) -> None:
        """
        No reward on path. Weaken edges and nodes proportionally.
        Strong edges resist weakening — stable knowledge is hard to shake.
        """
        edges = self.graph.activated_path(node_ids)
        for edge in edges:
            amount = intensity / (1.0 + edge.strength)
            edge.weaken(amount)

        for nid in node_ids:
            node = self.graph.get_node(nid)
            if node:
                amount = intensity / (1.0 + node.strength)
                node.strength = max(0.0, node.strength - amount)

    # ── Cross-modal coherence ─────────────────────────────────────

    def cross_modal_score(
        self,
        text_ids:   List[str],
        voice_ids:  List[str],
        vision_ids: List[str],
    ) -> float:
        """
        Score coherence across modalities.
        When text, voice, and vision paths all activate the same
        higher-level nodes, cross-modal coherence is high.
        This is how concepts form — same abstraction grounded
        in all three modalities simultaneously.

        Returns 0.0 if no cross-modal edges exist yet.
        Grows naturally as the system learns.
        """
        sets = []
        for ids in [text_ids, voice_ids, vision_ids]:
            if ids:
                sets.append(self._reachable(ids))

        if len(sets) < 2:
            return 0.0

        shared = sets[0]
        for s in sets[1:]:
            shared = shared & s

        total = sets[0]
        for s in sets[1:]:
            total = total | s

        if not total:
            return 0.0

        return len(shared) / len(total)

    def _reachable(self, node_ids: List[str], depth: int = 3) -> set:
        """
        All nodes reachable FROM the starting nodes within depth.
        Does NOT include the starting atoms themselves —
        cross-modal coherence measures convergence at higher levels,
        not shared atom names. Only follows edges with positive strength.
        """
        start    = set(node_ids)
        visited  = set(node_ids)
        frontier = set(node_ids)
        reached  = set()           # excludes start atoms

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                for neighbour, edge in self.graph.neighbours(nid):
                    if edge.strength > 0 and neighbour.id not in visited:
                        next_frontier.add(neighbour.id)
                        visited.add(neighbour.id)
                        if neighbour.id not in start:
                            reached.add(neighbour.id)
            frontier = next_frontier
            if not frontier:
                break

        return reached