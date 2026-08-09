"""
tinkering.py — Tinkering Engine: Novel Connection Proposer.

Runs during idle cycles. Takes existing graph edges and proposes
extensions, analogies, and compressions.

Three strategies:

  Extension   — follow existing edge chains outward.
                "water → evaporates → heat" → propose "water → heat" directly.

  Analogy     — find two subgraphs with similar topology in different domains.
                "water cycle" and "economic cycle" have same input→process→output.
                Propose analogy edge between them. Flag for LLM validation.

  Compression — find dense clusters that appear together so frequently
                they should become a primitive at the next level up.

Novel connections proposed here are pre-screened by emotional evaluator
before being sent to LLM validator. Only exciting ones make the cut.
"""

import random
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.graph               import MemoryGraph, Edge
    from hgls.library             import Library
    from hgls.emotional_evaluator import EmotionalEvaluator


# Min emotional score for a proposed connection to reach LLM
PRESCREEN_THRESHOLD = 0.55

# How many extensions to attempt per idle cycle
EXTENSIONS_PER_CYCLE  = 30
ANALOGIES_PER_CYCLE   = 10
COMPRESSIONS_PER_CYCLE = 5


class TinkeringEngine:

    def __init__(
        self,
        graph:     Optional['MemoryGraph']        = None,
        library:   Optional['Library']            = None,
        evaluator: Optional['EmotionalEvaluator'] = None,
    ):
        self.graph     = graph
        self.library   = library
        self.evaluator = evaluator

        self._proposals_generated = 0
        self._proposals_passed    = 0
        self._proposals_rejected  = 0

    # ── Main: one idle cycle ──────────────────────────────────────

    def tinker(self) -> List[Dict]:
        """
        Run one tinkering cycle. Returns list of proposals that
        passed emotional pre-screening, ready for LLM validation.

        Each proposal: {
            'source_id', 'target_id',
            'source_text', 'target_text',
            'strategy', 'emotional_score', 'emotional_state'
        }
        """
        if not self.graph or not self.library:
            return []

        proposals = []
        proposals.extend(self._extend(EXTENSIONS_PER_CYCLE))
        proposals.extend(self._analogise(ANALOGIES_PER_CYCLE))
        proposals.extend(self._compress(COMPRESSIONS_PER_CYCLE))

        # Pre-screen with emotional evaluator
        passed = []
        for prop in proposals:
            self._proposals_generated += 1
            if self._prescreen(prop):
                passed.append(prop)
                self._proposals_passed += 1
            else:
                self._proposals_rejected += 1

        return passed

    # ── Extension ─────────────────────────────────────────────────

    def _extend(self, n: int) -> List[Dict]:
        """
        Pick a random strong edge. Follow both endpoints outward.
        Propose a direct connection between the two outer nodes.

        A → B → C  →  propose A → C
        """
        proposals = []
        strong_edges = self.graph.strongest_edges(top_k=100)
        if len(strong_edges) < 2:
            return []

        for _ in range(n):
            edge = random.choice(strong_edges)
            # Get neighbours of target
            tgt_neighbours = self.graph.get_neighbours(
                edge.target_id, min_strength=0.3
            )
            if not tgt_neighbours:
                continue
            ext_edge = random.choice(tgt_neighbours)
            # The outer node
            outer_id = (
                ext_edge.target_id
                if ext_edge.source_id == edge.target_id
                else ext_edge.source_id
            )
            if outer_id == edge.source_id:
                continue  # loop
            if self.graph.is_rejected_novel(edge.source_id, outer_id):
                continue

            proposals.append({
                'source_id':   edge.source_id,
                'target_id':   outer_id,
                'source_text': self._get_text(edge.source_id),
                'target_text': self._get_text(outer_id),
                'strategy':    'extension',
                'chain':       [edge.source_id, edge.target_id, outer_id],
            })

        return proposals

    # ── Analogy ───────────────────────────────────────────────────

    def _analogise(self, n: int) -> List[Dict]:
        """
        Find two nodes from different regions with structurally similar
        neighbourhood topology. Propose an analogy edge between them.

        Structural similarity: both have similar degree and
        similar level_span distributions in their neighbourhood.
        """
        proposals = []
        all_edges = self.graph.strongest_edges(top_k=200)
        if len(all_edges) < 10:
            return []

        for _ in range(n):
            # Pick two random edges from (hopefully) different regions
            e1, e2 = random.sample(all_edges, 2)

            # Use source nodes of each edge as analogy candidates
            src1 = e1.source_id
            src2 = e2.source_id

            if src1 == src2:
                continue
            if self.graph.is_rejected_novel(src1, src2):
                continue

            # Topology similarity: compare neighbourhood sizes and level spans
            n1 = self.graph.get_neighbours(src1, min_strength=0.2)
            n2 = self.graph.get_neighbours(src2, min_strength=0.2)

            if not n1 or not n2:
                continue

            # Degree similarity
            degree_sim = 1.0 - abs(len(n1) - len(n2)) / max(len(n1) + len(n2), 1)
            if degree_sim < 0.5:
                continue  # too different in connectivity

            # Level span similarity
            spans1 = set(e.level_span for e in n1)
            spans2 = set(e.level_span for e in n2)
            span_sim = len(spans1 & spans2) / max(len(spans1 | spans2), 1)

            # Only propose if topologically similar
            if degree_sim + span_sim < 0.8:
                continue

            proposals.append({
                'source_id':   src1,
                'target_id':   src2,
                'source_text': self._get_text(src1),
                'target_text': self._get_text(src2),
                'strategy':    'analogy',
                'degree_sim':  round(degree_sim, 3),
                'span_sim':    round(span_sim, 3),
            })

        return proposals

    # ── Compression ───────────────────────────────────────────────

    def _compress(self, n: int) -> List[Dict]:
        """
        Find a cluster of nodes densely connected to each other.
        Propose the cluster as a new primitive at the next level.

        The cluster is returned as a compression proposal —
        not an edge but a set of nodes to be packaged.
        """
        proposals = []
        strong_edges = self.graph.strongest_edges(top_k=50)

        for _ in range(n):
            if not strong_edges:
                break
            seed_edge = random.choice(strong_edges)

            # Find cluster around this edge
            cluster = self._find_cluster(seed_edge.source_id, min_size=3)
            if not cluster:
                continue

            # Represent cluster as source→target of the strongest internal edge
            # Package metadata for LLM
            cluster_texts = [self._get_text(nid) for nid in cluster[:5]]

            proposals.append({
                'source_id':     seed_edge.source_id,
                'target_id':     seed_edge.target_id,
                'source_text':   self._get_text(seed_edge.source_id),
                'target_text':   self._get_text(seed_edge.target_id),
                'strategy':      'compression',
                'cluster_ids':   cluster,
                'cluster_texts': cluster_texts,
            })

        return proposals

    def _find_cluster(self, node_id: str, min_size: int = 3) -> List[str]:
        """Find a densely connected neighbourhood around node_id."""
        if not self.graph:
            return []

        region = self.graph.get_region(node_id, depth=1, min_strength=0.4)
        nodes  = list(region.keys())

        # Filter to nodes that connect back to each other
        dense = []
        for nid in nodes:
            connections_to_others = sum(
                1 for other in nodes
                if other != nid and any(
                    (e.source_id == nid and e.target_id == other) or
                    (e.target_id == nid and e.source_id == other)
                    for e in self.graph.get_neighbours(nid, min_strength=0.3)
                )
            )
            if connections_to_others >= min_size - 1:
                dense.append(nid)

        return dense if len(dense) >= min_size else []

    # ── Emotional pre-screening ───────────────────────────────────

    def _prescreen(self, proposal: Dict) -> bool:
        """
        Run emotional evaluator on proposed connection.
        Only pass if signal is exciting enough to warrant LLM call.
        """
        if not self.evaluator:
            return True  # no evaluator yet — pass everything

        src_text = proposal.get('source_text', '')
        tgt_text = proposal.get('target_text', '')

        if not src_text or not tgt_text:
            return False

        result = self.evaluator.evaluate(
            generated = src_text,
            target    = tgt_text,
            source_id = proposal['source_id'],
            target_id = proposal['target_id'],
        )

        proposal['emotional_score'] = result['score']
        proposal['emotional_state'] = result['state']

        # Pass if genuinely interesting — novel_exciting, deeply_surprising, curious
        if result['state'] in ('deeply_surprising', 'novel_exciting', 'curious'):
            return True

        # Also pass if score is high enough regardless of state
        if result['score'] >= PRESCREEN_THRESHOLD:
            return True

        return False

    # ── Helper ────────────────────────────────────────────────────

    def _get_text(self, struct_id: str) -> str:
        if not self.library:
            return struct_id
        struct = self.library.get(struct_id)
        if not struct:
            return struct_id
        return struct.generate(self.library)[:80]

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'proposals_generated': self._proposals_generated,
            'proposals_passed':    self._proposals_passed,
            'proposals_rejected':  self._proposals_rejected,
            'pass_rate': round(
                self._proposals_passed / max(self._proposals_generated, 1), 3
            ),
        }
