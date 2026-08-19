"""
Absorption Pipeline v2
Lean absorber — words + sentences only (phrases emerge from merges).
Merge check runs every N absorbs, not on every call.
O(1) node lookup via sequence index in GeometricSpace.
"""

import re
import numpy as np
from graph_engine import Node, GeometricSpace, RewardGraph


# ─────────────────────────────────────────────
# SEGMENTER
# ─────────────────────────────────────────────

class Segmenter:
    """
    Segments text at each level.
    Level 1 — words
    Level 3 — sentences
    Phrases (level 2) emerge from word-level merges naturally.
    Characters (level 0) available but optional — expensive at scale.
    """

    def segment(self, text: str, include_chars: bool = False) -> dict[int, list[str]]:
        result = {
            1: self._words(text),
            3: self._sentences(text),
        }
        if include_chars:
            result[0] = self._characters(text)
        return result

    def _characters(self, text: str) -> list[str]:
        return list(text.lower())

    def _words(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip().lower() for s in sentences if s.strip()]


# ─────────────────────────────────────────────
# MERGE ENGINE
# ─────────────────────────────────────────────

class MergeEngine:
    """
    When two nodes at the same level become geometrically close,
    merge them into a new higher-level parent node.
    Original nodes preserved. Parent links back to children.
    """

    def __init__(self, space: GeometricSpace, dim: int = 64):
        self.space = space
        self.dim = dim
        self.merge_log: list[dict] = []

    def merge(self, node_a: Node, node_b: Node, level: int) -> Node | None:
        # skip if either already has a parent
        if node_a.parent is not None or node_b.parent is not None:
            return None

        parent_position = (node_a.position + node_b.position) / 2
        parent_sequence = f"[{node_a.sequence}|{node_b.sequence}]"

        # reuse existing parent if already formed
        existing = self.space.get_node_by_sequence(parent_sequence, level + 1)
        if existing:
            if node_a not in existing.children:
                existing.children.append(node_a)
                node_a.parent = existing
            if node_b not in existing.children:
                existing.children.append(node_b)
                node_b.parent = existing
            return existing

        parent = Node(
            sequence=parent_sequence,
            level=level + 1,
            position=parent_position,
            dim=self.dim
        )
        parent.children = [node_a, node_b]
        node_a.parent = parent
        node_b.parent = parent
        self.space.add_node(parent)

        self.merge_log.append({
            "parent": parent_sequence,
            "children": [node_a.sequence, node_b.sequence],
            "level": level + 1
        })
        return parent

    def get_merge_log(self) -> list[dict]:
        return self.merge_log


# ─────────────────────────────────────────────
# ABSORBER
# ─────────────────────────────────────────────

class Absorber:
    """
    Absorbs text into the graph at all active levels simultaneously.
    - Updates co-occurrence counts between adjacent nodes
    - Shifts positions in geometric space (pull together)
    - Runs periodic merge checks (not every absorb — keeps it fast)
    """

    def __init__(
        self,
        space: GeometricSpace,
        reward_graph: RewardGraph,
        merge_engine: MergeEngine | None = None,
        dim: int = 64,
        co_occurrence_window: int = 2,
        merge_every: int = 20,      # run merge check every N absorbs
        max_merges_per_check: int = 5
    ):
        self.space = space
        self.reward_graph = reward_graph
        self.merge_engine = merge_engine
        self.dim = dim
        self.window = co_occurrence_window
        self.merge_every = merge_every
        self.max_merges_per_check = max_merges_per_check
        self.segmenter = Segmenter()
        self._absorb_count = 0

    # ── Main Entry Point ─────────────────────

    def absorb(self, text: str, include_chars: bool = False) -> dict:
        """Absorb text into the graph. Returns stats."""
        segments = self.segmenter.segment(text, include_chars=include_chars)
        stats = {"created": 0, "updated": 0, "merges": 0}

        for level, seqs in segments.items():
            if not seqs:
                continue
            nodes = [self._get_or_create(seq, level, stats) for seq in seqs]
            self._update_co_occurrence(nodes)

        self._absorb_count += 1

        # periodic merge check
        if self.merge_engine and self._absorb_count % self.merge_every == 0:
            stats["merges"] += self._run_merge_check()

        return stats

    # ── Node Management ──────────────────────

    def _get_or_create(self, sequence: str, level: int, stats: dict) -> Node:
        """O(1) lookup via sequence index. Create if missing."""
        node = self.space.get_node_by_sequence(sequence, level)
        if node:
            node.visit_count += 1
            stats["updated"] += 1
            return node

        node = Node(
            sequence=sequence,
            level=level,
            position=np.random.randn(self.dim) * 2.0,
            dim=self.dim
        )
        node.visit_count = 1
        self.space.add_node(node)
        stats["created"] += 1
        return node

    # ── Co-occurrence Updates ─────────────────

    def _update_co_occurrence(self, nodes: list[Node]):
        """Pull co-occurring nodes together within window."""
        for i, node in enumerate(nodes):
            start = max(0, i - self.window)
            end = min(len(nodes), i + self.window + 1)
            for neighbor in nodes[start:end]:
                if neighbor.id == node.id:
                    continue
                node.update_co_occurrence(neighbor.id)
                neighbor.update_co_occurrence(node.id)
                self.space.update_positions_from_co_occurrence(node, neighbor)

    # ── Merge Check ───────────────────────────

    def _run_merge_check(self) -> int:
        """Run merge candidates check on word level. Returns merge count."""
        if not self.merge_engine:
            return 0
        candidates = self.space.get_merge_candidates(level=1)
        count = 0
        for node_a, node_b, _ in candidates[:self.max_merges_per_check]:
            result = self.merge_engine.merge(node_a, node_b, level=1)
            if result:
                count += 1
        return count

    # ── Cross-Level Links ─────────────────────

    def absorb_cross_level(self, text: str) -> dict:
        """
        Create cross-level links between sentence nodes (level 3)
        and their constituent word nodes (level 1).

        This is what makes hierarchical rollout meaningful:
        - Sentence nodes gain reward_graph edges to their words
        - Word nodes gain edges back to their parent sentence
        - Positions are pulled together so geometry reflects membership

        Call this after absorb() with the same text so nodes already exist.
        """
        segments = self.segmenter.segment(text)
        stats = {"cross_links": 0}

        for sent_seq in segments.get(3, []):
            sent_node = self.space.get_node_by_sequence(sent_seq, level=3)
            if not sent_node:
                continue

            # re-segment just this sentence to get its own words
            word_seqs = self.segmenter._words(sent_seq)

            for ws in word_seqs:
                word_node = self.space.get_node_by_sequence(ws, level=1)
                if not word_node:
                    continue

                # bidirectional reward edges — rollout can cross levels both ways
                self.reward_graph.add_or_strengthen_edge(
                    sent_node.id, word_node.id, reward=0.1
                )
                self.reward_graph.add_or_strengthen_edge(
                    word_node.id, sent_node.id, reward=0.1
                )

                # geometric pull — sentence node moves toward its words and vice versa
                self.space.pull_together(sent_node, word_node, strength=0.5)

                # co-occurrence so positions keep updating on repeated text
                sent_node.update_co_occurrence(word_node.id)
                word_node.update_co_occurrence(sent_node.id)

                stats["cross_links"] += 1

        return stats

    def stats(self) -> dict:
        return {
            "absorb_count": self._absorb_count,
            "graph": self.space.stats()
        }


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Absorption v2 Test ===\n")

    space = GeometricSpace(dim=24, merge_threshold=1.5, pull_rate=0.1)
    rg = RewardGraph()
    me = MergeEngine(space=space, dim=24)
    ab = Absorber(space=space, reward_graph=rg, merge_engine=me,
                  dim=24, merge_every=3)

    corpus = [
        "the cat sat on the mat.",
        "the cat ate the rat.",
        "the dog sat on the log.",
        "cats and dogs are common pets.",
        "the cat chased the mouse around the house.",
        "a dog ran across the field quickly.",
        "the quick brown fox jumps over the lazy dog.",
        "dogs are loyal and friendly animals.",
        "cats are independent and curious creatures.",
        "the old cat and the young dog became friends.",
    ]

    import time
    t0 = time.time()
    for text in corpus:
        ab.absorb(text)
    print(f"Absorbed {len(corpus)} sentences in {time.time()-t0:.3f}s")
    print(space)

    # proximity check
    cat = space.get_node_by_sequence("cat", 1)
    dog = space.get_node_by_sequence("dog", 1)
    fox = space.get_node_by_sequence("fox", 1)
    the = space.get_node_by_sequence("the", 1)

    if cat and dog and fox and the:
        print(f"\nProximity:")
        print(f"  cat <-> dog: {space.distance(cat, dog):.4f}")
        print(f"  cat <-> fox: {space.distance(cat, fox):.4f}")
        print(f"  cat <-> the: {space.distance(cat, the):.4f}")

    print(f"\nMerges: {me.get_merge_log()[:5]}")
    print(f"\nStats: {ab.stats()}")
    print("\n=== OK ===")