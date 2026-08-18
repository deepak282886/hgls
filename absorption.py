"""
Absorption Pipeline
Takes raw text. Segments at all levels simultaneously.
Updates co-occurrence at every level. Shifts positions.
Triggers merge checks. All levels absorb the same sequence at once.
"""

import re
import numpy as np
from graph_engine import Node, GeometricSpace, RewardGraph


# ─────────────────────────────────────────────
# SEGMENTER
# Breaks a sequence into units at each level
# ─────────────────────────────────────────────

class Segmenter:
    """
    Given raw text, produces segments at each level.
    Level 0 — characters
    Level 1 — words
    Level 2 — phrases (noun/verb chunks via simple windowing for now)
    Level 3 — sentences
    Level 4 — paragraphs
    Higher levels — emerge from merge engine, not segmented directly
    """

    def segment(self, text: str) -> dict[int, list[str]]:
        return {
            0: self._characters(text),
            1: self._words(text),
            2: self._phrases(text),
            3: self._sentences(text),
            4: self._paragraphs(text),
        }

    def _characters(self, text: str) -> list[str]:
        return list(text.lower())

    def _words(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _phrases(self, text: str, window: int = 3) -> list[str]:
        """
        Sliding window over words to capture phrasal co-occurrence.
        Window of 3 gives bigrams and trigrams simultaneously.
        """
        words = self._words(text)
        phrases = []
        for size in range(2, window + 1):
            for i in range(len(words) - size + 1):
                phrases.append(" ".join(words[i:i+size]))
        return phrases

    def _sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip().lower() for s in sentences if s.strip()]

    def _paragraphs(self, text: str) -> list[str]:
        paras = re.split(r'\n\s*\n', text.strip())
        return [p.strip().lower() for p in paras if p.strip()]


# ─────────────────────────────────────────────
# ABSORBER
# Core pipeline — ingests text, updates graph
# ─────────────────────────────────────────────

class Absorber:
    """
    Absorbs a raw text sequence into the graph.
    At every level simultaneously:
      - Finds or creates nodes for each segment
      - Updates co-occurrence counts between adjacent segments
      - Shifts positions in geometric space
      - Queues merge candidates
    """

    def __init__(
        self,
        space: GeometricSpace,
        reward_graph: RewardGraph,
        merge_engine=None,       # injected later to avoid circular import
        dim: int = 64,
        co_occurrence_window: int = 2   # how many neighbors count as co-occurring
    ):
        self.space = space
        self.reward_graph = reward_graph
        self.merge_engine = merge_engine
        self.dim = dim
        self.window = co_occurrence_window
        self.segmenter = Segmenter()

    # ── Main Entry Point ─────────────────────

    def absorb(self, text: str) -> dict:
        """
        Absorb a piece of text into the graph at all levels.
        Returns stats about what was created/updated.
        """
        segments = self.segmenter.segment(text)
        stats = {"created": 0, "updated": 0, "merge_candidates": 0}

        for level, seqs in segments.items():
            if not seqs:
                continue

            # get or create nodes for all segments at this level
            nodes = [self._get_or_create_node(seq, level, stats) for seq in seqs]

            # update co-occurrence within window
            self._update_co_occurrence(nodes, level)

            # check for merge candidates at this level
            candidates = self.space.get_merge_candidates(level)
            stats["merge_candidates"] += len(candidates)

            # trigger merge engine if available
            if self.merge_engine and candidates:
                for node_a, node_b, dist in candidates:
                    self.merge_engine.merge(node_a, node_b, level)

        return stats

    # ── Node Management ──────────────────────

    def _get_or_create_node(self, sequence: str, level: int, stats: dict) -> Node:
        """
        Return existing node for this sequence or create a new one.
        New nodes get a position near their neighbors if possible.
        """
        existing = self.space.get_node_by_sequence(sequence)
        if existing and existing.level == level:
            existing.visit_count += 1
            stats["updated"] += 1
            return existing

        # create new node
        # position initialized near centroid of level if nodes exist, else random
        level_nodes = self.space.get_nodes_at_level(level)
        if level_nodes:
            centroid = np.mean([n.position for n in level_nodes], axis=0)
            noise = np.random.randn(self.dim) * 0.5
            position = centroid + noise
        else:
            position = np.random.randn(self.dim)

        node = Node(sequence=sequence, level=level, position=position, dim=self.dim)
        node.visit_count = 1
        self.space.add_node(node)
        stats["created"] += 1
        return node

    # ── Co-occurrence Updates ─────────────────

    def _update_co_occurrence(self, nodes: list[Node], level: int):
        """
        For each node in the sequence, update co-occurrence with
        nodes within the window. Then shift positions.
        """
        for i, node in enumerate(nodes):
            # window around current node
            start = max(0, i - self.window)
            end = min(len(nodes), i + self.window + 1)
            neighbors = nodes[start:end]

            for neighbor in neighbors:
                if neighbor.id == node.id:
                    continue

                # update counts
                node.update_co_occurrence(neighbor.id)
                neighbor.update_co_occurrence(node.id)

                # shift positions — pull together
                self.space.update_positions_from_co_occurrence(node, neighbor)

    # ── Cross Level Co-occurrence ─────────────

    def absorb_cross_level(self, text: str):
        """
        Optional — pull nodes closer across levels when they share content.
        E.g. word node 'the' should be near its character nodes 't','h','e'.
        Runs after standard absorption.
        """
        segments = self.segmenter.segment(text)

        # level 0 chars <-> level 1 words
        char_nodes = [self.space.get_node_by_sequence(c) for c in segments[0] if self.space.get_node_by_sequence(c)]
        word_nodes = [self.space.get_node_by_sequence(w) for w in segments[1] if self.space.get_node_by_sequence(w)]

        for w_node in word_nodes:
            if not w_node:
                continue
            for c in w_node.sequence:
                c_node = self.space.get_node_by_sequence(c)
                if c_node:
                    # gentle pull — cross level signal is weaker
                    self.space.pull_together(w_node, c_node, strength=0.3)


# ─────────────────────────────────────────────
# MERGE ENGINE
# Creates higher level nodes from close pairs
# ─────────────────────────────────────────────

class MergeEngine:
    """
    When two nodes at the same level get close enough,
    merge them into a new higher level node.
    Original nodes preserved. New node is their parent.
    """

    def __init__(self, space: GeometricSpace, dim: int = 64):
        self.space = space
        self.dim = dim
        self.merge_log: list[dict] = []

    def merge(self, node_a: Node, node_b: Node, level: int) -> Node:
        """
        Merge two nodes into a new parent node one level up.
        Parent position is midpoint of children.
        Parent sequence is concatenation (for now).
        """
        # avoid double merging
        if node_a.parent is not None or node_b.parent is not None:
            return None

        # parent position — midpoint
        parent_position = (node_a.position + node_b.position) / 2

        # parent sequence — combined representation
        parent_sequence = f"[{node_a.sequence}|{node_b.sequence}]"

        # check if this abstraction already exists
        existing = self.space.get_node_by_sequence(parent_sequence)
        if existing:
            # just link if not already linked
            if node_a not in existing.children:
                existing.children.append(node_a)
                node_a.parent = existing
            if node_b not in existing.children:
                existing.children.append(node_b)
                node_b.parent = existing
            return existing

        # create new parent node
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
            "parent": parent.sequence,
            "children": [node_a.sequence, node_b.sequence],
            "level": level + 1
        })

        return parent

    def get_merge_log(self) -> list[dict]:
        return self.merge_log


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Absorption Pipeline Test ===\n")

    # init
    space = GeometricSpace(dim=32, merge_threshold=2.0, pull_rate=0.1)
    reward_graph = RewardGraph()
    merge_engine = MergeEngine(space=space, dim=32)
    absorber = Absorber(space=space, reward_graph=reward_graph, merge_engine=merge_engine, dim=32)

    # absorb some text
    texts = [
        "the cat sat on the mat.",
        "the cat ate the rat.",
        "the dog sat on the log.",
        "a cat and a dog sat together.",
        "the quick brown fox jumps over the lazy dog.",
    ]

    print("Absorbing text sequences...\n")
    total_stats = {"created": 0, "updated": 0, "merge_candidates": 0}

    for text in texts:
        stats = absorber.absorb(text)
        absorber.absorb_cross_level(text)
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"  '{text[:40]}...' -> {stats}")

    print(f"\nTotal: {total_stats}")
    print(f"\n{space}")

    # check proximity — 'cat' and 'dog' should be closer than 'cat' and 'fox'
    cat = space.get_node_by_sequence("cat")
    dog = space.get_node_by_sequence("dog")
    fox = space.get_node_by_sequence("fox")
    the = space.get_node_by_sequence("the")

    if cat and dog and fox:
        print(f"\nProximity check:")
        print(f"  cat <-> dog : {space.distance(cat, dog):.4f}")
        print(f"  cat <-> fox : {space.distance(cat, fox):.4f}")
        print(f"  cat <-> the : {space.distance(cat, the):.4f}")

    # show merge log
    print(f"\nMerges formed ({len(merge_engine.get_merge_log())}):")
    for m in merge_engine.get_merge_log()[:5]:
        print(f"  Level {m['level']}: {m['parent']}")

    # show neighbors of 'the'
    if the:
        neighbors = space.nearest_neighbors(the, k=5)
        print(f"\nNearest to 'the': {[(n.sequence, round(d,3)) for n,d in neighbors]}")

    print("\n=== Absorption OK ===")