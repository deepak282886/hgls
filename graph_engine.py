"""
Graph Engine — Core Foundation
Nodes float in geometric space. Co-occurrence moves them closer.
Reward draws edges between them.
"""

import numpy as np
import networkx as nx
from typing import Optional
import uuid


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

class Node:
    """
    Every experienced sequence is a node.
    Nodes have a position in geometric space.
    Nodes have a level in the hierarchy.
    Nodes have reward weight and visit count.
    Nodes can have children (if they are abstractions).
    """

    def __init__(
        self,
        sequence: str,
        level: int = 0,
        position: Optional[np.ndarray] = None,
        dim: int = 64
    ):
        self.id = str(uuid.uuid4())
        self.sequence = sequence           # the actual content this node represents
        self.level = level                 # 0 = letter, 1 = word, 2 = phrase, etc.

        # position in geometric space — random init if not given
        self.position = position if position is not None else np.random.randn(dim)

        # hierarchy
        self.parent: Optional['Node'] = None
        self.children: list['Node'] = []

        # learning state
        self.visit_count: int = 0
        self.reward_weight: float = 0.0
        self.co_occurrence_counts: dict[str, int] = {}  # node_id -> count

    def update_co_occurrence(self, other_id: str):
        """Record that this node co-occurred with another node."""
        self.co_occurrence_counts[other_id] = self.co_occurrence_counts.get(other_id, 0) + 1

    def is_abstraction(self) -> bool:
        """A node is an abstraction if it has children."""
        return len(self.children) > 0

    def __repr__(self):
        return f"Node(seq={repr(self.sequence)}, level={self.level}, visits={self.visit_count})"


# ─────────────────────────────────────────────
# GEOMETRIC SPACE
# ─────────────────────────────────────────────

class GeometricSpace:
    """
    Holds all nodes and manages their positions.
    Co-occurrence pulls nodes closer.
    Non co-occurrence lets nodes drift apart.
    Proximity threshold triggers merge candidates.
    """

    def __init__(
        self,
        dim: int = 64,
        pull_rate: float = 0.05,
        push_rate: float = 0.01,
        merge_threshold: float = 0.15
    ):
        self.dim = dim
        self.pull_rate = pull_rate           # how fast co-occurring nodes move closer
        self.push_rate = push_rate           # how fast non co-occurring nodes drift apart
        self.merge_threshold = merge_threshold  # distance below which merge is triggered

        self.nodes: dict[str, Node] = {}     # node_id -> Node

    # ── Node Management ──────────────────────

    def add_node(self, node: Node):
        """Add a node to the space."""
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_node_by_sequence(self, sequence: str) -> Optional[Node]:
        """Find a node by its sequence content."""
        for node in self.nodes.values():
            if node.sequence == sequence:
                return node
        return None

    def get_nodes_at_level(self, level: int) -> list[Node]:
        return [n for n in self.nodes.values() if n.level == level]

    # ── Distance and Proximity ───────────────

    def distance(self, node_a: Node, node_b: Node) -> float:
        """Euclidean distance between two nodes in geometric space."""
        return float(np.linalg.norm(node_a.position - node_b.position))

    def nearest_neighbors(self, node: Node, k: int = 10, same_level: bool = True) -> list[tuple[Node, float]]:
        """Return k nearest nodes with their distances."""
        candidates = [
            n for n in self.nodes.values()
            if n.id != node.id and (not same_level or n.level == node.level)
        ]
        distances = [(n, self.distance(node, n)) for n in candidates]
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ── Position Updates ─────────────────────

    def pull_together(self, node_a: Node, node_b: Node, strength: float = 1.0):
        """
        Move two nodes closer together.
        Strength can be scaled by co-occurrence count.
        """
        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            return  # already at same point
        unit = direction / distance
        move = unit * self.pull_rate * strength
        node_a.position += move
        node_b.position -= move

    def push_apart(self, node_a: Node, node_b: Node):
        """
        Gently drift two nodes apart when they don't co-occur.
        Much weaker than pull to avoid destroying existing structure.
        """
        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            # same position — add small random perturbation
            direction = np.random.randn(self.dim)
            distance = np.linalg.norm(direction)
        unit = direction / distance
        move = unit * self.push_rate
        node_a.position -= move
        node_b.position += move

    def update_positions_from_co_occurrence(self, node_a: Node, node_b: Node):
        """
        Called during absorption.
        Pull co-occurring nodes together scaled by their co-occurrence count.
        """
        count = node_a.co_occurrence_counts.get(node_b.id, 0)
        strength = np.log1p(count)  # log scale — diminishing returns on repetition
        self.pull_together(node_a, node_b, strength=strength)

    # ── Merge Detection ──────────────────────

    def get_merge_candidates(self, level: int) -> list[tuple[Node, Node, float]]:
        """
        Find pairs of nodes at the same level that are close enough to merge.
        Returns list of (node_a, node_b, distance) sorted by distance.
        """
        candidates = []
        level_nodes = self.get_nodes_at_level(level)

        for i, node_a in enumerate(level_nodes):
            for node_b in level_nodes[i+1:]:
                dist = self.distance(node_a, node_b)
                if dist < self.merge_threshold:
                    candidates.append((node_a, node_b, dist))

        candidates.sort(key=lambda x: x[2])
        return candidates

    # ── Diagnostics ──────────────────────────

    def stats(self) -> dict:
        """Quick summary of the current state of the space."""
        levels = {}
        for node in self.nodes.values():
            levels[node.level] = levels.get(node.level, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "nodes_per_level": dict(sorted(levels.items())),
        }

    def __repr__(self):
        s = self.stats()
        return f"GeometricSpace(total={s['total_nodes']}, levels={s['nodes_per_level']})"


# ─────────────────────────────────────────────
# REWARD GRAPH
# ─────────────────────────────────────────────

class RewardGraph:
    """
    Holds directed edges between nodes.
    Edges are drawn by reward — only successful paths get edges.
    Edge weights reflect cumulative reward strength.
    """

    def __init__(self, decay: float = 0.95):
        self.graph = nx.DiGraph()
        self.decay = decay  # reward decay on normalization/failure

    # ── Edge Management ──────────────────────

    def add_or_strengthen_edge(self, from_id: str, to_id: str, reward: float):
        """Draw or strengthen an edge based on reward signal."""
        if self.graph.has_edge(from_id, to_id):
            current = self.graph[from_id][to_id]['weight']
            # running average — new reward blended in
            self.graph[from_id][to_id]['weight'] = current * 0.8 + reward * 0.2
            self.graph[from_id][to_id]['visits'] += 1
        else:
            self.graph.add_edge(from_id, to_id, weight=reward, visits=1)

    def normalize_path(self, path: list[str]):
        """
        On failure — reduce weight of all edges along this path.
        Edges don't disappear — they just become less preferred.
        """
        for i in range(len(path) - 1):
            from_id, to_id = path[i], path[i+1]
            if self.graph.has_edge(from_id, to_id):
                self.graph[from_id][to_id]['weight'] *= self.decay

    def strengthen_path(self, path: list[str], reward: float):
        """On success — strengthen all edges along this path."""
        for i in range(len(path) - 1):
            self.add_or_strengthen_edge(path[i], path[i+1], reward)

    def get_edge_weight(self, from_id: str, to_id: str) -> float:
        """Get reward weight of an edge. Returns 0 if edge doesn't exist."""
        if self.graph.has_edge(from_id, to_id):
            return self.graph[from_id][to_id]['weight']
        return 0.0

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        """Return outgoing neighbors with their edge weights."""
        neighbors = []
        for _, to_id, data in self.graph.out_edges(node_id, data=True):
            neighbors.append((to_id, data['weight']))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def stats(self) -> dict:
        return {
            "total_edges": self.graph.number_of_edges(),
            "total_nodes_with_edges": self.graph.number_of_nodes(),
        }

    def __repr__(self):
        s = self.stats()
        return f"RewardGraph(edges={s['total_edges']}, nodes={s['total_nodes_with_edges']})"


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Initializing Graph Engine ===\n")

    space = GeometricSpace(dim=64, merge_threshold=0.15)
    reward_graph = RewardGraph()

    # seed with some letter nodes
    letters = "abcdefghijklmnopqrstuvwxyz "
    letter_nodes = {}
    for ch in letters:
        node = Node(sequence=ch, level=0, dim=64)
        space.add_node(node)
        letter_nodes[ch] = node

    print(f"Seeded: {space}")

    # simulate co-occurrence from word "the"
    word = "the"
    for i in range(len(word) - 1):
        a = letter_nodes[word[i]]
        b = letter_nodes[word[i+1]]
        a.update_co_occurrence(b.id)
        b.update_co_occurrence(a.id)
        space.update_positions_from_co_occurrence(a, b)

    print(f"\nAfter absorbing '{word}':")
    t_node = letter_nodes['t']
    neighbors = space.nearest_neighbors(t_node, k=3)
    print(f"  Nearest to 't': {[(n.sequence, round(d, 4)) for n, d in neighbors]}")

    # simulate reward on a path t -> h -> e
    path = [letter_nodes['t'].id, letter_nodes['h'].id, letter_nodes['e'].id]
    reward_graph.strengthen_path(path, reward=1.0)

    print(f"\nAfter rewarding path t->h->e:")
    print(f"  t->h weight: {reward_graph.get_edge_weight(letter_nodes['t'].id, letter_nodes['h'].id):.3f}")
    print(f"  h->e weight: {reward_graph.get_edge_weight(letter_nodes['h'].id, letter_nodes['e'].id):.3f}")
    print(f"  t->e weight (no edge): {reward_graph.get_edge_weight(letter_nodes['t'].id, letter_nodes['e'].id):.3f}")

    print(f"\n{reward_graph}")
    print(f"\nMerge candidates at level 0: {space.get_merge_candidates(level=0)[:3]}")
    print("\n=== Graph Engine OK ===")