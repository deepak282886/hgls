"""
Graph Engine v2 — Optimized
Key fixes:
- sequence_index dict for O(1) node lookup instead of O(n) scan
- scipy cdist for vectorized merge candidate detection
- level_index for fast level queries
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Optional
import uuid


class Node:
    def __init__(self, sequence: str, level: int = 0,
                 position: Optional[np.ndarray] = None, dim: int = 64):
        self.id = str(uuid.uuid4())
        self.sequence = sequence
        self.level = level
        self.position = position if position is not None else np.random.randn(dim)
        self.parent: Optional['Node'] = None
        self.children: list['Node'] = []
        self.visit_count: int = 0
        self.reward_weight: float = 0.0
        self.co_occurrence_counts: dict[str, int] = {}

    def update_co_occurrence(self, other_id: str):
        self.co_occurrence_counts[other_id] = self.co_occurrence_counts.get(other_id, 0) + 1

    def is_abstraction(self) -> bool:
        return len(self.children) > 0

    def __repr__(self):
        return f"Node(seq={repr(self.sequence)}, level={self.level}, visits={self.visit_count})"


class GeometricSpace:
    def __init__(self, dim: int = 64, pull_rate: float = 0.05,
                 push_rate: float = 0.01, merge_threshold: float = 0.15):
        self.dim = dim
        self.pull_rate = pull_rate
        self.push_rate = push_rate
        self.merge_threshold = merge_threshold
        self.nodes: dict[str, Node] = {}
        # O(1) lookups
        self._sequence_index: dict[str, str] = {}   # sequence -> node_id
        self._level_index: dict[int, list[str]] = {} # level -> [node_ids]

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self._sequence_index[f"{node.sequence}::{node.level}"] = node.id
        if node.level not in self._level_index:
            self._level_index[node.level] = []
        self._level_index[node.level].append(node.id)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_node_by_sequence(self, sequence: str, level: int = None) -> Optional[Node]:
        if level is not None:
            nid = self._sequence_index.get(f"{sequence}::{level}")
            return self.nodes.get(nid) if nid else None
        # search across all levels
        for lvl in self._level_index:
            nid = self._sequence_index.get(f"{sequence}::{lvl}")
            if nid:
                return self.nodes.get(nid)
        return None

    def get_nodes_at_level(self, level: int) -> list[Node]:
        ids = self._level_index.get(level, [])
        return [self.nodes[i] for i in ids if i in self.nodes]

    def distance(self, node_a: Node, node_b: Node) -> float:
        return float(np.linalg.norm(node_a.position - node_b.position))

    def nearest_neighbors(self, node: Node, k: int = 10, same_level: bool = True) -> list[tuple[Node, float]]:
        candidates = [n for n in self.nodes.values()
                      if n.id != node.id and (not same_level or n.level == node.level)]
        if not candidates:
            return []
        positions = np.array([n.position for n in candidates])
        dists = np.linalg.norm(positions - node.position, axis=1)
        idx = np.argsort(dists)[:k]
        return [(candidates[i], float(dists[i])) for i in idx]

    def pull_together(self, node_a: Node, node_b: Node, strength: float = 1.0):
        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            return
        unit = direction / distance
        move = unit * self.pull_rate * strength
        node_a.position += move
        node_b.position -= move

    def push_apart(self, node_a: Node, node_b: Node):
        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            direction = np.random.randn(self.dim)
            distance = np.linalg.norm(direction)
        unit = direction / distance
        node_a.position -= unit * self.push_rate
        node_b.position += unit * self.push_rate

    def update_positions_from_co_occurrence(self, node_a: Node, node_b: Node):
        count = node_a.co_occurrence_counts.get(node_b.id, 0)
        strength = np.log1p(count)
        self.pull_together(node_a, node_b, strength=strength)

    def get_merge_candidates(self, level: int) -> list[tuple[Node, Node, float]]:
        level_nodes = self.get_nodes_at_level(level)
        if len(level_nodes) < 2:
            return []
        positions = np.array([n.position for n in level_nodes])
        # vectorized pairwise distances
        dm = cdist(positions, positions)
        rows, cols = np.where((dm < self.merge_threshold) & (dm > 0))
        candidates = []
        seen = set()
        for r, c in zip(rows, cols):
            if r >= c:
                continue
            key = (r, c)
            if key not in seen:
                seen.add(key)
                candidates.append((level_nodes[r], level_nodes[c], float(dm[r, c])))
        candidates.sort(key=lambda x: x[2])
        return candidates

    def stats(self) -> dict:
        levels = {lvl: len(ids) for lvl, ids in self._level_index.items()}
        return {"total_nodes": len(self.nodes), "nodes_per_level": dict(sorted(levels.items()))}

    def __repr__(self):
        s = self.stats()
        return f"GeometricSpace(total={s['total_nodes']}, levels={s['nodes_per_level']})"


class RewardGraph:
    def __init__(self, decay: float = 0.95):
        self.graph = nx.DiGraph()
        self.decay = decay

    def add_or_strengthen_edge(self, from_id: str, to_id: str, reward: float):
        if self.graph.has_edge(from_id, to_id):
            current = self.graph[from_id][to_id]['weight']
            self.graph[from_id][to_id]['weight'] = current * 0.8 + reward * 0.2
            self.graph[from_id][to_id]['visits'] += 1
        else:
            self.graph.add_edge(from_id, to_id, weight=reward, visits=1)

    def normalize_path(self, path: list[str]):
        for i in range(len(path) - 1):
            from_id, to_id = path[i], path[i+1]
            if self.graph.has_edge(from_id, to_id):
                self.graph[from_id][to_id]['weight'] *= self.decay

    def strengthen_path(self, path: list[str], reward: float):
        for i in range(len(path) - 1):
            self.add_or_strengthen_edge(path[i], path[i+1], reward)

    def get_edge_weight(self, from_id: str, to_id: str) -> float:
        if self.graph.has_edge(from_id, to_id):
            return self.graph[from_id][to_id]['weight']
        return 0.0

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        neighbors = [(to, data['weight'])
                     for _, to, data in self.graph.out_edges(node_id, data=True)]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def stats(self) -> dict:
        return {"total_edges": self.graph.number_of_edges(),
                "total_nodes_with_edges": self.graph.number_of_nodes()}

    def __repr__(self):
        s = self.stats()
        return f"RewardGraph(edges={s['total_edges']}, nodes={s['total_nodes_with_edges']})"
