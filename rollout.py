"""
Rollout Engine
MCTS with UCB across the hierarchy.
Coarse to fine — high level paths first, then fills downward.
Uses both geometric proximity (what's near) and reward edges (what worked).
"""

import math
import numpy as np
from typing import Optional
from graph_engine import Node, GeometricSpace, RewardGraph


# ─────────────────────────────────────────────
# UCB SCORER
# Balances exploration vs exploitation
# ─────────────────────────────────────────────

class UCBScorer:
    """
    Upper Confidence Bound scoring for node selection.
    High reward nodes preferred but unexplored nodes get bonus.
    As visits accumulate, exploration bonus shrinks.
    System learns to be more efficient with experience.
    """

    def __init__(self, exploration_constant: float = 1.414):
        self.c = exploration_constant  # sqrt(2) is standard UCB1

    def score(
        self,
        node: Node,
        edge_weight: float,
        proximity_score: float,
        total_visits: int
    ) -> float:
        """
        UCB score combining:
        - Exploitation: edge reward weight + proximity
        - Exploration: bonus for rarely visited nodes
        """
        if node.visit_count == 0:
            return float('inf')  # always explore unvisited nodes first

        exploitation = edge_weight * 0.7 + proximity_score * 0.3
        exploration = self.c * math.sqrt(math.log(total_visits + 1) / node.visit_count)

        return exploitation + exploration


# ─────────────────────────────────────────────
# ROLLOUT PATH
# A single candidate path through the graph
# ─────────────────────────────────────────────

class RolloutPath:
    """
    Represents one candidate path through the graph.
    Tracks nodes visited, cumulative reward, and depth.
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.cumulative_reward: float = 0.0
        self.depth: int = 0

    def add_step(self, node: Node, reward: float):
        self.nodes.append(node)
        self.cumulative_reward += reward
        self.depth += 1

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def expected_reward(self) -> float:
        """Average reward per step — normalizes for path length."""
        if self.depth == 0:
            return 0.0
        return self.cumulative_reward / self.depth

    def terminal_node(self) -> Optional[Node]:
        if self.nodes:
            return self.nodes[-1]
        return None

    def __repr__(self):
        seqs = [n.sequence[:15] for n in self.nodes]
        return f"Path(reward={self.expected_reward():.3f}, depth={self.depth}, nodes={seqs})"


# ─────────────────────────────────────────────
# ROLLOUT ENGINE
# Core MCTS loop across hierarchy levels
# ─────────────────────────────────────────────

class RolloutEngine:
    """
    Given an input activation (current state nodes),
    fans out candidate paths through the graph.
    Runs coarse to fine — high levels first, then fills downward.
    Returns ranked paths by expected reward.
    """

    def __init__(
        self,
        space: GeometricSpace,
        reward_graph: RewardGraph,
        max_depth: int = 10,
        n_rollouts: int = 20,
        exploration_constant: float = 1.414,
        proximity_weight: float = 0.3,
        reward_weight: float = 0.7
    ):
        self.space = space
        self.reward_graph = reward_graph
        self.max_depth = max_depth
        self.n_rollouts = n_rollouts
        self.ucb = UCBScorer(exploration_constant)
        self.proximity_weight = proximity_weight
        self.reward_weight = reward_weight

        # global visit count for UCB denominator
        self.total_visits: int = 0

    # ── Main Entry Point ─────────────────────

    def rollout(self, current_nodes: list[Node]) -> list[RolloutPath]:
        """
        Fan out from current activated nodes.
        Run n_rollouts paths. Return sorted by expected reward.
        """
        if not current_nodes:
            return []

        all_paths = []

        # determine max level in current activation
        max_level = max(n.level for n in current_nodes)

        # run rollouts starting from highest level down
        for level in range(max_level, -1, -1):
            level_starts = [n for n in current_nodes if n.level == level]
            if not level_starts:
                continue

            # allocate rollouts proportionally across levels
            # higher levels get more rollouts (coarse planning first)
            level_rollouts = max(1, self.n_rollouts // (max_level - level + 1))

            for start_node in level_starts:
                for _ in range(level_rollouts):
                    path = self._single_rollout(start_node, level)
                    if path.depth > 0:
                        all_paths.append(path)

        # sort by expected reward — best first
        all_paths.sort(key=lambda p: p.expected_reward(), reverse=True)
        return all_paths

    # ── Single Rollout ────────────────────────

    def _single_rollout(self, start_node: Node, level: int) -> RolloutPath:
        """
        Run one path from start_node.
        At each step use UCB to pick next node.
        Stops at max_depth or when no candidates exist.
        """
        path = RolloutPath()
        current = start_node
        visited = set()

        for _ in range(self.max_depth):
            visited.add(current.id)
            current.visit_count += 1
            self.total_visits += 1

            # get candidate next nodes
            candidates = self._get_candidates(current, level, visited)
            if not candidates:
                break

            # score all candidates with UCB
            scored = self._score_candidates(current, candidates)
            if not scored:
                break

            # pick best scored node
            next_node, step_reward = scored[0]

            path.add_step(next_node, step_reward)
            current = next_node

        return path

    # ── Candidate Selection ───────────────────

    def _get_candidates(
        self,
        current: Node,
        level: int,
        visited: set
    ) -> list[Node]:
        """
        Find candidate next nodes from current position.
        Combines reward graph neighbors and geometric neighbors.
        Stays at same level or moves up hierarchy.
        """
        candidates = set()

        # reward graph neighbors — nodes with existing edges
        for neighbor_id, _ in self.reward_graph.get_neighbors(current.id):
            node = self.space.get_node(neighbor_id)
            if node and node.id not in visited:
                candidates.add(node)

        # geometric neighbors — nodes that are close in space
        geo_neighbors = self.space.nearest_neighbors(current, k=5, same_level=True)
        for node, _ in geo_neighbors:
            if node.id not in visited:
                candidates.add(node)

        # also consider parent node if exists (moving up hierarchy)
        if current.parent and current.parent.id not in visited:
            candidates.add(current.parent)

        return list(candidates)

    # ── UCB Scoring ───────────────────────────

    def _score_candidates(
        self,
        current: Node,
        candidates: list[Node]
    ) -> list[tuple[Node, float]]:
        """
        Score each candidate using UCB.
        Returns sorted list of (node, reward_score).
        """
        scored = []

        for candidate in candidates:
            edge_weight = self.reward_graph.get_edge_weight(current.id, candidate.id)
            distance = self.space.distance(current, candidate)

            # proximity score — closer is better, normalized
            proximity_score = 1.0 / (1.0 + distance)

            ucb_score = self.ucb.score(
                candidate,
                edge_weight,
                proximity_score,
                self.total_visits
            )

            # step reward combines edge weight and proximity
            step_reward = (
                edge_weight * self.reward_weight +
                proximity_score * self.proximity_weight
            )

            scored.append((candidate, step_reward, ucb_score))

        # sort by UCB score for selection
        scored.sort(key=lambda x: x[2], reverse=True)

        # return (node, step_reward) — UCB used for selection only
        return [(node, reward) for node, reward, _ in scored]

    # ── Hierarchical Coarse to Fine ───────────

    def hierarchical_rollout(self, current_nodes: list[Node]) -> list[RolloutPath]:
        """
        Full hierarchical rollout.
        1. Roll out at highest level — get coarse path shape
        2. For each high level step, roll out at lower level within that region
        3. Returns full fine-grained best path
        """
        if not current_nodes:
            return []

        max_level = max(n.level for n in current_nodes)
        if max_level == 0:
            # already at lowest level — standard rollout
            return self.rollout(current_nodes)

        # step 1 — coarse rollout at highest level
        high_level_starts = [n for n in current_nodes if n.level == max_level]
        coarse_paths = []
        for start in high_level_starts:
            for _ in range(max(3, self.n_rollouts // 4)):
                path = self._single_rollout(start, max_level)
                if path.depth > 0:
                    coarse_paths.append(path)

        if not coarse_paths:
            return self.rollout(current_nodes)

        coarse_paths.sort(key=lambda p: p.expected_reward(), reverse=True)
        best_coarse = coarse_paths[0]

        # step 2 — for best coarse path, fill in lower level detail
        fine_paths = []
        for high_node in best_coarse.nodes:
            # find children of this high level node
            children = high_node.children if high_node.children else []
            low_starts = children if children else [high_node]

            for start in low_starts:
                for _ in range(max(2, self.n_rollouts // 8)):
                    path = self._single_rollout(start, start.level)
                    if path.depth > 0:
                        fine_paths.append(path)

        fine_paths.sort(key=lambda p: p.expected_reward(), reverse=True)

        # return coarse paths + fine paths merged
        all_paths = coarse_paths + fine_paths
        all_paths.sort(key=lambda p: p.expected_reward(), reverse=True)
        return all_paths

    def stats(self) -> dict:
        return {"total_visits": self.total_visits}


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from absorption import Absorber, MergeEngine

    print("=== Rollout Engine Test ===\n")

    # build graph
    space = GeometricSpace(dim=32, merge_threshold=2.0, pull_rate=0.1)
    reward_graph = RewardGraph()
    merge_engine = MergeEngine(space=space, dim=32)
    absorber = Absorber(
        space=space,
        reward_graph=reward_graph,
        merge_engine=merge_engine,
        dim=32
    )

    texts = [
        "the cat sat on the mat.",
        "the cat ate the rat.",
        "the dog sat on the log.",
        "a cat and a dog played together.",
        "the quick brown fox jumps over the lazy dog.",
        "cats and dogs are common pets.",
        "the cat chased the mouse around the house.",
    ]

    print("Absorbing sequences...")
    for text in texts:
        absorber.absorb(text)
        absorber.absorb_cross_level(text)

    print(f"{space}\n")

    # manually reward some paths so rollout has signal to work with
    cat = space.get_node_by_sequence("cat")
    sat = space.get_node_by_sequence("sat")
    the = space.get_node_by_sequence("the")
    dog = space.get_node_by_sequence("dog")
    mat = space.get_node_by_sequence("mat")

    if all([cat, sat, the, dog, mat]):
        reward_graph.strengthen_path([the.id, cat.id, sat.id, mat.id], reward=1.0)
        reward_graph.strengthen_path([the.id, dog.id, sat.id], reward=0.8)
        print("Reward paths seeded.\n")

    # init rollout engine
    engine = RolloutEngine(
        space=space,
        reward_graph=reward_graph,
        max_depth=5,
        n_rollouts=15
    )

    # activate current state — input "the cat"
    start_nodes = [n for n in [the, cat] if n]
    print(f"Rolling out from: {[n.sequence for n in start_nodes]}\n")

    # standard rollout
    paths = engine.rollout(start_nodes)
    print(f"Top 5 rollout paths:")
    for i, path in enumerate(paths[:5]):
        print(f"  [{i+1}] {path}")

    # hierarchical rollout
    print(f"\nHierarchical rollout:")
    h_paths = engine.hierarchical_rollout(start_nodes)
    print(f"Top 3 hierarchical paths:")
    for i, path in enumerate(h_paths[:3]):
        print(f"  [{i+1}] {path}")

    print(f"\nEngine stats: {engine.stats()}")
    print("\n=== Rollout Engine OK ===")
