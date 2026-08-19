"""
Reward Engine
Success draws and strengthens edges along winning paths.
Failure normalizes edges down — paths don't disappear, just lose priority.
External correction draws strong explicit edges — supervised signal.
All updates propagate across every level simultaneously.
"""

import numpy as np
from typing import Optional
from graph_engine import Node, GeometricSpace, RewardGraph
from rollout import RolloutPath


# ─────────────────────────────────────────────
# REWARD ENGINE
# ─────────────────────────────────────────────

class RewardEngine:
    """
    Manages all reward signal flow through the graph.

    Three signal types:
    1. Success   — path delivered expected outcome, strengthen edges
    2. Failure   — path did not deliver, normalize edges down
    3. Correction — external signal showing correct path, strengthen strongly
    """

    def __init__(
        self,
        space: GeometricSpace,
        reward_graph: RewardGraph,
        success_lr: float = 0.2,        # how much to strengthen on success
        failure_decay: float = 0.85,    # how much edges decay on failure
        correction_strength: float = 2.0,  # correction is stronger than natural reward
        propagate_to_parents: bool = True  # reward flows up hierarchy
    ):
        self.space = space
        self.reward_graph = reward_graph
        self.success_lr = success_lr
        self.failure_decay = failure_decay
        self.correction_strength = correction_strength
        self.propagate_to_parents = propagate_to_parents

        # history for analysis
        self.reward_history: list[dict] = []

    # ── Success Signal ────────────────────────

    def success(self, path: RolloutPath, outcome_reward: float):
        """
        Path delivered. Strengthen all edges along it.
        Reward flows back through every level the path touched.
        """
        if not path.nodes or path.depth == 0:
            return

        node_ids = path.node_ids()

        # strengthen edges along path
        self.reward_graph.strengthen_path(node_ids, reward=outcome_reward)

        # update node reward weights
        for node in path.nodes:
            node.reward_weight = (
                node.reward_weight * (1 - self.success_lr) +
                outcome_reward * self.success_lr
            )
            node.visit_count += 1

        # propagate reward up through hierarchy
        if self.propagate_to_parents:
            self._propagate_upward(path.nodes, outcome_reward * 0.5)

        self.reward_history.append({
            "type": "success",
            "path_length": path.depth,
            "reward": outcome_reward,
            "nodes": [n.sequence[:20] for n in path.nodes]
        })

    # ── Failure Signal ────────────────────────

    def failure(self, path: RolloutPath, competing_paths: Optional[list[RolloutPath]] = None):
        """
        Path did not deliver expected reward.
        Normalize edges down along failed path.
        Competing paths get a relative boost.
        """
        if not path.nodes:
            return

        node_ids = path.node_ids()

        # normalize down failed path
        self.reward_graph.normalize_path(node_ids)

        # reduce node reward weights slightly
        for node in path.nodes:
            node.reward_weight *= self.failure_decay

        # competing paths get gentle boost — relative preference shifts
        if competing_paths:
            for alt_path in competing_paths[:3]:  # top 3 alternatives
                for node in alt_path.nodes:
                    node.reward_weight = min(
                        1.0,
                        node.reward_weight * (1.0 / self.failure_decay)
                    )

        self.reward_history.append({
            "type": "failure",
            "path_length": path.depth,
            "nodes": [n.sequence[:20] for n in path.nodes]
        })

    # ── Correction Signal ─────────────────────

    def correction(self, correct_path_sequences: list[str], reward: float = 1.0):
        """
        External correction — a correct path is given directly.
        Find or create nodes for each sequence in path.
        Draw strong edges between them.
        This is the supervised/imitation learning channel.
        """
        nodes = []
        for seq in correct_path_sequences:
            node = self.space.get_node_by_sequence(seq)
            if node:
                nodes.append(node)

        if len(nodes) < 2:
            return  # not enough nodes found to draw a path

        node_ids = [n.id for n in nodes]

        # strengthen with correction_strength multiplier
        self.reward_graph.strengthen_path(
            node_ids,
            reward=reward * self.correction_strength
        )

        # update node weights strongly
        for node in nodes:
            node.reward_weight = min(
                1.0,
                node.reward_weight * (1 - self.success_lr) +
                reward * self.correction_strength * self.success_lr
            )
            node.visit_count += 1

        # propagate up
        if self.propagate_to_parents:
            self._propagate_upward(nodes, reward * self.correction_strength * 0.5)

        self.reward_history.append({
            "type": "correction",
            "path": correct_path_sequences,
            "reward": reward * self.correction_strength
        })

    # ── Hierarchical Propagation ──────────────

    def _propagate_upward(self, nodes: list[Node], reward: float):
        """
        When a path at level N succeeds, propagate reward
        to parent nodes at level N+1.
        Reward decays as it moves up — distant abstractions
        get weaker signal than direct nodes.
        """
        parent_nodes = set()
        for node in nodes:
            if node.parent:
                parent_nodes.add(node.parent)

        if not parent_nodes:
            return

        parent_list = list(parent_nodes)
        parent_ids = [n.id for n in parent_list]

        # strengthen edges between parents
        if len(parent_ids) > 1:
            self.reward_graph.strengthen_path(parent_ids, reward=reward * 0.5)

        # update parent reward weights
        for parent in parent_list:
            parent.reward_weight = (
                parent.reward_weight * (1 - self.success_lr) +
                reward * self.success_lr
            )

        # recurse one more level up — grandparent signal very weak
        grandparents = set()
        for p in parent_list:
            if p.parent:
                grandparents.add(p.parent)

        for gp in grandparents:
            gp.reward_weight = (
                gp.reward_weight * (1 - self.success_lr) +
                reward * 0.25 * self.success_lr
            )

    # ── Path Comparison ───────────────────────

    def compare_and_update(
        self,
        winning_path: RolloutPath,
        losing_paths: list[RolloutPath],
        outcome_reward: float
    ):
        """
        Combined update — one path won, others lost.
        Winner gets strengthened. Losers normalize.
        This is the core learning loop.
        """
        self.success(winning_path, outcome_reward)
        for loser in losing_paths:
            self.failure(loser, competing_paths=[winning_path])

    # ── Diagnostics ──────────────────────────

    def top_reward_nodes(self, level: Optional[int] = None, k: int = 10) -> list[Node]:
        """Return nodes with highest reward weights."""
        nodes = self.space.nodes.values()
        if level is not None:
            nodes = [n for n in nodes if n.level == level]
        sorted_nodes = sorted(nodes, key=lambda n: n.reward_weight, reverse=True)
        return sorted_nodes[:k]

    def stats(self) -> dict:
        successes = sum(1 for h in self.reward_history if h["type"] == "success")
        failures = sum(1 for h in self.reward_history if h["type"] == "failure")
        corrections = sum(1 for h in self.reward_history if h["type"] == "correction")
        return {
            "successes": successes,
            "failures": failures,
            "corrections": corrections,
            "total_events": len(self.reward_history)
        }


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from absorption import Absorber, MergeEngine
    from rollout import RolloutEngine

    print("=== Reward Engine Test ===\n")

    # build full stack
    space = GeometricSpace(dim=32, merge_threshold=2.0, pull_rate=0.1)
    reward_graph = RewardGraph()
    merge_engine = MergeEngine(space=space, dim=32)
    absorber = Absorber(
        space=space,
        reward_graph=reward_graph,
        merge_engine=merge_engine,
        dim=32
    )
    rollout_engine = RolloutEngine(
        space=space,
        reward_graph=reward_graph,
        max_depth=5,
        n_rollouts=15
    )
    reward_engine = RewardEngine(
        space=space,
        reward_graph=reward_graph
    )

    # absorb corpus
    texts = [
        "the cat sat on the mat.",
        "the cat ate the rat.",
        "the dog sat on the log.",
        "cats and dogs are common pets.",
        "the cat chased the mouse around the house.",
        "a dog ran across the field.",
        "the quick brown fox jumps over the lazy dog.",
    ]

    print("Absorbing sequences...")
    for text in texts:
        absorber.absorb(text)
    print(f"{space}\n")

    # ── test 1: success signal ────────────────
    cat = space.get_node_by_sequence("cat")
    sat = space.get_node_by_sequence("sat")
    mat = space.get_node_by_sequence("mat")
    the = space.get_node_by_sequence("the")

    print("--- Test 1: Success Signal ---")
    paths = rollout_engine.rollout([cat, the])
    if paths:
        best = paths[0]
        print(f"Best path before reward: {best}")
        print(f"  cat reward_weight before: {cat.reward_weight:.4f}")

        reward_engine.success(best, outcome_reward=1.0)
        print(f"  cat reward_weight after:  {cat.reward_weight:.4f}")

    # ── test 2: failure + normalization ───────
    print("\n--- Test 2: Failure + Normalization ---")
    if len(paths) > 1:
        worst = paths[-1]
        first_node = worst.nodes[0] if worst.nodes else None
        if first_node:
            weight_before = reward_graph.get_edge_weight(
                worst.nodes[0].id,
                worst.nodes[1].id
            ) if len(worst.nodes) > 1 else 0
            print(f"Worst path: {worst}")

            reward_engine.failure(worst, competing_paths=[paths[0]])
            weight_after = reward_graph.get_edge_weight(
                worst.nodes[0].id,
                worst.nodes[1].id
            ) if len(worst.nodes) > 1 else 0
            print(f"  Edge weight before failure: {weight_before:.4f}")
            print(f"  Edge weight after failure:  {weight_after:.4f}")

    # ── test 3: external correction ──────────
    print("\n--- Test 3: External Correction ---")
    correction_path = ["the", "cat", "sat", "mat"]
    print(f"Correction given: {correction_path}")

    mat_before = mat.reward_weight if mat else 0
    reward_engine.correction(correction_path, reward=1.0)
    mat_after = mat.reward_weight if mat else 0

    print(f"  mat reward_weight before correction: {mat_before:.4f}")
    print(f"  mat reward_weight after correction:  {mat_after:.4f}")

    if mat and sat:
        edge = reward_graph.get_edge_weight(sat.id, mat.id)
        print(f"  sat->mat edge weight after correction: {edge:.4f}")

    # ── test 4: top reward nodes ──────────────
    print("\n--- Top Reward Nodes (level 1 — words) ---")
    top_nodes = reward_engine.top_reward_nodes(level=1, k=5)
    for node in top_nodes:
        print(f"  {node.sequence:<20} reward={node.reward_weight:.4f}  visits={node.visit_count}")

    print(f"\nReward engine stats: {reward_engine.stats()}")
    print("\n=== Reward Engine OK ===")
