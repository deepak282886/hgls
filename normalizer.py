"""
normalizer.py

All normalization logic for the graph.

Four normalization operations, each with a specific purpose:

    1. NODE REWARD — per level
       Normalize node rewards within each hierarchy level separately.
       Prevents higher levels from dominating just because they
       accumulate more signal. Uses Welford running stats from graph.

    2. EDGE REWARD — per source node
       Normalize outgoing edge rewards relative to each other.
       Every node's best edge is comparably good regardless of
       how many edges it has or what absolute scale it sits on.

    3. BELLMAN V — global with running statistics
       V values must be comparable across the whole graph for
       MCTS to navigate. Normalized using global running stats.
       Updated incrementally — no full recompute needed.

    4. ACTIVATION WEIGHTS — incoming edges per node
       When a query activates a neighborhood, normalize incoming
       edge weights so no node gets over-activated just because
       many things point to it. Softmax-style, conserves budget.

All operations are incremental where possible — designed for
a continuously growing graph. No batch recomputation from scratch.
"""

import numpy as np
from typing import Optional

from core.atoms import Level
from core.graph import Graph
from core.node  import Node
from core.edge  import Edge


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EPSILON = 1e-9   # prevents division by zero throughout

# How often (in new nodes) to run a full normalization sweep
# Between sweeps, only the newly added nodes are normalized
SWEEP_INTERVAL = 500


# ─────────────────────────────────────────────
# 1. NODE REWARD NORMALIZATION — per level
# ─────────────────────────────────────────────

def normalize_node_reward(node: Node, graph: Graph) -> float:
    """
    Normalize a single node's reward within its level.

    Uses the running mean and std for that level stored in the graph.
    Returns the normalized reward — does not write to DB.
    Caller decides when to persist (batch or immediate).

    Formula:
        norm_reward = (reward - level_mean) / (level_std + epsilon)

    Clipped to [-3, +3] standard deviations to handle outliers.
    """
    mean = graph.level_mean(node.level)
    std  = graph.level_std(node.level)

    norm = (node.reward - mean) / (std + EPSILON)
    return float(np.clip(norm, -3.0, 3.0))


def normalize_all_node_rewards(graph: Graph, level: Optional[Level] = None):
    """
    Recompute normalized rewards for all nodes at a given level
    (or all levels if level is None).

    Run after the Welford stats have been updated by enough new nodes
    to meaningfully shift the distribution — i.e. every SWEEP_INTERVAL nodes.

    Writes norm_reward back to the database.
    """
    levels = [level] if level is not None else list(Level)

    for lv in levels:
        mean = graph.level_mean(lv)
        std  = graph.level_std(lv)

        if std < EPSILON:
            continue   # not enough data at this level yet

        nodes = graph.nodes_by_level(lv, limit=100_000)
        for node in nodes:
            norm = (node.reward - mean) / (std + EPSILON)
            norm = float(np.clip(norm, -3.0, 3.0))
            graph.update_node_reward(node.node_id, node.reward, norm_reward=norm)


# ─────────────────────────────────────────────
# 2. EDGE REWARD NORMALIZATION — per source node
# ─────────────────────────────────────────────

def normalize_node_edges(node_id: int, graph: Graph):
    """
    Normalize all outgoing edge rewards from a given node
    relative to each other.

    After normalization, the best edge from any node has comparable
    norm_reward regardless of how many edges that node has or what
    absolute reward scale it sits on.

    Formula (per-node z-score):
        norm_reward_ij = (reward_ij - mu_i) / (sigma_i + epsilon)

    Clipped to [-3, +3].

    Writes norm_reward back to the database for each edge.
    """
    edges = graph.get_outgoing_edges(node_id)

    if not edges:
        return

    rewards = np.array([e.reward for e in edges], dtype=np.float32)

    if len(rewards) == 1:
        # Only one edge — normalized reward is 0.0 (it's the only option)
        # but we still write it so the field is populated
        graph.update_edge_norm_reward(edges[0].edge_id, 0.0)
        return

    mu    = float(np.mean(rewards))
    sigma = float(np.std(rewards))

    for edge in edges:
        if sigma < EPSILON:
            norm = 0.0
        else:
            norm = (edge.reward - mu) / (sigma + EPSILON)
            norm = float(np.clip(norm, -3.0, 3.0))
        graph.update_edge_norm_reward(edge.edge_id, norm)


def normalize_edges_for_new_node(node_id: int, graph: Graph):
    """
    Called immediately after a new edge is added from node_id.
    Re-normalizes all outgoing edges of that node to account
    for the new addition.

    This is the incremental path — no full sweep needed.
    """
    normalize_node_edges(node_id, graph)


# ─────────────────────────────────────────────
# 3. BELLMAN V NORMALIZATION — global running
# ─────────────────────────────────────────────

class VNormalizer:
    """
    Maintains running global statistics for Bellman V values.
    Uses Welford's online algorithm — O(1) per update.

    V values span the entire graph and must be globally comparable
    for MCTS to navigate across levels and domains.

    Usage:
        vnorm = VNormalizer()
        vnorm.update(new_V_value)
        normalized = vnorm.normalize(raw_V)
    """

    def __init__(self):
        self._count : int   = 0
        self._mean  : float = 0.0
        self._M2    : float = 0.0    # sum of squared deviations (Welford)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return float(np.sqrt(self._M2 / (self._count - 1)) + EPSILON)

    @property
    def count(self) -> int:
        return self._count

    def update(self, value: float):
        """
        Incorporate a new V value into running statistics.
        Called every time a node's V is updated by bellman.py.
        """
        self._count += 1
        delta        = value - self._mean
        self._mean  += delta / self._count
        delta2       = value - self._mean
        self._M2    += delta * delta2

    def normalize(self, raw_V: float) -> float:
        """
        Map raw V to normalized V using current running stats.
        Returns value typically in [-3, +3].
        """
        norm = (raw_V - self._mean) / (self.std + EPSILON)
        return float(np.clip(norm, -5.0, 5.0))

    def normalize_batch(self, values: list[float]) -> list[float]:
        """Normalize a batch of V values efficiently."""
        arr  = np.array(values, dtype=np.float32)
        norm = (arr - self._mean) / (self.std + EPSILON)
        return list(np.clip(norm, -5.0, 5.0).astype(float))

    def to_dict(self) -> dict:
        return {
            "count": self._count,
            "mean" : self._mean,
            "M2"   : self._M2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VNormalizer":
        vn = cls()
        vn._count = d["count"]
        vn._mean  = d["mean"]
        vn._M2    = d["M2"]
        return vn

    def __repr__(self) -> str:
        return (
            f"VNormalizer(count={self._count}, "
            f"mean={self._mean:.4f}, std={self.std:.4f})"
        )


def normalize_all_V(graph: Graph, vnorm: VNormalizer):
    """
    Full sweep — recompute norm_V for every node using current
    VNormalizer state. Run after a Bellman sweep completes.

    Memory-efficient: processes nodes in batches.
    """
    for batch in graph.iter_all_nodes(batch_size=500):
        for node in batch:
            norm_V = vnorm.normalize(node.V)
            graph.update_node_V(node.node_id, node.V, norm_V=norm_V)


# ─────────────────────────────────────────────
# 4. ACTIVATION WEIGHT NORMALIZATION
#    Incoming edges per node — softmax style
# ─────────────────────────────────────────────

def normalize_activation_weights(
    node_id    : int,
    graph      : Graph,
    temperature: float = 1.0,
) -> dict[int, float]:
    """
    Normalize incoming edge rewards for a node so that
    activation budget is conserved — total activation flowing
    in sums to 1.0 regardless of how many things point to it.

    Uses softmax over incoming edge rewards weighted by temperature.
    Higher temperature = more uniform (explore).
    Lower temperature  = more peaked (exploit).

    Returns:
        dict mapping source_node_id → normalized activation weight

    This is computed fresh at query time — not stored in DB.
    The activation pattern is query-dependent and changes every call.
    """
    incoming = graph.get_incoming_edges(node_id)

    if not incoming:
        return {}

    rewards = np.array([e.reward for e in incoming], dtype=np.float32)

    # Softmax with temperature
    scaled  = rewards / (temperature + EPSILON)
    shifted = scaled - np.max(scaled)    # numerical stability
    exp     = np.exp(shifted)
    weights = exp / (np.sum(exp) + EPSILON)

    return {
        e.source_id: float(w)
        for e, w in zip(incoming, weights)
    }


# ─────────────────────────────────────────────
# MASTER NORMALIZER
#
# Single object that coordinates all four normalization
# operations. Passed around to bellman.py, mcts.py,
# activation.py so they all use the same state.
# ─────────────────────────────────────────────

class Normalizer:
    """
    Coordinates all normalization operations for the graph.

    Holds the VNormalizer instance (stateful running stats).
    All other operations are stateless functions that read
    from the graph's per-level Welford stats.

    Usage:
        normalizer = Normalizer()

        # After adding a new node:
        normalizer.on_node_added(node, graph)

        # After adding a new edge from source_id:
        normalizer.on_edge_added(source_id, graph)

        # After a Bellman sweep updates V values:
        normalizer.on_bellman_sweep(graph)

        # At query time for activation:
        weights = normalizer.activation_weights(node_id, graph)
    """

    def __init__(self):
        self.vnorm        = VNormalizer()
        self._nodes_since_sweep = 0

    def on_node_added(self, node: Node, graph: Graph):
        """
        Called by ingestion pipeline immediately after a node is added.
        Updates running stats and normalizes the new node's reward.
        """
        # Normalize node reward using current level stats
        norm_r = normalize_node_reward(node, graph)
        graph.update_node_reward(node.node_id, node.reward, norm_reward=norm_r)

        # Update VNormalizer with initial V (= reward at creation)
        self.vnorm.update(node.V)

        self._nodes_since_sweep += 1

        # Periodic full sweep to keep all old nodes' norm_reward current
        if self._nodes_since_sweep >= SWEEP_INTERVAL:
            self._full_reward_sweep(graph)
            self._nodes_since_sweep = 0

    def on_edge_added(self, source_id: int, graph: Graph):
        """
        Called immediately after a new edge is added from source_id.
        Re-normalizes all outgoing edges of that node.
        """
        normalize_edges_for_new_node(source_id, graph)

    def on_bellman_sweep(self, graph: Graph):
        """
        Called after bellman.py completes a V update sweep.
        Recomputes norm_V for all nodes using updated VNormalizer.
        """
        # First update VNormalizer with all current V values
        for batch in graph.iter_all_nodes(batch_size=500):
            for node in batch:
                self.vnorm.update(node.V)

        # Then normalize all V values
        normalize_all_V(graph, self.vnorm)

    def _full_reward_sweep(self, graph: Graph):
        """
        Recompute norm_reward for all nodes at all levels.
        Run periodically as the reward distribution shifts
        with new data.
        """
        normalize_all_node_rewards(graph)

    def activation_weights(
        self,
        node_id    : int,
        graph      : Graph,
        temperature: float = 1.0,
    ) -> dict[int, float]:
        """
        Compute activation weights for incoming edges to node_id.
        Fresh computation at query time.
        """
        return normalize_activation_weights(node_id, graph, temperature)

    def node_ucb_score(
        self,
        node      : Node,
        parent_visits: int,
        c_explore : float = 1.414,
    ) -> float:
        """
        UCB score for MCTS node selection.
        Uses normalized V and visit counts.

        UCB = norm_V + c * sqrt(ln(parent_visits) / (visits + 1))

        The exploration term uses normalized V so exploitation
        and exploration are on comparable scales.
        """
        exploit = node.norm_V
        if parent_visits <= 0 or node.visit_count <= 0:
            explore = c_explore * 2.0   # high exploration for unvisited
        else:
            explore = c_explore * np.sqrt(
                np.log(parent_visits + 1) / (node.visit_count + 1)
            )
        return float(exploit + explore)

    def to_dict(self) -> dict:
        return {"vnorm": self.vnorm.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls()
        n.vnorm = VNormalizer.from_dict(d["vnorm"])
        return n

    def __repr__(self) -> str:
        return f"Normalizer(vnorm={self.vnorm})"


if __name__ == "__main__":
    import tempfile, os
    from core.atoms import Level

    print("=== normalizer.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()

        sentences = [
            ("Detective Maria arrived at the warehouse.", 2.0),
            ("The building smelled of rust.",             1.7),
            ("She found footprints.",                     1.5),
            ("The footprints were fresh.",                1.3),
            ("Her flashlight flickered and died.",        1.5),
            ("She replaced the batteries.",               0.6),
            ("Scrawled on the wall were numbers.",        1.4),
            ("She entered the codes.",                    0.6),
            ("The box clicked open.",                     0.6),
            ("Inside were photographs.",                  0.8),
        ]

        nodes = []
        for text, reward in sentences:
            n = graph.add_node(text, Level.SENTENCE, reward=reward)
            norm.on_node_added(n, graph)
            nodes.append(n)

        print("Node rewards after normalization:")
        for n in nodes:
            fresh = graph.get_node(n.node_id)
            print(f"  [{fresh.node_id}] raw={fresh.reward:+.3f}  "
                  f"norm={fresh.norm_reward:+.3f}  {fresh.text[:45]}")

        # Add edges
        print("\nAdding edges and normalizing...")
        for i in range(len(nodes) - 1):
            src, tgt = nodes[i], nodes[i+1]
            graph.add_edge(src.node_id, tgt.node_id,
                           src.text, tgt.text, Level.SENTENCE)
            norm.on_edge_added(src.node_id, graph)

        # Check edge normalization
        print("\nOutgoing edges from node 0 (normalized):")
        for e in graph.get_outgoing_edges(nodes[0].node_id):
            print(f"  {e.source_id}→{e.target_id}  "
                  f"raw={e.reward:+.3f}  norm={e.norm_reward:+.3f}")

        # VNormalizer state
        print(f"\nVNormalizer: {norm.vnorm}")

        # UCB scores
        print("\nUCB scores (parent_visits=10):")
        for n in nodes[:4]:
            fresh = graph.get_node(n.node_id)
            ucb = norm.node_ucb_score(fresh, parent_visits=10)
            print(f"  [{fresh.node_id}] norm_V={fresh.norm_V:+.3f}  UCB={ucb:.4f}")

        # Activation weights for node 5 (which has incoming edge from node 4)
        print("\nActivation weights for node 1 (incoming from node 0):")
        weights = norm.activation_weights(nodes[1].node_id, graph)
        for src_id, w in weights.items():
            print(f"  from node {src_id}: weight={w:.4f}")

        graph.close()
        print("\nAll normalization operations complete.")