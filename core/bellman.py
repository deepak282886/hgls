"""
bellman.py

Bellman value computation for all nodes in the graph.

The Bellman value V(i) encodes the best possible cumulative
reward reachable from node i following the optimal path.

    V(i) = r(i) + max over outgoing edges of ( e(i,j) + V(j) )
    V(i) = r(i)   if node i is terminal

Every node gets a globally informed future estimate — not just
its own reward but the best reward reachable through it.
This is what makes greedy traversal globally optimal.

Two modes of operation:

    1. FULL SWEEP
       Process all nodes in reverse topological order until
       convergence. Run at startup and periodically as the
       graph grows substantially.

    2. DELTA PROPAGATION
       When a new node or edge arrives, propagate its V value
       change backward through incoming edges only to affected
       ancestors. Surgical — touches only what changed.
       Run continuously during ingestion.

The combination of both keeps V values current without
recomputing the entire graph after every ingestion event.
"""

import numpy as np
import time
from typing import Optional
from collections import deque

from core.atoms      import Level
from core.graph      import Graph
from core.node       import Node
from core.normalizer import Normalizer


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Minimum V change to consider a node "changed"
# — smaller changes are noise, not signal
DELTA_THRESHOLD   = 0.001

# Convergence criterion for full sweep
# — stop iterating when max change drops below this
CONVERGENCE_EPS   = 0.01

# Maximum iterations for full sweep
# — safety limit, convergence usually happens in 3-5
MAX_SWEEP_ITER    = 20

# Maximum hops for delta propagation backward
# — prevents runaway propagation in densely connected graphs
MAX_DELTA_HOPS    = 100

# How often (new edges) to trigger a full sweep
# — between sweeps, delta propagation handles incremental updates
FULL_SWEEP_EVERY  = 1000


# ─────────────────────────────────────────────
# CORE V COMPUTATION — single node
# ─────────────────────────────────────────────

def compute_V(node_id: int, graph: Graph) -> float:
    """
    Compute V for a single node given current V values of neighbors.

    V(i) = r(i) + max_j ( e(i,j) + V(j) )

    Uses raw reward and raw edge reward — normalization happens
    after Bellman sweep completes (normalizer.on_bellman_sweep).

    Returns raw V value. Does not write to DB — caller decides when.
    """
    node = graph.get_node(node_id)
    if node is None:
        return 0.0

    edges = graph.get_outgoing_edges(node_id)

    if not edges:
        # Terminal node — V equals own reward
        return node.reward

    best_future = -np.inf
    for edge in edges:
        neighbor = graph.get_node(edge.target_id)
        if neighbor is None:
            continue
        future = edge.reward + neighbor.V
        if future > best_future:
            best_future = future

    if best_future == -np.inf:
        # All neighbors missing — treat as terminal
        return node.reward

    return node.reward + best_future


# ─────────────────────────────────────────────
# FULL SWEEP
# ─────────────────────────────────────────────

def full_sweep(
    graph      : Graph,
    normalizer : Normalizer,
    verbose    : bool = False,
) -> dict:
    """
    Full Bellman sweep over all nodes until convergence.

    Processes nodes in reverse creation order (terminals first,
    roots last) and iterates until max V change drops below
    CONVERGENCE_EPS or MAX_SWEEP_ITER is reached.

    After convergence, triggers normalizer.on_bellman_sweep to
    recompute norm_V for all nodes.

    Returns diagnostic dict with sweep statistics.
    """
    t0 = time.time()

    # Load all node IDs in reverse creation order
    # (newer nodes tend to be terminals — process them first
    #  so their V propagates backward to older nodes)
    all_ids = graph.all_node_ids()
    if not all_ids:
        return {"iterations": 0, "nodes": 0, "elapsed": 0.0}

    all_ids_rev = list(reversed(all_ids))
    n_nodes     = len(all_ids_rev)

    iterations  = 0
    max_change  = np.inf

    while iterations < MAX_SWEEP_ITER and max_change > CONVERGENCE_EPS:
        max_change  = 0.0
        n_changed   = 0

        for node_id in all_ids_rev:
            node  = graph.get_node(node_id)
            if node is None:
                continue

            old_V = node.V
            new_V = compute_V(node_id, graph)
            change = abs(new_V - old_V)

            if change > DELTA_THRESHOLD:
                graph.update_node_V(node_id, new_V)
                max_change = max(max_change, change)
                n_changed += 1

        iterations += 1

        if verbose:
            print(f"  Sweep iter {iterations}: "
                  f"max_change={max_change:.5f}  "
                  f"nodes_updated={n_changed}")

    # Normalize all V values now that sweep has converged
    normalizer.on_bellman_sweep(graph)

    elapsed = time.time() - t0

    return {
        "iterations"   : iterations,
        "nodes"        : n_nodes,
        "max_change"   : round(max_change, 5),
        "converged"    : max_change <= CONVERGENCE_EPS,
        "elapsed_sec"  : round(elapsed, 3),
    }


# ─────────────────────────────────────────────
# DELTA PROPAGATION
# ─────────────────────────────────────────────

def delta_propagate(
    node_id    : int,
    graph      : Graph,
    normalizer : Normalizer,
    max_hops   : int = MAX_DELTA_HOPS,
) -> int:
    """
    Propagate a V value change backward from node_id
    to all ancestors that might be affected.

    Uses BFS over incoming edges. Only propagates if
    the ancestor's V actually changes by more than DELTA_THRESHOLD.

    Called:
        - After a new edge is added (edge.target_id changed V landscape)
        - After a node reward is updated

    Returns number of nodes whose V was updated.
    """
    queue    = deque([node_id])
    visited  = set()
    updated  = 0
    hops     = 0

    while queue and hops < max_hops:
        current_id = queue.popleft()

        if current_id in visited:
            continue
        visited.add(current_id)

        # Find all nodes that have an outgoing edge TO current_id
        # — they may need their V updated now that current_id's V changed
        incoming = graph.get_incoming_edges(current_id)

        for edge in incoming:
            src_id   = edge.source_id
            src_node = graph.get_node(src_id)

            if src_node is None or src_id in visited:
                continue

            old_V = src_node.V
            new_V = compute_V(src_id, graph)
            change = abs(new_V - old_V)

            if change > DELTA_THRESHOLD:
                norm_V = normalizer.vnorm.normalize(new_V)
                graph.update_node_V(src_id, new_V, norm_V=norm_V)
                normalizer.vnorm.update(new_V)
                updated += 1

                # Propagate further backward if change was significant
                if change > DELTA_THRESHOLD * 10:
                    queue.append(src_id)

        hops += 1

    return updated


def on_edge_added(
    source_id  : int,
    target_id  : int,
    graph      : Graph,
    normalizer : Normalizer,
) -> int:
    """
    Called by ingestion pipeline immediately after a new edge
    (source_id → target_id) is added.

    Steps:
        1. Recompute V for source_id — it now has a new outgoing path
        2. If source_id's V changed, propagate backward through
           its ancestors via delta_propagate

    Returns number of nodes updated.
    """
    src_node = graph.get_node(source_id)
    if src_node is None:
        return 0

    old_V = src_node.V
    new_V = compute_V(source_id, graph)
    change = abs(new_V - old_V)

    if change <= DELTA_THRESHOLD:
        return 0

    norm_V = normalizer.vnorm.normalize(new_V)
    graph.update_node_V(source_id, new_V, norm_V=norm_V)
    normalizer.vnorm.update(new_V)

    # Propagate backward from source_id
    n_updated = delta_propagate(source_id, graph, normalizer)
    return n_updated + 1


def on_node_reward_updated(
    node_id    : int,
    new_reward : float,
    graph      : Graph,
    normalizer : Normalizer,
) -> int:
    """
    Called when a node's reward changes — e.g. after contrastive
    training sharpens the reward estimate.

    Recomputes V for the node and propagates backward.
    """
    node = graph.get_node(node_id)
    if node is None:
        return 0

    # Recompute with new reward
    edges = graph.get_outgoing_edges(node_id)
    if not edges:
        new_V = new_reward
    else:
        best  = max(e.reward + graph.get_node(e.target_id).V
                    for e in edges
                    if graph.get_node(e.target_id) is not None)
        new_V = new_reward + best

    norm_V = normalizer.vnorm.normalize(new_V)
    graph.update_node_V(node_id, new_V, norm_V=norm_V)
    normalizer.vnorm.update(new_V)

    return delta_propagate(node_id, graph, normalizer)


# ─────────────────────────────────────────────
# BELLMAN MANAGER
# Coordinates sweep scheduling and delta propagation
# ─────────────────────────────────────────────

class BellmanManager:
    """
    Manages when to run full sweeps vs delta propagation.

    Tracks edges added since last full sweep. When the count
    exceeds FULL_SWEEP_EVERY, schedules a full sweep.
    Between sweeps, all updates go through delta_propagate.

    Usage:
        bm = BellmanManager(graph, normalizer)

        # After ingestion pipeline adds an edge:
        bm.on_edge_added(source_id, target_id)

        # Force a full sweep:
        stats = bm.sweep(verbose=True)
    """

    def __init__(self, graph: Graph, normalizer: Normalizer):
        self.graph         = graph
        self.normalizer    = normalizer
        self._edges_since_sweep = 0
        self._total_sweeps = 0
        self._total_deltas = 0

    def on_edge_added(self, source_id: int, target_id: int) -> int:
        """
        Called after every new edge.
        Runs delta propagation or schedules a full sweep.
        """
        self._edges_since_sweep += 1

        # Delta propagation for incremental update
        n = on_edge_added(source_id, target_id,
                          self.graph, self.normalizer)
        self._total_deltas += n

        # Schedule full sweep if enough edges have accumulated
        if self._edges_since_sweep >= FULL_SWEEP_EVERY:
            self.sweep()

        return n

    def sweep(self, verbose: bool = False) -> dict:
        """Force a full Bellman sweep now."""
        stats = full_sweep(self.graph, self.normalizer, verbose=verbose)
        self._edges_since_sweep = 0
        self._total_sweeps     += 1
        return stats

    def status(self) -> dict:
        return {
            "edges_since_sweep": self._edges_since_sweep,
            "total_sweeps"     : self._total_sweeps,
            "total_delta_updates": self._total_deltas,
            "sweep_threshold"  : FULL_SWEEP_EVERY,
        }


if __name__ == "__main__":
    import tempfile, os

    print("=== bellman.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer

        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # Build a chain of sentence nodes
        sentences = [
            ("Detective Maria arrived at the warehouse.",              2.0),
            ("The building smelled of rust and old machinery.",        1.7),
            ("She found footprints leading to the back room.",         1.5),
            ("The footprints were fresh, made within the last hour.",  1.3),
            ("In the corner she discovered a locked metal box.",       1.8),
            ("The box had three combination locks.",                   1.5),
            ("Scrawled on the wall were the numbers 1987 2034 0451.", 2.1),
            ("She entered the codes and the box clicked open.",        1.6),
            ("Inside were photographs and a folded letter.",           1.8),
            ("The letter named someone she recognized.",               2.2),
        ]

        nodes = []
        for text, reward in sentences:
            n = graph.add_node(text, Level.SENTENCE, reward=reward)
            norm.on_node_added(n, graph)
            nodes.append(n)

        print(f"Added {len(nodes)} nodes\n")
        print("V before Bellman (= raw reward only):")
        for n in nodes:
            fresh = graph.get_node(n.node_id)
            print(f"  [{fresh.node_id:02d}] V={fresh.V:+.3f}  {fresh.text[:50]}")

        # Add sequential edges and trigger delta propagation
        print("\nAdding edges + delta propagation...")
        for i in range(len(nodes) - 1):
            src, tgt = nodes[i], nodes[i+1]
            graph.add_edge(src.node_id, tgt.node_id,
                           src.text, tgt.text, Level.SENTENCE)
            norm.on_edge_added(src.node_id, graph)
            n_updated = bm.on_edge_added(src.node_id, tgt.node_id)

        # Full sweep
        print("\nRunning full sweep...")
        stats = bm.sweep(verbose=True)
        print(f"\nSweep stats: {stats}")

        # V after Bellman
        print("\nV after Bellman sweep (reward propagated from terminal):")
        max_V = max(graph.get_node(n.node_id).V for n in nodes)
        for n in nodes:
            fresh = graph.get_node(n.node_id)
            bar   = "█" * int((fresh.V / max_V) * 20)
            print(f"  [{fresh.node_id:02d}] V={fresh.V:+.3f}  "
                  f"norm_V={fresh.norm_V:+.3f}  {bar}")

        # Demonstrate delta propagation — add a high-reward terminal
        print("\nAdding high-reward terminal node...")
        big = graph.add_node(
            "The mystery was solved — the evidence was conclusive.",
            Level.SENTENCE, reward=3.0
        )
        norm.on_node_added(big, graph)
        last = nodes[-1]
        graph.add_edge(last.node_id, big.node_id,
                       last.text, big.text, Level.SENTENCE)
        norm.on_edge_added(last.node_id, graph)
        n_delta = bm.on_edge_added(last.node_id, big.node_id)

        print(f"Delta propagation updated {n_delta} nodes\n")
        print("V after high-reward terminal added:")
        for n in nodes[:5]:
            fresh = graph.get_node(n.node_id)
            print(f"  [{fresh.node_id:02d}] V={fresh.V:+.3f}  {fresh.text[:50]}")

        print(f"\nBellman manager status: {bm.status()}")
        graph.close()