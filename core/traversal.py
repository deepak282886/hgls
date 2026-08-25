"""
traversal.py

Traversal — extracting the optimal path through the graph.

Two modes:

    1. GREEDY TRAVERSAL
       Follow highest (edge_reward + V) at each step.
       No search — just harvests the Bellman landscape.
       O(depth) per query. Instant.
       Best when the graph is well-trained and V values
       are reliable — the answer is already encoded.

    2. MCTS TRAVERSAL
       Run MCTS search then extract the best path found.
       More expensive but better on uncertain terrain —
       sparse graphs, new domains, ambiguous queries.
       Falls back to greedy when MCTS finds no improvement.

Both modes operate within the lit subgraph from activation.py.
Unlit nodes are invisible regardless of their V value.

Traversal also handles:
    - Replay — re-run a known path ID sequence
    - Path rendering — convert node IDs to readable output
    - Path comparison — which path has higher total reward
"""

import time
from dataclasses import dataclass, field
from typing      import Optional

from core.atoms      import Level
from core.graph      import Graph
from core.node       import Node
from core.activation import ActivationResult, activate, activate_from_node, activate_from_path
from core.normalizer import Normalizer
from core.mcts       import MCTS, MCTSResult


# ─────────────────────────────────────────────
# TRAVERSAL RESULT
# ─────────────────────────────────────────────

@dataclass
class TraversalResult:
    """
    The output of a traversal — a path through the graph.

    path         : ordered list of node IDs
    texts        : text content of each node in path
    total_reward : cumulative reward along path
    mode         : "greedy" or "mcts"
    level        : hierarchy level traversed
    elapsed      : wall clock seconds
    mcts_result  : MCTSResult if mode="mcts", else None
    """
    path         : list[int]
    texts        : list[str]
    total_reward : float
    mode         : str
    level        : Level
    elapsed      : float
    mcts_result  : Optional[MCTSResult] = None

    def __len__(self) -> int:
        return len(self.path)

    def text_chain(self, separator: str = " ") -> str:
        """Join all texts into a single string."""
        return separator.join(self.texts)

    def summary(self) -> str:
        return (
            f"TraversalResult("
            f"mode={self.mode}, "
            f"steps={len(self.path)}, "
            f"reward={self.total_reward:.3f}, "
            f"elapsed={self.elapsed:.4f}s)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────
# GREEDY TRAVERSAL
# ─────────────────────────────────────────────

def greedy_traverse(
    start_id   : int,
    activation : ActivationResult,
    graph      : Graph,
    max_depth  : int = 100,
) -> list[int]:
    """
    Follow highest (edge_reward + V) at each step.
    Stays within lit subgraph. Never revisits a node.

    This is O(depth * avg_edges) — fast even on large graphs
    because activation bounds the search space.

    Returns ordered list of node IDs from start to terminal.
    """
    path    = [start_id]
    current = start_id
    visited = {start_id}
    depth   = 0

    while depth < max_depth:
        out_edges = graph.get_outgoing_edges(current)

        # Filter: lit, unvisited, positive reward
        candidates = [
            e for e in out_edges
            if activation.is_lit(e.target_id)
            and e.target_id not in visited
        ]

        if not candidates:
            break

        # Score: edge_reward + V(neighbor)
        def score(edge):
            neighbor = graph.get_node(edge.target_id)
            v        = neighbor.V if neighbor else 0.0
            act      = activation.activation(edge.target_id)
            return edge.reward + v + act * 0.5

        best_edge = max(candidates, key=score)
        next_id   = best_edge.target_id

        path.append(next_id)
        visited.add(next_id)
        current = next_id
        depth  += 1

    return path


# ─────────────────────────────────────────────
# PATH UTILITIES
# ─────────────────────────────────────────────

def path_texts(path: list[int], graph: Graph) -> list[str]:
    """Extract text content for each node in a path."""
    texts = []
    for nid in path:
        node = graph.get_node(nid)
        texts.append(node.text if node else f"[missing:{nid}]")
    return texts


def path_reward(path: list[int], graph: Graph) -> float:
    """Compute total reward (node + edge) along a path."""
    if not path:
        return 0.0

    total = 0.0
    for i, nid in enumerate(path):
        node = graph.get_node(nid)
        if node:
            total += node.reward

        if i + 1 < len(path):
            edges = graph.get_outgoing_edges(nid)
            for e in edges:
                if e.target_id == path[i+1]:
                    total += e.reward
                    break

    return round(total, 4)


def compare_paths(
    path_a : list[int],
    path_b : list[int],
    graph  : Graph,
) -> tuple[list[int], float, float]:
    """
    Compare two paths by total reward.
    Returns (winner, reward_a, reward_b).
    """
    ra = path_reward(path_a, graph)
    rb = path_reward(path_b, graph)
    return (path_a if ra >= rb else path_b), ra, rb


# ─────────────────────────────────────────────
# REPLAY
# ─────────────────────────────────────────────

def replay(
    path  : list[int],
    graph : Graph,
    level : Optional[Level] = None,
) -> TraversalResult:
    """
    Replay a known path — re-traverse a sequence of node IDs.

    Used for:
        - Re-running a previously found optimal path
        - Replaying an ingested sequence (audit / review)
        - Continuing from a checkpoint mid-path

    Returns TraversalResult with texts and reward recomputed.
    """
    t0    = time.time()
    texts = path_texts(path, graph)
    total = path_reward(path, graph)

    # Infer level from first node if not given
    if level is None and path:
        first = graph.get_node(path[0])
        level = first.level if first else Level.SENTENCE

    return TraversalResult(
        path         = path,
        texts        = texts,
        total_reward = total,
        mode         = "replay",
        level        = level or Level.SENTENCE,
        elapsed      = round(time.time() - t0, 4),
    )


# ─────────────────────────────────────────────
# MAIN TRAVERSAL ENTRY POINT
# ─────────────────────────────────────────────

def traverse(
    query      : str,
    graph      : Graph,
    normalizer : Normalizer,
    level      : Level         = Level.SENTENCE,
    mode       : str           = "mcts",
    start_id   : Optional[int] = None,
    max_depth  : int           = 50,
    max_iter   : int           = 100,
    max_hops   : int           = 6,
) -> TraversalResult:
    """
    Main traversal entry point.

    Given a query string (or a start node ID), activate the
    relevant subgraph and traverse to the highest-reward terminal.

    Parameters:
        query      : text query to activate the graph
        graph      : the persistent graph
        normalizer : shared normalizer instance
        level      : which hierarchy level to traverse
        mode       : "greedy" or "mcts"
        start_id   : if given, start from this node (skip similarity)
        max_depth  : maximum path length
        max_iter   : MCTS iterations (ignored in greedy mode)
        max_hops   : activation spread hops

    Returns TraversalResult with path, texts, and reward.
    """
    t0 = time.time()

    # ── 1. Activation ───────────────────────
    if start_id is not None:
        activation = activate_from_node(
            start_id, graph,
            max_hops = max_hops,
        )
        # If no lit nodes found, fall back to query
        if not activation.lit:
            activation = activate(
                query, graph, level=level, max_hops=max_hops
            )
    else:
        activation = activate(
            query, graph, level=level, max_hops=max_hops
        )

    if not activation.lit:
        return TraversalResult(
            path=[], texts=[], total_reward=0.0,
            mode=mode, level=level,
            elapsed=round(time.time()-t0, 4),
        )

    # ── 2. Determine start node ──────────────
    if start_id is not None and activation.is_lit(start_id):
        actual_start = start_id
    elif activation.seeds:
        # Use highest-V seed as start
        seed_vs = [
            (sid, graph.get_node(sid).V if graph.get_node(sid) else 0.0)
            for sid in activation.seeds
            if graph.get_node(sid) is not None
        ]
        actual_start = max(seed_vs, key=lambda x: x[1])[0] if seed_vs else activation.seeds[0]
    else:
        actual_start = next(iter(activation.lit))

    # ── 3. Traverse ─────────────────────────
    mcts_result = None

    if mode == "greedy":
        path = greedy_traverse(actual_start, activation, graph, max_depth)

    elif mode == "mcts":
        mcts   = MCTS(graph, normalizer)
        mcts_result = mcts.search(
            start_node_id = actual_start,
            activation    = activation,
            max_iter      = max_iter,
            max_depth     = max_depth,
            level         = level,
        )
        path = mcts_result.best_path

        # If MCTS found nothing useful, fall back to greedy
        if not path:
            path = greedy_traverse(actual_start, activation, graph, max_depth)
            mode = "greedy_fallback"

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'greedy' or 'mcts'.")

    # ── 4. Build result ──────────────────────
    texts = path_texts(path, graph)
    total = path_reward(path, graph)

    return TraversalResult(
        path         = path,
        texts        = texts,
        total_reward = total,
        mode         = mode,
        level        = level,
        elapsed      = round(time.time() - t0, 4),
        mcts_result  = mcts_result,
    )


def traverse_from_node(
    node_id    : int,
    graph      : Graph,
    normalizer : Normalizer,
    level      : Optional[Level] = None,
    mode       : str             = "mcts",
    max_depth  : int             = 50,
    max_iter   : int             = 100,
    max_hops   : int             = 6,
) -> TraversalResult:
    """
    Traverse from a known node ID.
    Convenience wrapper for traverse() with start_id set.
    Used when MCTS resumes mid-graph or when
    a specific node is the entry point.
    """
    node = graph.get_node(node_id)
    if node is None:
        return TraversalResult(
            path=[], texts=[], total_reward=0.0,
            mode=mode, level=level or Level.SENTENCE,
            elapsed=0.0,
        )

    return traverse(
        query      = node.text,
        graph      = graph,
        normalizer = normalizer,
        level      = level or node.level,
        mode       = mode,
        start_id   = node_id,
        max_depth  = max_depth,
        max_iter   = max_iter,
        max_hops   = max_hops,
    )


# ─────────────────────────────────────────────
# TWO-PHASE RESULT
# ─────────────────────────────────────────────

@dataclass
class TwoPhaseResult:
    """
    Output of a two-phase query — understanding then generation.

    understanding : phase 1 — what the graph knows about the query.
                    MCTS navigated from the query-activated subgraph.
                    This path IS the comprehension.

    generation    : phase 2 — the answer.
                    MCTS navigated from the subgraph activated by
                    the understanding path. This path IS the response.

    total_elapsed : wall clock seconds for both phases combined.
    """
    understanding : TraversalResult
    generation    : TraversalResult
    total_elapsed : float

    def summary(self) -> str:
        return (
            f"TwoPhaseResult("
            f"understanding={len(self.understanding.path)} steps "
            f"reward={self.understanding.total_reward:.3f} | "
            f"generation={len(self.generation.path)} steps "
            f"reward={self.generation.total_reward:.3f} | "
            f"elapsed={self.total_elapsed:.4f}s)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────
# TWO-PHASE TRAVERSAL
# ─────────────────────────────────────────────

def traverse_two_phase(
    query      : str,
    graph      : Graph,
    normalizer : Normalizer,
    level      : Level         = Level.SENTENCE,
    mode       : str           = "mcts",
    max_depth  : int           = 50,
    max_iter   : int           = 100,
    max_hops   : int           = 6,
) -> TwoPhaseResult:
    """
    Two-phase query: understanding then generation.

    Phase 1 — Understanding
        Activate from the raw query text.
        MCTS searches the query-shaped subgraph.
        The resulting path is the system's comprehension of
        what was asked — which nodes, in which order, are
        most relevant to the query.

    Phase 2 — Generation
        Activate from the understanding path using
        activate_from_path(). The new subgraph is shaped by
        what the system understood, not by the raw query text.
        Later nodes in the understanding path seed stronger
        because they represent where comprehension landed.
        MCTS searches this knowledge-shaped subgraph.
        The resulting path is the answer.

    The start node for phase 2 is the last node of the
    understanding path — generation continues forward from
    where comprehension ended. If the last node produces an
    empty activation (isolated node), the highest-V seed in
    the phase 2 subgraph is used instead.

    Falls back gracefully:
        - Phase 1 empty  → returns two empty TraversalResults
        - Phase 2 empty  → generation falls back to greedy from
                           the phase 2 activation seeds
    """
    t0 = time.time()

    # ── Phase 1: Understanding ───────────────
    understanding = traverse(
        query      = query,
        graph      = graph,
        normalizer = normalizer,
        level      = level,
        mode       = mode,
        max_depth  = max_depth,
        max_iter   = max_iter,
        max_hops   = max_hops,
    )

    # If phase 1 found nothing, bail — no comprehension, no generation
    if not understanding.path:
        empty = TraversalResult(
            path=[], texts=[], total_reward=0.0,
            mode=mode, level=level,
            elapsed=0.0,
        )
        return TwoPhaseResult(
            understanding = understanding,
            generation    = empty,
            total_elapsed = round(time.time() - t0, 4),
        )

    # ── Phase 2: Generation ──────────────────
    # Activate from the understanding path — broader spread
    # (activate_from_path uses max_hops + 2 by default)
    gen_activation = activate_from_path(
        path      = understanding.path,
        graph     = graph,
        threshold = 0.03,           # slightly lower — cast wider net
        max_nodes = 750,            # more nodes available for generation
    )

    if not gen_activation.lit:
        empty = TraversalResult(
            path=[], texts=[], total_reward=0.0,
            mode=mode, level=level,
            elapsed=0.0,
        )
        return TwoPhaseResult(
            understanding = understanding,
            generation    = empty,
            total_elapsed = round(time.time() - t0, 4),
        )

    # Start generation from the endpoint of comprehension
    gen_start = understanding.path[-1]

    # If the comprehension endpoint is not in the gen subgraph,
    # use the highest-V seed from the gen activation instead
    if not gen_activation.is_lit(gen_start):
        seed_vs = [
            (sid, graph.get_node(sid).V if graph.get_node(sid) else 0.0)
            for sid in gen_activation.seeds
            if graph.get_node(sid) is not None
        ]
        if seed_vs:
            gen_start = max(seed_vs, key=lambda x: x[1])[0]
        else:
            gen_start = next(iter(gen_activation.lit))

    # Run MCTS (or greedy) within the generation subgraph
    t_gen = time.time()
    gen_mcts_result = None
    gen_mode = mode

    if mode == "greedy":
        gen_path = greedy_traverse(gen_start, gen_activation, graph, max_depth)

    elif mode == "mcts":
        mcts = MCTS(graph, normalizer)
        gen_mcts_result = mcts.search(
            start_node_id = gen_start,
            activation    = gen_activation,
            max_iter      = max_iter,
            max_depth     = max_depth,
            level         = level,
        )
        gen_path = gen_mcts_result.best_path

        if not gen_path:
            gen_path = greedy_traverse(gen_start, gen_activation, graph, max_depth)
            gen_mode = "greedy_fallback"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'greedy' or 'mcts'.")

    gen_texts  = path_texts(gen_path, graph)
    gen_reward = path_reward(gen_path, graph)

    generation = TraversalResult(
        path         = gen_path,
        texts        = gen_texts,
        total_reward = gen_reward,
        mode         = gen_mode,
        level        = level,
        elapsed      = round(time.time() - t_gen, 4),
        mcts_result  = gen_mcts_result,
    )

    return TwoPhaseResult(
        understanding = understanding,
        generation    = generation,
        total_elapsed = round(time.time() - t0, 4),
    )


if __name__ == "__main__":
    import tempfile, os

    print("=== traversal.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager

        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # Two branches from node 2 — A is high reward, B is low
        sentences = [
            (0, "Detective Maria arrived at the warehouse.",              2.0),
            (1, "The building smelled of rust and old machinery.",        1.7),
            (2, "She found footprints leading to the back room.",         1.5),
            (3, "The footprints led to a locked room with a metal box.",  2.1),
            (4, "Inside the box were photographs and a folded letter.",   2.3),
            (5, "The letter named someone she recognized from the case.", 2.5),
            (6, "She noticed a broken window on the far wall.",           1.2),
            (7, "The window had been forced open from the outside.",      1.1),
            (8, "She made a note and continued searching.",               0.9),
        ]

        nodes = {}
        for idx, text, reward in sentences:
            n = graph.add_node(text, Level.SENTENCE, reward=reward)
            norm.on_node_added(n, graph)
            nodes[idx] = n

        edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(2,6),(6,7),(7,8)]
        for s, t in edges:
            src, tgt = nodes[s], nodes[t]
            graph.add_edge(src.node_id, tgt.node_id,
                           src.text, tgt.text, Level.SENTENCE)
            norm.on_edge_added(src.node_id, graph)
            bm.on_edge_added(src.node_id, tgt.node_id)

        bm.sweep()

        # ── Greedy traversal ──────────────────
        print("── Greedy traversal from node 0 ──")
        result_g = traverse_from_node(
            nodes[0].node_id, graph, norm,
            mode="greedy", max_depth=10,
        )
        print(f"  {result_g.summary()}")
        for i, (nid, text) in enumerate(zip(result_g.path, result_g.texts)):
            print(f"  [{i}] id={nid}  {text}")

        print()

        # ── MCTS traversal ────────────────────
        print("── MCTS traversal from node 0 ──")
        result_m = traverse_from_node(
            nodes[0].node_id, graph, norm,
            mode="mcts", max_depth=10, max_iter=60,
        )
        print(f"  {result_m.summary()}")
        for i, (nid, text) in enumerate(zip(result_m.path, result_m.texts)):
            print(f"  [{i}] id={nid}  {text}")

        print()

        # ── Compare paths ─────────────────────
        winner, ra, rb = compare_paths(
            result_g.path, result_m.path, graph
        )
        print(f"Path comparison:")
        print(f"  Greedy reward: {ra:.3f}")
        print(f"  MCTS reward:   {rb:.3f}")
        print(f"  Winner: {'greedy' if winner is result_g.path else 'mcts'}")

        print()

        # ── Replay ────────────────────────────
        known_path = [nodes[0].node_id, nodes[1].node_id,
                      nodes[2].node_id, nodes[3].node_id]
        print("── Replay known path ──")
        replayed = replay(known_path, graph)
        print(f"  {replayed.summary()}")
        for i, text in enumerate(replayed.texts):
            print(f"  [{i}] {text}")

        print()

        # ── Text chain output ─────────────────
        print("── Full text chain (MCTS path) ──")
        print(result_m.text_chain(separator="\n  "))

        # ── Branch validation ─────────────────
        branch_a = {nodes[k].node_id for k in [3,4,5]}
        branch_b = {nodes[k].node_id for k in [6,7,8]}
        path_set = set(result_m.path)
        print(f"\nBranch A chosen: {bool(path_set & branch_a)}")
        print(f"Branch B chosen: {bool(path_set & branch_b)}")

        graph.close()