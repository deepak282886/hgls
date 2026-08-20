"""
activation.py

Activation — the mechanism by which a query lights a neighborhood
of the graph, defining the search space for MCTS traversal.

A query arrives in any form — text, a node ID, or a set of
seed nodes. Activation spreads from the query through the graph
along edges, weighted by:

    - Embedding similarity (how similar is this node to the query)
    - Edge reward (how strong is the connection)
    - Distance decay (activation weakens with each hop)

The result is a lit subgraph — a subset of nodes with activation
values above a threshold. MCTS operates only within this subgraph.
Unlit nodes do not exist for this query.

Two spreading strategies:

    1. SIMILARITY SPREAD
       From query embedding, compute similarity to all nodes
       in a candidate pool. Nodes above threshold are lit.
       Fast but requires comparing against many embeddings.

    2. EDGE SPREAD (BFS-based)
       From seed node(s), spread activation along edges for
       N hops, decaying with distance. Naturally follows the
       causal structure of the graph.

In practice: similarity spread identifies seed nodes,
then edge spread expands the neighborhood from those seeds.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from core.atoms  import Level, EMBEDDING_DIM
from core.graph  import Graph
from core.node   import Node, similarity


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Minimum activation to include a node in the lit subgraph
ACTIVATION_THRESHOLD = 0.05

# Decay factor per hop during edge spread
# 1.0 = no decay, 0.5 = halves each hop
HOP_DECAY = 0.75

# Maximum hops for edge spread
MAX_HOPS = 6

# Maximum nodes in lit subgraph
# — prevents MCTS from working over too large a space
MAX_LIT_NODES = 500

# Temperature for activation softmax
# Higher = more uniform spread, Lower = more peaked at best matches
ACTIVATION_TEMPERATURE = 2.0


# ─────────────────────────────────────────────
# ACTIVATION RESULT
# ─────────────────────────────────────────────

@dataclass
class ActivationResult:
    """
    The lit subgraph returned after activation spreads.

    Contains:
        lit     : dict of node_id → activation_value
        seeds   : list of seed node IDs (highest similarity to query)
        query   : the query embedding used
        level   : which level was primarily activated
    """
    lit    : dict[int, float]         # node_id → activation [0,1]
    seeds  : list[int]                # starting nodes
    query  : Optional[np.ndarray]     # query embedding (may be None)
    level  : Optional[Level]          # primary level activated

    def __len__(self) -> int:
        return len(self.lit)

    def top_nodes(self, n: int = 10) -> list[tuple[int, float]]:
        """Top N most activated nodes by activation value."""
        return sorted(self.lit.items(),
                      key=lambda x: x[1], reverse=True)[:n]

    def is_lit(self, node_id: int) -> bool:
        return node_id in self.lit

    def activation(self, node_id: int) -> float:
        return self.lit.get(node_id, 0.0)

    def __repr__(self) -> str:
        return (
            f"ActivationResult("
            f"lit={len(self.lit)}, "
            f"seeds={self.seeds}, "
            f"level={self.level})"
        )


# ─────────────────────────────────────────────
# QUERY EMBEDDING
# ─────────────────────────────────────────────

def embed_query(query_text: str, level: Level) -> np.ndarray:
    """
    Compute embedding for a query string at a given level.
    Uses the same embedding function as node.py — same space,
    so similarity is meaningful.
    """
    from core.node import compute_embedding
    return compute_embedding(query_text, level)


# ─────────────────────────────────────────────
# SIMILARITY SPREAD
# Fast top-K similarity search over candidate nodes
# ─────────────────────────────────────────────

def similarity_spread(
    query_emb  : np.ndarray,
    graph      : Graph,
    level      : Optional[Level] = None,
    top_k      : int = 20,
    threshold  : float = ACTIVATION_THRESHOLD,
) -> dict[int, float]:
    """
    Compute cosine similarity between query_emb and all nodes
    at the given level (or all levels if None).

    Returns dict of node_id → activation for nodes above threshold.

    For large graphs this is the expensive step — comparing against
    every node. In a production system this would use an ANN index
    (e.g. FAISS). For now: exact search over batches from DB.
    """
    activations = {}

    levels = [level] if level is not None else list(Level)

    for lv in levels:
        nodes = graph.nodes_by_level(lv, limit=10_000)

        for node in nodes:
            # Embeddings may differ in dim across levels
            # — use the similarity function which handles this
            qe = query_emb
            ne = node.embedding

            # Align dimensions
            if qe.shape != ne.shape:
                max_dim = max(len(qe), len(ne))
                tmp_q   = np.zeros(max_dim, dtype=np.float32)
                tmp_n   = np.zeros(max_dim, dtype=np.float32)
                tmp_q[:len(qe)] = qe
                tmp_n[:len(ne)] = ne
                sim = float(np.dot(tmp_q, tmp_n))
            else:
                sim = float(np.dot(qe, ne))

            sim = max(-1.0, min(1.0, sim))

            # Map similarity [-1,1] → activation [0,1]
            activation = (sim + 1.0) / 2.0

            if activation >= threshold:
                activations[node.node_id] = activation

    # Keep top_k by activation value
    if len(activations) > top_k:
        sorted_items = sorted(activations.items(),
                              key=lambda x: x[1], reverse=True)
        activations  = dict(sorted_items[:top_k])

    return activations


# ─────────────────────────────────────────────
# EDGE SPREAD
# BFS from seed nodes along edges with decay
# ─────────────────────────────────────────────

def edge_spread(
    seeds      : list[int],
    graph      : Graph,
    max_hops   : int   = MAX_HOPS,
    decay      : float = HOP_DECAY,
    threshold  : float = ACTIVATION_THRESHOLD,
    initial_activations: Optional[dict[int, float]] = None,
) -> dict[int, float]:
    """
    Spread activation from seed nodes along edges (BFS).

    At each hop, activation is:
        a(neighbor) = max(a(neighbor),
                          a(current) * decay * edge_weight)

    Edge weight is the normalized edge reward mapped to [0,1].
    Negative edges transmit less activation — they are weak links.

    Spreads both forward (outgoing) and backward (incoming)
    so the lit subgraph includes both what follows and what
    precedes the seeds — full causal neighborhood.

    Returns dict of node_id → activation.
    """
    activations = dict(initial_activations or {})

    # Initialize seeds at full activation if not already set
    for seed_id in seeds:
        if seed_id not in activations:
            activations[seed_id] = 1.0
        else:
            activations[seed_id] = max(activations[seed_id], 1.0)

    # BFS queue: (node_id, current_activation, hop_count)
    queue   = deque([(sid, activations[sid], 0) for sid in seeds])
    visited = set()

    while queue:
        node_id, current_act, hop = queue.popleft()

        if node_id in visited:
            continue
        visited.add(node_id)

        if hop >= max_hops:
            continue

        next_act = current_act * decay

        if next_act < threshold:
            continue

        # Spread forward along outgoing edges
        out_edges = graph.get_outgoing_edges(node_id)
        for edge in out_edges:
            tid = edge.target_id

            # Edge weight: map reward to [0.1, 1.0]
            # Negative edges still transmit some activation (dim connection)
            # but less than positive edges
            scale      = ACTIVATION_TEMPERATURE
            raw_reward = edge.reward
            # Sigmoid-like mapping: heavily negative → 0.1, positive → 1.0
            edge_weight = 0.1 + 0.9 * (1.0 / (1.0 + np.exp(-raw_reward / scale)))
            neighbor_act = next_act * edge_weight

            if neighbor_act >= threshold:
                if tid not in activations or activations[tid] < neighbor_act:
                    activations[tid] = neighbor_act
                    if tid not in visited:
                        queue.append((tid, neighbor_act, hop + 1))

        # Spread backward along incoming edges (causal ancestors)
        in_edges = graph.get_incoming_edges(node_id)
        for edge in in_edges:
            sid = edge.source_id

            scale        = ACTIVATION_TEMPERATURE
            edge_weight  = 0.1 + 0.9 * (1.0 / (1.0 + np.exp(-edge.reward / scale)))
            neighbor_act = next_act * edge_weight * 0.7  # backward spread weaker

            if neighbor_act >= threshold:
                if sid not in activations or activations[sid] < neighbor_act:
                    activations[sid] = neighbor_act
                    if sid not in visited:
                        queue.append((sid, neighbor_act, hop + 1))

    return activations


# ─────────────────────────────────────────────
# NORMALIZE ACTIVATION
# ─────────────────────────────────────────────

def normalize_activation(
    activations : dict[int, float],
    max_nodes   : int = MAX_LIT_NODES,
) -> dict[int, float]:
    """
    Trim and normalize activation values so they sum to 1.0.
    Keeps top max_nodes nodes by activation strength.

    Normalization conserves the activation budget —
    MCTS knows the total signal is 1.0 regardless of
    how many nodes are lit.
    """
    if not activations:
        return {}

    # Trim to max_nodes
    if len(activations) > max_nodes:
        items       = sorted(activations.items(),
                             key=lambda x: x[1], reverse=True)
        activations = dict(items[:max_nodes])

    # Normalize to sum to 1.0
    total = sum(activations.values())
    if total < 1e-9:
        return activations

    return {k: v / total for k, v in activations.items()}


# ─────────────────────────────────────────────
# ACTIVATE — main entry point
# ─────────────────────────────────────────────

def activate(
    query      : str,
    graph      : Graph,
    level      : Optional[Level] = None,
    top_k_seeds: int   = 10,
    max_hops   : int   = MAX_HOPS,
    threshold  : float = ACTIVATION_THRESHOLD,
    max_nodes  : int   = MAX_LIT_NODES,
) -> ActivationResult:
    """
    Main activation entry point.

    Given a query string:
    1. Embed the query at the given level (or all levels)
    2. Find top-K seed nodes by similarity
    3. Spread activation from seeds along edges
    4. Trim and normalize the lit subgraph
    5. Return ActivationResult

    The level parameter controls which level of the hierarchy
    is primarily activated:
        None     → search all levels, let similarity decide
        SENTENCE → activate at sentence level (most common)
        WORD     → activate at word level (keyword queries)
        etc.
    """
    if not query.strip():
        return ActivationResult(lit={}, seeds=[], query=None, level=level)

    # Determine primary level for embedding
    primary_level = level if level is not None else Level.SENTENCE

    # 1. Embed query
    query_emb = embed_query(query, primary_level)

    # 2. Similarity spread — find seed nodes
    sim_acts = similarity_spread(
        query_emb, graph,
        level     = level,
        top_k     = top_k_seeds,
        threshold = threshold,
    )

    if not sim_acts:
        # No similar nodes found — return empty
        return ActivationResult(
            lit=[], seeds=[], query=query_emb, level=primary_level
        )

    seed_ids = list(sim_acts.keys())

    # 3. Edge spread from seeds
    full_acts = edge_spread(
        seeds      = seed_ids,
        graph      = graph,
        max_hops   = max_hops,
        threshold  = threshold,
        initial_activations = sim_acts,
    )

    # 4. Trim and normalize
    norm_acts = normalize_activation(full_acts, max_nodes=max_nodes)

    return ActivationResult(
        lit   = norm_acts,
        seeds = seed_ids,
        query = query_emb,
        level = primary_level,
    )


def activate_from_node(
    node_id    : int,
    graph      : Graph,
    max_hops   : int   = MAX_HOPS,
    threshold  : float = ACTIVATION_THRESHOLD,
    max_nodes  : int   = MAX_LIT_NODES,
) -> ActivationResult:
    """
    Activate from a known node ID rather than a text query.
    Used when MCTS is resuming from a mid-graph position
    or when a specific node is the query.
    """
    node = graph.get_node(node_id)
    if node is None:
        return ActivationResult(lit={}, seeds=[], query=None, level=None)

    # Start at full activation for the given node
    initial = {node_id: 1.0}

    full_acts = edge_spread(
        seeds       = [node_id],
        graph       = graph,
        max_hops    = max_hops,
        threshold   = threshold,
        initial_activations = initial,
    )

    norm_acts = normalize_activation(full_acts, max_nodes=max_nodes)

    return ActivationResult(
        lit   = norm_acts,
        seeds = [node_id],
        query = node.embedding,
        level = node.level,
    )


if __name__ == "__main__":
    import tempfile, os

    print("=== activation.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager

        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # Build two separate narrative chains
        detective = [
            "Detective Maria arrived at the abandoned warehouse at midnight.",
            "The building smelled of rust and old machinery.",
            "She found a trail of footprints leading to the back room.",
            "The footprints were fresh, made within the last hour.",
            "In the corner she discovered a locked metal box.",
        ]

        cooking = [
            "Heat the olive oil in a large pan over medium heat.",
            "Add the diced onions and cook until they are translucent.",
            "Stir in the garlic and cook for another minute.",
            "Pour in the tomato sauce and season with salt and pepper.",
            "Simmer the sauce for twenty minutes before serving.",
        ]

        all_nodes = []
        for text in detective + cooking:
            n = graph.add_node(text, Level.SENTENCE, reward=1.5)
            norm.on_node_added(n, graph)
            all_nodes.append(n)

        # Chain detective nodes
        det_nodes = all_nodes[:5]
        for i in range(len(det_nodes)-1):
            graph.add_edge(det_nodes[i].node_id, det_nodes[i+1].node_id,
                           det_nodes[i].text, det_nodes[i+1].text,
                           Level.SENTENCE)
            norm.on_edge_added(det_nodes[i].node_id, graph)
            bm.on_edge_added(det_nodes[i].node_id, det_nodes[i+1].node_id)

        # Chain cooking nodes
        cook_nodes = all_nodes[5:]
        for i in range(len(cook_nodes)-1):
            graph.add_edge(cook_nodes[i].node_id, cook_nodes[i+1].node_id,
                           cook_nodes[i].text, cook_nodes[i+1].text,
                           Level.SENTENCE)
            norm.on_edge_added(cook_nodes[i].node_id, graph)
            bm.on_edge_added(cook_nodes[i].node_id, cook_nodes[i+1].node_id)

        bm.sweep()

        print(f"Graph: {graph.stats()['node_count']} nodes, "
              f"{graph.stats()['edge_count']} edges\n")

        # ── Query 1: detective topic ──
        query1 = "footprints warehouse detective mystery"
        print(f"Query 1: '{query1}'")
        result1 = activate(query1, graph, level=Level.SENTENCE)
        print(f"  Lit nodes: {len(result1)}")
        print(f"  Seeds: {result1.seeds}")
        print(f"  Top activated nodes:")
        for nid, act in result1.top_nodes(5):
            node = graph.get_node(nid)
            print(f"    [{nid:02d}] act={act:.4f}  {node.text[:55]}")

        print()

        # ── Query 2: cooking topic ──
        query2 = "olive oil garlic sauce cooking pan"
        print(f"Query 2: '{query2}'")
        result2 = activate(query2, graph, level=Level.SENTENCE)
        print(f"  Lit nodes: {len(result2)}")
        print(f"  Seeds: {result2.seeds}")
        print(f"  Top activated nodes:")
        for nid, act in result2.top_nodes(5):
            node = graph.get_node(nid)
            print(f"    [{nid:02d}] act={act:.4f}  {node.text[:55]}")

        print()

        # ── Activate from specific node ──
        print(f"Activate from node 2 (footprints):")
        result3 = activate_from_node(det_nodes[2].node_id, graph, max_hops=3)
        print(f"  Lit nodes: {len(result3)}")
        print(f"  Top activated nodes:")
        for nid, act in result3.top_nodes(5):
            node = graph.get_node(nid)
            print(f"    [{nid:02d}] act={act:.4f}  {node.text[:55]}")

        # ── Verify domain separation ──
        print()
        det_ids  = {n.node_id for n in det_nodes}
        cook_ids = {n.node_id for n in cook_nodes}

        q1_lit = set(result1.lit.keys())
        q2_lit = set(result2.lit.keys())

        det_in_q1  = len(q1_lit & det_ids)
        cook_in_q1 = len(q1_lit & cook_ids)
        det_in_q2  = len(q2_lit & det_ids)
        cook_in_q2 = len(q2_lit & cook_ids)

        print(f"Domain separation:")
        print(f"  Detective query → detective nodes: {det_in_q1}/5  "
              f"cooking nodes: {cook_in_q1}/5")
        print(f"  Cooking query   → detective nodes: {det_in_q2}/5  "
              f"cooking nodes: {cook_in_q2}/5")

        graph.close()