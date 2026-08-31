"""
activation.py

Activation — the mechanism by which a query lights a neighborhood
of the graph, defining the search space for MCTS traversal.

A query arrives as text. Activation works in two stages:

    1. KEYWORD SEEDING
       Extract meaningful keywords from the query.
       Search the graph for nodes at every level whose text
       contains those keywords, ordered by V value.
       The graph's own value landscape picks the best entry points.
       AXIOM nodes are searched first — they are foundational
       and connect to the most downstream content.

    2. EDGE SPREAD (BFS-based)
       From seed nodes, spread activation along edges for
       N hops, decaying with distance. Naturally follows the
       causal structure of the graph.

This is correct for the architecture:
    The query activates nodes that literally contain the query's
    concepts at every level of the hierarchy. AXIOM nodes for
    the relevant domain light up first. Edge spread then lights
    everything connected to those axioms. MCTS navigates within
    this subgraph, sequencing axioms toward the answer.

    No embedding similarity needed — the graph's edge structure
    and V values already encode what is relevant to what.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from core.atoms  import Level, EMBEDDING_DIM
from core.graph  import Graph
from core.node   import Node


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

def _extract_keywords(query: str, max_keywords: int = 8) -> list[str]:
    """
    Extract meaningful keywords from a query string.
    Filters stop words. Returns longer words first — more distinctive.
    """
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'that', 'this', 'it', 'its', 'or', 'and', 'but', 'not',
        'if', 'then', 'so', 'as', 'up', 'out', 'about', 'into',
        'than', 'we', 'i', 'you', 'he', 'she', 'they', 'what',
        'which', 'who', 'how', 'when', 'where', 'why', 'show',
        'prove', 'find', 'let', 'give', 'take', 'make', 'get',
    }
    tokens   = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = [t for t in tokens if t not in STOP_WORDS]
    keywords = sorted(set(keywords), key=len, reverse=True)
    return keywords[:max_keywords]


# Search priority per level — AXIOM first since it is
# the most load-bearing level. SENTENCE next. WORD for
# exact single-word matches. CHARACTER not useful for
# keyword search (single chars match everything).
_SEARCH_LEVELS = [
    (Level.AXIOM,    2.0),   # (level, priority_weight)
    (Level.SENTENCE, 1.0),
    (Level.WORD,     0.5),
]


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
    1. Extract keywords from the query
    2. Search all meaningful levels for nodes containing keywords
       — AXIOM nodes have 2× priority weight because they are
         foundational and connect to the most downstream content
       — SENTENCE nodes next
       — WORD nodes for exact word matches
    3. Weight seeds by: level_priority × keyword_rank × node_V
    4. Spread activation from seeds along edges
    5. Trim and normalize the lit subgraph

    With an axiom-structured graph:
    The relevant axioms for the query domain light up first.
    Edge spread then lights everything connected to those axioms.
    MCTS navigates this subgraph, sequencing axioms toward the answer.
    Understanding and generation happen in one traversal.
    """
    if not query.strip():
        return ActivationResult(lit={}, seeds=[], query=None, level=level)

    primary_level = level if level is not None else Level.SENTENCE

    # 1. Extract keywords
    keywords = _extract_keywords(query)
    if not keywords:
        return ActivationResult(lit={}, seeds=[], query=None, level=primary_level)

    # 2. Search all meaningful levels for seeds
    seed_activations: dict[int, float] = {}
    seeds_per_keyword = max(2, top_k_seeds // max(1, len(keywords)))

    # If a specific level was requested, only search that level
    # (plus AXIOM which is always searched)
    if level is not None and level != Level.AXIOM:
        search_levels = [(Level.AXIOM, 2.0), (level, 1.0)]
    else:
        search_levels = _SEARCH_LEVELS

    for rank, keyword in enumerate(keywords):
        keyword_weight = 1.0 / (1.0 + rank * 0.2)

        for search_level, level_priority in search_levels:
            matches = graph.search_nodes_by_keyword(
                keyword, search_level,
                limit = seeds_per_keyword,
            )
            for node in matches:
                nid      = node.node_id
                v_weight = max(0.1, (node.norm_V + 1.0) / 2.0)
                act      = keyword_weight * level_priority * v_weight

                if nid in seed_activations:
                    # Node matched multiple keywords or levels — boost
                    seed_activations[nid] = min(2.0, seed_activations[nid] + act * 0.5)
                else:
                    seed_activations[nid] = act

    if not seed_activations:
        return ActivationResult(
            lit={}, seeds=[], query=None, level=primary_level
        )

    # Normalize seed activations to [0, 1]
    max_seed = max(seed_activations.values())
    if max_seed > 1e-9:
        seed_activations = {k: v / max_seed for k, v in seed_activations.items()}

    seed_ids = list(seed_activations.keys())

    # 3. Edge spread from seeds
    full_acts = edge_spread(
        seeds               = seed_ids,
        graph               = graph,
        max_hops            = max_hops,
        threshold           = threshold,
        initial_activations = seed_activations,
    )

    # 4. Trim and normalize
    norm_acts = normalize_activation(full_acts, max_nodes=max_nodes)

    return ActivationResult(
        lit   = norm_acts,
        seeds = seed_ids,
        query = None,
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