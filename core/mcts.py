"""
mcts.py

Monte Carlo Tree Search over the lit subgraph.

MCTS is the thinking engine. It operates entirely on the
numerical graph — node IDs, weights, rewards, edges.
No symbols. No domain knowledge. Pure reward navigation.

Four operations per iteration:

    1. SELECTION
       Descend the search tree from root using UCB score.
       UCB = norm_V + c * sqrt(ln(parent_visits) / visits) * activation
       Stay within the lit subgraph — dark nodes do not exist.

    2. EXPANSION
       At a leaf node, generate candidate next nodes:
           a. Existing graph neighbors (already discovered)
           b. Generated candidates via transition function (new)
       Score each candidate with the reward function.
       Keep only positive-reward candidates.

    3. SIMULATION (rollout)
       From the expanded node, roll forward greedily using
       Bellman V values until terminal or max depth.
       Accumulate reward. This is the fast path estimate.

    4. BACKPROPAGATION
       Propagate the simulation reward back up the search tree.
       Update visit counts and total reward at each ancestor.

Dynamic depth:
    Depth is not fixed. MCTS monitors reward variance across
    branches. When one branch dominates clearly, it commits
    early. When uncertain, it goes deeper. This is the
    "thinking harder on hard problems" property.

The search tree is temporary — built during a query and
discarded after. The permanent graph is not modified during
MCTS (unless a new high-reward candidate is found and
the ingestion pipeline decides to add it).
"""

import numpy as np
import time
from dataclasses   import dataclass, field
from typing        import Optional
from collections   import defaultdict

from core.atoms      import Level
from core.graph      import Graph
from core.node       import Node
from core.edge       import EdgeType, transition
from core.reward     import node_reward, edge_reward, path_reward
from core.activation import ActivationResult
from core.normalizer import Normalizer


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# UCB exploration constant
# sqrt(2) ≈ 1.414 is the theoretical default
C_EXPLORE = 1.414

# Maximum rollout depth — hard ceiling
MAX_ROLLOUT_DEPTH = 50

# Minimum rollout depth — always think at least this far
MIN_ROLLOUT_DEPTH = 3

# Early commitment threshold
# When best branch UCB exceeds second best by this margin,
# commit without more rollouts
COMMIT_MARGIN = 2.0

# Minimum rollout iterations before dynamic depth kicks in
MIN_ITERATIONS = 5

# Maximum total MCTS iterations per query
MAX_ITERATIONS = 200

# Reward discount per step in rollout
# Rewards further in the future are worth slightly less
DISCOUNT = 0.95

# Minimum reward to keep a candidate during expansion
EXPANSION_REWARD_THRESHOLD = 0.0


# ─────────────────────────────────────────────
# MCTS TREE NODE
# Separate from graph Node — temporary search state
# ─────────────────────────────────────────────

@dataclass
class TreeNode:
    """
    A node in the MCTS search tree.
    Corresponds to a node in the permanent graph (via node_id)
    but holds temporary search statistics.
    """
    node_id     : int
    parent      : Optional["TreeNode"] = field(default=None, repr=False)
    children    : list["TreeNode"]     = field(default_factory=list, repr=False)

    visits      : int   = 0
    total_reward: float = 0.0
    is_expanded : bool  = False
    is_terminal : bool  = False

    # Cache of graph node properties for fast access
    _norm_V     : float = 0.0
    _activation : float = 0.0
    _reward     : float = 0.0

    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb(self, parent_visits: int,
            c: float = C_EXPLORE) -> float:
        """
        UCB1 score weighted by normalized V and activation.

        UCB = mean_reward
            + c * sqrt(ln(parent_visits + 1) / (visits + 1))
            + norm_V_bonus
            + activation_bonus
        """
        if self.visits == 0:
            # Unvisited — very high UCB to ensure exploration
            return float("inf")

        exploit = self.mean_reward()
        explore = c * np.sqrt(
            np.log(parent_visits + 1) / (self.visits + 1)
        )
        v_bonus  = self._norm_V   * 0.3
        act_bonus = self._activation * 0.5

        return exploit + explore + v_bonus + act_bonus

    def best_child(self, c: float = C_EXPLORE) -> Optional["TreeNode"]:
        """Select child with highest UCB score."""
        if not self.children:
            return None
        return max(self.children, key=lambda ch: ch.ucb(self.visits, c))

    def best_child_greedy(self) -> Optional["TreeNode"]:
        """Select child with highest mean reward (no exploration)."""
        if not self.children:
            return None
        return max(self.children, key=lambda ch: ch.mean_reward())

    def is_leaf(self) -> bool:
        return not self.children or not self.is_expanded

    def __repr__(self) -> str:
        return (
            f"TreeNode(id={self.node_id}, "
            f"visits={self.visits}, "
            f"mean_r={self.mean_reward():.3f}, "
            f"children={len(self.children)})"
        )


# ─────────────────────────────────────────────
# MCTS RESULT
# ─────────────────────────────────────────────

@dataclass
class MCTSResult:
    """
    Result of an MCTS search.

    best_path   : list of node IDs from start to terminal
    total_reward: cumulative reward along best path
    iterations  : number of MCTS iterations run
    depth       : depth of best path found
    elapsed     : wall clock seconds
    committed_early: True if dynamic depth triggered early commitment
    """
    best_path      : list[int]
    total_reward   : float
    iterations     : int
    depth          : int
    elapsed        : float
    committed_early: bool = False
    confidence     : float = 0.0   # how dominant was the best branch

    def __repr__(self) -> str:
        return (
            f"MCTSResult("
            f"path={self.best_path}, "
            f"reward={self.total_reward:.3f}, "
            f"iters={self.iterations}, "
            f"depth={self.depth}, "
            f"committed_early={self.committed_early})"
        )


# ─────────────────────────────────────────────
# MCTS ENGINE
# ─────────────────────────────────────────────

class MCTS:
    """
    Monte Carlo Tree Search engine.

    Usage:
        mcts   = MCTS(graph, normalizer)
        result = mcts.search(
            start_node_id = 0,
            activation    = activation_result,
        )
    """

    def __init__(
        self,
        graph      : Graph,
        normalizer : Normalizer,
        c_explore  : float = C_EXPLORE,
        discount   : float = DISCOUNT,
    ):
        self.graph      = graph
        self.normalizer = normalizer
        self.c_explore  = c_explore
        self.discount   = discount


    # ─────────────────────────────────────────
    # MAIN SEARCH
    # ─────────────────────────────────────────

    def search(
        self,
        start_node_id : int,
        activation    : ActivationResult,
        max_iter      : int = MAX_ITERATIONS,
        max_depth     : int = MAX_ROLLOUT_DEPTH,
        level         : Optional[Level] = None,
    ) -> MCTSResult:
        """
        Run MCTS from start_node_id within the lit subgraph.

        Returns MCTSResult with best path found.
        """
        t0 = time.time()

        start_node = self.graph.get_node(start_node_id)
        if start_node is None:
            return MCTSResult(
                best_path=[], total_reward=0.0,
                iterations=0, depth=0, elapsed=0.0
            )

        # Use start node's level if not specified
        if level is None:
            level = start_node.level

        # Build root of search tree
        root = self._make_tree_node(start_node_id, activation)

        committed_early = False
        confidence      = 0.0
        iteration       = 0

        while iteration < max_iter:
            iteration += 1

            # ── 1. SELECTION ──────────────────────
            leaf = self._select(root, activation)

            # ── 2. EXPANSION ──────────────────────
            if not leaf.is_terminal and not leaf.is_expanded:
                self._expand(leaf, activation, level, max_depth)

            # ── 3. SIMULATION ─────────────────────
            sim_reward = self._simulate(leaf, activation, level, max_depth)

            # ── 4. BACKPROPAGATION ────────────────
            self._backpropagate(leaf, sim_reward)

            # ── DYNAMIC DEPTH CHECK ───────────────
            if iteration >= MIN_ITERATIONS and root.children:
                confidence, should_commit = self._check_commitment(root)
                if should_commit:
                    committed_early = True
                    break

        # Extract best path from root
        best_path    = self._extract_best_path(root, activation, max_depth)
        total_reward = self._path_total_reward(best_path, level)
        elapsed      = time.time() - t0

        return MCTSResult(
            best_path       = best_path,
            total_reward    = total_reward,
            iterations      = iteration,
            depth           = len(best_path),
            elapsed         = round(elapsed, 4),
            committed_early = committed_early,
            confidence      = round(confidence, 4),
        )


    # ─────────────────────────────────────────
    # SELECTION — descend tree with UCB
    # ─────────────────────────────────────────

    def _select(self, root: TreeNode,
                activation: ActivationResult) -> TreeNode:
        """
        Descend from root to a leaf using UCB.
        Only follow children that are in the lit subgraph.
        """
        node = root

        while not node.is_leaf() and not node.is_terminal:
            # Filter children to lit subgraph
            lit_children = [
                ch for ch in node.children
                if activation.is_lit(ch.node_id)
            ]

            if not lit_children:
                break

            node = max(lit_children,
                       key=lambda ch: ch.ucb(node.visits, self.c_explore))

        return node


    # ─────────────────────────────────────────
    # EXPANSION — generate children
    # ─────────────────────────────────────────

    def _expand(
        self,
        tree_node  : TreeNode,
        activation : ActivationResult,
        level      : Level,
        max_depth  : int,
    ):
        """
        Generate child nodes for a leaf.

        Sources:
            a. Existing graph edges (already discovered neighbors)
            b. Transition function candidates (generated)

        Only adds children that are lit (in activation subgraph)
        and have positive edge reward.
        """
        graph_node = self.graph.get_node(tree_node.node_id)
        if graph_node is None:
            tree_node.is_terminal = True
            tree_node.is_expanded = True
            return

        # ── a. Existing graph neighbors ───────
        out_edges = self.graph.get_outgoing_edges(tree_node.node_id)
        added_ids = set()

        for edge in out_edges:
            tid = edge.target_id

            # Must be lit
            if not activation.is_lit(tid):
                continue

            # Positive edge reward only
            if edge.reward <= EXPANSION_REWARD_THRESHOLD:
                continue

            if tid in added_ids:
                continue

            target_node = self.graph.get_node(tid)
            if target_node is None:
                continue

            child = self._make_tree_node(tid, activation, parent=tree_node)
            tree_node.children.append(child)
            added_ids.add(tid)

        # ── b. Transition function candidates ─
        # Only generate new candidates if few neighbors found
        if len(added_ids) < 3:
            context = self._get_context(tree_node, level)
            candidates = transition(graph_node.text, level, context)

            for cand_text in candidates:
                # Check if this candidate already exists in graph
                existing_id = self.graph.text_exists(cand_text, level)

                if existing_id is not None:
                    if existing_id not in added_ids and activation.is_lit(existing_id):
                        child = self._make_tree_node(
                            existing_id, activation, parent=tree_node
                        )
                        tree_node.children.append(child)
                        added_ids.add(existing_id)
                else:
                    # New candidate — score it
                    er = edge_reward(graph_node.text, cand_text, level)
                    if er > EXPANSION_REWARD_THRESHOLD:
                        # Create a temporary virtual node ID
                        # (negative = not in permanent graph)
                        virt_id   = -(hash(cand_text) % 10_000_000)
                        virt_child = TreeNode(
                            node_id     = virt_id,
                            parent      = tree_node,
                            _reward     = node_reward(cand_text, level),
                            _norm_V     = 0.0,
                            _activation = activation.activation(
                                tree_node.node_id
                            ) * 0.5,  # inherit partial activation
                        )
                        tree_node.children.append(virt_child)

        # Terminal if no children could be added
        if not tree_node.children:
            tree_node.is_terminal = True

        tree_node.is_expanded = True


    # ─────────────────────────────────────────
    # SIMULATION — fast rollout
    # ─────────────────────────────────────────

    def _simulate(
        self,
        tree_node  : TreeNode,
        activation : ActivationResult,
        level      : Level,
        max_depth  : int,
    ) -> float:
        """
        Fast greedy rollout from tree_node using Bellman V values.
        Stays within lit subgraph. Applies discount per step.

        Returns cumulative discounted reward.
        """
        current_id  = tree_node.node_id
        total       = 0.0
        depth       = 0
        discount    = 1.0
        visited     = {current_id}

        while depth < max_depth:
            # Virtual nodes (not in graph) — use cached reward only
            if current_id < 0:
                total += tree_node._reward * discount
                break

            node = self.graph.get_node(current_id)
            if node is None:
                break

            total += node.reward * discount
            discount *= self.discount

            # Get best lit outgoing edge
            out_edges = self.graph.get_outgoing_edges(current_id)
            lit_edges = [
                e for e in out_edges
                if activation.is_lit(e.target_id)
                and e.target_id not in visited
                and e.reward > 0
            ]

            if not lit_edges:
                # Terminal or no lit neighbors — rollout ends
                break

            # Greedy: pick highest (edge_reward + V)
            best_edge = max(
                lit_edges,
                key=lambda e: (
                    e.reward
                    + (self.graph.get_node(e.target_id).V
                       if self.graph.get_node(e.target_id) else 0.0)
                )
            )

            total += best_edge.reward * discount
            discount *= self.discount

            visited.add(best_edge.target_id)
            current_id = best_edge.target_id
            depth += 1

        return total


    # ─────────────────────────────────────────
    # BACKPROPAGATION
    # ─────────────────────────────────────────

    def _backpropagate(self, tree_node: TreeNode, reward: float):
        """
        Propagate simulation reward back up the tree.
        Updates visit count and total reward at each ancestor.
        """
        node = tree_node
        while node is not None:
            node.visits       += 1
            node.total_reward += reward
            node = node.parent


    # ─────────────────────────────────────────
    # DYNAMIC DEPTH — commitment check
    # ─────────────────────────────────────────

    def _check_commitment(
        self, root: TreeNode
    ) -> tuple[float, bool]:
        """
        Check whether MCTS should commit early.

        Commits when the best child's UCB exceeds the second
        best by COMMIT_MARGIN — one branch is clearly dominant.

        Returns (confidence, should_commit).
        """
        lit_children = [ch for ch in root.children if ch.visits > 0]

        if len(lit_children) < 2:
            return 0.0, False

        scores = sorted(
            [ch.ucb(root.visits, self.c_explore) for ch in lit_children],
            reverse=True
        )

        best   = scores[0]
        second = scores[1]

        if second == 0.0:
            return 1.0, True

        confidence = (best - second) / (abs(second) + 1e-9)
        return confidence, confidence >= COMMIT_MARGIN


    # ─────────────────────────────────────────
    # EXTRACT BEST PATH
    # ─────────────────────────────────────────

    def _extract_best_path(
        self,
        root       : TreeNode,
        activation : ActivationResult,
        max_depth  : int,
    ) -> list[int]:
        """
        Extract the best path from root by following
        the highest mean-reward child at each step.

        Returns list of node IDs (excludes virtual negative IDs).
        """
        path    = []
        current = root
        visited = set()
        depth   = 0

        while current is not None and depth < max_depth:
            nid = current.node_id

            # Skip virtual nodes in the final path
            if nid >= 0:
                if nid in visited:
                    break
                path.append(nid)
                visited.add(nid)

            if not current.children or current.is_terminal:
                break

            # Among visited children, pick highest mean reward
            # Fallback to Bellman V for unvisited children
            best = None
            best_score = -np.inf

            for ch in current.children:
                if ch.node_id < 0:
                    continue  # skip virtual
                if ch.node_id in visited:
                    continue
                if not activation.is_lit(ch.node_id):
                    continue

                if ch.visits > 0:
                    score = ch.mean_reward()
                else:
                    # Unvisited — use Bellman V as proxy
                    gn = self.graph.get_node(ch.node_id)
                    score = gn.V if gn else -np.inf

                if score > best_score:
                    best_score = score
                    best = ch

            if best is None:
                break

            current = best
            depth  += 1

        # If path is just the root, try extending via graph Bellman
        if len(path) <= 1:
            path = self._bellman_extend(path, activation, max_depth)

        return path


    def _bellman_extend(
        self,
        path       : list[int],
        activation : ActivationResult,
        max_depth  : int,
    ) -> list[int]:
        """
        Extend a short path using greedy Bellman traversal.
        Used as fallback when MCTS tree is shallow.
        """
        if not path:
            return path

        current_id = path[-1]
        visited    = set(path)

        while len(path) < max_depth:
            out_edges = self.graph.get_outgoing_edges(current_id)
            lit_edges = [
                e for e in out_edges
                if activation.is_lit(e.target_id)
                and e.target_id not in visited
            ]

            if not lit_edges:
                break

            best_edge = max(
                lit_edges,
                key=lambda e: (
                    e.reward
                    + (self.graph.get_node(e.target_id).V
                       if self.graph.get_node(e.target_id) else 0.0)
                )
            )

            path.append(best_edge.target_id)
            visited.add(best_edge.target_id)
            current_id = best_edge.target_id

        return path


    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _make_tree_node(
        self,
        node_id    : int,
        activation : ActivationResult,
        parent     : Optional[TreeNode] = None,
    ) -> TreeNode:
        """Create a TreeNode with cached graph properties."""
        gn  = self.graph.get_node(node_id)
        act = activation.activation(node_id)

        return TreeNode(
            node_id     = node_id,
            parent      = parent,
            _norm_V     = gn.norm_V    if gn else 0.0,
            _activation = act,
            _reward     = gn.reward    if gn else 0.0,
            is_terminal = gn.is_terminal if gn else True,
        )

    def _get_context(
        self,
        tree_node : TreeNode,
        level     : Level,
    ) -> list[str]:
        """
        Get text context from ancestors for transition function.
        Walk up the tree to collect preceding node texts.
        """
        context = []
        node    = tree_node.parent
        while node is not None and len(context) < 5:
            if node.node_id >= 0:
                gn = self.graph.get_node(node.node_id)
                if gn and gn.level == level:
                    context.insert(0, gn.text)
            node = node.parent
        return context

    def _path_total_reward(
        self,
        path  : list[int],
        level : Level,
    ) -> float:
        """Sum node + edge rewards along a path."""
        if not path:
            return 0.0
        total = 0.0
        for i, nid in enumerate(path):
            gn = self.graph.get_node(nid)
            if gn:
                total += gn.reward
            if i + 1 < len(path):
                edges = self.graph.get_outgoing_edges(nid)
                for e in edges:
                    if e.target_id == path[i+1]:
                        total += e.reward
                        break
        return round(total, 4)


if __name__ == "__main__":
    import tempfile, os

    print("=== mcts.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager
        from core.activation import activate_from_node

        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # Build a branching graph — two possible continuations at node 2
        sentences = [
            (0,  "Detective Maria arrived at the warehouse.",              2.0),
            (1,  "The building smelled of rust and old machinery.",        1.7),
            (2,  "She found footprints leading to the back room.",         1.5),
            # Branch A — high reward continuation
            (3,  "The footprints led to a locked room with a metal box.",  2.1),
            (4,  "Inside the box were photographs and a folded letter.",   2.3),
            (5,  "The letter named someone she recognized from the case.", 2.5),
            # Branch B — lower reward continuation
            (6,  "She noticed a broken window on the far wall.",           1.2),
            (7,  "The window had been forced open from outside.",          1.1),
            (8,  "She made a note and continued searching.",               0.9),
        ]

        nodes = {}
        for idx, text, reward in sentences:
            n = graph.add_node(text, Level.SENTENCE, reward=reward)
            norm.on_node_added(n, graph)
            nodes[idx] = n

        # Linear chain 0→1→2
        for i in range(2):
            s, t = nodes[i], nodes[i+1]
            graph.add_edge(s.node_id, t.node_id, s.text, t.text, Level.SENTENCE)
            norm.on_edge_added(s.node_id, graph)
            bm.on_edge_added(s.node_id, t.node_id)

        # Branch A: 2→3→4→5
        for i in [2, 3, 4]:
            s, t = nodes[i], nodes[i+1]
            graph.add_edge(s.node_id, t.node_id, s.text, t.text, Level.SENTENCE)
            norm.on_edge_added(s.node_id, graph)
            bm.on_edge_added(s.node_id, t.node_id)

        # Branch B: 2→6→7→8
        for i in [2, 6, 7]:
            s, t = nodes[i], nodes[i+1]
            graph.add_edge(s.node_id, t.node_id, s.text, t.text, Level.SENTENCE)
            norm.on_edge_added(s.node_id, graph)
            bm.on_edge_added(s.node_id, t.node_id)

        # Full sweep
        stats = bm.sweep(verbose=False)
        print(f"Graph: {graph.stats()['node_count']} nodes, "
              f"{graph.stats()['edge_count']} edges")
        print(f"Bellman converged: {stats['converged']}\n")

        # V values — branch A should be higher
        print("V values:")
        for idx, n in nodes.items():
            fresh = graph.get_node(n.node_id)
            branch = "A" if idx in [3,4,5] else ("B" if idx in [6,7,8] else " ")
            bar = "█" * int(max(0, fresh.V) / 2)
            print(f"  [{idx}]{branch} V={fresh.V:+.3f}  {bar}  "
                  f"{fresh.text[:45]}")

        # ── MCTS from node 0 ──────────────────────────
        print("\nMCTS search from node 0 (full graph activation)...")
        activation = activate_from_node(nodes[0].node_id, graph, max_hops=6)
        print(f"Lit nodes: {len(activation)}")

        mcts   = MCTS(graph, norm)
        result = mcts.search(
            start_node_id = nodes[0].node_id,
            activation    = activation,
            max_iter      = 50,
            level         = Level.SENTENCE,
        )

        print(f"\nMCTS result: {result}")
        print(f"\nBest path:")
        for i, nid in enumerate(result.best_path):
            gn = graph.get_node(nid)
            if gn:
                print(f"  [{i}] id={nid}  V={gn.V:+.3f}  {gn.text}")

        # ── MCTS from node 2 (branch point) ───────────
        print("\nMCTS from node 2 (branch point — which branch wins?)...")
        activation2 = activate_from_node(nodes[2].node_id, graph, max_hops=4)
        result2 = mcts.search(
            start_node_id = nodes[2].node_id,
            activation    = activation2,
            max_iter      = 50,
            level         = Level.SENTENCE,
        )
        print(f"Path: {result2.best_path}")
        print(f"Total reward: {result2.total_reward:+.3f}")
        print(f"Committed early: {result2.committed_early}  "
              f"Confidence: {result2.confidence:.3f}")

        # Verify branch A was chosen (higher reward)
        branch_a_ids = {n.node_id for k,n in nodes.items() if k in [3,4,5]}
        branch_b_ids = {n.node_id for k,n in nodes.items() if k in [6,7,8]}
        path_set     = set(result2.best_path)
        chose_a      = bool(path_set & branch_a_ids)
        chose_b      = bool(path_set & branch_b_ids)
        print(f"\nChose branch A (high reward): {chose_a}")
        print(f"Chose branch B (low reward):  {chose_b}")

        graph.close()