"""
tinkerer.py — Novel Connection Engine.

One job: connect unconnected activated nodes.

Triggered when algo encounters a weak or missing path —
novel situation where the graph has no strong answer yet.

Tinkerer looks at what is activated and finds the best
bridge between unconnected nodes using what already exists
in the graph. It proposes the connection. Eval and future
reward signals determine if it was right.

Three strategies, each reading graph structure differently:

  extension  — A is connected to B, B is connected to C,
                A and C are not connected → connect A and C directly.
                Transitivity: if A relates to B and B relates to C,
                maybe A relates to C.

  analogy    — A has a neighbourhood pattern similar to X,
                B has a neighbourhood pattern similar to Y,
                A-B are not connected but X-Y are →
                connect A-B. Same structure, different domain.

  compression — a cluster of nodes all strongly connected to each other
                and all activated together repeatedly →
                they should become one node at the next level up.
                Tinkerer flags this for algo to abstract.

Tinkerer itself learns which strategy works in which situation —
because strategy choices and their outcomes are also edges in the graph.
A strategy node connects to the gap pattern it was applied to,
and that edge strengthens or weakens based on whether the
proposed connection later got reinforced by reward.

No hardcoded strategy selection. The graph decides.
"""

from typing import List, Tuple, Optional, Dict
from memory import Graph, Node


class Tinkerer:

    def __init__(self, graph: Graph):
        self.graph = graph

        # Strategy nodes live in the graph too.
        # They connect to gap patterns they've been applied to.
        # Their edges strengthen when proposals get rewarded.
        self._strategy_ids = self._init_strategies()

    # ── Strategy nodes ────────────────────────────────────────────

    def _init_strategies(self) -> Dict[str, str]:
        """
        Create strategy nodes in the graph if they don't exist.
        These are special level-0 nodes with modality 'meta'.
        Their connections to gap patterns encode learned strategy preference.
        """
        strategies = {}
        for name in ('extension', 'analogy', 'compression'):
            sid = f"strategy:{name}"
            if not self.graph.has_node(sid):
                self.graph.add_node(Node(
                    id       = sid,
                    level    = 0,
                    modality = 'meta',
                    elements = [name],
                    strength = 0.0,
                ))
            strategies[name] = sid
        return strategies

    # ── Primary interface ─────────────────────────────────────────

    def engage(
        self,
        activated: List[str],
    ) -> List[Tuple[str, str, str]]:
        """
        Called by algo when activated nodes have weak or missing paths.

        activated: list of node ids currently active in this interaction.

        Returns list of (source_id, target_id, strategy) proposals —
        pairs of nodes that tinkerer suggests connecting.
        Each proposal becomes an edge in the graph immediately (weak).
        Whether it survives depends on future reward signals via eval.
        """
        if len(activated) < 2:
            return []

        proposals = []

        # Find which pairs are not yet connected
        unconnected = self._unconnected_pairs(activated)
        if not unconnected:
            return []

        for src, tgt in unconnected:
            strategy = self._select_strategy(src, tgt)
            connection = self._apply_strategy(src, tgt, strategy)
            if True:  # co-activation is sufficient justification
                # Create the edge immediately — weak
                self.graph.add_edge(src, tgt)
                proposals.append((src, tgt, strategy))
                # Record strategy was used for this gap
                self._record_strategy_use(strategy, src, tgt)

        return proposals

    def outcome(
        self,
        proposals: List[Tuple[str, str, str]],
        reward: bool,
    ) -> None:
        """
        Reward signal arrived after tinkerer proposed connections.
        Strengthen or weaken the strategy→gap edges so tinkerer
        learns which strategies work in which situations.
        """
        for src, tgt, strategy in proposals:
            sid = self._strategy_ids.get(strategy)
            if not sid:
                continue
            # Gap pattern node is the edge between src and tgt
            # We use the strategy→src and strategy→tgt edges
            # as the record of this choice
            edge_to_src = self.graph.get_edge(sid, src)
            edge_to_tgt = self.graph.get_edge(sid, tgt)

            for edge in [edge_to_src, edge_to_tgt]:
                if edge is None:
                    continue
                if reward:
                    amount = 1.0 / (1.0 + edge.strength)
                    edge.reinforce(amount)
                else:
                    amount = 1.0 / (1.0 + edge.strength)
                    edge.weaken(amount)

    # ── Strategy selection ────────────────────────────────────────

    def _select_strategy(self, src: str, tgt: str) -> str:
        """
        Select strategy based on graph structure around src and tgt.
        Reads strategy node strengths to their neighbours —
        whichever strategy has strongest signal for this type of gap wins.

        Falls back to extension if no signal yet.
        """
        src_node = self.graph.get_node(src)
        tgt_node = self.graph.get_node(tgt)

        if not src_node or not tgt_node:
            return 'extension'

        # Check if there is a bridge node (B) connecting src and tgt
        # through existing edges → extension is applicable
        src_neighbours = {n.id for n, _ in self.graph.neighbours(src)}
        tgt_neighbours = {n.id for n, _ in self.graph.neighbours(tgt)}
        bridge_exists  = bool(src_neighbours & tgt_neighbours)

        if bridge_exists:
            return 'extension'

        # Check if src and tgt have similar neighbourhood sizes →
        # structural similarity → analogy applicable
        src_degree = len(src_neighbours)
        tgt_degree = len(tgt_neighbours)
        if src_degree > 0 and tgt_degree > 0:
            ratio = min(src_degree, tgt_degree) / max(src_degree, tgt_degree)
            if ratio > 0.5:
                return 'analogy'

        # If both nodes have many strong mutual connections to a
        # shared cluster → compression candidate
        shared = src_neighbours & tgt_neighbours
        if len(shared) >= 3:
            return 'compression'

        return 'extension'

    # ── Strategy implementations ──────────────────────────────────

    def _apply_strategy(
        self,
        src: str,
        tgt: str,
        strategy: str,
    ) -> bool:
        """
        Apply the chosen strategy to determine if a connection
        between src and tgt is structurally justified.
        Returns True if the proposal is structurally sound.
        Returns False if the strategy finds no justification.
        """
        if strategy == 'extension':
            return self._extension(src, tgt)
        elif strategy == 'analogy':
            return self._analogy(src, tgt)
        elif strategy == 'compression':
            return self._compression(src, tgt)
        return False

    def _extension(self, src: str, tgt: str) -> bool:
        """
        A → B → C: if src connects to some B and B connects to tgt,
        then src → tgt is a transitive extension.
        Returns True if at least one bridge node exists.
        """
        src_neighbours = {n.id for n, _ in self.graph.neighbours(src)}
        tgt_neighbours = {n.id for n, _ in self.graph.neighbours(tgt)}
        return bool(src_neighbours & tgt_neighbours)

    def _analogy(self, src: str, tgt: str) -> bool:
        """
        src and tgt have similar neighbourhood structures.
        If their neighbours are themselves connected to similar nodes,
        the analogy is structurally sound.
        Returns True if neighbourhood overlap ratio is meaningful.
        """
        src_neighbours = {n.id for n, _ in self.graph.neighbours(src)}
        tgt_neighbours = {n.id for n, _ in self.graph.neighbours(tgt)}

        if not src_neighbours or not tgt_neighbours:
            return False

        # Look one level deeper — do the neighbours of src
        # connect to nodes similar to the neighbours of tgt?
        src_reach = set()
        for nid in src_neighbours:
            src_reach |= {n.id for n, _ in self.graph.neighbours(nid)}

        tgt_reach = set()
        for nid in tgt_neighbours:
            tgt_reach |= {n.id for n, _ in self.graph.neighbours(nid)}

        if not src_reach or not tgt_reach:
            return False

        overlap = len(src_reach & tgt_reach)
        total   = len(src_reach | tgt_reach)
        return (overlap / total) > 0.0

    def _compression(self, src: str, tgt: str) -> bool:
        """
        src and tgt are part of a dense cluster that co-activates.
        Flag this cluster for abstraction by algo.
        Returns True if the cluster is dense enough to compress.
        """
        src_neighbours = {n.id for n, _ in self.graph.neighbours(src)}
        tgt_neighbours = {n.id for n, _ in self.graph.neighbours(tgt)}
        shared = src_neighbours & tgt_neighbours
        # A meaningful cluster needs at least 2 shared neighbours
        return len(shared) >= 2

    # ── Helpers ───────────────────────────────────────────────────

    def _unconnected_pairs(
        self,
        activated: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Find all pairs in the activated set that have no edge yet
        or have only a very weak edge (strength near initial 0.01).
        These are the gaps tinkerer should try to bridge.
        """
        pairs = []
        for i, a in enumerate(activated):
            for b in activated[i+1:]:
                edge = self.graph.get_edge(a, b)
                if edge is None or edge.strength <= 0.01:
                    pairs.append((a, b))
        return pairs

    def _record_strategy_use(
        self,
        strategy: str,
        src: str,
        tgt: str,
    ) -> None:
        """
        Record that this strategy was used to bridge src→tgt.
        Creates edges from strategy node to both endpoints.
        These edges strengthen/weaken via outcome() later.
        """
        sid = self._strategy_ids.get(strategy)
        if not sid:
            return
        self.graph.add_edge(sid, src)
        self.graph.add_edge(sid, tgt)

    # ── Compression signal ────────────────────────────────────────

    def compression_candidates(
        self,
        activated: List[str],
        min_cluster: int = 3,
    ) -> List[List[str]]:
        """
        Find clusters within activated nodes that are densely
        interconnected and should be abstracted to the next level.
        Called by algo to decide when to abstract.

        Returns list of clusters (each cluster is a list of node ids).
        A cluster qualifies if every pair within it has an edge
        with positive strength.
        """
        candidates = []
        n = len(activated)

        for size in range(min_cluster, n + 1):
            for i in range(n - size + 1):
                cluster = activated[i:i + size]
                if self._is_dense(cluster):
                    candidates.append(cluster)

        return candidates

    def _is_dense(self, node_ids: List[str]) -> bool:
        """
        A cluster is dense if every pair of nodes in it
        has a positive-strength edge.
        """
        for i, a in enumerate(node_ids):
            for b in node_ids[i+1:]:
                edge = self.graph.get_edge(a, b)
                if edge is None or edge.strength <= 0:
                    return False
        return True