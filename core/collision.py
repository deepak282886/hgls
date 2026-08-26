"""
collision.py

The continuous collision loop — the core learning mechanism.

Every collision is a question:
    Can I navigate from node A to node B through what I already know?

The impossibility checker fires first. If provably impossible — killed.
If possible — MCTS navigates A → B within the activated subgraph.
If it finds a high-reward path — the connection is real, an edge forms.
Bellman delta propagates. The graph learned this connection.

Sampling distribution:
    70%  same-level pairs     (most productive)
    20%  adjacent-level pairs (hierarchical connections)
    10%  cross-level pairs    (exploration — mostly killed by checker)

Nodes are sampled with V-weighted probability — high-V nodes are more
likely to form meaningful new connections because they already lead
somewhere good.

The loop runs forever or until max_iter is reached.
Stop with Ctrl+C. State persists — resume any time.
"""

import time
import random
import signal
import numpy as np
from dataclasses import dataclass, field
from typing      import Optional
from collections import deque

from core.atoms      import Level
from core.graph      import Graph
from core.node       import Node
from core.edge       import EdgeType
from core.normalizer import Normalizer
from core.bellman    import BellmanManager
from core.activation import activate_from_node, edge_spread, normalize_activation
from core.mcts       import MCTS, MCTSResult
from core.checker    import Checker, CheckResult


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Sampling distribution across level pairs
SAME_LEVEL_PROB     = 0.70
ADJACENT_LEVEL_PROB = 0.20
# Cross-level is the remainder: 0.10

# Pool size for V-weighted sampling — top N nodes per level
SAMPLE_POOL_SIZE = 5000

# Levels eligible for collision sampling.
# Restricted to SENTENCE only until CHARACTER and WORD nodes
# are populated — PARAGRAPH and DOCUMENT nodes produce subgraphs
# too large for MCTS to navigate in reasonable time.
COLLISION_LEVELS = {Level.SENTENCE}

# Minimum V value for a node to enter the sampling pool
# — set well below the graph's average V so most nodes are included
MIN_V_FOR_SAMPLING = -25.0

# Minimum path reward for a collision to form an edge
MIN_PATH_REWARD = 0.0

# How often to refresh the sampling pool (iterations)
POOL_REFRESH_INTERVAL = 500

# MCTS settings for collision — tighter than query-time
# because we run thousands of collisions, not dozens of queries
COLLISION_MCTS_ITER  = 30
COLLISION_MAX_DEPTH  = 10
COLLISION_MAX_HOPS   = 3

# How often to run a full Bellman sweep (iterations)
BELLMAN_SWEEP_INTERVAL = 2000

# How often to print progress (iterations)
LOG_INTERVAL = 100


# ─────────────────────────────────────────────
# COLLISION STATS
# ─────────────────────────────────────────────

@dataclass
class CollisionStats:
    """Running statistics for the collision loop."""
    iterations       : int   = 0
    killed_layer1    : int   = 0
    killed_layer2    : int   = 0
    killed_layer3    : int   = 0
    mcts_attempted   : int   = 0
    mcts_succeeded   : int   = 0   # path found with reward > 0
    edges_formed     : int   = 0
    elapsed_sec      : float = 0.0

    # Rolling window for success rate (last 1000 collisions)
    _recent          : deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_kill(self, layer: int):
        self.iterations += 1
        self._recent.append(0)
        if layer == 1:
            self.killed_layer1 += 1
        elif layer == 2:
            self.killed_layer2 += 1
        elif layer == 3:
            self.killed_layer3 += 1

    def record_mcts(self, succeeded: bool):
        self.mcts_attempted += 1
        if succeeded:
            self.mcts_succeeded += 1
            self.edges_formed   += 1
            self._recent.append(1)
        else:
            self._recent.append(0)

    @property
    def kill_rate(self) -> float:
        killed = self.killed_layer1 + self.killed_layer2 + self.killed_layer3
        if self.iterations == 0:
            return 0.0
        return killed / self.iterations

    @property
    def recent_success_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    @property
    def mcts_success_rate(self) -> float:
        if self.mcts_attempted == 0:
            return 0.0
        return self.mcts_succeeded / self.mcts_attempted

    def report(self) -> str:
        rate = self.iterations / (self.elapsed_sec + 1e-9)
        return (
            f"iterations={self.iterations}  "
            f"edges_formed={self.edges_formed}  "
            f"kill_rate={self.kill_rate:.1%}  "
            f"mcts_success={self.mcts_success_rate:.1%}  "
            f"recent_success={self.recent_success_rate:.1%}  "
            f"rate={rate:.1f}iter/s"
        )


# ─────────────────────────────────────────────
# NODE SAMPLING POOL
# ─────────────────────────────────────────────

class SamplingPool:
    """
    Maintains a pool of nodes per level for V-weighted sampling.

    Refreshed every POOL_REFRESH_INTERVAL iterations so the
    pool reflects new nodes added during the collision loop.

    V-weighted sampling: probability proportional to
        max(0, V - V_min) + epsilon
    so higher-V nodes are sampled more often but low-V nodes
    still get some chance.
    """

    def __init__(self, graph: Graph):
        self.graph     = graph
        self.pools     : dict[Level, list[Node]]    = {}
        self.weights   : dict[Level, np.ndarray]    = {}
        # Set to -POOL_REFRESH_INTERVAL so the very first
        # call to refresh() triggers immediately at iteration 0
        self.last_refresh = -POOL_REFRESH_INTERVAL

    def refresh(self, iteration: int):
        """Rebuild pools from current graph state."""
        if (iteration - self.last_refresh) < POOL_REFRESH_INTERVAL:
            return
        self.last_refresh = iteration

        for level in Level:
            nodes = self.graph.nodes_by_level(level, limit=SAMPLE_POOL_SIZE)
            nodes = [n for n in nodes if n.V >= MIN_V_FOR_SAMPLING]
            if not nodes:
                self.pools[level]   = []
                self.weights[level] = np.array([])
                continue

            v_min = min(n.V for n in nodes)
            raw_weights = np.array([
                max(0.0, n.V - v_min) + 1e-3
                for n in nodes
            ], dtype=np.float64)
            raw_weights /= raw_weights.sum()

            self.pools[level]   = nodes
            self.weights[level] = raw_weights

    def sample_from_level(self, level: Level) -> Optional[Node]:
        """Sample one node from the given level using V-weighted probability."""
        pool    = self.pools.get(level, [])
        weights = self.weights.get(level, np.array([]))
        if not pool or len(weights) == 0:
            return None
        idx = np.random.choice(len(pool), p=weights)
        return pool[idx]

    def available_levels(self) -> list[Level]:
        """Levels with at least one node in their pool, restricted to COLLISION_LEVELS."""
        return [
            lv for lv in Level
            if lv in COLLISION_LEVELS
            and len(self.pools.get(lv, [])) >= 2
        ]


# ─────────────────────────────────────────────
# PAIR SAMPLER
# ─────────────────────────────────────────────

def sample_pair(pool: SamplingPool) -> Optional[tuple[Node, Node]]:
    """
    Sample (node_a, node_b) according to the distribution:
        70%  same level
        20%  adjacent levels (one step apart)
        10%  cross levels (two steps apart, within checker limit)

    Only samples from levels that actually have nodes in their pool.
    Returns None if the pool is too sparse.
    """
    available = pool.available_levels()
    if not available:
        return None

    roll = random.random()

    if roll < SAME_LEVEL_PROB:
        # Same level — pick from available levels weighted by pool size
        weights = [len(pool.pools.get(lv, [])) for lv in available]
        total   = sum(weights)
        if total == 0:
            return None
        probs  = [w / total for w in weights]
        level  = available[np.random.choice(len(available), p=probs)]
        a = pool.sample_from_level(level)
        b = pool.sample_from_level(level)

    elif roll < SAME_LEVEL_PROB + ADJACENT_LEVEL_PROB:
        # Adjacent levels
        if len(available) < 2:
            level = available[0]
            a = pool.sample_from_level(level)
            b = pool.sample_from_level(level)
        else:
            level_a = random.choice(available)
            adjacent = [
                lv for lv in available
                if abs(int(lv) - int(level_a)) == 1
            ]
            level_b = random.choice(adjacent) if adjacent else level_a
            a = pool.sample_from_level(level_a)
            b = pool.sample_from_level(level_b)

    else:
        # Cross level (2 steps apart)
        if len(available) < 2:
            level = available[0]
            a = pool.sample_from_level(level)
            b = pool.sample_from_level(level)
        else:
            level_a = random.choice(available)
            cross = [
                lv for lv in available
                if abs(int(lv) - int(level_a)) == 2
            ]
            level_b = random.choice(cross) if cross else level_a
            a = pool.sample_from_level(level_a)
            b = pool.sample_from_level(level_b)

    if a is None or b is None:
        return None
    if a.node_id == b.node_id:
        return None

    return a, b


# ─────────────────────────────────────────────
# COLLISION ACTIVATION
# Merge neighborhoods of both nodes for MCTS
# ─────────────────────────────────────────────

def activate_collision(
    node_a : Node,
    node_b : Node,
    graph  : Graph,
) :
    """
    Activate the merged neighbourhood of node_a and node_b.

    Both nodes seed at full activation. The merged subgraph
    is what MCTS navigates — it can use any node reachable
    from either seed.

    Returns ActivationResult covering both seeds.
    """
    from core.activation import ActivationResult, normalize_activation

    initial = {
        node_a.node_id: 1.0,
        node_b.node_id: 1.0,
    }

    full_acts = edge_spread(
        seeds               = [node_a.node_id, node_b.node_id],
        graph               = graph,
        max_hops            = COLLISION_MAX_HOPS,
        initial_activations = initial,
    )

    norm_acts = normalize_activation(full_acts, max_nodes=300)

    return ActivationResult(
        lit   = norm_acts,
        seeds = [node_a.node_id, node_b.node_id],
        query = node_a.embedding,
        level = node_a.level,
    )


# ─────────────────────────────────────────────
# EDGE FORMATION
# ─────────────────────────────────────────────

def form_edge_from_path(
    path       : list[int],
    graph      : Graph,
    normalizer : Normalizer,
    bm         : BellmanManager,
) -> int:
    """
    Form edges along the entire discovered path, not just A→B.

    Each consecutive pair in the path gets an ASSOCIATIVE edge
    if it doesn't already exist. This maximises the learning
    signal from each successful MCTS navigation.

    Returns number of new edges formed.
    """
    from core.activation import ACTIVATION_TEMPERATURE
    new_edges = 0

    for i in range(len(path) - 1):
        src_id = path[i]
        tgt_id = path[i + 1]

        if graph.edge_exists(src_id, tgt_id):
            continue

        src_node = graph.get_node(src_id)
        tgt_node = graph.get_node(tgt_id)
        if src_node is None or tgt_node is None:
            continue

        graph.add_edge(
            src_id, tgt_id,
            src_node.text, tgt_node.text,
            src_node.level,
            edge_type = EdgeType.ASSOCIATIVE,
        )
        normalizer.on_edge_added(src_id, graph)
        bm.on_edge_added(src_id, tgt_id)
        new_edges += 1

    return new_edges


# ─────────────────────────────────────────────
# COLLISION LOOP
# ─────────────────────────────────────────────

class CollisionLoop:
    """
    The continuous collision loop.

    Usage:
        loop = CollisionLoop(graph, normalizer, bm, checker)
        loop.run()                    # runs forever
        loop.run(max_iter=10_000)     # runs for 10k iterations
    """

    def __init__(
        self,
        graph        : Graph,
        normalizer   : Normalizer,
        bm           : BellmanManager,
        checker      : Checker,
        verbose      : bool = True,
    ):
        self.graph      = graph
        self.normalizer = normalizer
        self.bm         = bm
        self.checker    = checker
        self.verbose    = verbose

        self.pool  = SamplingPool(graph)
        self.stats = CollisionStats()
        self._stop = False

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        if self.verbose:
            print("\n\nCollision loop interrupted. Finishing current iteration...")
        self._stop = True

    def run(self, max_iter: int = 0) -> CollisionStats:
        """
        Run the collision loop.

        max_iter=0 means run forever (until Ctrl+C).
        Returns CollisionStats when done.
        """
        t0 = time.time()

        if self.verbose:
            gs = self.graph.stats()
            print(f"\nCollision loop starting.")
            print(f"  Graph: {gs['node_count']} nodes, "
                  f"{gs['edge_count']} edges")
            print(f"  Max iter: {'∞' if max_iter == 0 else max_iter}")
            print(f"  Commandments: {len(self.checker.commandments)}")
            print(f"\nPress Ctrl+C to stop.\n")

        iteration = 0

        while not self._stop:
            if max_iter > 0 and iteration >= max_iter:
                break

            # Refresh sampling pool periodically
            self.pool.refresh(iteration)

            # Skip if pool is empty
            if not self.pool.available_levels():
                if self.verbose:
                    print("  Pool empty — waiting for nodes...")
                time.sleep(1.0)
                iteration += 1
                continue

            # Sample a pair
            pair = sample_pair(self.pool)
            if pair is None:
                iteration += 1
                continue

            node_a, node_b = pair

            # Run the collision
            self._attempt_collision(node_a, node_b, iteration)
            iteration += 1

            # Periodic Bellman sweep
            if iteration % BELLMAN_SWEEP_INTERVAL == 0:
                if self.verbose:
                    print(f"\n  Bellman sweep at iteration {iteration}...")
                self.bm.sweep(verbose=False)
                if self.verbose:
                    print(f"  Sweep complete.\n")

            # Progress logging
            if self.verbose and iteration % LOG_INTERVAL == 0:
                self.stats.elapsed_sec = time.time() - t0
                gs = self.graph.stats()
                print(
                    f"  iter={iteration:7d}  "
                    f"{self.stats.report()}  "
                    f"graph_edges={gs['edge_count']}"
                )

        self.stats.elapsed_sec   = time.time() - t0
        self.stats.iterations    = iteration

        if self.verbose:
            print(f"\nCollision loop finished.")
            print(f"  {self.stats.report()}")

        return self.stats

    def _attempt_collision(
        self,
        node_a    : Node,
        node_b    : Node,
        iteration : int,
    ):
        """
        Single collision attempt.

        1. Skip if A→B already directly connected.
        2. Run impossibility checker.
        3. Activate merged neighbourhood.
        4. Run MCTS from A.
        5. If path reward > MIN_PATH_REWARD:
               - Form direct A→B associative edge (the collision product)
               - Form edges along intermediate path (bonus signal)
               - Propagate Bellman delta
        """
        # ── Skip if already directly connected ──
        if self.graph.edge_exists(node_a.node_id, node_b.node_id):
            return

        # ── Layer check ──────────────────────
        result = self.checker.check(node_a, node_b)
        if not result:
            self.stats.record_kill(result.layer)
            return

        self.stats.iterations += 1

        # ── Activate merged neighbourhood ────
        activation = activate_collision(node_a, node_b, self.graph)

        if not activation.lit:
            self.stats.record_mcts(succeeded=False)
            return

        # ── MCTS navigation A → B ───────────
        mcts        = MCTS(self.graph, self.normalizer)
        mcts_result = mcts.search(
            start_node_id = node_a.node_id,
            activation    = activation,
            max_iter      = COLLISION_MCTS_ITER,
            max_depth     = COLLISION_MAX_DEPTH,
            level         = node_a.level,
        )

        path        = mcts_result.best_path
        path_reward = mcts_result.total_reward

        if not path or path_reward <= MIN_PATH_REWARD:
            self.stats.record_mcts(succeeded=False)
            return

        # ── Form direct A→B edge (the collision product) ─
        # This is the core of what collision means:
        # A and B proved they can be connected → form a direct bond
        new_edges = 0
        if not self.graph.edge_exists(node_a.node_id, node_b.node_id):
            self.graph.add_edge(
                node_a.node_id, node_b.node_id,
                node_a.text, node_b.text,
                node_a.level,
                edge_type = EdgeType.ASSOCIATIVE,
            )
            self.normalizer.on_edge_added(node_a.node_id, self.graph)
            self.bm.on_edge_added(node_a.node_id, node_b.node_id)
            new_edges += 1

        # ── Also form edges along intermediate path ──
        # These are bonus connections discovered along the way
        new_edges += form_edge_from_path(
            path, self.graph, self.normalizer, self.bm
        )

        if new_edges > 0:
            self.stats.record_mcts(succeeded=True)
        else:
            self.stats.record_mcts(succeeded=False)


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# Used by main.py's collide command
# ─────────────────────────────────────────────

def run_collision_loop(
    graph            : Graph,
    normalizer       : Normalizer,
    bm               : BellmanManager,
    commandments_path: Optional[str] = None,
    max_iter         : int  = 0,
    verbose          : bool = True,
) -> CollisionStats:
    """
    Convenience wrapper — load checker and run the loop.

    Called by main.py:
        python -m core.main collide [--max-iter N] [--commandments path.json]
    """
    from core.checker import Checker, load_commandments

    commandments = load_commandments(commandments_path)
    checker      = Checker(commandments=commandments)
    loop         = CollisionLoop(graph, normalizer, bm, checker, verbose=verbose)

    return loop.run(max_iter=max_iter)


if __name__ == "__main__":
    import tempfile, os

    print("=== collision.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        from core.graph      import Graph
        from core.normalizer import Normalizer
        from core.bellman    import BellmanManager
        from core.ingestion  import ingest_text, IngestionConfig

        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        norm    = Normalizer()
        bm      = BellmanManager(graph, norm)

        # Ingest sample text to populate the graph
        sample = """
The proof begins with the fundamental theorem of arithmetic.
Every integer greater than one is either prime or a product of primes.
This factorization is unique up to the order of the factors.
Euclid demonstrated this result using a simple contradiction argument.

Assume the set of primes is finite with largest prime P.
Multiply all primes together and add one to get N.
N is not divisible by any prime in our finite list.
Therefore N is either prime or has a prime factor outside the list.
This contradicts our assumption that P was the largest prime.
Therefore there are infinitely many primes.
""".strip()

        config = IngestionConfig(max_level=Level.SENTENCE, verbose=False)
        ingest_text(sample, graph, norm, bm, config)

        gs = graph.stats()
        print(f"Graph after ingestion: {gs['node_count']} nodes, "
              f"{gs['edge_count']} edges\n")

        # Run a short collision loop
        checker = Checker(commandments=["how to harm"])
        loop    = CollisionLoop(graph, norm, bm, checker, verbose=True)
        stats   = loop.run(max_iter=200)

        print(f"\nFinal stats:")
        print(f"  {stats.report()}")

        gs = graph.stats()
        print(f"  Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")

        graph.close()