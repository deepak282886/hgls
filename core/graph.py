"""
graph.py

The graph — persistent storage and interface for all nodes and edges.

Uses SQLite as the backend. The graph lives on disk and grows
continuously across sessions. No full reload needed — nodes and
edges are queried lazily as MCTS and traversal need them.

Responsibilities:
    - Add nodes and edges
    - Query nodes by ID, level, reward, V value
    - Query edges by source, target, type
    - Maintain an in-memory cache of recently accessed nodes/edges
    - Assign stable integer IDs to new nodes and edges
    - Track graph-level statistics for the normalizer

Design principles:
    - Every write is immediate — no batching that could lose data
    - Reads are cached — hot nodes stay in memory
    - The graph never deletes — only grows
    - Thread-safe writes via SQLite's built-in locking
"""

import sqlite3
import time
import os
import numpy as np
from collections import OrderedDict
from typing import Optional, Iterator

from core.atoms import Level
from core.node import Node, make_node
from core.edge import Edge, EdgeType, make_edge


# ─────────────────────────────────────────────
# LRU CACHE
# Simple bounded cache for hot nodes/edges
# ─────────────────────────────────────────────

class LRUCache:
    def __init__(self, capacity: int = 10_000):
        self.cache    = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def invalidate(self, key):
        self.cache.pop(key, None)

    def __len__(self):
        return len(self.cache)


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;

CREATE TABLE IF NOT EXISTS nodes (
    node_id      INTEGER PRIMARY KEY,
    text         TEXT    NOT NULL,
    level        INTEGER NOT NULL,
    embedding    BLOB    NOT NULL,
    reward       REAL    DEFAULT 0.0,
    norm_reward  REAL    DEFAULT 0.0,
    V            REAL    DEFAULT 0.0,
    norm_V       REAL    DEFAULT 0.0,
    visit_count  INTEGER DEFAULT 0,
    total_reward REAL    DEFAULT 0.0,
    created_at   REAL    NOT NULL,
    is_terminal  INTEGER DEFAULT 0,
    children     TEXT    DEFAULT '',
    parents      TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id      INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL,
    target_id    INTEGER NOT NULL,
    edge_type    INTEGER NOT NULL,
    reward       REAL    DEFAULT 0.0,
    norm_reward  REAL    DEFAULT 0.0,
    visit_count  INTEGER DEFAULT 0,
    total_reward REAL    DEFAULT 0.0,
    source_level INTEGER NOT NULL,
    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
);

CREATE TABLE IF NOT EXISTS graph_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_level    ON nodes(level);
CREATE INDEX IF NOT EXISTS idx_nodes_V        ON nodes(V);
CREATE INDEX IF NOT EXISTS idx_nodes_reward   ON nodes(reward);
CREATE INDEX IF NOT EXISTS idx_edges_source   ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target   ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type     ON edges(edge_type);
"""


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

class Graph:
    def __init__(self, db_path: str = "graph.db",
                 cache_size: int = 10_000):
        """
        Open or create the graph database.

        db_path    : path to SQLite file. Created if it does not exist.
        cache_size : max nodes/edges to keep in memory simultaneously.
        """
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        self._node_cache = LRUCache(cache_size)
        self._edge_cache = LRUCache(cache_size)

        # Running ID counters — loaded from DB on open
        self._next_node_id = self._load_counter("next_node_id", 0)
        self._next_edge_id = self._load_counter("next_edge_id", 0)

        # Running statistics for normalizer (Welford online algorithm)
        self._node_count  = self._load_counter("node_count", 0)
        self._edge_count  = self._load_counter("edge_count", 0)

        # Per-level reward stats for normalization
        # Format: {level: {"mean": float, "M2": float, "count": int}}
        self._level_stats = self._load_level_stats()


    # ─────────────────────────────────────────
    # SCHEMA INIT
    # ─────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()


    # ─────────────────────────────────────────
    # METADATA PERSISTENCE
    # ─────────────────────────────────────────

    def _load_counter(self, key: str, default: int) -> int:
        row = self._conn.execute(
            "SELECT value FROM graph_meta WHERE key=?", (key,)
        ).fetchone()
        return int(row["value"]) if row else default

    def _save_counter(self, key: str, value: int):
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?,?)",
            (key, str(value))
        )

    def _load_level_stats(self) -> dict:
        stats = {}
        for level in Level:
            key = f"level_stats_{int(level)}"
            row = self._conn.execute(
                "SELECT value FROM graph_meta WHERE key=?", (key,)
            ).fetchone()
            if row:
                import json
                stats[level] = json.loads(row["value"])
            else:
                stats[level] = {"mean": 0.0, "M2": 0.0, "count": 0}
        return stats

    def _save_level_stats(self, level: Level):
        import json
        key = f"level_stats_{int(level)}"
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?,?)",
            (key, json.dumps(self._level_stats[level]))
        )


    # ─────────────────────────────────────────
    # WELFORD ONLINE STATISTICS UPDATE
    # Updates running mean and variance incrementally.
    # Called when a new node reward is added at a level.
    # ─────────────────────────────────────────

    def _welford_update(self, level: Level, new_value: float):
        s = self._level_stats[level]
        s["count"] += 1
        delta  = new_value - s["mean"]
        s["mean"] += delta / s["count"]
        delta2 = new_value - s["mean"]
        s["M2"] += delta * delta2

    def level_mean(self, level: Level) -> float:
        return self._level_stats[level]["mean"]

    def level_std(self, level: Level) -> float:
        s = self._level_stats[level]
        if s["count"] < 2:
            return 1.0
        return float(np.sqrt(s["M2"] / (s["count"] - 1)) + 1e-9)


    # ─────────────────────────────────────────
    # ADD NODE
    # ─────────────────────────────────────────

    def add_node(self, text: str, level: Level,
                 reward: float = 0.0,
                 children: Optional[list[int]] = None,
                 parents:  Optional[list[int]] = None) -> Node:
        """
        Create and persist a new node.
        Returns the Node with its assigned stable ID.
        """
        nid  = self._next_node_id
        node = make_node(nid, text, level, reward, children, parents)

        # Persist
        d = node.to_dict()
        self._conn.execute("""
            INSERT INTO nodes
            (node_id, text, level, embedding, reward, norm_reward,
             V, norm_V, visit_count, total_reward, created_at,
             is_terminal, children, parents)
            VALUES
            (:node_id,:text,:level,:embedding,:reward,:norm_reward,
             :V,:norm_V,:visit_count,:total_reward,:created_at,
             :is_terminal,:children,:parents)
        """, d)

        # Update counters
        self._next_node_id += 1
        self._node_count   += 1
        self._save_counter("next_node_id", self._next_node_id)
        self._save_counter("node_count",   self._node_count)

        # Update Welford stats for this level
        self._welford_update(level, reward)
        self._save_level_stats(level)

        self._conn.commit()

        # Cache
        self._node_cache.put(nid, node)
        return node


    # ─────────────────────────────────────────
    # ADD EDGE
    # ─────────────────────────────────────────

    def add_edge(self, source_id: int, target_id: int,
                 source_text: str, target_text: str,
                 level: Level,
                 edge_type: EdgeType = EdgeType.SEQUENTIAL) -> Edge:
        """
        Create and persist a new edge between two existing nodes.
        Returns the Edge with its assigned stable ID.

        Also updates the is_terminal flag on the source node —
        a node with at least one outgoing edge is not terminal.
        """
        eid  = self._next_edge_id
        edge = make_edge(eid, source_id, target_id,
                         source_text, target_text, level, edge_type)

        d = edge.to_dict()
        self._conn.execute("""
            INSERT INTO edges
            (edge_id, source_id, target_id, edge_type, reward,
             norm_reward, visit_count, total_reward, source_level)
            VALUES
            (:edge_id,:source_id,:target_id,:edge_type,:reward,
             :norm_reward,:visit_count,:total_reward,:source_level)
        """, d)

        # Source node is no longer terminal
        self._conn.execute(
            "UPDATE nodes SET is_terminal=0 WHERE node_id=?",
            (source_id,)
        )

        # Update counters
        self._next_edge_id += 1
        self._edge_count   += 1
        self._save_counter("next_edge_id", self._next_edge_id)
        self._save_counter("edge_count",   self._edge_count)

        self._conn.commit()

        # Invalidate source node cache (is_terminal changed)
        self._node_cache.invalidate(source_id)
        self._edge_cache.put(eid, edge)
        return edge


    # ─────────────────────────────────────────
    # GET NODE
    # ─────────────────────────────────────────

    def get_node(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID. Cache-first."""
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached

        row = self._conn.execute(
            "SELECT * FROM nodes WHERE node_id=?", (node_id,)
        ).fetchone()

        if row is None:
            return None

        node = Node.from_dict(dict(row))
        self._node_cache.put(node_id, node)
        return node


    # ─────────────────────────────────────────
    # GET EDGE
    # ─────────────────────────────────────────

    def get_edge(self, edge_id: int) -> Optional[Edge]:
        """Retrieve an edge by ID. Cache-first."""
        cached = self._edge_cache.get(edge_id)
        if cached is not None:
            return cached

        row = self._conn.execute(
            "SELECT * FROM edges WHERE edge_id=?", (edge_id,)
        ).fetchone()

        if row is None:
            return None

        edge = Edge.from_dict(dict(row))
        self._edge_cache.put(edge_id, edge)
        return edge


    # ─────────────────────────────────────────
    # GET OUTGOING EDGES
    # ─────────────────────────────────────────

    def get_outgoing_edges(self, node_id: int) -> list[Edge]:
        """All edges where source_id = node_id."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_id=?", (node_id,)
        ).fetchall()
        edges = []
        for row in rows:
            e = Edge.from_dict(dict(row))
            self._edge_cache.put(e.edge_id, e)
            edges.append(e)
        return edges


    # ─────────────────────────────────────────
    # GET INCOMING EDGES
    # ─────────────────────────────────────────

    def get_incoming_edges(self, node_id: int) -> list[Edge]:
        """All edges where target_id = node_id."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target_id=?", (node_id,)
        ).fetchall()
        edges = []
        for row in rows:
            e = Edge.from_dict(dict(row))
            self._edge_cache.put(e.edge_id, e)
            edges.append(e)
        return edges


    # ─────────────────────────────────────────
    # UPDATE NODE V AND VISIT STATS
    # ─────────────────────────────────────────

    def update_node_V(self, node_id: int, V: float, norm_V: float = 0.0):
        """Update Bellman value for a node. Invalidates cache."""
        self._conn.execute(
            "UPDATE nodes SET V=?, norm_V=? WHERE node_id=?",
            (V, norm_V, node_id)
        )
        self._conn.commit()
        self._node_cache.invalidate(node_id)

    def update_node_reward(self, node_id: int,
                           reward: float, norm_reward: float = 0.0):
        """Update reward for a node. Invalidates cache."""
        self._conn.execute(
            "UPDATE nodes SET reward=?, norm_reward=? WHERE node_id=?",
            (reward, norm_reward, node_id)
        )
        self._conn.commit()
        self._node_cache.invalidate(node_id)

    def record_node_visit(self, node_id: int, reward: float):
        """Record an MCTS visit to a node."""
        self._conn.execute("""
            UPDATE nodes
            SET visit_count  = visit_count + 1,
                total_reward = total_reward + ?
            WHERE node_id = ?
        """, (reward, node_id))
        self._conn.commit()
        self._node_cache.invalidate(node_id)

    def record_edge_visit(self, edge_id: int, reward: float):
        """Record an MCTS visit through an edge."""
        self._conn.execute("""
            UPDATE edges
            SET visit_count  = visit_count + 1,
                total_reward = total_reward + ?
            WHERE edge_id = ?
        """, (reward, edge_id))
        self._conn.commit()
        self._edge_cache.invalidate(edge_id)

    def update_edge_norm_reward(self, edge_id: int, norm_reward: float):
        """Update normalized reward for an edge."""
        self._conn.execute(
            "UPDATE edges SET norm_reward=? WHERE edge_id=?",
            (norm_reward, edge_id)
        )
        self._conn.commit()
        self._edge_cache.invalidate(edge_id)


    # ─────────────────────────────────────────
    # QUERY NODES
    # ─────────────────────────────────────────

    def nodes_by_level(self, level: Level,
                       limit: int = 1000) -> list[Node]:
        """All nodes at a given hierarchy level."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE level=? ORDER BY V DESC LIMIT ?",
            (int(level), limit)
        ).fetchall()
        return [Node.from_dict(dict(r)) for r in rows]

    def top_nodes(self, n: int = 20, level: Optional[Level] = None) -> list[Node]:
        """Top N nodes by Bellman V value, optionally filtered by level."""
        if level is not None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE level=? ORDER BY V DESC LIMIT ?",
                (int(level), n)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes ORDER BY V DESC LIMIT ?", (n,)
            ).fetchall()
        return [Node.from_dict(dict(r)) for r in rows]

    def terminal_nodes(self, level: Optional[Level] = None) -> list[Node]:
        """Nodes with no outgoing edges — natural endpoints."""
        if level is not None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE is_terminal=1 AND level=? ORDER BY V DESC",
                (int(level),)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE is_terminal=1 ORDER BY V DESC"
            ).fetchall()
        return [Node.from_dict(dict(r)) for r in rows]

    def nodes_since(self, created_after: float) -> list[Node]:
        """Nodes created after a given timestamp. Used by delta propagator."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE created_at > ? ORDER BY created_at ASC",
            (created_after,)
        ).fetchall()
        return [Node.from_dict(dict(r)) for r in rows]

    def all_node_ids(self, level: Optional[Level] = None) -> list[int]:
        """All node IDs, optionally filtered by level."""
        if level is not None:
            rows = self._conn.execute(
                "SELECT node_id FROM nodes WHERE level=?", (int(level),)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT node_id FROM nodes"
            ).fetchall()
        return [r[0] for r in rows]

    def text_exists(self, text: str, level: Level) -> Optional[int]:
        """
        Check if a node with exact text already exists at this level.
        Returns node_id if found, None otherwise.
        Prevents duplicate nodes for identical text units.
        """
        row = self._conn.execute(
            "SELECT node_id FROM nodes WHERE text=? AND level=? LIMIT 1",
            (text, int(level))
        ).fetchone()
        return row[0] if row else None


    # ─────────────────────────────────────────
    # EDGE EXISTS CHECK
    # ─────────────────────────────────────────

    def edge_exists(self, source_id: int, target_id: int) -> bool:
        """Check if a directed edge already exists between two nodes."""
        row = self._conn.execute(
            "SELECT edge_id FROM edges WHERE source_id=? AND target_id=? LIMIT 1",
            (source_id, target_id)
        ).fetchone()
        return row is not None


    # ─────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────

    def stats(self) -> dict:
        """Graph-level statistics for monitoring."""
        node_count = self._conn.execute(
            "SELECT COUNT(*) FROM nodes"
        ).fetchone()[0]
        edge_count = self._conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        level_counts = {}
        for level in Level:
            c = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE level=?", (int(level),)
            ).fetchone()[0]
            level_counts[level.name] = c
        avg_V = self._conn.execute(
            "SELECT AVG(V) FROM nodes"
        ).fetchone()[0] or 0.0
        max_V = self._conn.execute(
            "SELECT MAX(V) FROM nodes"
        ).fetchone()[0] or 0.0
        terminal_count = self._conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE is_terminal=1"
        ).fetchone()[0]

        return {
            "node_count"    : node_count,
            "edge_count"    : edge_count,
            "level_counts"  : level_counts,
            "avg_V"         : round(avg_V, 4),
            "max_V"         : round(max_V, 4),
            "terminal_count": terminal_count,
            "cache_size"    : len(self._node_cache),
            "db_path"       : self.db_path,
        }


    # ─────────────────────────────────────────
    # ITERATE ALL NODES (for Bellman sweep)
    # ─────────────────────────────────────────

    def iter_all_nodes(self, batch_size: int = 500) -> Iterator[list[Node]]:
        """
        Iterate all nodes in batches ordered by creation time.
        Used by bellman.py for full graph sweeps.
        Memory-efficient — does not load entire graph at once.
        """
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT * FROM nodes ORDER BY created_at ASC LIMIT ? OFFSET ?",
                (batch_size, offset)
            ).fetchall()
            if not rows:
                break
            yield [Node.from_dict(dict(r)) for r in rows]
            offset += batch_size


    # ─────────────────────────────────────────
    # CLOSE
    # ─────────────────────────────────────────

    def close(self):
        """Flush all pending writes and close the database."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import tempfile, os

    print("=== graph.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        with Graph(db_path) as G:
            # Add nodes
            n0 = G.add_node("Detective Maria arrived at the warehouse.", Level.SENTENCE, reward=2.0)
            n1 = G.add_node("The building smelled of rust.", Level.SENTENCE, reward=1.7)
            n2 = G.add_node("She found footprints leading to the back room.", Level.SENTENCE, reward=1.5)
            n3 = G.add_node("The footprints were fresh.", Level.SENTENCE, reward=1.3)

            print(f"Added nodes: {n0.node_id}, {n1.node_id}, {n2.node_id}, {n3.node_id}")

            # Add edges
            e01 = G.add_edge(n0.node_id, n1.node_id, n0.text, n1.text, Level.SENTENCE)
            e12 = G.add_edge(n1.node_id, n2.node_id, n1.text, n2.text, Level.SENTENCE)
            e23 = G.add_edge(n2.node_id, n3.node_id, n2.text, n3.text, Level.SENTENCE)

            print(f"\nEdges:")
            for e in [e01, e12, e23]:
                verdict = "✓" if e.reward > 0 else "✗"
                print(f"  {e}  {verdict}")

            # Query
            print(f"\nGet node 2: {G.get_node(2)}")

            # Outgoing edges from node 1
            out = G.get_outgoing_edges(n1.node_id)
            print(f"\nOutgoing from node 1: {out}")

            # Terminal nodes
            terminals = G.terminal_nodes()
            print(f"\nTerminal nodes: {[t.node_id for t in terminals]}")

            # Duplicate text check
            existing = G.text_exists("The building smelled of rust.", Level.SENTENCE)
            print(f"\nDuplicate check for n1 text: exists at node_id={existing}")

            # Update V
            G.update_node_V(n3.node_id, V=5.5, norm_V=1.2)
            n3_updated = G.get_node(n3.node_id)
            print(f"\nAfter V update: {n3_updated}")

            # Stats
            print(f"\nGraph stats:")
            for k, v in G.stats().items():
                print(f"  {k}: {v}")