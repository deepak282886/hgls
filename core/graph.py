"""
graph.py

The data structure. One persistent weighted directed hypergraph.
SQLite backed. Three tables: nodes, edges, hyperedges.

Design principles:
    - Identity is structure, not surface form
      Level 0 nodes store raw text. Level N nodes store component IDs.
      Surface text is always derived, never stored as identity.

    - One unified node table for all states
      provisional, stable, hypothetical, decaying — all in same table.
      State is just a field. No separate provisional buffer.

    - Decay runs on everything
      Every node and edge has a decay rate derived from its level.
      Nothing is exempt.

    - Three parameters only
      BASE_DECAY, COMPRESSION_THRESHOLD, BOUNDARY_SENSITIVITY.
      Everything else derived.
"""

import sqlite3
import json
import time
from typing import Optional, Iterator
from collections import OrderedDict

from core.atoms import (
    State, decay_rate, now,
    DERIVED, SLOT,
    BASE_DECAY, ACTIVATION_BOOST,
    STABLE_THRESHOLD, DISSOLVE_THRESHOLD, ACTIVATION_MIN,
)


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-32000;

CREATE TABLE IF NOT EXISTS nodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    level        INTEGER NOT NULL DEFAULT 0,
    components   TEXT    NOT NULL DEFAULT '[]',
    raw          TEXT    NOT NULL DEFAULT '',
    weight       REAL    NOT NULL DEFAULT 0.5,
    state        TEXT    NOT NULL DEFAULT 'provisional',
    created_at   REAL    NOT NULL,
    activated_at REAL    NOT NULL,
    decay_rate   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL REFERENCES nodes(id),
    target_id    INTEGER NOT NULL REFERENCES nodes(id),
    weight       REAL    NOT NULL DEFAULT 0.5,
    observed     INTEGER NOT NULL DEFAULT 1,
    created_at   REAL    NOT NULL,
    activated_at REAL    NOT NULL,
    decay_rate   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS hyperedges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    level        INTEGER NOT NULL DEFAULT 0,
    structure    TEXT    NOT NULL DEFAULT '[]',
    constraints  TEXT    NOT NULL DEFAULT '{}',
    instances    TEXT    NOT NULL DEFAULT '[]',
    weight       REAL    NOT NULL DEFAULT 0.5,
    state        TEXT    NOT NULL DEFAULT 'provisional',
    created_at   REAL    NOT NULL,
    activated_at REAL    NOT NULL,
    decay_rate   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_level       ON nodes(level);
CREATE INDEX IF NOT EXISTS idx_nodes_state       ON nodes(state);
CREATE INDEX IF NOT EXISTS idx_nodes_weight      ON nodes(weight);
CREATE INDEX IF NOT EXISTS idx_nodes_raw         ON nodes(raw);
CREATE INDEX IF NOT EXISTS idx_edges_source      ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target      ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_hyperedges_level  ON hyperedges(level);
CREATE INDEX IF NOT EXISTS idx_hyperedges_state  ON hyperedges(state);
"""


# ─────────────────────────────────────────────
# LRU CACHE
# ─────────────────────────────────────────────

class LRUCache:
    def __init__(self, capacity: int = 5000):
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


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

class Graph:
    def __init__(self, db_path: str = "graph.db"):
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        self._node_cache = LRUCache(5000)
        self._edge_cache = LRUCache(5000)


    # ─────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()


    # ─────────────────────────────────────────
    # NODE — ADD
    # ─────────────────────────────────────────

    def add_node(
        self,
        level      : int,
        components : list[int],
        raw        : str   = DERIVED,
        weight     : float = 0.5,
        state      : State = State.PROVISIONAL,
    ) -> int:
        """
        Add a new node. Returns node ID.

        Level 0 nodes: components=[], raw=the actual token
        Level N nodes: components=[list of level N-1 node IDs], raw=DERIVED
        """
        t = now()
        dr = decay_rate(level)

        cur = self._conn.execute("""
            INSERT INTO nodes
            (level, components, raw, weight, state, created_at, activated_at, decay_rate)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            level,
            json.dumps(components),
            raw,
            weight,
            state.value,
            t, t, dr,
        ))
        self._conn.commit()
        return cur.lastrowid


    def get_node(self, node_id: int) -> Optional[dict]:
        """Get node by ID. Cache-first."""
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached

        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id=?", (node_id,)
        ).fetchone()

        if row is None:
            return None

        node = dict(row)
        node["components"] = json.loads(node["components"])
        self._node_cache.put(node_id, node)
        return node


    def update_node_weight(self, node_id: int, weight: float):
        """Update node weight. Invalidates cache."""
        weight = max(0.0, min(1.0, weight))
        self._conn.execute(
            "UPDATE nodes SET weight=? WHERE id=?", (weight, node_id)
        )
        self._conn.commit()
        self._node_cache.invalidate(node_id)


    def update_node_state(self, node_id: int, state: State):
        """Update node state. Invalidates cache."""
        self._conn.execute(
            "UPDATE nodes SET state=? WHERE id=?", (state.value, node_id)
        )
        self._conn.commit()
        self._node_cache.invalidate(node_id)


    def activate_node(self, node_id: int):
        """
        Record activation — boost weight, reset activated_at.
        This is what prevents decay from dissolving useful nodes.
        """
        t = now()
        self._conn.execute("""
            UPDATE nodes
            SET weight       = MIN(1.0, weight + ?),
                activated_at = ?
            WHERE id = ?
        """, (ACTIVATION_BOOST, t, node_id))
        self._conn.commit()
        self._node_cache.invalidate(node_id)


    def find_node(
        self,
        level      : int,
        components : list[int],
        raw        : str = DERIVED,
    ) -> Optional[int]:
        """
        Find an existing node by its identity (level + components or raw).
        Returns node ID if found, None otherwise.
        """
        if level == 0:
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE level=0 AND raw=? LIMIT 1",
                (raw,)
            ).fetchone()
        else:
            comp_str = json.dumps(sorted(components))
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE level=? AND components=? LIMIT 1",
                (level, json.dumps(components))
            ).fetchone()

        return row[0] if row else None


    def get_or_create_node(
        self,
        level      : int,
        components : list[int],
        raw        : str   = DERIVED,
        weight     : float = 0.5,
        state      : State = State.PROVISIONAL,
    ) -> tuple[int, bool]:
        """
        Get existing node or create new one.
        Returns (node_id, is_new).
        """
        existing = self.find_node(level, components, raw)
        if existing is not None:
            return existing, False
        nid = self.add_node(level, components, raw, weight, state)
        return nid, True


    def nodes_by_state(self, state: State, limit: int = 1000) -> list[dict]:
        """All nodes in a given state."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE state=? ORDER BY weight DESC LIMIT ?",
            (state.value, limit)
        ).fetchall()
        nodes = []
        for row in rows:
            n = dict(row)
            n["components"] = json.loads(n["components"])
            nodes.append(n)
        return nodes


    def nodes_by_level(self, level: int, limit: int = 1000) -> list[dict]:
        """All stable nodes at a given level."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE level=? AND state=? ORDER BY weight DESC LIMIT ?",
            (level, State.STABLE.value, limit)
        ).fetchall()
        nodes = []
        for row in rows:
            n = dict(row)
            n["components"] = json.loads(n["components"])
            nodes.append(n)
        return nodes


    def search_nodes_by_raw(
        self,
        text  : str,
        level : int = 0,
        limit : int = 10,
    ) -> list[dict]:
        """Find level-0 nodes whose raw text contains the given string."""
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE level=? AND raw LIKE ? ORDER BY weight DESC LIMIT ?",
            (level, f"%{text}%", limit)
        ).fetchall()
        nodes = []
        for row in rows:
            n = dict(row)
            n["components"] = json.loads(n["components"])
            nodes.append(n)
        return nodes


    def all_node_ids(
        self,
        state : Optional[State] = None,
        level : Optional[int]   = None,
    ) -> list[int]:
        """All node IDs, optionally filtered."""
        if state and level is not None:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE state=? AND level=?",
                (state.value, level)
            ).fetchall()
        elif state:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE state=?", (state.value,)
            ).fetchall()
        elif level is not None:
            rows = self._conn.execute(
                "SELECT id FROM nodes WHERE level=?", (level,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT id FROM nodes").fetchall()
        return [r[0] for r in rows]


    # ─────────────────────────────────────────
    # EDGE — ADD
    # ─────────────────────────────────────────

    def add_edge(
        self,
        source_id : int,
        target_id : int,
        weight    : float = 0.5,
    ) -> int:
        """Add a directed edge. Returns edge ID."""
        t  = now()
        # Edge level = min of node levels — conservative
        src = self.get_node(source_id)
        tgt = self.get_node(target_id)
        level = min(
            src["level"] if src else 0,
            tgt["level"] if tgt else 0,
        )
        dr = decay_rate(level)

        cur = self._conn.execute("""
            INSERT INTO edges
            (source_id, target_id, weight, observed, created_at, activated_at, decay_rate)
            VALUES (?,?,?,1,?,?,?)
        """, (source_id, target_id, weight, t, t, dr))
        self._conn.commit()
        return cur.lastrowid


    def get_edge(self, edge_id: int) -> Optional[dict]:
        """Get edge by ID. Cache-first."""
        cached = self._edge_cache.get(edge_id)
        if cached is not None:
            return cached

        row = self._conn.execute(
            "SELECT * FROM edges WHERE id=?", (edge_id,)
        ).fetchone()

        if row is None:
            return None

        edge = dict(row)
        self._edge_cache.put(edge_id, edge)
        return edge


    def find_edge(self, source_id: int, target_id: int) -> Optional[int]:
        """Find existing edge between two nodes. Returns edge ID or None."""
        row = self._conn.execute(
            "SELECT id FROM edges WHERE source_id=? AND target_id=? LIMIT 1",
            (source_id, target_id)
        ).fetchone()
        return row[0] if row else None


    def get_or_create_edge(
        self,
        source_id : int,
        target_id : int,
        weight    : float = 0.5,
    ) -> tuple[int, bool]:
        """Get existing edge or create. Returns (edge_id, is_new)."""
        existing = self.find_edge(source_id, target_id)
        if existing is not None:
            return existing, False
        eid = self.add_edge(source_id, target_id, weight)
        return eid, True


    def activate_edge(self, edge_id: int):
        """Boost edge weight and reset decay timer."""
        t = now()
        self._conn.execute("""
            UPDATE edges
            SET weight       = MIN(1.0, weight + ?),
                observed     = observed + 1,
                activated_at = ?
            WHERE id = ?
        """, (ACTIVATION_BOOST, t, edge_id))
        self._conn.commit()
        self._edge_cache.invalidate(edge_id)


    def get_outgoing_edges(self, node_id: int) -> list[dict]:
        """All edges where source_id = node_id."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_id=? ORDER BY weight DESC",
            (node_id,)
        ).fetchall()
        edges = []
        for row in rows:
            e = dict(row)
            self._edge_cache.put(e["id"], e)
            edges.append(e)
        return edges


    def get_incoming_edges(self, node_id: int) -> list[dict]:
        """All edges where target_id = node_id."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target_id=? ORDER BY weight DESC",
            (node_id,)
        ).fetchall()
        edges = []
        for row in rows:
            e = dict(row)
            self._edge_cache.put(e["id"], e)
            edges.append(e)
        return edges


    # ─────────────────────────────────────────
    # HYPEREDGE — ADD
    # ─────────────────────────────────────────

    def add_hyperedge(
        self,
        level       : int,
        structure   : list,       # list of (node_id | SLOT)
        constraints : dict = None,
        instances   : list = None,
        weight      : float = 0.5,
        state       : State = State.PROVISIONAL,
    ) -> int:
        """Add a hyperedge (schema). Returns hyperedge ID."""
        t  = now()
        dr = decay_rate(level)

        cur = self._conn.execute("""
            INSERT INTO hyperedges
            (level, structure, constraints, instances, weight, state,
             created_at, activated_at, decay_rate)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            level,
            json.dumps(structure),
            json.dumps(constraints or {}),
            json.dumps(instances   or []),
            weight,
            state.value,
            t, t, dr,
        ))
        self._conn.commit()
        return cur.lastrowid


    def get_hyperedge(self, hid: int) -> Optional[dict]:
        """Get hyperedge by ID."""
        row = self._conn.execute(
            "SELECT * FROM hyperedges WHERE id=?", (hid,)
        ).fetchone()
        if row is None:
            return None
        h = dict(row)
        h["structure"]   = json.loads(h["structure"])
        h["constraints"] = json.loads(h["constraints"])
        h["instances"]   = json.loads(h["instances"])
        return h


    def add_instance_to_hyperedge(self, hid: int, node_id: int):
        """Add a node as a known instance of a hyperedge."""
        h = self.get_hyperedge(hid)
        if h is None:
            return
        instances = h["instances"]
        if node_id not in instances:
            instances.append(node_id)
            self._conn.execute(
                "UPDATE hyperedges SET instances=? WHERE id=?",
                (json.dumps(instances), hid)
            )
            self._conn.commit()


    def activate_hyperedge(self, hid: int):
        """Boost hyperedge weight and reset decay timer."""
        t = now()
        self._conn.execute("""
            UPDATE hyperedges
            SET weight       = MIN(1.0, weight + ?),
                activated_at = ?
            WHERE id = ?
        """, (ACTIVATION_BOOST, t, hid))
        self._conn.commit()


    def hyperedges_by_level(self, level: int) -> list[dict]:
        """All stable hyperedges at a given level."""
        rows = self._conn.execute(
            "SELECT * FROM hyperedges WHERE level=? AND state=? ORDER BY weight DESC",
            (level, State.STABLE.value)
        ).fetchall()
        result = []
        for row in rows:
            h = dict(row)
            h["structure"]   = json.loads(h["structure"])
            h["constraints"] = json.loads(h["constraints"])
            h["instances"]   = json.loads(h["instances"])
            result.append(h)
        return result


    # ─────────────────────────────────────────
    # SURFACE TEXT — derived, never stored
    # ─────────────────────────────────────────

    def surface(self, node_id: int, max_depth: int = 10) -> str:
        """
        Derive surface text from a node by walking down to level-0 components.
        Level 0: return raw field directly.
        Level N: recursively derive from components, join with space.
        """
        node = self.get_node(node_id)
        if node is None:
            return f"[missing:{node_id}]"

        if node["level"] == 0:
            return node["raw"]

        if max_depth <= 0:
            return f"[level:{node['level']}]"

        parts = []
        for comp_id in node["components"]:
            parts.append(self.surface(comp_id, max_depth - 1))

        return " ".join(parts)


    # ─────────────────────────────────────────
    # CO-OCCURRENCE COUNTING
    # Used by compression for PMI calculation
    # ─────────────────────────────────────────

    def count_node_observations(self, node_id: int) -> int:
        """How many times has this node been observed (edge activations)."""
        row = self._conn.execute(
            "SELECT SUM(observed) FROM edges WHERE source_id=? OR target_id=?",
            (node_id, node_id)
        ).fetchone()
        return int(row[0] or 0)


    def count_edge_observations(self, source_id: int, target_id: int) -> int:
        """How many times has this specific transition been observed."""
        row = self._conn.execute(
            "SELECT observed FROM edges WHERE source_id=? AND target_id=? LIMIT 1",
            (source_id, target_id)
        ).fetchone()
        return int(row[0]) if row else 0


    def total_observations(self) -> int:
        """Total edge observations across the entire graph."""
        row = self._conn.execute(
            "SELECT SUM(observed) FROM edges"
        ).fetchone()
        return int(row[0] or 1)


    # ─────────────────────────────────────────
    # DECAY ITERATION
    # Used by decay process to find nodes needing update
    # ─────────────────────────────────────────

    def iter_all_nodes_for_decay(
        self,
        batch_size: int = 200,
    ) -> Iterator[list[dict]]:
        """Iterate all nodes in batches for decay processing."""
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT * FROM nodes ORDER BY activated_at ASC LIMIT ? OFFSET ?",
                (batch_size, offset)
            ).fetchall()
            if not rows:
                break
            batch = []
            for row in rows:
                n = dict(row)
                n["components"] = json.loads(n["components"])
                batch.append(n)
            yield batch
            offset += batch_size


    def iter_all_edges_for_decay(
        self,
        batch_size: int = 200,
    ) -> Iterator[list[dict]]:
        """Iterate all edges in batches for decay processing."""
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT * FROM edges ORDER BY activated_at ASC LIMIT ? OFFSET ?",
                (batch_size, offset)
            ).fetchall()
            if not rows:
                break
            yield [dict(row) for row in rows]
            offset += batch_size


    def delete_node(self, node_id: int):
        """Remove a dissolved node and its edges."""
        self._conn.execute("DELETE FROM edges WHERE source_id=? OR target_id=?",
                           (node_id, node_id))
        self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self._conn.commit()
        self._node_cache.invalidate(node_id)


    def delete_edge(self, edge_id: int):
        """Remove a dissolved edge."""
        self._conn.execute("DELETE FROM edges WHERE id=?", (edge_id,))
        self._conn.commit()
        self._edge_cache.invalidate(edge_id)


    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def stats(self) -> dict:
        """Graph health metrics."""
        node_count = self._conn.execute(
            "SELECT COUNT(*) FROM nodes"
        ).fetchone()[0]

        edge_count = self._conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]

        hyperedge_count = self._conn.execute(
            "SELECT COUNT(*) FROM hyperedges"
        ).fetchone()[0]

        state_counts = {}
        for state in State:
            c = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE state=?", (state.value,)
            ).fetchone()[0]
            state_counts[state.value] = c

        max_level = self._conn.execute(
            "SELECT MAX(level) FROM nodes WHERE state=?",(State.STABLE.value,)
        ).fetchone()[0] or 0

        avg_weight = self._conn.execute(
            "SELECT AVG(weight) FROM nodes WHERE state=?",(State.STABLE.value,)
        ).fetchone()[0] or 0.0

        return {
            "nodes"          : node_count,
            "edges"          : edge_count,
            "hyperedges"     : hyperedge_count,
            "state_counts"   : state_counts,
            "max_level"      : max_level,
            "avg_weight"     : round(avg_weight, 4),
            "db_path"        : self.db_path,
        }


    # ─────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────

    def close(self):
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

        with Graph(db_path) as g:
            # Add level-0 nodes (primitives)
            h_id, _ = g.get_or_create_node(0, [], raw="hello",
                                            state=State.STABLE)
            w_id, _ = g.get_or_create_node(0, [], raw="world",
                                            state=State.STABLE)
            print(f"Level-0 nodes: {h_id}, {w_id}")

            # Add level-1 node (composition of level-0)
            l1_id, _ = g.get_or_create_node(1, [h_id, w_id],
                                             state=State.STABLE)
            print(f"Level-1 node: {l1_id}")
            print(f"Surface text: '{g.surface(l1_id)}'")

            # Add edge
            eid, is_new = g.get_or_create_edge(h_id, w_id)
            print(f"Edge {eid} (new={is_new})")

            # Activate
            g.activate_node(h_id)
            g.activate_edge(eid)

            # Add hyperedge (schema)
            hype_id = g.add_hyperedge(
                level      = 1,
                structure  = [SLOT, 0, SLOT],
                constraints= {"slot_2": "slot_0 + slot_1"},
                instances  = [l1_id],
                state      = State.STABLE,
            )
            print(f"Hyperedge: {hype_id}")

            # Stats
            print(f"\nStats: {g.stats()}")

            # Find node
            found = g.find_node(0, [], raw="hello")
            print(f"Find 'hello': node_id={found}")

            # Duplicate prevention
            h_id2, is_new2 = g.get_or_create_node(0, [], raw="hello",
                                                   state=State.STABLE)
            print(f"Duplicate 'hello': id={h_id2}, is_new={is_new2}")
            assert h_id2 == h_id, "Deduplication failed"
            print("Deduplication: OK")
