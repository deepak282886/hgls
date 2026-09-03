"""
compression.py

Process 2: Compression — the metabolism.

Watches provisional nodes. When patterns have recurred enough
with consistent internal structure — promotes them to stable nodes
and compresses them upward to the next level.

Two directions:
    Bottom-up  — provisional patterns → stable level-0 nodes
    Lateral    — stable level-N nodes co-occurring → level-N+1 nodes

Slot detection:
    When multiple instances of a pattern share structure but vary
    in some positions — those varying positions become slots.
    The pattern becomes a hyperedge (schema).

What compression does NOT do:
    - Read raw input (ingestion does that)
    - Generate hypotheticals (projection does that)
    - Remove anything (decay does that)

Compression only promotes. It is the only process that creates
new stable nodes and hyperedges.
"""

import json
import math
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from core.atoms import (
    State, now, pmi,
    COMPRESSION_THRESHOLD,
    MIN_INSTANCES_FOR_SCHEMA,
    MIN_RECURRENCE,
    PROVISIONAL_TTL,
    SLOT,
)
from core.graph import Graph


# ─────────────────────────────────────────────
# COMPRESSION RESULT
# ─────────────────────────────────────────────

@dataclass
class CompressionResult:
    nodes_promoted  : int = 0   # provisional → stable
    nodes_dissolved : int = 0   # provisional → gone (expired, no recurrence)
    level1_created  : int = 0   # new level-1 nodes from co-occurrence
    schemas_created : int = 0   # new hyperedges from slot detection
    schemas_promoted: int = 0   # provisional hyperedges → stable


# ─────────────────────────────────────────────
# COMPRESSION ENGINE
# ─────────────────────────────────────────────

class Compression:
    """
    The metabolism. Promotes structure from provisional to stable.
    Runs periodically — not on every token.

    Usage:
        compression = Compression(graph)
        result      = compression.run()
    """

    def __init__(self, graph: Graph):
        self.graph = graph


    # ─────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────

    def run(self) -> CompressionResult:
        """
        Run one compression cycle.
        Checks all provisional nodes and level-0 stable co-occurrences.
        """
        result = CompressionResult()

        # Step 1: promote or dissolve provisional level-0 nodes
        self._compress_provisional(result)

        # Step 2: find co-occurring stable level-0 nodes → level-1
        self._compress_cooccurrence(result)

        # Step 3: detect slots in level-1 patterns → hyperedges
        self._detect_schemas(result)

        # Step 4: promote provisional hyperedges that have enough instances
        self._promote_schemas(result)

        return result


    # ─────────────────────────────────────────
    # STEP 1: PROVISIONAL → STABLE or DISSOLVE
    # ─────────────────────────────────────────

    def _compress_provisional(self, result: CompressionResult):
        """
        Examine all provisional level-0 nodes.

        Promote if: observed enough times (edge count crosses threshold)
        Dissolve if: too old without recurrence (TTL expired)
        """
        provisional = self.graph.nodes_by_state(State.PROVISIONAL)
        t           = now()

        for node in provisional:
            if node["level"] != 0:
                continue

            node_id  = node["id"]
            age      = t - node["created_at"]
            obs      = self.graph.count_node_observations(node_id)

            # Check recurrence — how many times has this token appeared?
            # We use incoming + outgoing edge observation count as proxy
            if obs >= MIN_RECURRENCE:
                # Promote to stable
                self.graph.update_node_state(node_id, State.STABLE)
                # Boost weight based on observation count
                new_weight = min(1.0, 0.3 + obs * 0.05)
                self.graph.update_node_weight(node_id, new_weight)
                result.nodes_promoted += 1

            elif age > PROVISIONAL_TTL:
                # Too old, never recurred — dissolve
                self.graph.delete_node(node_id)
                result.nodes_dissolved += 1


    # ─────────────────────────────────────────
    # STEP 2: CO-OCCURRENCE → LEVEL-1 NODES
    # ─────────────────────────────────────────

    def _compress_cooccurrence(self, result: CompressionResult):
        """
        Find stable level-0 nodes that reliably co-occur.
        When PMI exceeds COMPRESSION_THRESHOLD — create level-1 node.

        This is how words become phrases, phrases become patterns.
        """
        total = self.graph.total_observations()
        if total < 10:
            return  # not enough data yet

        # Get all stable level-0 nodes
        stable_nodes = self.graph.nodes_by_state(State.STABLE)
        stable_l0    = [n for n in stable_nodes if n["level"] == 0]

        if len(stable_l0) < 2:
            return

        # For each stable node, look at its outgoing edges
        for node in stable_l0:
            nid    = node["id"]
            edges  = self.graph.get_outgoing_edges(nid)

            for edge in edges:
                tid = edge["target_id"]
                target = self.graph.get_node(tid)

                if target is None or target["state"] != State.STABLE.value:
                    continue
                if target["level"] != 0:
                    continue

                # Compute PMI between this pair
                count_a  = self.graph.count_node_observations(nid)
                count_b  = self.graph.count_node_observations(tid)
                count_ab = self.graph.count_edge_observations(nid, tid)

                score = pmi(count_ab, count_a, count_b, total)

                if score >= COMPRESSION_THRESHOLD:
                    # Strong enough co-occurrence — create level-1 node
                    l1_id, is_new = self.graph.get_or_create_node(
                        level      = 1,
                        components = [nid, tid],
                        weight     = score,
                        state      = State.STABLE,
                    )
                    if is_new:
                        result.level1_created += 1

                        # Create edges: each component → level-1 node
                        self.graph.get_or_create_edge(nid, l1_id, weight=score)
                        self.graph.get_or_create_edge(tid, l1_id, weight=score)


    # ─────────────────────────────────────────
    # STEP 3: SLOT DETECTION → HYPEREDGES
    # ─────────────────────────────────────────

    def _detect_schemas(self, result: CompressionResult):
        """
        Look at level-1 nodes that share structural similarity.
        When multiple level-1 nodes have the same fixed positions
        but varying component positions — those varying positions are slots.
        The shared structure becomes a hyperedge (schema).

        Example:
            level-1 node: [hello, world]
            level-1 node: [hello, there]
            level-1 node: [hello, friend]
            → Schema: [hello, SLOT]
              Instances: all three nodes
        """
        # Get all stable level-1 nodes
        l1_nodes = self.graph.nodes_by_state(State.STABLE)
        l1_nodes = [n for n in l1_nodes if n["level"] == 1]

        if len(l1_nodes) < MIN_INSTANCES_FOR_SCHEMA:
            return

        # Group by first component (find shared anchors)
        by_first  = defaultdict(list)
        by_second = defaultdict(list)

        for node in l1_nodes:
            comps = node["components"]
            if len(comps) == 2:
                by_first[comps[0]].append(node)
                by_second[comps[1]].append(node)

        # Find groups with enough instances — fixed first, varying second
        for anchor_id, group in by_first.items():
            if len(group) < MIN_INSTANCES_FOR_SCHEMA:
                continue

            # Check if second components vary
            second_ids = [n["components"][1] for n in group]
            if len(set(second_ids)) < MIN_INSTANCES_FOR_SCHEMA:
                continue  # second position doesn't vary enough

            # Schema found: [anchor_id, SLOT]
            structure  = [anchor_id, SLOT]
            instances  = [n["id"] for n in group]

            self._create_or_update_schema(
                level      = 1,
                structure  = structure,
                instances  = instances,
                result     = result,
            )

        # Fixed second, varying first
        for anchor_id, group in by_second.items():
            if len(group) < MIN_INSTANCES_FOR_SCHEMA:
                continue

            first_ids = [n["components"][0] for n in group]
            if len(set(first_ids)) < MIN_INSTANCES_FOR_SCHEMA:
                continue

            structure = [SLOT, anchor_id]
            instances = [n["id"] for n in group]

            self._create_or_update_schema(
                level      = 1,
                structure  = structure,
                instances  = instances,
                result     = result,
            )


    def _create_or_update_schema(
        self,
        level     : int,
        structure : list,
        instances : list[int],
        result    : CompressionResult,
    ):
        """
        Create a new hyperedge or update an existing one with new instances.
        """
        structure_str = json.dumps(structure)

        # Check if this schema already exists
        existing = self.graph._conn.execute(
            "SELECT id FROM hyperedges WHERE level=? AND structure=? LIMIT 1",
            (level, structure_str)
        ).fetchone()

        if existing:
            hid = existing[0]
            for inst_id in instances:
                self.graph.add_instance_to_hyperedge(hid, inst_id)
            self.graph.activate_hyperedge(hid)
        else:
            # New schema
            hid = self.graph.add_hyperedge(
                level      = level,
                structure  = structure,
                constraints= {},
                instances  = instances,
                weight     = 0.5,
                state      = State.PROVISIONAL,
            )
            result.schemas_created += 1


    # ─────────────────────────────────────────
    # STEP 4: PROMOTE PROVISIONAL HYPEREDGES
    # ─────────────────────────────────────────

    def _promote_schemas(self, result: CompressionResult):
        """
        Provisional hyperedges with enough instances → stable.
        """
        rows = self.graph._conn.execute(
            "SELECT * FROM hyperedges WHERE state=?",
            (State.PROVISIONAL.value,)
        ).fetchall()

        for row in rows:
            import json as _json
            instances = _json.loads(row["instances"])
            if len(instances) >= MIN_INSTANCES_FOR_SCHEMA:
                self.graph._conn.execute(
                    "UPDATE hyperedges SET state=? WHERE id=?",
                    (State.STABLE.value, row["id"])
                )
                self.graph._conn.commit()
                result.schemas_promoted += 1


    # ─────────────────────────────────────────
    # COMPRESSION SIGNAL
    # How much compression activity is happening?
    # Used by the main loop to gauge learning rate.
    # ─────────────────────────────────────────

    def compression_signal(self) -> float:
        """
        Current compression activity level.
        High = system is finding lots of new structure.
        Low  = system is saturated or data is unstructured.
        Returns [0, 1].
        """
        stats = self.graph.stats()
        total = stats["nodes"]
        if total == 0:
            return 0.0

        provisional = stats["state_counts"].get("provisional", 0)
        # High ratio of provisional to total = active learning
        return min(1.0, provisional / max(1, total))


if __name__ == "__main__":
    import tempfile, os
    from core.ingestion import Ingestion

    print("=== compression.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        ing     = Ingestion(graph)
        comp    = Compression(graph)

        # Feed repeated patterns so compression has signal
        sentences = [
            "the cat sat on the mat",
            "the cat sat on the roof",
            "the cat sat on the floor",
            "the dog sat on the mat",
            "the dog sat on the roof",
            "the bird sat on the mat",
            "the cat ran on the mat",
            "the cat ran on the floor",
        ]

        print("Ingesting sentences...")
        for sent in sentences:
            ing.reset_context()
            ing.process_text(sent)

        print(f"After ingestion: {graph.stats()}\n")

        # Run compression
        print("Running compression...")
        result = comp.run()
        print(f"Result: {result}")
        print(f"After compression: {graph.stats()}\n")

        # Run again — more data might compress now
        print("Feeding more data and compressing again...")
        for sent in sentences * 3:
            ing.reset_context()
            ing.process_text(sent)

        result2 = comp.run()
        print(f"Result 2: {result2}")
        print(f"Final stats: {graph.stats()}")

        # Show what level-1 nodes formed
        l1 = graph.nodes_by_level(1)
        print(f"\nLevel-1 nodes formed ({len(l1)}):")
        for n in l1[:10]:
            surface = graph.surface(n["id"])
            print(f"  [{n['id']}] '{surface}'  weight={n['weight']:.3f}")

        # Show schemas
        schemas = graph.hyperedges_by_level(1)
        print(f"\nSchemas formed ({len(schemas)}):")
        for s in schemas:
            structure_readable = []
            for pos in s["structure"]:
                if pos == SLOT:
                    structure_readable.append("SLOT")
                else:
                    structure_readable.append(
                        f"'{graph.surface(pos)}'"
                    )
            print(f"  [{s['id']}] {structure_readable}  "
                  f"instances={len(s['instances'])}  "
                  f"weight={s['weight']:.3f}")

        graph.close()
