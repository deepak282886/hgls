"""
projection.py

Process 3: Projection — anticipation.

Takes stable schemas and projects them forward.
Asks: if this schema is true, what instances should exist
that we haven't seen yet?

Creates hypothetical nodes as targets for exploration.
When ingestion confirms a hypothetical — it becomes stable.
When ingestion contradicts one — it dissolves.

Also handles analogy: finding schemas with similar structure
across different domains and creating meta-schema hypotheticals.

What projection does NOT do:
    - Read raw input (ingestion does that)
    - Compress observations (compression does that)
    - Remove anything (decay does that)
    - Create stable nodes directly — only hypotheticals

Projection only generates hypotheticals and tracks resolution.
"""

import json
import random
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from core.atoms import (
    State, now, SLOT,
    MIN_INSTANCES_FOR_SCHEMA,
    ACTIVATION_BOOST,
)
from core.graph import Graph


# ─────────────────────────────────────────────
# PROJECTION RESULT
# ─────────────────────────────────────────────

@dataclass
class ProjectionResult:
    hypotheticals_created  : int = 0
    hypotheticals_confirmed: int = 0   # by checking against existing stable nodes
    hypotheticals_dissolved: int = 0
    analogies_found        : int = 0
    meta_schemas_created   : int = 0


# ─────────────────────────────────────────────
# HYPOTHESIS
# A predicted node that hasn't been observed yet
# ─────────────────────────────────────────────

@dataclass
class Hypothesis:
    parent_schema_id : int         # which schema generated this
    slot_bindings    : dict        # {slot_position: node_id}
    predicted_level  : int
    confidence       : float       # how strongly schema predicts this


# ─────────────────────────────────────────────
# PROJECTION ENGINE
# ─────────────────────────────────────────────

class Projection:
    """
    The anticipation engine. Generates hypothetical nodes
    from stable schemas by filling slots with known values.

    Usage:
        projection = Projection(graph)
        result     = projection.run()
    """

    def __init__(self, graph: Graph):
        self.graph = graph


    # ─────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────

    def run(self) -> ProjectionResult:
        """
        Run one projection cycle.
        Generates hypotheticals from all stable schemas.
        Checks if any existing stable nodes confirm hypotheticals.
        """
        result = ProjectionResult()

        # Step 1: generate hypotheticals from stable schemas
        self._project_from_schemas(result)

        # Step 2: check hypotheticals against existing stable nodes
        self._check_hypotheticals(result)

        # Step 3: find analogies between schemas
        self._find_analogies(result)

        return result


    # ─────────────────────────────────────────
    # STEP 1: PROJECT FROM SCHEMAS
    # ─────────────────────────────────────────

    def _project_from_schemas(self, result: ProjectionResult):
        """
        For each stable schema — generate hypothetical instances
        by combining known slot values in new ways.
        """
        # Get all stable schemas at each level
        for level in range(0, 4):
            schemas = self.graph.hyperedges_by_level(level)
            for schema in schemas:
                self._project_schema(schema, result)


    def _project_schema(self, schema: dict, result: ProjectionResult):
        """
        Generate new hypothetical instances from one schema.

        Strategy:
            1. Collect all known slot values from existing instances
            2. Generate new combinations not yet in instances
            3. Create hypothetical nodes for each new combination
        """
        structure  = schema["structure"]
        instances  = schema["instances"]
        schema_id  = schema["id"]

        if not instances:
            return

        # Find slot positions
        slot_positions = [
            i for i, s in enumerate(structure) if s == SLOT
        ]
        fixed_positions = [
            i for i, s in enumerate(structure) if s != SLOT
        ]

        if not slot_positions:
            return  # no slots — nothing to project

        # Collect known slot values from existing instances
        slot_values = defaultdict(set)

        for inst_id in instances:
            inst_node = self.graph.get_node(inst_id)
            if inst_node is None:
                continue
            comps = inst_node["components"]
            if len(comps) != len(structure):
                continue
            for slot_pos in slot_positions:
                if slot_pos < len(comps):
                    slot_values[slot_pos].add(comps[slot_pos])

        if not slot_values:
            return

        # Generate new combinations
        # Simple 2-slot case: try all pairs of known values
        if len(slot_positions) == 1:
            sp = slot_positions[0]
            known_values = list(slot_values[sp])

            # Project: each known value is a valid slot fill
            # The hypothetical is: the same fixed positions + this slot value
            for val_id in known_values[:5]:  # limit projection breadth
                hypothesis = self._build_hypothesis(
                    schema_id   = schema_id,
                    structure   = structure,
                    slot_fills  = {sp: val_id},
                    fixed_fills = {fp: structure[fp] for fp in fixed_positions},
                    level       = schema["level"],
                    confidence  = schema["weight"],
                )
                if hypothesis:
                    self._create_hypothetical(hypothesis, result)

        elif len(slot_positions) == 2:
            sp0, sp1 = slot_positions[0], slot_positions[1]
            vals0    = list(slot_values[sp0])
            vals1    = list(slot_values[sp1])

            # Cross-combine known values
            for v0 in vals0[:3]:
                for v1 in vals1[:3]:
                    hypothesis = self._build_hypothesis(
                        schema_id   = schema_id,
                        structure   = structure,
                        slot_fills  = {sp0: v0, sp1: v1},
                        fixed_fills = {fp: structure[fp] for fp in fixed_positions},
                        level       = schema["level"],
                        confidence  = schema["weight"],
                    )
                    if hypothesis:
                        self._create_hypothetical(hypothesis, result)


    def _build_hypothesis(
        self,
        schema_id   : int,
        structure   : list,
        slot_fills  : dict,
        fixed_fills : dict,
        level       : int,
        confidence  : float,
    ) -> Optional[Hypothesis]:
        """
        Build a hypothesis by filling slots with specific values.
        Returns None if this combination already exists as a stable node.
        """
        # Build the component list for the hypothetical node
        components = []
        for i, pos in enumerate(structure):
            if i in slot_fills:
                components.append(slot_fills[i])
            elif i in fixed_fills:
                val = fixed_fills[i]
                if isinstance(val, int):
                    components.append(val)
                # else skip non-node fixed positions

        if not components:
            return None

        # Check if this already exists as a stable node
        existing = self.graph.find_node(level + 1, components)
        if existing is not None:
            node = self.graph.get_node(existing)
            if node and node["state"] == State.STABLE.value:
                return None  # already confirmed, no need for hypothesis

        return Hypothesis(
            parent_schema_id = schema_id,
            slot_bindings    = slot_fills,
            predicted_level  = level + 1,
            confidence       = confidence,
        )


    def _create_hypothetical(
        self,
        hypothesis : Hypothesis,
        result     : ProjectionResult,
    ):
        """
        Create a hypothetical node in the graph.
        """
        # Build components from slot bindings
        components = list(hypothesis.slot_bindings.values())

        if not components:
            return

        # Check if already exists
        existing = self.graph.find_node(
            hypothesis.predicted_level, components
        )
        if existing is not None:
            return  # already exists

        # Create hypothetical node
        self.graph.add_node(
            level      = hypothesis.predicted_level,
            components = components,
            weight     = hypothesis.confidence * 0.5,  # starts at half confidence
            state      = State.HYPOTHETICAL,
        )
        result.hypotheticals_created += 1


    # ─────────────────────────────────────────
    # STEP 2: CHECK HYPOTHETICALS
    # ─────────────────────────────────────────

    def _check_hypotheticals(self, result: ProjectionResult):
        """
        Check all hypothetical nodes against existing stable nodes.
        If a stable node matches a hypothetical's structure — confirm it.
        """
        hypotheticals = self.graph.nodes_by_state(State.HYPOTHETICAL)

        for hyp in hypotheticals:
            hyp_id    = hyp["id"]
            hyp_comps = hyp["components"]
            hyp_level = hyp["level"]

            # Look for a stable node with the same components and level
            existing = self.graph.find_node(hyp_level, hyp_comps)

            if existing is not None:
                node = self.graph.get_node(existing)
                if node and node["state"] == State.STABLE.value:
                    # Confirmed — dissolve the hypothetical
                    # (the stable node is the confirmation)
                    self.graph.delete_node(hyp_id)
                    result.hypotheticals_confirmed += 1

                    # Boost the parent schema weight
                    # (it made a correct prediction)
                    # Find which schema spawned this — look at hyperedges
                    # that contain this node as instance
                    self._boost_parent_schemas(existing)


    def _boost_parent_schemas(self, node_id: int):
        """Boost weight of schemas that predicted this node."""
        # Find hyperedges that have this node as an instance
        rows = self.graph._conn.execute(
            "SELECT id, instances FROM hyperedges WHERE state=?",
            (State.STABLE.value,)
        ).fetchall()

        for row in rows:
            instances = json.loads(row["instances"])
            if node_id in instances:
                self.graph.activate_hyperedge(row["id"])


    # ─────────────────────────────────────────
    # STEP 3: FIND ANALOGIES
    # ─────────────────────────────────────────

    def _find_analogies(self, result: ProjectionResult):
        """
        Find schemas with structurally similar patterns.
        Two schemas are analogous if they have the same slot/fixed arrangement
        but different fixed nodes.

        Example:
            Schema A: [hello, SLOT]   — "hello ___"
            Schema B: [goodbye, SLOT] — "goodbye ___"
            These are analogous — same structure, different anchor.

        Analogous schemas suggest a meta-schema:
            [SLOT, SLOT] — "[greeting] ___"
        """
        all_schemas = []
        for level in range(0, 4):
            schemas = self.graph.hyperedges_by_level(level)
            all_schemas.extend(schemas)

        if len(all_schemas) < 2:
            return

        # Group schemas by their slot pattern (which positions are slots)
        by_slot_pattern = defaultdict(list)
        for schema in all_schemas:
            pattern = tuple(
                "SLOT" if s == SLOT else "FIXED"
                for s in schema["structure"]
            )
            by_slot_pattern[pattern].append(schema)

        # Groups with 2+ schemas sharing same slot pattern are analogous
        for pattern, group in by_slot_pattern.items():
            if len(group) < 2:
                continue

            # Create a meta-schema where fixed positions also become slots
            meta_structure = [SLOT] * len(pattern)

            # Check if this meta-schema already exists
            meta_str = json.dumps(meta_structure)
            existing = self.graph._conn.execute(
                "SELECT id FROM hyperedges WHERE structure=? LIMIT 1",
                (meta_str,)
            ).fetchone()

            if existing:
                continue

            # New meta-schema — create as hypothetical
            # Level = level of schemas + 1
            base_level = group[0]["level"]
            meta_instances = [s["id"] for s in group]

            self.graph.add_hyperedge(
                level      = base_level + 1,
                structure  = meta_structure,
                constraints= {},
                instances  = meta_instances,
                weight     = 0.3,
                state      = State.HYPOTHETICAL,
            )
            result.meta_schemas_created += 1
            result.analogies_found      += len(group)


    # ─────────────────────────────────────────
    # CONFIRM HYPOTHETICAL
    # Called by ingestion when observation matches
    # ─────────────────────────────────────────

    def confirm_hypothetical(self, node_id: int) -> bool:
        """
        Called when ingestion observes something that matches
        a hypothetical node. Promotes it to stable.
        Returns True if a hypothetical was confirmed.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return False

        if node["state"] != State.HYPOTHETICAL.value:
            return False

        # Promote to stable
        self.graph.update_node_state(node_id, State.STABLE)
        self.graph.update_node_weight(node_id, min(1.0, node["weight"] + 0.2))
        self._boost_parent_schemas(node_id)
        return True


    # ─────────────────────────────────────────
    # UNCERTAINTY MAP
    # Where is the system most uncertain?
    # This drives the exploration signal.
    # ─────────────────────────────────────────

    def uncertainty_map(self, top_n: int = 10) -> list[dict]:
        """
        Return the most uncertain regions of the graph.
        These are hypothetical nodes that have been waiting longest
        without confirmation or refutation.

        High uncertainty = the system has a prediction but no feedback.
        This is where exploration attention should go.
        """
        rows = self.graph._conn.execute("""
            SELECT id, level, weight, created_at, activated_at
            FROM nodes
            WHERE state = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (State.HYPOTHETICAL.value, top_n)).fetchall()

        result = []
        t = now()
        for row in rows:
            age = t - row["created_at"]
            result.append({
                "node_id"    : row["id"],
                "level"      : row["level"],
                "weight"     : row["weight"],
                "age_seconds": round(age, 1),
                "uncertainty": 1.0 - row["weight"],
            })

        return result


if __name__ == "__main__":
    import tempfile, os
    from core.ingestion   import Ingestion
    from core.compression import Compression

    print("=== projection.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        ing     = Ingestion(graph)
        comp    = Compression(graph)
        proj    = Projection(graph)

        # Feed structured patterns — same structure, different words
        patterns = [
            "hello world",
            "hello friend",
            "hello there",
            "goodbye world",
            "goodbye friend",
            "goodbye there",
            "hello everyone",
            "goodbye everyone",
        ]

        print("Ingesting patterns...")
        for p in patterns * 5:  # repeat for recurrence
            ing.reset_context()
            ing.process_text(p)

        # Compress
        r1 = comp.run()
        print(f"Compression: {r1}")

        # Project
        r2 = proj.run()
        print(f"Projection: {r2}")

        stats = graph.stats()
        print(f"Stats: {stats}\n")

        # Show hypotheticals
        hyps = graph.nodes_by_state(State.HYPOTHETICAL)
        print(f"Hypothetical nodes ({len(hyps)}):")
        for h in hyps[:5]:
            print(f"  [{h['id']}] level={h['level']} "
                  f"components={h['components']} "
                  f"weight={h['weight']:.3f}")

        # Uncertainty map
        print(f"\nUncertainty map (top 5):")
        for u in proj.uncertainty_map(5):
            print(f"  node={u['node_id']}  "
                  f"uncertainty={u['uncertainty']:.3f}  "
                  f"age={u['age_seconds']}s")

        graph.close()
