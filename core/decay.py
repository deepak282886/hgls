"""
decay.py

Process 4: Decay — homeostasis.

Everything in the graph loses weight continuously.
Nodes and edges that are activated reset their decay.
Nodes and edges that aren't — weaken and eventually dissolve.

This is what makes the system living rather than just growing.
Without decay the graph only accumulates.
With decay the graph stays current and lean.

Four outcomes for any node:
    Reinforcement    — activated before weight drops too low. survives.
    Graceful fade    — weight drops slowly. still accessible but weaker.
    Threshold cross  — weight drops below activation threshold. dormant.
    Dissolution      — weight hits zero. removed from graph.

What decay does NOT do:
    - Read raw input (ingestion does that)
    - Create new structure (compression does that)
    - Generate hypotheticals (projection does that)

Decay only removes. It is the only process that shrinks the graph.
"""

import time
from dataclasses import dataclass
from typing import Optional

from core.atoms import (
    State, now,
    STABLE_THRESHOLD,
    DISSOLVE_THRESHOLD,
    ACTIVATION_MIN,
    BASE_DECAY,
)
from core.graph import Graph


# ─────────────────────────────────────────────
# DECAY RESULT
# ─────────────────────────────────────────────

@dataclass
class DecayResult:
    nodes_weakened   : int = 0   # weight reduced
    nodes_dormant    : int = 0   # fell below activation threshold
    nodes_dissolved  : int = 0   # removed from graph
    nodes_demoted    : int = 0   # stable → decaying state
    edges_weakened   : int = 0
    edges_dissolved  : int = 0


# ─────────────────────────────────────────────
# DECAY ENGINE
# ─────────────────────────────────────────────

class Decay:
    """
    Homeostasis engine. Runs continuously.
    Applies weight decay to all nodes and edges.

    Usage:
        decay  = Decay(graph)
        result = decay.run()   # call periodically
    """

    def __init__(self, graph: Graph):
        self.graph       = graph
        self._last_run   = now()


    # ─────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────

    def run(self) -> DecayResult:
        """
        Apply decay to all nodes and edges.
        Amount of decay is proportional to time since last run.
        """
        result    = DecayResult()
        t         = now()
        elapsed   = t - self._last_run
        self._last_run = t

        if elapsed <= 0:
            return result

        # Decay all nodes
        self._decay_nodes(elapsed, result)

        # Decay all edges
        self._decay_edges(elapsed, result)

        return result


    # ─────────────────────────────────────────
    # DECAY NODES
    # ─────────────────────────────────────────

    def _decay_nodes(self, elapsed: float, result: DecayResult):
        """Apply decay to all nodes."""
        for batch in self.graph.iter_all_nodes_for_decay():
            to_delete = []
            to_update = []

            for node in batch:
                node_id    = node["id"]
                weight     = node["weight"]
                dr         = node["decay_rate"]
                state      = node["state"]

                # Compute new weight
                # Decay is proportional to elapsed time
                decay_amount = dr * elapsed
                new_weight   = max(0.0, weight - decay_amount)

                if new_weight <= DISSOLVE_THRESHOLD:
                    # Dissolve — remove from graph
                    to_delete.append(node_id)
                    result.nodes_dissolved += 1

                elif new_weight != weight:
                    # Weakened — update weight
                    to_update.append((node_id, new_weight, state))
                    result.nodes_weakened += 1

                    # Check for state transitions
                    if (state == State.STABLE.value
                            and new_weight < STABLE_THRESHOLD):
                        result.nodes_demoted += 1

                    if new_weight < ACTIVATION_MIN:
                        result.nodes_dormant += 1

            # Batch updates
            for node_id in to_delete:
                try:
                    self.graph.delete_node(node_id)
                except Exception:
                    pass  # already deleted by cascade

            for node_id, new_weight, state in to_update:
                self.graph.update_node_weight(node_id, new_weight)

                # Demote stable nodes that have fallen too low
                if (state == State.STABLE.value
                        and new_weight < STABLE_THRESHOLD):
                    try:
                        self.graph.update_node_state(node_id, State.DECAYING)
                    except Exception:
                        pass


    # ─────────────────────────────────────────
    # DECAY EDGES
    # ─────────────────────────────────────────

    def _decay_edges(self, elapsed: float, result: DecayResult):
        """Apply decay to all edges."""
        for batch in self.graph.iter_all_edges_for_decay():
            to_delete = []
            to_update = []

            for edge in batch:
                edge_id    = edge["id"]
                weight     = edge["weight"]
                dr         = edge["decay_rate"]

                decay_amount = dr * elapsed
                new_weight   = max(0.0, weight - decay_amount)

                if new_weight <= DISSOLVE_THRESHOLD:
                    to_delete.append(edge_id)
                    result.edges_dissolved += 1
                elif new_weight != weight:
                    to_update.append((edge_id, new_weight))
                    result.edges_weakened += 1

            # Batch updates
            for edge_id in to_delete:
                try:
                    self.graph.delete_edge(edge_id)
                except Exception:
                    pass

            for edge_id, new_weight in to_update:
                self.graph._conn.execute(
                    "UPDATE edges SET weight=? WHERE id=?",
                    (new_weight, edge_id)
                )

            if to_update:
                self.graph._conn.commit()


    # ─────────────────────────────────────────
    # HEALTH CHECK
    # Is the graph in homeostatic balance?
    # ─────────────────────────────────────────

    def health(self) -> dict:
        """
        Report on graph health.
        Healthy = growth and decay roughly in balance.
        """
        stats = self.graph.stats()

        total      = stats["nodes"]
        stable     = stats["state_counts"].get(State.STABLE.value,     0)
        provisional= stats["state_counts"].get(State.PROVISIONAL.value,0)
        decaying   = stats["state_counts"].get(State.DECAYING.value,   0)
        hypothetical=stats["state_counts"].get(State.HYPOTHETICAL.value,0)

        # Stability ratio — what fraction of nodes are stable
        stability = stable / max(1, total)

        # Churn ratio — what fraction are in flux (provisional or decaying)
        churn = (provisional + decaying) / max(1, total)

        # Hypothesis ratio — what fraction are anticipated but unconfirmed
        hypothesis_ratio = hypothetical / max(1, total)

        return {
            "total_nodes"      : total,
            "stable"           : stable,
            "provisional"      : provisional,
            "decaying"         : decaying,
            "hypothetical"     : hypothetical,
            "stability_ratio"  : round(stability,         4),
            "churn_ratio"      : round(churn,             4),
            "hypothesis_ratio" : round(hypothesis_ratio,  4),
            "is_healthy"       : stability > 0.3 and churn < 0.5,
        }


if __name__ == "__main__":
    import tempfile, os, time
    from core.ingestion   import Ingestion
    from core.compression import Compression

    print("=== decay.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        ing     = Ingestion(graph)
        comp    = Compression(graph)
        dec     = Decay(graph)

        # Feed data
        sentences = [
            "the cat sat on the mat",
            "the cat sat on the roof",
            "the dog ran in the park",
        ]

        print("Ingesting...")
        for s in sentences * 5:
            ing.reset_context()
            ing.process_text(s)

        comp.run()
        print(f"After compression: {graph.stats()}\n")
        print(f"Health: {dec.health()}\n")

        # Run decay immediately — should change little
        r1 = dec.run()
        print(f"Decay run 1 (immediate): {r1}")
        print(f"Health: {dec.health()}\n")

        # Simulate time passing by manually lowering weights
        # (in real system decay accumulates over actual time)
        print("Simulating time passing — lowering weights manually...")
        graph._conn.execute(
            "UPDATE nodes SET weight = weight * 0.3, "
            "activated_at = activated_at - 86400"  # 1 day ago
        )
        graph._conn.execute(
            "UPDATE edges SET weight = weight * 0.3, "
            "activated_at = activated_at - 86400"
        )
        graph._conn.commit()

        # Now decay should find things to dissolve
        r2 = dec.run()
        print(f"Decay run 2 (after weight drop): {r2}")
        print(f"Health: {dec.health()}\n")

        # Re-feed some data — should reinforce surviving nodes
        print("Re-feeding data to reinforce...")
        for s in sentences[:1] * 3:
            ing.reset_context()
            ing.process_text(s)

        r3 = dec.run()
        print(f"Decay run 3 (after reinforcement): {r3}")
        print(f"Final health: {dec.health()}")
        print(f"Final stats: {graph.stats()}")

        graph.close()
