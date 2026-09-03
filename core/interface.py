"""
interface.py

The conversation interface. The system's mouth and ears.

A query arrives as text. The system:
    1. Ingests it (updates the graph — queries are observations too)
    2. Activates the relevant subgraph
    3. Runs rollouts from the activated region
    4. Returns the highest-confidence continuation

The response is whatever the system predicts comes next
given the query as context. If the system knows the domain
well — the prediction is good. If not — it says so honestly.

Crucially: the query feeds the metabolism.
Every exchange is training data for the next one.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from core.atoms import (
    State, now,
    MAX_ROLLOUT_DEPTH, MIN_CONFIDENCE,
    PARALLEL_ROLLOUTS, CONTEXT_WINDOW_SIZE,
    ACTIVATION_MIN,
)
from core.graph    import Graph
from core.loop     import LivingLoop
from core.memory   import WorkingMemory
from core.thinking import Thinker


# ─────────────────────────────────────────────
# ROLLOUT
# One path through the graph from a start node
# ─────────────────────────────────────────────

@dataclass
class Rollout:
    path        : list[int]   # node IDs visited
    surfaces    : list[str]   # surface text at each node
    confidence  : float       # product of edge weights along path
    total_weight: float       # sum of node weights along path


# ─────────────────────────────────────────────
# RESPONSE
# What the system returns to the conversation
# ─────────────────────────────────────────────

@dataclass
class Response:
    text        : str
    confidence  : float
    rollout     : Optional[Rollout]
    elapsed     : float


# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────

class Interface:
    """
    Conversation interface. Connects the living loop to dialogue.

    Usage:
        interface = Interface(loop)
        response  = interface.respond("what is a cat")
        print(response.text)
    """

    def __init__(self, loop: LivingLoop):
        self.loop    = loop
        self.graph   = loop.graph
        self.thinker = Thinker(loop.graph)


    # ─────────────────────────────────────────
    # RESPOND — main entry point
    # ─────────────────────────────────────────

    def respond(
        self,
        query       : str,
        speaker     : str = "user",
        max_depth   : int = MAX_ROLLOUT_DEPTH,
    ) -> Response:
        """
        Process a query and generate a response.

        Steps:
            1. Ingest the query (it's an observation)
            2. Activate relevant subgraph from current context
            3. Run parallel rollouts
            4. Return best rollout as response
        """
        t0 = time.time()

        # Step 1: ingest the query — it feeds the metabolism
        results = self.loop.ingest_turn(speaker, query)

        # Step 2: build working memory from query context
        memory    = WorkingMemory(self.graph)
        activated = self._activate_from_query(query, results)
        memory.load_context(
            activated + list(self.loop.state.context_window)
        )

        if not memory.attention_map:
            return Response(
                text       = "...",
                confidence = 0.0,
                rollout    = None,
                elapsed    = round(time.time() - t0, 4),
            )

        # Step 3: think — parallel rollouts with schema guidance
        rollout_results = self.thinker.think(memory, max_depth=max_depth)

        if not rollout_results:
            return Response(
                text       = "...",
                confidence = 0.0,
                rollout    = None,
                elapsed    = round(time.time() - t0, 4),
            )

        # Step 4: best rollout becomes response
        best = rollout_results[0]
        text = self.thinker.best_response_text(rollout_results)

        return Response(
            text       = text,
            confidence = round(best.confidence, 4),
            rollout    = None,   # RolloutResult not Rollout — fine for now
            elapsed    = round(time.time() - t0, 4),
        )


    # ─────────────────────────────────────────
    # ACTIVATE FROM QUERY
    # Find which nodes the query lights up
    # ─────────────────────────────────────────

    def _activate_from_query(
        self,
        query   : str,
        results : list,
    ) -> list[int]:
        """
        Find the most activated nodes relevant to this query.

        Uses three signals:
            1. Nodes directly mentioned in the query (just ingested)
            2. High-weight neighbors of those nodes
            3. Current context window nodes

        Returns list of node IDs ordered by activation strength.
        """
        activation = {}  # node_id → activation score

        # Signal 1: nodes from the query itself
        for r in results:
            if r.node_id > 0:
                node = self.graph.get_node(r.node_id)
                if node and node["weight"] >= ACTIVATION_MIN:
                    activation[r.node_id] = node["weight"]

        # Signal 2: high-weight neighbors
        for node_id in list(activation.keys()):
            edges = self.graph.get_outgoing_edges(node_id)
            for edge in edges[:3]:  # top 3 outgoing
                tid  = edge["target_id"]
                tnode = self.graph.get_node(tid)
                if tnode and tnode["weight"] >= ACTIVATION_MIN:
                    score = edge["weight"] * tnode["weight"]
                    if tid not in activation or activation[tid] < score:
                        activation[tid] = score

        # Signal 3: context window (what was recently active)
        for nid in self.loop.state.context_window:
            if nid not in activation:
                node = self.graph.get_node(nid)
                if node and node["weight"] >= ACTIVATION_MIN:
                    activation[nid] = node["weight"] * 0.5  # lower weight

        # Sort by activation, return top nodes
        sorted_nodes = sorted(
            activation.items(), key=lambda x: x[1], reverse=True
        )
        return [nid for nid, _ in sorted_nodes[:PARALLEL_ROLLOUTS * 2]]


    # ─────────────────────────────────────────
    # RUN ROLLOUTS
    # Parallel forward paths through the graph
    # ─────────────────────────────────────────

    def _run_rollouts(
        self,
        start_nodes : list[int],
        max_depth   : int,
    ) -> list[Rollout]:
        """
        Run parallel rollouts from multiple start nodes.
        Each rollout follows the highest-weight path forward.
        Returns all rollouts that reached a meaningful depth.
        """
        rollouts = []

        for start_id in start_nodes[:PARALLEL_ROLLOUTS]:
            rollout = self._single_rollout(start_id, max_depth)
            if rollout and len(rollout.path) > 1:
                rollouts.append(rollout)

        return rollouts


    def _single_rollout(
        self,
        start_id  : int,
        max_depth : int,
    ) -> Optional[Rollout]:
        """
        One greedy rollout from start_id.
        Follow highest (edge_weight × target_weight) at each step.
        Stop when confidence drops below MIN_CONFIDENCE or no edges.
        """
        path         = [start_id]
        surfaces     = [self.graph.surface(start_id)]
        confidence   = 1.0
        total_weight = 0.0
        visited      = {start_id}

        current_id = start_id

        for _ in range(max_depth):
            node = self.graph.get_node(current_id)
            if node:
                total_weight += node["weight"]

            edges = self.graph.get_outgoing_edges(current_id)

            # Filter: unvisited, positive weight
            candidates = [
                e for e in edges
                if e["target_id"] not in visited
                and e["weight"] > 0
            ]

            if not candidates:
                break

            # Score: edge weight × target node weight
            def score(edge):
                tgt = self.graph.get_node(edge["target_id"])
                tw  = tgt["weight"] if tgt else 0.0
                return edge["weight"] * tw

            best_edge = max(candidates, key=score)
            step_conf = best_edge["weight"]

            # Update cumulative confidence
            confidence *= step_conf

            if confidence < MIN_CONFIDENCE:
                break

            next_id = best_edge["target_id"]
            path.append(next_id)
            surfaces.append(self.graph.surface(next_id))
            visited.add(next_id)
            current_id = next_id

        if len(path) < 2:
            return None

        return Rollout(
            path         = path,
            surfaces     = surfaces,
            confidence   = confidence,
            total_weight = total_weight,
        )


    # ─────────────────────────────────────────
    # PATH TO TEXT
    # Convert a rollout path to readable text
    # ─────────────────────────────────────────

    def _path_to_text(self, rollout: Rollout) -> str:
        """
        Convert rollout path to a readable response.
        Skip speaker label tokens. Join with spaces.
        """
        words = []
        for surface in rollout.surfaces:
            # Skip speaker labels like [parent], [child]
            if surface.startswith("[") and surface.endswith("]"):
                continue
            words.append(surface)

        return " ".join(words) if words else "..."


    # ─────────────────────────────────────────
    # CONVERSATION LOOP — interactive
    # ─────────────────────────────────────────

    def conversation(self):
        """
        Run an interactive conversation loop in the terminal.
        The system ingests each turn and responds.
        Type 'quit' to exit, 'status' to see graph state.
        """
        print("\nLiving System — Conversation Interface")
        print("=" * 40)
        print("Commands: 'quit', 'status', 'context'\n")

        self.loop.start()

        try:
            while True:
                try:
                    user_input = input("you: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    break

                if user_input.lower() == "status":
                    status = self.loop.status()
                    print(f"\n  tokens={status['tokens_processed']} "
                          f"compressions={status['compressions_run']} "
                          f"nodes={status['graph']['nodes']} "
                          f"edges={status['graph']['edges']}")
                    health = status["health"]
                    print(f"  stable={health['stable']} "
                          f"provisional={health['provisional']} "
                          f"decaying={health['decaying']} "
                          f"hypothetical={health['hypothetical']}\n")
                    continue

                if user_input.lower() == "context":
                    ctx = self.loop.current_context()
                    print(f"\n  Context window ({len(ctx)} nodes):")
                    for n in ctx:
                        print(f"    '{n['surface']}'  "
                              f"w={n['weight']:.3f}  l={n['level']}")
                    print()
                    continue

                # Generate response
                response = self.respond(user_input, speaker="user")
                print(f"system: {response.text}  "
                      f"[conf={response.confidence:.3f}  "
                      f"t={response.elapsed}s]")

        finally:
            self.loop.stop()


if __name__ == "__main__":
    import tempfile, os

    print("=== interface.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)

        from core.loop import LoopConfig
        config = LoopConfig(
            compress_every   = 10,
            project_every    = 2,
            decay_every_sec  = 30.0,
            verbose          = True,
        )
        loop      = LivingLoop(graph, config)
        interface = Interface(loop)

        # Seed with some conversation data first
        seed_conversation = [
            ("parent", "what is that"),
            ("child",  "that is a cat"),
            ("parent", "yes that is a cat"),
            ("child",  "the cat is big"),
            ("parent", "yes the cat is big"),
            ("parent", "what is that over there"),
            ("child",  "that is a dog"),
            ("parent", "yes that is a dog"),
            ("child",  "the dog is small"),
            ("parent", "yes the dog is small"),
            ("parent", "is the cat big"),
            ("child",  "yes the cat is big"),
            ("parent", "is the dog small"),
            ("child",  "yes the dog is small"),
        ]

        print("Seeding with conversation data...\n")
        for speaker, text in seed_conversation:
            loop.ingest_turn(speaker, text)

        # Run some ticks
        for _ in range(3):
            loop.tick()

        print(f"\nGraph after seeding: {graph.stats()}\n")

        # Now query
        queries = [
            "what is a cat",
            "is the dog big",
            "the cat",
        ]

        print("Querying...\n")
        for q in queries:
            response = interface.respond(q)
            print(f"Q: {q}")
            print(f"A: {response.text}  "
                  f"[conf={response.confidence:.3f}  "
                  f"elapsed={response.elapsed}s]")
            if response.rollout:
                print(f"   path length: {len(response.rollout.path)}")
            print()

        # Final status
        print("Final status:")
        status = loop.status()
        print(f"  tokens processed : {status['tokens_processed']}")
        print(f"  compressions run : {status['compressions_run']}")
        print(f"  graph nodes      : {status['graph']['nodes']}")
        print(f"  graph edges      : {status['graph']['edges']}")
        print(f"  health           : {status['health']}")

        graph.close()
