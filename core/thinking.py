"""
thinking.py

The thinking engine. Parallel rollouts with schema guidance.

Three modes of rollout running simultaneously:

    1. EXPLOITATION rollout
       Follow highest-weight edges from the most attended node.
       Fast. Confident. Uses what the system knows best.
       Produces the safe, reliable response.

    2. EXPLORATION rollout
       Follow highest-uncertainty paths — low visit count,
       high hypothetical density. Slow. Risky. May find
       new structure. Produces the creative response.

    3. SCHEMA-GUIDED rollout
       When a schema fires on the current context — follow
       the schema's structure. Fill slots from context.
       Read off the conclusion. O(1) when schema matches well.
       Produces the structured, generalizing response.

After all rollouts complete:
    - Compare by confidence × schema_alignment × coherence
    - Best rollout becomes the response
    - All rollouts update graph weights (mini training episode)
    - New hypotheticals posted to projection

Dynamic depth:
    Rollout continues while confidence > MIN_CONFIDENCE.
    Well-known domains → deep rollouts (confidence stays high).
    Unknown domains → shallow rollouts (confidence drops fast).
    System naturally thinks harder about what it knows.

Backpropagation:
    Every rollout — win or lose — updates the graph.
    Successful paths get weight boost.
    Dead-end paths get slight weight decrease.
    This is how thinking improves the graph.
    Every thought is a mini training episode.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from core.atoms import (
    State, now, SLOT,
    MAX_ROLLOUT_DEPTH, MIN_CONFIDENCE,
    PARALLEL_ROLLOUTS, ACTIVATION_BOOST,
    ACTIVATION_MIN,
)
from core.graph  import Graph
from core.memory import WorkingMemory, SchemaMatch


# ─────────────────────────────────────────────
# ROLLOUT RESULT
# One completed path through the graph
# ─────────────────────────────────────────────

@dataclass
class RolloutResult:
    path        : list[int]    # node IDs
    surfaces    : list[str]    # surface text at each node
    confidence  : float        # cumulative confidence
    total_weight: float        # sum of node weights
    mode        : str          # "exploit" | "explore" | "schema"
    schema_id   : Optional[int] = None   # if schema-guided
    depth       : int          = 0

    def score(self) -> float:
        """Combined quality score for ranking rollouts."""
        schema_bonus = 0.2 if self.schema_id else 0.0
        depth_bonus  = min(0.3, self.depth * 0.03)
        return (
            self.confidence * 0.5
            + (self.total_weight / max(1, self.depth)) * 0.3
            + schema_bonus
            + depth_bonus
        )


# ─────────────────────────────────────────────
# THINKING ENGINE
# ─────────────────────────────────────────────

class Thinker:
    """
    The thinking engine. Runs parallel rollouts and picks the best.

    Usage:
        thinker = Thinker(graph)
        memory  = WorkingMemory(graph)
        memory.load_context(node_ids)

        results = thinker.think(memory)
        best    = results[0]
    """

    def __init__(self, graph: Graph):
        self.graph = graph


    # ─────────────────────────────────────────
    # MAIN THINK
    # Run all rollout modes and return ranked results
    # ─────────────────────────────────────────

    def think(
        self,
        memory    : WorkingMemory,
        max_depth : int = MAX_ROLLOUT_DEPTH,
    ) -> list[RolloutResult]:
        """
        Run parallel rollouts from working memory state.
        Returns rollouts ranked by score — best first.

        Each rollout uses a different strategy:
            exploitation  — follow known high-weight paths
            exploration   — follow uncertain low-weight paths
            schema-guided — use active schemas to structure path
        """
        results = []

        # Get starting nodes from attention map
        start_nodes = [
            nid for nid, _ in memory.top_attention_nodes(PARALLEL_ROLLOUTS)
            if nid not in memory.visited
        ]

        if not start_nodes:
            return results

        # ── Exploitation rollout ──────────────────
        exploit = self._exploit_rollout(
            start_id  = start_nodes[0],
            memory    = memory,
            max_depth = max_depth,
        )
        if exploit:
            results.append(exploit)

        # ── Schema-guided rollout ─────────────────
        if memory.has_strong_schema():
            schema_result = self._schema_rollout(
                memory    = memory,
                max_depth = max_depth,
            )
            if schema_result:
                results.append(schema_result)

        # ── Exploration rollout ───────────────────
        if len(start_nodes) > 1:
            explore = self._explore_rollout(
                start_id  = start_nodes[-1],  # least attended = most uncertain
                memory    = memory,
                max_depth = max_depth // 2,   # shorter — we're in unknown territory
            )
            if explore:
                results.append(explore)

        # ── Analogical rollout ────────────────────
        analogy = self._analogy_rollout(
            memory    = memory,
            max_depth = max_depth // 2,
        )
        if analogy:
            results.append(analogy)

        # Rank by score
        results.sort(key=lambda r: r.score(), reverse=True)

        # Backpropagate all rollouts — every thought updates the graph
        self._backpropagate(results)

        # Record best path in working memory trace
        if results:
            best = results[0]
            for i, nid in enumerate(best.path):
                conf = best.confidence * (0.9 ** i)
                memory.record_step(nid, conf, source=best.mode)

        return results


    # ─────────────────────────────────────────
    # EXPLOITATION ROLLOUT
    # Follow highest-weight path greedily
    # ─────────────────────────────────────────

    def _exploit_rollout(
        self,
        start_id  : int,
        memory    : WorkingMemory,
        max_depth : int,
    ) -> Optional[RolloutResult]:
        """
        Greedy exploitation: follow highest (edge_weight × node_weight).
        Stays on the beaten path. Fast and confident.
        """
        path         = [start_id]
        surfaces     = [self.graph.surface(start_id)]
        confidence   = 1.0
        total_weight = 0.0
        visited      = set(memory.visited) | {start_id}
        current_id   = start_id

        for depth in range(max_depth):
            node = self.graph.get_node(current_id)
            if node:
                total_weight += node["weight"]

            edges = self.graph.get_outgoing_edges(current_id)
            candidates = [
                e for e in edges
                if e["target_id"] not in visited
                and e["weight"] > 0
            ]

            if not candidates:
                break

            # Score: edge weight × target node weight
            best_edge = max(
                candidates,
                key=lambda e: self._edge_score(e),
            )

            step_conf   = best_edge["weight"]
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

        return RolloutResult(
            path         = path,
            surfaces     = surfaces,
            confidence   = confidence,
            total_weight = total_weight,
            mode         = "exploit",
            depth        = len(path),
        )


    # ─────────────────────────────────────────
    # SCHEMA-GUIDED ROLLOUT
    # Use active schema structure to guide path
    # ─────────────────────────────────────────

    def _schema_rollout(
        self,
        memory    : WorkingMemory,
        max_depth : int,
    ) -> Optional[RolloutResult]:
        """
        Schema-guided rollout. Uses the strongest active schema
        to structure the path.

        If the schema is [A, SLOT, B] and context has A:
            - Start from A
            - Fill SLOT from best available node
            - Move to B
            - Continue from B greedily

        This is how the system applies learned patterns to new inputs.
        """
        match = memory.best_schema()
        if not match:
            return None

        schema    = match.schema
        structure = schema["structure"]
        bindings  = match.bindings

        path         = []
        surfaces     = []
        confidence   = match.match_score
        total_weight = 0.0
        visited      = set(memory.visited)

        # Walk through schema structure
        for i, pos in enumerate(structure):
            if confidence < MIN_CONFIDENCE:
                break

            if pos == SLOT:
                # Fill slot from bindings or find best available
                if i in bindings:
                    nid = bindings[i]
                else:
                    # Find best unvisited neighbor of last node
                    nid = self._best_slot_fill(
                        path[-1] if path else None,
                        visited,
                    )

                if nid is None:
                    continue

                node = self.graph.get_node(nid)
                if node:
                    step_conf   = node["weight"]
                    confidence *= step_conf
                    total_weight += node["weight"]

                    path.append(nid)
                    surfaces.append(self.graph.surface(nid))
                    visited.add(nid)

            else:
                # Fixed position — must be a node ID
                if not isinstance(pos, int):
                    continue

                nid  = pos
                node = self.graph.get_node(nid)
                if node is None:
                    continue

                if nid in visited:
                    continue

                step_conf   = node["weight"]
                confidence *= step_conf
                total_weight += node["weight"]

                path.append(nid)
                surfaces.append(self.graph.surface(nid))
                visited.add(nid)

        if len(path) < 2:
            return None

        # After schema structure, continue greedily
        if path and len(path) < max_depth:
            extension = self._greedy_extend(
                start_id  = path[-1],
                visited   = visited,
                max_steps = max_depth - len(path),
            )
            if extension:
                ext_path, ext_surfaces = extension
                path.extend(ext_path)
                surfaces.extend(ext_surfaces)

        return RolloutResult(
            path         = path,
            surfaces     = surfaces,
            confidence   = confidence,
            total_weight = total_weight,
            mode         = "schema",
            schema_id    = match.schema_id,
            depth        = len(path),
        )


    def _best_slot_fill(
        self,
        prev_node_id : Optional[int],
        visited      : set[int],
    ) -> Optional[int]:
        """Find the best node to fill a schema slot."""
        if prev_node_id is None:
            return None

        edges = self.graph.get_outgoing_edges(prev_node_id)
        candidates = [
            e for e in edges
            if e["target_id"] not in visited
            and e["weight"] > 0
        ]

        if not candidates:
            return None

        best = max(candidates, key=lambda e: self._edge_score(e))
        return best["target_id"]


    # ─────────────────────────────────────────
    # EXPLORATION ROLLOUT
    # Follow uncertain low-weight paths
    # ─────────────────────────────────────────

    def _explore_rollout(
        self,
        start_id  : int,
        memory    : WorkingMemory,
        max_depth : int,
    ) -> Optional[RolloutResult]:
        """
        Exploration rollout. Follows paths with high uncertainty.
        Prefers nodes with low weight but active hypotheticals nearby.
        May discover new structure. Often produces worse immediate
        responses but feeds the projection process.
        """
        path         = [start_id]
        surfaces     = [self.graph.surface(start_id)]
        confidence   = 1.0
        total_weight = 0.0
        visited      = set(memory.visited) | {start_id}
        current_id   = start_id

        for depth in range(max_depth):
            node = self.graph.get_node(current_id)
            if node:
                total_weight += node["weight"]

            edges = self.graph.get_outgoing_edges(current_id)
            candidates = [
                e for e in edges
                if e["target_id"] not in visited
            ]

            if not candidates:
                break

            # Exploration score: prefer lower-weight edges (less trodden)
            # but not zero — completely unknown paths have no signal
            def explore_score(edge):
                tgt = self.graph.get_node(edge["target_id"])
                tw  = tgt["weight"] if tgt else 0.0
                # Prefer moderate weight — not too known, not unknown
                # Peak at weight=0.3
                return 1.0 - abs(tw - 0.3)

            best_edge   = max(candidates, key=explore_score)
            step_conf   = max(0.1, best_edge["weight"])
            confidence *= step_conf

            # Exploration allows lower confidence — we expect uncertainty
            if confidence < MIN_CONFIDENCE * 0.1:
                break

            next_id = best_edge["target_id"]
            path.append(next_id)
            surfaces.append(self.graph.surface(next_id))
            visited.add(next_id)
            current_id = next_id

        if len(path) < 2:
            return None

        return RolloutResult(
            path         = path,
            surfaces     = surfaces,
            confidence   = confidence * 0.5,  # penalize for being exploratory
            total_weight = total_weight,
            mode         = "explore",
            depth        = len(path),
        )


    # ─────────────────────────────────────────
    # ANALOGICAL ROLLOUT
    # Find structural parallels across domains
    # ─────────────────────────────────────────

    def _analogy_rollout(
        self,
        memory    : WorkingMemory,
        max_depth : int,
    ) -> Optional[RolloutResult]:
        """
        Analogical rollout. Find nodes at higher levels that share
        structural components with the current context.
        Then follow their edges.

        This is lateral thinking — connecting the current situation
        to structurally similar situations in different domains.
        """
        context_nodes = list(memory.context_window)
        if not context_nodes:
            return None

        # Find level-1 nodes that contain context nodes as components
        analogues = []
        for nid in context_nodes:
            rows = self.graph._conn.execute(
                "SELECT id, components, weight FROM nodes "
                "WHERE level > 0 AND state = ? AND weight > ?",
                (State.STABLE.value, ACTIVATION_MIN)
            ).fetchall()

            import json
            for row in rows:
                comps = json.loads(row["components"])
                if nid in comps:
                    analogues.append(row["id"])

        if not analogues:
            return None

        # Start from the highest-weight analogue
        analogues_with_weights = []
        for aid in analogues:
            node = self.graph.get_node(aid)
            if node:
                analogues_with_weights.append((aid, node["weight"]))

        if not analogues_with_weights:
            return None

        best_analogue = max(analogues_with_weights, key=lambda x: x[1])[0]

        # Exploit from the analogue node
        visited = set(memory.visited)
        return self._exploit_rollout(
            start_id  = best_analogue,
            memory    = memory,
            max_depth = max_depth,
        )


    # ─────────────────────────────────────────
    # GREEDY EXTENSION
    # Extend a path greedily after schema structure ends
    # ─────────────────────────────────────────

    def _greedy_extend(
        self,
        start_id  : int,
        visited   : set[int],
        max_steps : int,
    ) -> Optional[tuple[list[int], list[str]]]:
        """Extend a path greedily from start_id."""
        path     = []
        surfaces = []
        current  = start_id

        for _ in range(max_steps):
            edges = self.graph.get_outgoing_edges(current)
            candidates = [
                e for e in edges
                if e["target_id"] not in visited
                and e["weight"] > 0
            ]

            if not candidates:
                break

            best  = max(candidates, key=lambda e: self._edge_score(e))
            nid   = best["target_id"]
            path.append(nid)
            surfaces.append(self.graph.surface(nid))
            visited.add(nid)
            current = nid

        return (path, surfaces) if path else None


    # ─────────────────────────────────────────
    # EDGE SCORING
    # ─────────────────────────────────────────

    def _edge_score(self, edge: dict) -> float:
        """Score an edge for exploitation rollout."""
        tgt = self.graph.get_node(edge["target_id"])
        tw  = tgt["weight"] if tgt else 0.0
        return edge["weight"] * tw


    # ─────────────────────────────────────────
    # BACKPROPAGATION
    # Every rollout updates graph weights
    # ─────────────────────────────────────────

    def _backpropagate(self, results: list[RolloutResult]):
        """
        Update graph weights based on rollout results.

        Best rollout → weight boost along path.
        Worse rollouts → smaller boost.
        Failed rollouts (short, low confidence) → slight decrease.

        This is the mini training episode — every thought
        leaves a trace in the graph.
        """
        if not results:
            return

        best_score = results[0].score()

        for rank, result in enumerate(results):
            relative = result.score() / max(best_score, 1e-9)

            # Boost factor: best rollout gets full boost,
            # worse rollouts get proportionally less
            boost = ACTIVATION_BOOST * relative * 0.5

            if boost <= 0:
                continue

            # Update nodes along path
            for i, nid in enumerate(result.path):
                # Discount further along path — nearby steps
                # get more credit than distant ones
                step_boost = boost * (0.9 ** i)
                node = self.graph.get_node(nid)
                if node:
                    new_weight = min(1.0, node["weight"] + step_boost)
                    self.graph.update_node_weight(nid, new_weight)

            # Update edges along path
            for i in range(len(result.path) - 1):
                src_id = result.path[i]
                tgt_id = result.path[i + 1]
                eid    = self.graph.find_edge(src_id, tgt_id)
                if eid:
                    self.graph.activate_edge(eid)

            # Boost the schema if schema-guided
            if result.schema_id:
                self.graph.activate_hyperedge(result.schema_id)


    # ─────────────────────────────────────────
    # BEST RESPONSE
    # Extract text from the best rollout
    # ─────────────────────────────────────────

    def best_response_text(
        self,
        results  : list[RolloutResult],
        skip_tags: bool = True,
    ) -> str:
        """
        Extract readable text from the best rollout result.
        Skips speaker label tokens if skip_tags=True.
        """
        if not results:
            return "..."

        best   = results[0]
        words  = []

        for surface in best.surfaces:
            if skip_tags and surface.startswith("[") and surface.endswith("]"):
                continue
            words.append(surface)

        return " ".join(words) if words else "..."


if __name__ == "__main__":
    import tempfile, os
    from core.ingestion   import Ingestion
    from core.compression import Compression
    from core.loop        import LivingLoop, LoopConfig

    print("=== thinking.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)

        config = LoopConfig(
            compress_every  = 15,
            project_every   = 2,
            decay_every_sec = 60.0,
            verbose         = True,
        )
        loop = LivingLoop(graph, config)

        # Seed with structured conversation
        conversation = [
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
            ("parent", "what is that"),
            ("child",  "that is a bird"),
            ("parent", "the bird is small"),
            ("child",  "yes the bird is small"),
        ]

        print("Seeding with conversation...\n")
        for speaker, text in conversation * 2:
            loop.ingest_turn(speaker, text)

        for _ in range(5):
            loop.tick()

        print(f"\nGraph: {graph.stats()}\n")

        # Create thinking engine and working memory
        thinker = Thinker(graph)
        memory  = WorkingMemory(graph)

        # Load context from loop state
        memory.load_context(loop.state.context_window)

        print(f"Working memory: {memory.summary()}\n")

        # Run thinking
        print("Running thinking...\n")
        results = thinker.think(memory, max_depth=8)

        print(f"Rollouts produced: {len(results)}\n")
        for i, r in enumerate(results):
            print(f"  [{i+1}] mode={r.mode:8s}  "
                  f"score={r.score():.4f}  "
                  f"conf={r.confidence:.4f}  "
                  f"depth={r.depth}")
            text = thinker.best_response_text([r])
            print(f"       text: '{text}'")
            print()

        # Best response
        best_text = thinker.best_response_text(results)
        print(f"Best response: '{best_text}'")

        # Now simulate a query context
        print("\n── Query: 'what is the cat' ──\n")
        memory2 = WorkingMemory(graph)

        # Find nodes for "what", "is", "the", "cat"
        query_nodes = []
        for word in ["what", "is", "the", "cat"]:
            nid = graph.find_node(0, [], raw=word)
            if nid:
                query_nodes.append(nid)
                print(f"  Found '{word}' → node {nid}")

        memory2.load_context(query_nodes)
        print(f"\n  Memory: {memory2.summary()}")

        results2 = thinker.think(memory2, max_depth=6)
        best2    = thinker.best_response_text(results2)
        print(f"\n  Response: '{best2}'")
        if results2:
            print(f"  Confidence: {results2[0].confidence:.4f}")
            print(f"  Mode: {results2[0].mode}")

        graph.close()
