"""
memory.py

Working memory — the scratch pad for active thought.

Unlike the graph which is permanent, working memory is temporary.
It holds what the system is currently thinking about and dissolves
when the thinking episode ends.

Working memory has four components:

    context_window  — recent nodes, ordered by recency
                      what the system just processed
    active_schemas  — hyperedges currently firing on context
                      what structural patterns are active
    attention_map   — node_id → attention weight
                      where focus is concentrated right now
    thought_trace   — sequence of rollout steps taken so far
                      what path has been explored this episode

The context window is the most important piece.
It connects consecutive thoughts — without it each rollout
starts fresh with no memory of what came before.

Schema activation is what separates structured reasoning
from random graph walking. When a schema fires on the
current context — the system knows what KIND of situation
it is in and can use the schema's structure to guide rollout.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from core.atoms import (
    State, now,
    CONTEXT_WINDOW_SIZE, SLOT,
)
from core.graph import Graph


# ─────────────────────────────────────────────
# SCHEMA MATCH
# A hyperedge that fired on the current context
# ─────────────────────────────────────────────

@dataclass
class SchemaMatch:
    schema_id    : int
    schema       : dict          # the full hyperedge dict
    bindings     : dict          # {position: node_id} — how slots are filled
    match_score  : float         # how strongly this schema matches context
    is_partial   : bool          # True if some slots are unfilled


# ─────────────────────────────────────────────
# THOUGHT STEP
# One step in the current rollout trace
# ─────────────────────────────────────────────

@dataclass
class ThoughtStep:
    node_id    : int
    surface    : str
    confidence : float   # confidence at this step
    source     : str     # "context" | "rollout" | "schema" | "hypothetical"


# ─────────────────────────────────────────────
# WORKING MEMORY
# ─────────────────────────────────────────────

class WorkingMemory:
    """
    Temporary scratch pad for one thinking episode.

    Created fresh for each query or internal thinking session.
    Populated from the graph's current state and the incoming context.
    Dissolves when the episode ends.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

        # Recent node IDs — ordered, most recent last
        self.context_window : deque[int] = deque(maxlen=CONTEXT_WINDOW_SIZE)

        # Schemas currently firing
        self.active_schemas : list[SchemaMatch] = []

        # Attention weights — node_id → float [0,1]
        self.attention_map  : dict[int, float] = {}

        # Rollout trace — steps taken this episode
        self.thought_trace  : list[ThoughtStep] = []

        # Visited nodes — prevent loops in rollout
        self.visited        : set[int] = set()

        # Episode start time
        self.started_at     : float = now()


    # ─────────────────────────────────────────
    # POPULATE FROM CONTEXT
    # Load working memory from recent loop state
    # ─────────────────────────────────────────

    def load_context(self, node_ids: list[int]):
        """
        Prime working memory from a list of recent node IDs.
        These become the starting context window.
        """
        for nid in node_ids[-CONTEXT_WINDOW_SIZE:]:
            self.context_window.append(nid)
            node = self.graph.get_node(nid)
            if node:
                self.attention_map[nid] = node["weight"]

        # Match schemas against the loaded context
        self._match_schemas()


    # ─────────────────────────────────────────
    # SCHEMA MATCHING
    # Which schemas fire on the current context?
    # ─────────────────────────────────────────

    def _match_schemas(self):
        """
        Check all stable schemas against the current context window.
        A schema matches if its fixed positions appear in context
        and its slot positions could be filled from context.
        """
        self.active_schemas = []
        context_set = set(self.context_window)

        if not context_set:
            return

        # Check schemas at each level
        for level in range(0, 5):
            schemas = self.graph.hyperedges_by_level(level)
            for schema in schemas:
                match = self._try_match(schema, context_set)
                if match:
                    self.active_schemas.append(match)

        # Sort by match score — strongest matches first
        self.active_schemas.sort(key=lambda m: m.match_score, reverse=True)


    def _try_match(
        self,
        schema      : dict,
        context_set : set[int],
    ) -> Optional[SchemaMatch]:
        """
        Try to match a schema against the current context.

        A schema matches if:
            - Its fixed positions appear in context
            - Its slot positions can be filled from context or graph

        Returns SchemaMatch if matched, None otherwise.
        """
        structure = schema["structure"]
        bindings  = {}
        fixed_matches = 0
        fixed_total   = 0
        slot_fills    = 0
        slot_total    = 0

        for i, pos in enumerate(structure):
            if pos == SLOT:
                slot_total += 1
                # Can this slot be filled from context?
                # For now: yes if context has any node at compatible level
                if context_set:
                    # Take any context node as a possible slot fill
                    fill = next(iter(context_set))
                    bindings[i] = fill
                    slot_fills += 1
            else:
                # Fixed position — must appear in context
                fixed_total += 1
                if pos in context_set:
                    bindings[i] = pos
                    fixed_matches += 1

        if fixed_total == 0 and slot_total == 0:
            return None

        # Match score: fraction of fixed positions that matched
        if fixed_total > 0:
            match_score = fixed_matches / fixed_total
        else:
            match_score = 0.3  # slot-only schemas get low score

        # Threshold: at least half of fixed positions must match
        if fixed_total > 0 and match_score < 0.5:
            return None

        is_partial = slot_fills < slot_total

        return SchemaMatch(
            schema_id  = schema["id"],
            schema     = schema,
            bindings   = bindings,
            match_score= round(match_score * schema["weight"], 4),
            is_partial = is_partial,
        )


    # ─────────────────────────────────────────
    # ATTENTION
    # Where should the rollout focus?
    # ─────────────────────────────────────────

    def update_attention(self, node_id: int, weight: float):
        """Update attention for a node. Higher = more likely to explore."""
        current = self.attention_map.get(node_id, 0.0)
        self.attention_map[node_id] = min(1.0, current + weight)


    def top_attention_nodes(self, n: int = 5) -> list[tuple[int, float]]:
        """Top N nodes by current attention weight."""
        return sorted(
            self.attention_map.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]


    def most_attended(self) -> Optional[int]:
        """The single most attended node. Start rollout from here."""
        if not self.attention_map:
            return None
        return max(self.attention_map, key=self.attention_map.get)


    # ─────────────────────────────────────────
    # THOUGHT TRACE
    # Record what's been explored this episode
    # ─────────────────────────────────────────

    def record_step(
        self,
        node_id    : int,
        confidence : float,
        source     : str = "rollout",
    ):
        """Record one step in the thought trace."""
        surface = self.graph.surface(node_id)
        self.thought_trace.append(ThoughtStep(
            node_id    = node_id,
            surface    = surface,
            confidence = confidence,
            source     = source,
        ))
        self.visited.add(node_id)
        self.update_attention(node_id, confidence * 0.3)


    def trace_surfaces(self) -> list[str]:
        """Surface text of all steps in the trace."""
        return [step.surface for step in self.thought_trace]


    def trace_confidence(self) -> float:
        """Overall confidence of the current trace."""
        if not self.thought_trace:
            return 0.0
        confs = [step.confidence for step in self.thought_trace]
        result = 1.0
        for c in confs:
            result *= c
        return result


    # ─────────────────────────────────────────
    # BEST SCHEMA
    # The strongest firing schema guides rollout
    # ─────────────────────────────────────────

    def best_schema(self) -> Optional[SchemaMatch]:
        """The highest-scoring active schema."""
        if not self.active_schemas:
            return None
        return self.active_schemas[0]


    def has_strong_schema(self, threshold: float = 0.5) -> bool:
        """Is there a schema firing with high confidence?"""
        best = self.best_schema()
        return best is not None and best.match_score >= threshold


    # ─────────────────────────────────────────
    # CONTEXT SUMMARY
    # What is currently in working memory?
    # ─────────────────────────────────────────

    def summary(self) -> dict:
        """Human-readable summary of current working memory state."""
        return {
            "context_window"   : [
                self.graph.surface(nid)
                for nid in self.context_window
            ],
            "active_schemas"   : len(self.active_schemas),
            "best_schema_score": (
                self.active_schemas[0].match_score
                if self.active_schemas else 0.0
            ),
            "attention_nodes"  : len(self.attention_map),
            "thought_steps"    : len(self.thought_trace),
            "visited_nodes"    : len(self.visited),
            "trace_confidence" : round(self.trace_confidence(), 4),
            "age_ms"           : round((now() - self.started_at) * 1000, 1),
        }


    # ─────────────────────────────────────────
    # RESET
    # Start fresh for a new thinking episode
    # ─────────────────────────────────────────

    def reset(self):
        """Clear the working memory for a new episode."""
        self.context_window.clear()
        self.active_schemas  = []
        self.attention_map   = {}
        self.thought_trace   = []
        self.visited         = set()
        self.started_at      = now()


if __name__ == "__main__":
    import tempfile, os
    from core.ingestion   import Ingestion
    from core.compression import Compression

    print("=== memory.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        ing     = Ingestion(graph)
        comp    = Compression(graph)

        # Seed with data
        sentences = [
            "the cat sat on the mat",
            "the cat sat on the roof",
            "the cat sat on the floor",
            "the dog sat on the mat",
            "the dog sat on the floor",
            "the bird sat on the roof",
        ] * 5

        for s in sentences:
            ing.reset_context()
            ing.process_text(s)

        comp.run()
        comp.run()  # second pass finds more

        stats = graph.stats()
        print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges\n")

        # Create working memory and load context
        wm = WorkingMemory(graph)

        # Simulate context: "the cat"
        the_id  = graph.find_node(0, [], raw="the")
        cat_id  = graph.find_node(0, [], raw="cat")
        sat_id  = graph.find_node(0, [], raw="sat")

        print(f"Key nodes: the={the_id} cat={cat_id} sat={sat_id}\n")

        if the_id and cat_id and sat_id:
            wm.load_context([the_id, cat_id, sat_id])

        print(f"Working memory summary:")
        for k, v in wm.summary().items():
            print(f"  {k}: {v}")

        print(f"\nTop attention nodes:")
        for nid, att in wm.top_attention_nodes(5):
            surface = graph.surface(nid)
            print(f"  '{surface}'  attention={att:.3f}")

        print(f"\nActive schemas: {len(wm.active_schemas)}")
        for match in wm.active_schemas[:3]:
            structure_readable = []
            for pos in match.schema["structure"]:
                if pos == SLOT:
                    structure_readable.append("SLOT")
                else:
                    structure_readable.append(f"'{graph.surface(pos)}'")
            print(f"  [{match.schema_id}] {structure_readable}  "
                  f"score={match.match_score:.3f}  "
                  f"partial={match.is_partial}")

        # Record some thought steps
        if cat_id:
            wm.record_step(cat_id, 0.9, source="context")
        if sat_id:
            wm.record_step(sat_id, 0.7, source="rollout")

        print(f"\nAfter recording steps:")
        print(f"  trace: {wm.trace_surfaces()}")
        print(f"  confidence: {wm.trace_confidence():.4f}")

        graph.close()
