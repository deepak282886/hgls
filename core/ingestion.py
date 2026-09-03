"""
ingestion.py

Process 1: Ingestion — the membrane.

Reads the input stream. Tests each incoming token against the boundary.
Routes it:
    - Familiar     → reinforce existing nodes and edges
    - Boundary     → create or strengthen provisional node
    - Novel        → hold in provisional state, watch for recurrence

What ingestion does NOT do:
    - Decide what level a pattern belongs to (compression does that)
    - Create schemas (compression does that)
    - Generate hypotheticals (projection does that)
    - Remove anything (decay does that)

Ingestion only routes. Thin and fast.
"""

import re
from typing import Optional, Iterator
from dataclasses import dataclass, field

from core.atoms import (
    State, now,
    BOUNDARY_SENSITIVITY, ACTIVATION_BOOST,
    ACTIVATION_MIN, PROVISIONAL_TTL,
    novelty,
)
from core.graph import Graph


# ─────────────────────────────────────────────
# TOKEN
# The atomic unit ingestion works with.
# At level 0 this is a raw string token.
# ─────────────────────────────────────────────

@dataclass
class Token:
    raw      : str
    position : int   # position in the input stream


# ─────────────────────────────────────────────
# INGESTION RESULT
# What happened to this token
# ─────────────────────────────────────────────

@dataclass
class IngestionResult:
    token      : Token
    node_id    : int
    is_new     : bool       # was a new node created?
    is_novel   : bool       # did this cross the novelty threshold?
    edge_id    : Optional[int] = None   # edge from previous token


# ─────────────────────────────────────────────
# TOKENIZER
# Break raw text into a stream of tokens.
# Simple for now — can be replaced later.
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[Token]:
    """
    Break text into word-level tokens.
    Preserves punctuation as separate tokens.
    Position is character offset.
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append(Token(
            raw      = match.group(),
            position = match.start(),
        ))
    return tokens


def tokenize_chars(text: str) -> list[Token]:
    """
    Character-level tokenization.
    Used for building the lowest level of the graph.
    """
    return [Token(raw=c, position=i) for i, c in enumerate(text)]


# ─────────────────────────────────────────────
# INGESTION ENGINE
# ─────────────────────────────────────────────

class Ingestion:
    """
    The membrane. Reads token streams and routes them into the graph.

    Usage:
        ingestion = Ingestion(graph)
        results   = ingestion.process_text("hello world")
    """

    def __init__(self, graph: Graph):
        self.graph        = graph
        self._prev_node   : Optional[int] = None   # last processed node
        self._position    : int = 0                # stream position


    # ─────────────────────────────────────────
    # PROCESS TEXT
    # Main entry point for string input
    # ─────────────────────────────────────────

    def process_text(
        self,
        text     : str,
        reset_context: bool = False,
    ) -> list[IngestionResult]:
        """
        Process a text string through the membrane.
        Returns one IngestionResult per token.

        reset_context: if True, clear the previous node context.
                       Use at document boundaries.
        """
        if reset_context:
            self._prev_node = None

        tokens  = tokenize(text)
        results = []

        for token in tokens:
            result = self._process_token(token)
            results.append(result)

        return results


    def process_conversation_turn(
        self,
        speaker : str,
        text    : str,
    ) -> list[IngestionResult]:
        """
        Process one turn of a conversation.
        Speaker label is prepended as a special token.
        Context is maintained across turns.
        """
        # Speaker token — marks turn boundary
        speaker_token = Token(raw=f"[{speaker}]", position=self._position)
        results = [self._process_token(speaker_token)]

        # Then process the text maintaining context
        text_results = self.process_text(text, reset_context=False)
        results.extend(text_results)

        return results


    # ─────────────────────────────────────────
    # PROCESS ONE TOKEN
    # ─────────────────────────────────────────

    def _process_token(self, token: Token) -> IngestionResult:
        """
        Route one token through the membrane.

        Three outcomes:
            1. Familiar   → reinforce existing node and edge
            2. Boundary   → create provisional node, connect
            3. Novel      → create provisional node, connect
        """
        self._position += 1
        raw = token.raw.lower().strip()

        if not raw:
            return IngestionResult(
                token   = token,
                node_id = -1,
                is_new  = False,
                is_novel= False,
            )

        # ── Find best existing match ───────────
        existing_id = self.graph.find_node(0, [], raw=raw)

        if existing_id is not None:
            # Familiar — exact match found
            node    = self.graph.get_node(existing_id)
            nov     = novelty(node["weight"])
            is_novel = nov > BOUNDARY_SENSITIVITY

            # Reinforce
            self.graph.activate_node(existing_id)

            # Create or reinforce edge from previous
            edge_id = self._connect_to_previous(existing_id)

            result = IngestionResult(
                token   = token,
                node_id = existing_id,
                is_new  = False,
                is_novel= is_novel,
                edge_id = edge_id,
            )

        else:
            # Novel — no exact match
            # Create provisional node
            node_id, is_new = self.graph.get_or_create_node(
                level  = 0,
                components = [],
                raw    = raw,
                weight = 0.3,   # starts low — must earn stability
                state  = State.PROVISIONAL,
            )

            # Connect to previous
            edge_id = self._connect_to_previous(node_id)

            result = IngestionResult(
                token   = token,
                node_id = node_id,
                is_new  = is_new,
                is_novel= True,
                edge_id = edge_id,
            )

        self._prev_node = result.node_id
        return result


    # ─────────────────────────────────────────
    # CONNECT TO PREVIOUS
    # Create or reinforce sequential edge
    # ─────────────────────────────────────────

    def _connect_to_previous(self, node_id: int) -> Optional[int]:
        """
        Create or reinforce the sequential edge from previous node
        to current node. Returns edge ID if created/found.
        """
        if self._prev_node is None or self._prev_node == node_id:
            return None
        if self._prev_node < 0:
            return None

        edge_id, is_new = self.graph.get_or_create_edge(
            self._prev_node,
            node_id,
            weight = 0.3,
        )
        self.graph.activate_edge(edge_id)
        return edge_id


    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────

    def reset_context(self):
        """Clear the running context. Call at document boundaries."""
        self._prev_node = None


    # ─────────────────────────────────────────
    # STREAM PROCESSING
    # For continuous input streams
    # ─────────────────────────────────────────

    def process_stream(
        self,
        stream: Iterator[str],
    ) -> Iterator[IngestionResult]:
        """
        Process a continuous stream of text chunks.
        Yields IngestionResult for each token.
        Maintains context across chunks.
        """
        for chunk in stream:
            results = self.process_text(chunk)
            yield from results


    # ─────────────────────────────────────────
    # NOVELTY SUMMARY
    # What did we just learn?
    # ─────────────────────────────────────────

    def novelty_summary(
        self,
        results: list[IngestionResult],
    ) -> dict:
        """Summarize what was novel in a batch of results."""
        total   = len(results)
        novel   = sum(1 for r in results if r.is_novel)
        new     = sum(1 for r in results if r.is_new)
        known   = total - novel

        return {
            "total"         : total,
            "known"         : known,
            "novel"         : novel,
            "new_nodes"     : new,
            "novelty_rate"  : round(novel / total, 4) if total > 0 else 0.0,
        }


if __name__ == "__main__":
    import tempfile, os

    print("=== ingestion.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)
        ing     = Ingestion(graph)

        # First pass — everything novel
        print("── Pass 1: 'the cat sat on the mat' ──")
        results = ing.process_text("the cat sat on the mat")
        for r in results:
            marker = "NEW" if r.is_new else "known"
            print(f"  [{marker}] '{r.token.raw}' → node {r.node_id}")

        summary = ing.novelty_summary(results)
        print(f"  Summary: {summary}\n")

        # Second pass — some familiar, some new
        print("── Pass 2: 'the cat sat on the roof' ──")
        ing.reset_context()
        results2 = ing.process_text("the cat sat on the roof")
        for r in results2:
            marker = "NEW" if r.is_new else "known"
            print(f"  [{marker}] '{r.token.raw}' → node {r.node_id}")

        summary2 = ing.novelty_summary(results2)
        print(f"  Summary: {summary2}\n")

        # Conversation turn
        print("── Conversation turn ──")
        ing.reset_context()
        results3 = ing.process_conversation_turn(
            "parent", "what is that over there"
        )
        for r in results3:
            print(f"  '{r.token.raw}' → node {r.node_id}")

        # Graph stats
        print(f"\nGraph stats: {graph.stats()}")
        graph.close()
