"""
loop.py

The main living loop. Coordinates all four processes.

This is the organism. Not a pipeline. Not a request handler.
A continuously running process that:

    - Ingests when input is available
    - Compresses periodically
    - Projects periodically
    - Decays continuously
    - Thinks internally when idle

The loop never stops. External input is a perturbation.
Queries are handled as part of the loop, not as the driver.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator

from core.atoms import (
    State, now,
    CONTEXT_WINDOW_SIZE,
)
from core.graph       import Graph
from core.ingestion   import Ingestion
from core.compression import Compression
from core.projection  import Projection
from core.decay       import Decay


# ─────────────────────────────────────────────
# LOOP CONFIG
# How often each process runs
# ─────────────────────────────────────────────

@dataclass
class LoopConfig:
    # How many tokens between compression runs
    compress_every  : int   = 50

    # How many compression runs between projection runs
    project_every   : int   = 3

    # How many seconds between decay runs
    decay_every_sec : float = 10.0

    # How many seconds of idle before internal thinking starts
    idle_think_after: float = 5.0

    # Verbose output
    verbose         : bool  = False


# ─────────────────────────────────────────────
# LOOP STATE
# What the loop knows about itself right now
# ─────────────────────────────────────────────

@dataclass
class LoopState:
    tokens_processed    : int   = 0
    compressions_run    : int   = 0
    projections_run     : int   = 0
    decay_runs          : int   = 0
    last_input_time     : float = field(default_factory=now)
    last_decay_time     : float = field(default_factory=now)
    last_compress_time  : float = field(default_factory=now)
    is_running          : bool  = False

    # Recent node IDs — working context window
    context_window      : list  = field(default_factory=list)

    def update_context(self, node_id: int):
        """Add node to context window. Trim to size."""
        if node_id > 0:
            self.context_window.append(node_id)
            if len(self.context_window) > CONTEXT_WINDOW_SIZE:
                self.context_window = self.context_window[-CONTEXT_WINDOW_SIZE:]

    def idle_seconds(self) -> float:
        return now() - self.last_input_time


# ─────────────────────────────────────────────
# LIVING LOOP
# ─────────────────────────────────────────────

class LivingLoop:
    """
    The organism. Runs all four processes in coordination.

    Usage:
        loop = LivingLoop(graph)

        # Feed input
        loop.ingest("the cat sat on the mat")

        # Run one tick manually
        loop.tick()

        # Or run continuously in background
        loop.start()
        loop.stop()
    """

    def __init__(
        self,
        graph  : Graph,
        config : LoopConfig = None,
    ):
        self.graph      = graph
        self.config     = config or LoopConfig()
        self.state      = LoopState()

        # Four processes
        self.ingestion   = Ingestion(graph)
        self.compression = Compression(graph)
        self.projection  = Projection(graph)
        self.decay       = Decay(graph)

        # Background thread
        self._thread    : Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

        # Callbacks — called when things happen
        self._on_schema_formed   : Optional[Callable] = None
        self._on_hypothetical    : Optional[Callable] = None


    # ─────────────────────────────────────────
    # INGEST — primary input interface
    # ─────────────────────────────────────────

    def ingest(
        self,
        text           : str,
        reset_context  : bool = False,
    ) -> list:
        """
        Feed text into the loop.
        Updates context window. Triggers compression if threshold crossed.
        """
        self.state.last_input_time = now()

        results = self.ingestion.process_text(text, reset_context)

        for r in results:
            self.state.tokens_processed += 1
            self.state.update_context(r.node_id)

            # Check compression threshold inside loop — catches every Nth token
            if self.state.tokens_processed % self.config.compress_every == 0:
                self._run_compression()

        return results


    def ingest_turn(self, speaker: str, text: str) -> list:
        """Feed one conversational turn."""
        self.state.last_input_time = now()

        results = self.ingestion.process_conversation_turn(speaker, text)

        for r in results:
            self.state.tokens_processed += 1
            self.state.update_context(r.node_id)

            if self.state.tokens_processed % self.config.compress_every == 0:
                self._run_compression()

        return results


    # ─────────────────────────────────────────
    # TICK — one loop cycle
    # ─────────────────────────────────────────

    def tick(self):
        """
        Run one loop cycle.
        Handles decay, internal thinking when idle.
        Call this periodically when running manually.
        """
        t = now()

        # Decay — runs on time schedule
        if t - self.state.last_decay_time >= self.config.decay_every_sec:
            self._run_decay()

        # Internal thinking when idle
        if self.state.idle_seconds() >= self.config.idle_think_after:
            self._think_internally()


    # ─────────────────────────────────────────
    # PROCESS RUNNERS
    # ─────────────────────────────────────────

    def _run_compression(self):
        """Run compression cycle."""
        result = self.compression.run()
        self.state.compressions_run += 1

        if self.config.verbose and (
            result.nodes_promoted > 0
            or result.level1_created > 0
            or result.schemas_created > 0
        ):
            print(f"  [compress] promoted={result.nodes_promoted} "
                  f"level1={result.level1_created} "
                  f"schemas={result.schemas_created}")

        # Run projection every N compressions
        if self.state.compressions_run % self.config.project_every == 0:
            self._run_projection()


    def _run_projection(self):
        """Run projection cycle."""
        result = self.projection.run()
        self.state.projections_run += 1

        if self.config.verbose and result.hypotheticals_created > 0:
            print(f"  [project] hypotheticals={result.hypotheticals_created} "
                  f"confirmed={result.hypotheticals_confirmed} "
                  f"analogies={result.analogies_found}")

        if self._on_hypothetical and result.hypotheticals_created > 0:
            self._on_hypothetical(result)


    def _run_decay(self):
        """Run decay cycle."""
        result = self.decay.run()
        self.state.decay_runs     += 1
        self.state.last_decay_time = now()

        if self.config.verbose and result.nodes_dissolved > 0:
            print(f"  [decay] dissolved={result.nodes_dissolved} "
                  f"demoted={result.nodes_demoted}")


    def _think_internally(self):
        """
        Internal thinking when no external input.
        Explores uncertain regions of the graph.
        Runs projection on highest-uncertainty hypotheticals.
        """
        uncertain = self.projection.uncertainty_map(top_n=3)

        if not uncertain:
            return

        if self.config.verbose:
            print(f"  [think] exploring {len(uncertain)} uncertain regions")

        # Activate uncertain nodes — gives them a chance to connect
        for u in uncertain:
            self.graph.activate_node(u["node_id"])
            self.state.update_context(u["node_id"])

        # Run a projection cycle from this activated state
        self._run_projection()


    # ─────────────────────────────────────────
    # BACKGROUND RUNNING
    # ─────────────────────────────────────────

    def start(self):
        """Start the loop running in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self.state.is_running = True
        self._thread = threading.Thread(
            target = self._run_forever,
            daemon = True,
        )
        self._thread.start()

        if self.config.verbose:
            print("[loop] started")


    def stop(self):
        """Stop the background loop."""
        self._stop_event.set()
        self.state.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        if self.config.verbose:
            print("[loop] stopped")


    def _run_forever(self):
        """Background loop — runs until stopped."""
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:
                if self.config.verbose:
                    print(f"[loop] error in tick: {e}")
            time.sleep(0.1)  # 100ms tick rate


    # ─────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────

    def status(self) -> dict:
        """Current loop status."""
        graph_stats = self.graph.stats()
        health      = self.decay.health()

        return {
            "running"           : self.state.is_running,
            "tokens_processed"  : self.state.tokens_processed,
            "compressions_run"  : self.state.compressions_run,
            "projections_run"   : self.state.projections_run,
            "decay_runs"        : self.state.decay_runs,
            "idle_seconds"      : round(self.state.idle_seconds(), 1),
            "context_window"    : self.state.context_window,
            "graph"             : graph_stats,
            "health"            : health,
        }


    # ─────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────

    def on_schema_formed(self, callback: Callable):
        """Register callback for when a new schema is stabilized."""
        self._on_schema_formed = callback

    def on_hypothetical(self, callback: Callable):
        """Register callback for when hypotheticals are generated."""
        self._on_hypothetical = callback


    # ─────────────────────────────────────────
    # CONTEXT
    # What is the system currently thinking about?
    # ─────────────────────────────────────────

    def current_context(self) -> list[dict]:
        """
        The current working context — recent nodes in focus.
        This is what the system is currently thinking about.
        """
        result = []
        for nid in self.state.context_window:
            node = self.graph.get_node(nid)
            if node:
                result.append({
                    "node_id": nid,
                    "surface": self.graph.surface(nid),
                    "weight" : node["weight"],
                    "level"  : node["level"],
                })
        return result


if __name__ == "__main__":
    import tempfile, os

    print("=== loop.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        graph   = Graph(db_path)

        config = LoopConfig(
            compress_every   = 20,
            project_every    = 2,
            decay_every_sec  = 5.0,
            idle_think_after = 2.0,
            verbose          = True,
        )

        loop = LivingLoop(graph, config)

        # Feed conversation data
        conversation = [
            ("parent", "what is that over there"),
            ("child",  "that is a cat"),
            ("parent", "yes that is a cat"),
            ("child",  "the cat is big"),
            ("parent", "yes the cat is very big"),
            ("child",  "what is that"),
            ("parent", "that is a dog"),
            ("child",  "the dog is small"),
            ("parent", "yes the dog is small"),
            ("child",  "is the cat bigger than the dog"),
            ("parent", "yes the cat is bigger than the dog"),
            ("child",  "what is that over there"),
            ("parent", "that is a bird"),
            ("child",  "the bird is small"),
            ("parent", "yes the bird is very small"),
        ]

        print("Processing conversation...\n")
        for speaker, text in conversation:
            print(f"  {speaker}: {text}")
            loop.ingest_turn(speaker, text)

        # Manual ticks
        print("\nRunning ticks...")
        for _ in range(5):
            loop.tick()

        # Status
        print("\nStatus:")
        status = loop.status()
        for k, v in status.items():
            if k not in ("graph", "health", "context_window"):
                print(f"  {k}: {v}")

        print(f"\nGraph: {status['graph']}")
        print(f"Health: {status['health']}")

        print(f"\nCurrent context:")
        for node in loop.current_context():
            print(f"  '{node['surface']}'  level={node['level']}  "
                  f"weight={node['weight']:.3f}")

        graph.close()
