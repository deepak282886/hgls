"""
Generation Engine
Takes input. Activates current state in graph.
Runs hierarchical rollout. Commits best path top down.
Traverses committed path to produce output sequence.
Feeds output back into absorption. Loop closes.
"""

import numpy as np
from typing import Optional
from graph_engine import Node, GeometricSpace, RewardGraph
from absorption import Absorber, MergeEngine
from rollout import RolloutEngine, RolloutPath
from reward import RewardEngine


# ─────────────────────────────────────────────
# INPUT ACTIVATOR
# Maps input text to activated nodes in graph
# ─────────────────────────────────────────────

class InputActivator:
    """
    Given input text, finds which nodes it activates
    across all levels of the hierarchy.
    Activation strength decays with distance from exact match.
    """

    def __init__(self, space: GeometricSpace, top_k: int = 5):
        self.space = space
        self.top_k = top_k

    def activate(self, text: str) -> list[tuple[Node, float]]:
        """
        Find nodes activated by input text.
        Returns list of (node, activation_strength) sorted by strength.
        """
        text = text.lower().strip()
        activated = []

        for node in self.space.nodes.values():
            strength = self._match_strength(text, node)
            if strength > 0:
                activated.append((node, strength))

        # sort by activation strength
        activated.sort(key=lambda x: x[1], reverse=True)
        return activated[:self.top_k * 3]  # keep top matches across levels

    def activate_nodes_only(self, text: str) -> list[Node]:
        """Return just the nodes, no strengths."""
        return [node for node, _ in self.activate(text)]

    def _match_strength(self, text: str, node: Node) -> float:
        """
        How strongly does this input activate this node.
        Exact match = 1.0
        Partial match = proportional
        Substring = 0.5
        No match = 0.0
        """
        seq = node.sequence.lower()

        # exact match
        if text == seq:
            return 1.0

        # input contains node sequence
        if seq in text and len(seq) > 1:
            return 0.8 * (len(seq) / len(text))

        # node sequence contains input
        if text in seq and len(text) > 1:
            return 0.6 * (len(text) / len(seq))

        # word level partial — any word in common
        text_words = set(text.split())
        seq_words = set(seq.split())
        common = text_words & seq_words
        if common:
            return 0.4 * (len(common) / max(len(text_words), len(seq_words)))

        return 0.0


# ─────────────────────────────────────────────
# PATH DECODER
# Traverses committed path and produces output
# ─────────────────────────────────────────────

class PathDecoder:
    """
    Given a committed rollout path, decode it into output text.
    Works top down — high level path commits shape,
    lower levels fill content, lowest levels produce actual tokens.
    """

    def __init__(self, space: GeometricSpace):
        self.space = space

    def decode(self, path: RolloutPath) -> str:
        """
        Decode a path into output text.
        Strategy: collect all leaf-level sequences from path nodes,
        ordered by path traversal, deduplicated.
        """
        if not path.nodes:
            return ""

        # collect output tokens from path
        output_parts = []
        seen = set()

        for node in path.nodes:
            tokens = self._extract_tokens(node)
            for token in tokens:
                if token not in seen and len(token) > 1:
                    seen.add(token)
                    output_parts.append(token)

        return " ".join(output_parts)

    def _extract_tokens(self, node: Node) -> list[str]:
        """
        Extract meaningful tokens from a node.
        If node has children, prefer children's sequences.
        If leaf node, return its sequence directly.
        """
        if not node.children:
            # leaf node — return sequence directly if it's a word or phrase
            seq = node.sequence
            # filter out merge notation artifacts
            if seq.startswith('[') and '|' in seq:
                # extract the original sequences from merge notation
                return self._unpack_merge_node(seq)
            return [seq]
        else:
            # abstraction node — return children sequences
            tokens = []
            for child in node.children:
                tokens.extend(self._extract_tokens(child))
            return tokens

    def _unpack_merge_node(self, seq: str) -> list[str]:
        """Extract readable content from merge node notation like [cat|dog]."""
        # strip brackets and split on pipe
        inner = seq.strip('[]')
        parts = inner.split('|')
        results = []
        for part in parts:
            part = part.strip('[]').strip()
            if len(part) > 1 and not part.startswith('['):
                results.append(part)
        return results


# ─────────────────────────────────────────────
# GENERATION ENGINE
# Full pipeline — input to output
# ─────────────────────────────────────────────

class GenerationEngine:
    """
    Main generation loop.
    Input → activate → rollout → commit best path → decode → output
    Output feeds back into absorption.
    Reward signal updates graph after outcome known.
    """

    def __init__(
        self,
        space: GeometricSpace,
        reward_graph: RewardGraph,
        absorber: Absorber,
        rollout_engine: RolloutEngine,
        reward_engine: RewardEngine,
        top_k_paths: int = 5,
        auto_absorb_output: bool = True
    ):
        self.space = space
        self.reward_graph = reward_graph
        self.absorber = absorber
        self.rollout_engine = rollout_engine
        self.reward_engine = reward_engine
        self.top_k_paths = top_k_paths
        self.auto_absorb_output = auto_absorb_output

        self.activator = InputActivator(space)
        self.decoder = PathDecoder(space)

        # generation history
        self.history: list[dict] = []
        self._pending_paths: list[RolloutPath] = []

    # ── Main Generate ─────────────────────────

    def generate(self, input_text: str) -> dict:
        """
        Full generation cycle.
        Returns dict with output, paths considered, best path.
        """
        # 1. activate current state from input
        activated_nodes = self.activator.activate_nodes_only(input_text)

        if not activated_nodes:
            return {
                "input": input_text,
                "output": "[no activation — input not in graph]",
                "paths_considered": 0,
                "best_path": None
            }

        # 2. run hierarchical rollout from activated nodes
        paths = self.rollout_engine.hierarchical_rollout(activated_nodes)

        if not paths:
            return {
                "input": input_text,
                "output": "[no paths found]",
                "paths_considered": 0,
                "best_path": None
            }

        # 3. commit best path
        best_path = paths[0]
        competing_paths = paths[1:self.top_k_paths]

        # store for later reward update
        self._pending_paths = paths

        # 4. decode path to output
        output = self.decoder.decode(best_path)

        # 5. auto absorb output back into graph
        if self.auto_absorb_output and output:
            self.absorber.absorb(output)

        # 6. log
        record = {
            "input": input_text,
            "output": output,
            "paths_considered": len(paths),
            "best_path": best_path,
            "activated_nodes": [n.sequence for n in activated_nodes[:5]]
        }
        self.history.append(record)

        return record

    # ── Feedback Signals ─────────────────────

    def feedback_success(self, outcome_reward: float = 1.0):
        """Call after generate() when outcome was good."""
        if not self._pending_paths:
            return
        best = self._pending_paths[0]
        competing = self._pending_paths[1:self.top_k_paths]
        self.reward_engine.compare_and_update(best, competing, outcome_reward)

    def feedback_failure(self):
        """Call after generate() when outcome was bad."""
        if not self._pending_paths:
            return
        best = self._pending_paths[0]
        competing = self._pending_paths[1:self.top_k_paths]
        self.reward_engine.failure(best, competing_paths=competing)

    def feedback_correction(self, correct_sequence: list[str], reward: float = 1.0):
        """
        Call with correct path when system got it wrong.
        Directly carves the right path into the graph.
        """
        self.reward_engine.correction(correct_sequence, reward)

    # ── Diagnostics ──────────────────────────

    def stats(self) -> dict:
        return {
            "total_generated": len(self.history),
            "graph": self.space.stats(),
            "reward_graph": self.reward_graph.stats(),
            "rollout": self.rollout_engine.stats(),
            "reward": self.reward_engine.stats()
        }


# ─────────────────────────────────────────────
# FULL SYSTEM ASSEMBLER
# Convenience builder for the complete stack
# ─────────────────────────────────────────────

def build_system(dim: int = 32, merge_threshold: float = 2.0) -> dict:
    """Build and return the complete system."""
    space = GeometricSpace(dim=dim, merge_threshold=merge_threshold, pull_rate=0.1)
    reward_graph = RewardGraph()
    merge_engine = MergeEngine(space=space, dim=dim)
    absorber = Absorber(
        space=space,
        reward_graph=reward_graph,
        merge_engine=merge_engine,
        dim=dim
    )
    rollout_engine = RolloutEngine(
        space=space,
        reward_graph=reward_graph,
        max_depth=6,
        n_rollouts=20
    )
    reward_engine = RewardEngine(
        space=space,
        reward_graph=reward_graph
    )
    generation_engine = GenerationEngine(
        space=space,
        reward_graph=reward_graph,
        absorber=absorber,
        rollout_engine=rollout_engine,
        reward_engine=reward_engine
    )

    return {
        "space": space,
        "reward_graph": reward_graph,
        "merge_engine": merge_engine,
        "absorber": absorber,
        "rollout_engine": rollout_engine,
        "reward_engine": reward_engine,
        "reward": reward_engine,
        "generation": generation_engine
    }


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generation Engine Test ===\n")

    # build full system
    sys = build_system(dim=32, merge_threshold=2.0)
    gen = sys["generation"]
    absorber = sys["absorber"]
    reward_engine = sys["reward"]

    # absorb training corpus
    corpus = [
        "the cat sat on the mat.",
        "the cat ate the rat.",
        "the dog sat on the log.",
        "cats and dogs are common pets.",
        "the cat chased the mouse around the house.",
        "a dog ran across the field.",
        "the quick brown fox jumps over the lazy dog.",
        "dogs are loyal and friendly animals.",
        "cats are independent and curious creatures.",
        "the old cat and the young dog became friends.",
    ]

    print("Absorbing corpus...")
    for text in corpus:
        absorber.absorb(text)
        absorber.absorb_cross_level(text)
    print(f"{sys['space']}\n")

    # ── test 1: basic generation ──────────────
    print("--- Test 1: Generate from 'the cat' ---")
    result = gen.generate("the cat")
    print(f"  Input:     {result['input']}")
    print(f"  Output:    {result['output']}")
    print(f"  Activated: {result['activated_nodes']}")
    print(f"  Paths:     {result['paths_considered']}")

    # signal success
    gen.feedback_success(outcome_reward=1.0)
    print(f"  → feedback: success\n")

    # ── test 2: generation after correction ───
    print("--- Test 2: Correction then regenerate ---")
    gen.feedback_correction(["the", "cat", "sat", "mat"], reward=1.0)
    result2 = gen.generate("the cat")
    print(f"  Input:  {result2['input']}")
    print(f"  Output: {result2['output']}")
    gen.feedback_success(outcome_reward=1.0)
    print(f"  → feedback: success\n")

    # ── test 3: different input ───────────────
    print("--- Test 3: Generate from 'dog' ---")
    result3 = gen.generate("dog")
    print(f"  Input:  {result3['input']}")
    print(f"  Output: {result3['output']}")
    print(f"  Paths:  {result3['paths_considered']}")
    gen.feedback_failure()
    print(f"  → feedback: failure\n")

    # ── test 4: learning over iterations ─────
    print("--- Test 4: Learning loop — 5 iterations on 'the cat' ---")
    gen.feedback_correction(["cat", "sat", "mat"], reward=1.0)

    for i in range(5):
        r = gen.generate("the cat")
        gen.feedback_success(outcome_reward=1.0)
        print(f"  iter {i+1}: {r['output']}")

    # ── top reward nodes ──────────────────────
    print("\n--- Top Reward Nodes (words) ---")
    top = reward_engine.top_reward_nodes(level=1, k=6)
    for node in top:
        print(f"  {node.sequence:<20} reward={node.reward_weight:.4f}  visits={node.visit_count}")

    print(f"\nFull system stats:")
    stats = gen.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Generation Engine OK ===")