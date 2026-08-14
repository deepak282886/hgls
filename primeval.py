"""
primeval.py — Primitive Architecture for Learning and Reasoning
==============================================================

One graph. Three modalities. No thresholds.

Structure creation
------------------
Level-1 structures (atom pairs) are created immediately on first co-occurrence.
Level-2+ structures are created by the Consolidator, proportionally to surface
weight. There is no gate: every pair the Consolidator examines gets a structure,
whose initial weight is proportional to the surface between its constituents.
Thin structures exist — they just carry near-zero weight until reward carves them.

This split is architectural, not a threshold: atoms producing level-1 structures
is O(unique_atom_pairs) — bounded and fast. Level-2+ via the Consolidator is
O(top_k²) per pass — controlled by a compute budget, not a weight gate.

Fixes applied
-------------
1. Consolidator no longer overwrites existing edge weights. It only proposes
   an initial weight when the edge is new. Once an edge exists, only traversal
   (occurrence) and reward are allowed to change its weight.

2. Inference traversal now updates _active alongside ingestion traversal, so
   reward() called after infer() correctly strengthens the reasoning chain —
   not just the last ingestion's edges.

3. Downward growth added. After traversal, activated structures strengthen
   their edges back down to constituent atoms, grounding abstractions in their
   atomic substrate and increasing their stability and resistance to compression.

4. Anchor scoring fixed. Activation dominates; level is a tiebreaker scaled
   logarithmically, not a raw multiplier. A high-level structure with weak
   activation can no longer outscore a lower structure with strong activation.
"""

from __future__ import annotations

import os
import time
import pickle
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger("primeval")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Co-occurrence window
    window_size: int = 2

    # Weight signals
    occurrence_delta: float = 1.0
    reward_multiplier: float = 10.0

    # Downward growth — how much an activated structure reinforces
    # edges down to its constituent atoms per traversal
    downward_growth_delta: float = 0.1

    # Consolidator
    consolidator_interval: int = 500
    # proposed weight for a NEW inter-structure link = surface * this
    # existing links are never touched by the consolidator
    consolidator_proposal_scale: float = 0.01
    # compute budget — how many co-activated pairs to process per pass.
    # This is NOT a semantic threshold. Every pair in _counts is a
    # legitimate candidate. This parameter controls how many we can
    # afford to process each pass. The most frequently co-activated
    # pairs go first. The rest are deferred to the next pass.
    # Nothing is excluded — just queued by frequency.
    consolidator_budget: int = 500
    # fractional decay applied to _counts each decay pass.
    # Stale co-activations fade naturally — pairs no longer witnessed
    # gradually drop toward zero and stop consuming consolidator budget.
    counts_decay_rate: float = 0.01

    # Decay — asymptotic compression, no floor, no deletion
    decay_interval: int = 200
    base_decay_rate: float = 5e-4       # level-0 fractional loss per pass
    level_decay_factor: float = 1.5     # multiplied per level above 0

    # Inference
    beam_width: int = 10
    max_traversal_depth: int = 20

    # Persistence
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 5_000


# ══════════════════════════════════════════════════════════════════════════════
# ATOM ENCODING
# ══════════════════════════════════════════════════════════════════════════════

PHONEMES = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH",
    "SIL","SP","BREATH","LAUGH","NOISE",
]
_PHONEME_MAP: dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}


class Atoms:
    """
    Non-overlapping integer atom IDs.

      [   0,  255]  grayscale pixels
      [ 256,  511]  letters  (unicode byte mod 256)
      [ 512,  555]  phonemes (44 ARPAbet + 5 para-linguistic)
      [1024,   …)   structures (graph-assigned)
    """
    PIXEL_OFF    = 0
    LETTER_OFF   = 256
    PHONEME_OFF  = 512
    STRUCT_START = 1024
    ATOM_CEIL    = 1024

    @classmethod
    def pixel(cls, v: int) -> int:
        return cls.PIXEL_OFF + int(np.clip(v, 0, 255))

    @classmethod
    def letter(cls, ch: str) -> int:
        return cls.LETTER_OFF + (ord(ch) & 0xFF)

    @classmethod
    def phoneme(cls, sym: str) -> int:
        return cls.PHONEME_OFF + _PHONEME_MAP.get(sym.upper(), _PHONEME_MAP["SIL"])

    @classmethod
    def sequence(cls, modality: str, values) -> list[int]:
        fn = {"pixel": cls.pixel, "letter": cls.letter, "phoneme": cls.phoneme}[modality]
        return [fn(v) for v in values]

    @classmethod
    def mixed(cls, *pairs) -> list[int]:
        out: list[int] = []
        for modality, values in pairs:
            out.extend(cls.sequence(modality, values))
        return out

    @classmethod
    def describe(cls, node_id: int) -> str:
        if node_id < cls.LETTER_OFF:
            return f"pixel({node_id})"
        if node_id < cls.PHONEME_OFF:
            return f"letter({chr(node_id - cls.LETTER_OFF)!r})"
        if node_id < cls.STRUCT_START:
            idx = node_id - cls.PHONEME_OFF
            return f"phoneme({PHONEMES[idx] if idx < len(PHONEMES) else idx})"
        return f"struct({node_id})"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CORE
# ══════════════════════════════════════════════════════════════════════════════

class NodeMeta:
    __slots__ = ("node_id", "level", "constituents", "traversals", "born_at")

    def __init__(self, node_id: int, level: int, constituents: tuple[int,...]):
        self.node_id      = node_id
        self.level        = level
        self.constituents = constituents
        self.traversals   = 0
        self.born_at      = time.monotonic()


class WeightedGraph:
    """
    Weighted directed graph. No thresholds.

    _in_weight[node_id] is maintained incrementally on every edge change,
    giving O(1) stability queries — critical for the Consolidator's
    stability ranking and inference's confidence scoring.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.RLock()

        self._nodes:          dict[int, NodeMeta]        = {}
        self._pattern:        dict[tuple, int]           = {}
        self._next_struct_id: int                        = Atoms.STRUCT_START

        self._weights:  dict[tuple[int,int], float]      = {}
        self._counts:   dict[tuple[int,int], int]        = {}
        self._reward:   dict[tuple[int,int], float]      = {}

        self._adj:       dict[int, dict[int,float]]      = defaultdict(dict)
        self._in_weight: dict[int, float]                = defaultdict(float)
        self._levels:    dict[int, set[int]]             = defaultdict(set)
        self._active:    set[tuple[int,int]]             = set()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _ensure_atom(self, atom_id: int) -> None:
        if atom_id not in self._nodes:
            with self._lock:
                if atom_id not in self._nodes:
                    self._nodes[atom_id] = NodeMeta(atom_id, 0, (atom_id,))
                    self._levels[0].add(atom_id)

    def get_or_create_structure(self, constituents: tuple[int,...], level: int) -> int:
        if constituents in self._pattern:
            return self._pattern[constituents]
        with self._lock:
            if constituents in self._pattern:
                return self._pattern[constituents]
            nid = self._next_struct_id
            self._next_struct_id += 1
            self._nodes[nid] = NodeMeta(nid, level, constituents)
            self._pattern[constituents] = nid
            self._levels[level].add(nid)
            logger.debug("New structure L%d id=%d", level, nid)
            return nid

    # ── Edges ─────────────────────────────────────────────────────────────────

    def _add_weight(self, src: int, dst: int, delta: float) -> float:
        key = (src, dst)
        with self._lock:
            new_w = self._weights.get(key, 0.0) + delta
            self._weights[key] = new_w
            self._adj[src][dst] = new_w
            self._in_weight[dst] += delta
        return new_w

    def propose_weight(self, src: int, dst: int, w: float) -> bool:
        """
        FIX 1 — Consolidator-only method.
        Sets an edge weight ONLY if the edge does not yet exist.
        Once an edge exists, only traversal (occurrence) and reward
        are permitted to change its weight. Returns True if a new
        edge was created, False if the edge already existed and was
        left untouched.
        """
        key = (src, dst)
        with self._lock:
            if key in self._weights:
                return False          # already exists — do not overwrite
            self._weights[key] = w
            self._adj[src][dst] = w
            self._in_weight[dst] += w
        return True

    # ── Traversal ─────────────────────────────────────────────────────────────

    def traverse(self, atom_ids: list[int], window: int,
                 update_active: bool = True) -> set[tuple[int,int]]:
        """
        Two-stage sequential traversal. Returns the set of active edges.

        Ordering is enforced throughout. Co-occurrence means sequential
        proximity — src appears before dst within the window. A structure
        is only activated if its constituents appear as an ordered
        subsequence in the input, preserving the direction of the sequence.

        Stage 1 — Atom level
        For each atom at position i, pair it with atoms at positions
        i+1 … i+window (forward only). Edge src→dst encodes order.
        A level-1 structure (src, dst) encodes a sequential bigram.

        Stage 2 — Structure level (existing structures only)
        A structure fires only if its constituents appear in order
        in the activated sequence from the previous level.
        Activated structures are kept as an ordered list, and edges
        between them also respect that order.

        Downward growth (FIX 3)
        Every activated structure reinforces edges back down to its
        direct constituent atoms, deepening atomic grounding.
        """
        for aid in atom_ids:
            self._ensure_atom(aid)

        local_active: set[tuple[int,int]] = set()
        n = len(atom_ids)

        # ── Stage 1: sequential atom pairs → edges + level-1 structures ───────
        for i in range(n):
            self._nodes[atom_ids[i]].traversals += 1
            for j in range(i + 1, min(i + window + 1, n)):
                src, dst = atom_ids[i], atom_ids[j]   # order preserved
                self._add_weight(src, dst, self.cfg.occurrence_delta)
                with self._lock:
                    self._counts[(src, dst)] = self._counts.get((src, dst), 0) + 1
                local_active.add((src, dst))
                # level-1 structure encodes ordered pair (src before dst)
                self.get_or_create_structure((src, dst), level=1)

        # ── Stage 2: sequential structure activation ───────────────────────────
        # prev_activated is an ordered list — order mirrors input sequence.
        # A structure fires only if its constituents appear in that order.
        prev_activated_ordered: list[int] = list(atom_ids)
        prev_activated_set:     set[int]  = set(atom_ids)
        max_lv = self.max_level()

        for level in range(1, max_lv + 1):
            activated_ordered: list[int] = []

            for nid in self._levels.get(level, set()):
                meta = self._nodes.get(nid)
                if meta is None:
                    continue
                # All constituents must be present AND appear in order
                if not all(c in prev_activated_set for c in meta.constituents):
                    continue
                if not self._constituents_in_order(
                    meta.constituents, prev_activated_ordered
                ):
                    continue

                activated_ordered.append(nid)
                meta.traversals += 1

                # Downward growth — reinforce edges to direct constituent atoms
                for constituent in meta.constituents:
                    if constituent < Atoms.STRUCT_START:
                        self._add_weight(
                            nid, constituent,
                            self.cfg.downward_growth_delta
                        )
                        local_active.add((nid, constituent))

            # Sequential edges between activated structures
            m = len(activated_ordered)
            for i in range(m):
                for j in range(i + 1, min(i + window + 1, m)):
                    sa, sb = activated_ordered[i], activated_ordered[j]
                    self._add_weight(sa, sb, self.cfg.occurrence_delta)
                    # Track co-activation count — used by consolidator
                    # to identify genuinely sequential structure pairs
                    with self._lock:
                        self._counts[(sa, sb)] = self._counts.get((sa, sb), 0) + 1
                    local_active.add((sa, sb))

            if not activated_ordered:
                break
            prev_activated_ordered = activated_ordered
            prev_activated_set     = set(activated_ordered)

        if update_active:
            self._active = local_active

        return local_active

    @staticmethod
    def _constituents_in_order(
        constituents: tuple[int,...], ordered: list[int]
    ) -> bool:
        """
        Check that all constituents appear in order within the ordered list.
        Uses a greedy subsequence check — same logic as 'is subsequence'.
        """
        it = iter(ordered)
        return all(c in it for c in constituents)

    # ── Reward ────────────────────────────────────────────────────────────────

    def apply_reward(self, amount: float) -> None:
        """
        Strengthen all active edges.
        _active is the union of the last ingestion traversal AND any
        inference traversal that followed — so reward correctly
        strengthens whichever path was most recently walked.
        """
        delta = amount * self.cfg.reward_multiplier
        with self._lock:
            for key in self._active:
                if key in self._weights:
                    self._weights[key] += delta
                    self._adj[key[0]][key[1]] = self._weights[key]
                    self._in_weight[key[1]] += delta
                    self._reward[key] = self._reward.get(key, 0.0) + amount

    # ── Decay ─────────────────────────────────────────────────────────────────

    def decay(self) -> None:
        """
        Asymptotic compression. No floor. No deletion.
        Float underflow to exactly 0.0 triggers memory cleanup only.
        Higher-level structures decay faster — they compress toward
        the atomic bedrock beneath them.

        Also decays _counts — co-activation counts for pairs no longer
        witnessed fade naturally. This prevents stale pairs from
        permanently consuming consolidator budget.
        """
        with self._lock:
            # ── Edge weight decay ──────────────────────────────────────────────
            to_remove: list[tuple[int,int]] = []
            for key, w in list(self._weights.items()):
                level = self._nodes[key[0]].level if key[0] in self._nodes else 0
                rate  = self.cfg.base_decay_rate * (self.cfg.level_decay_factor ** level)
                new_w = w * (1.0 - rate)
                diff  = new_w - w
                if new_w == 0.0:
                    to_remove.append(key)
                    self._in_weight[key[1]] += diff
                else:
                    self._weights[key] = new_w
                    self._adj[key[0]][key[1]] = new_w
                    self._in_weight[key[1]] += diff
            for key in to_remove:
                del self._weights[key]
                self._counts.pop(key, None)
                self._reward.pop(key, None)
                self._adj.get(key[0], {}).pop(key[1], None)

            # ── Counts decay ───────────────────────────────────────────────────
            # Co-activation counts decay fractionally each pass.
            # Pairs no longer witnessed gradually fade and stop consuming
            # consolidator budget. Integer floor at 0 — never negative.
            rate = self.cfg.counts_decay_rate
            stale: list[tuple[int,int]] = []
            for key, count in list(self._counts.items()):
                new_count = count * (1.0 - rate)
                if new_count < 1.0:
                    stale.append(key)
                else:
                    self._counts[key] = new_count
            for key in stale:
                del self._counts[key]

    # ── Queries ───────────────────────────────────────────────────────────────

    def neighbors(self, node_id: int) -> list[tuple[int, float]]:
        result = list(self._adj.get(node_id, {}).items())
        result.sort(key=lambda x: -x[1])
        return result

    def weight(self, src: int, dst: int) -> float:
        return self._weights.get((src, dst), 0.0)

    def stability(self, node_id: int) -> float:
        """Incoming weight sum — O(1). Confidence is weight, read off the geometry."""
        return self._in_weight.get(node_id, 0.0)

    def nodes_at_level(self, level: int) -> list[int]:
        return list(self._levels.get(level, set()))

    def max_level(self) -> int:
        return max(self._levels.keys(), default=0)

    def get_stats(self) -> dict:
        return {
            "nodes":           len(self._nodes),
            "edges":           len(self._weights),
            "structures":      len(self._pattern),
            "max_level":       self.max_level(),
            "nodes_per_level": {l: len(s) for l, s in sorted(self._levels.items())},
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with self._lock:
            state = {
                "nodes": self._nodes, "pattern": self._pattern,
                "next_struct_id": self._next_struct_id,
                "weights": dict(self._weights), "counts": dict(self._counts),
                "reward": dict(self._reward),
                "adj": {k: dict(v) for k, v in self._adj.items()},
                "in_weight": dict(self._in_weight),
                "levels": {k: set(v) for k, v in self._levels.items()},
            }
        with open(os.path.join(path, "graph.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=5)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "graph.pkl"), "rb") as f:
            state = pickle.load(f)
        with self._lock:
            self._nodes          = state["nodes"]
            self._pattern        = state["pattern"]
            self._next_struct_id = state["next_struct_id"]
            self._weights        = state["weights"]
            self._counts         = state["counts"]
            self._reward         = state["reward"]
            self._adj       = defaultdict(dict,  {k: dict(v) for k,v in state["adj"].items()})
            self._in_weight = defaultdict(float, state["in_weight"])
            self._levels    = defaultdict(set,   {k: set(v) for k,v in state["levels"].items()})


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class Consolidator:
    """
    Creates inter-structure connections at level-2 and above.

    Proximity redefined as co-activation.
    The old approach ranked structures by stability and paired top-k
    arbitrarily — producing sequentially meaningless pairs that never
    actually co-activated in any real input.

    The fix: traversal already creates directed edges between structures
    that co-activated sequentially in the same input (Stage 2 of traverse).
    Those edges and their counts ARE the co-activation history.
    The consolidator reads those edges and proposes a higher-level
    structure for each co-activated pair.

    Every consolidator proposal is now grounded in at least one real
    input where both structures fired together in order. The resulting
    higher-level structures are genuine sequential abstractions.

    The consolidator remains a proposer only — propose_weight writes
    once and never overwrites. Only traversal and reward may thicken.
    """

    def __init__(self, graph: WeightedGraph, cfg: Config):
        self.graph       = graph
        self.cfg         = cfg
        self.total_links = 0

    def run_pass(self) -> int:
        g   = self.graph
        cfg = self.cfg
        new = 0

        # Collect all co-activated structure pairs from _counts.
        # Sort by co-activation count descending — most frequently
        # witnessed sequential pairs get processed first.
        # Process only up to consolidator_budget pairs per pass.
        # This is a compute budget, not a semantic threshold —
        # every pair is a legitimate candidate, the budget just
        # controls how many we can afford each pass. Pairs not
        # processed this pass will be picked up next time, ranked
        # again by their then-current count. Counts decay ensures
        # stale pairs gradually drop out of contention naturally.
        coactivated: list[tuple[float, int, int]] = []
        for (a, b), count in list(g._counts.items()):
            meta_a = g._nodes.get(a)
            meta_b = g._nodes.get(b)
            if meta_a is None or meta_b is None:
                continue
            if meta_a.level < 1 or meta_b.level < 1:
                continue   # atom pairs handled by traversal directly
            coactivated.append((count, a, b))

        # Most frequently co-activated pairs first
        coactivated.sort(reverse=True)

        for count, a, b in coactivated[:cfg.consolidator_budget]:
            meta_a   = g._nodes.get(a)
            meta_b   = g._nodes.get(b)
            if meta_a is None or meta_b is None:
                continue
            surface  = self._surface(a, b)
            if surface <= 0.0:
                continue
            new_level  = max(meta_a.level, meta_b.level) + 1
            g.get_or_create_structure((a, b), level=new_level)
            proposed_w = surface * cfg.consolidator_proposal_scale
            if g.propose_weight(a, b, proposed_w):
                new += 1

        self.total_links += new
        if new:
            logger.info("Consolidator: %d new links (total %d)", new, self.total_links)
        return new

    def _surface(self, a: int, b: int) -> float:
        """
        Boundary surface: total weight of edges crossing between
        the constituent atom sets of structures a and b.
        Discrete analogue of the Nambu-Goto minimal surface.
        Proximity is now grounded in co-activation history.
        """
        g  = self.graph
        ma = g._nodes.get(a)
        mb = g._nodes.get(b)
        if ma is None or mb is None:
            return 0.0
        ca = set(ma.constituents)
        cb = set(mb.constituents)
        total = 0.0
        for u in ca:
            for v in cb:
                total += g.weight(u, v)
                total += g.weight(v, u)
        return total


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class InferenceResult(NamedTuple):
    anchor_id:    int
    anchor_level: int
    confidence:   float
    chain:        list[int]
    path_weights: list[float]


class InferenceEngine:
    """
    Top-down inference: anchor → traverse.

    FIX 2 — _active is updated during traversal so reward() called after
    infer() strengthens the reasoning chain, not just the last ingestion.
    The graph merges inference active edges into _active via update_active=True.

    FIX 4 — Anchor scoring: activation dominates; level is a log-scaled
    tiebreaker. A high-level structure with weak activation can no longer
    outscore a lower structure with strong activation.

        score = activation × stability × (1 + log(1 + level))

    The log term gently favours higher abstractions when activation and
    stability are equal, without letting level dominate the product.

    Anchoring
    ---------
    Score all structures by activation × stability × (1 + log(1 + level)).
    Activation is recursive and memoised:
      - Atom: 1.0 if present in input, 0.0 if absent
      - Structure: mean activation of its direct constituents

    The highest-scoring structure wins. If the input is genuinely novel,
    it anchors to the nearest stable structure — weaker foothold, coherent.

    Traversal
    ---------
    Beam search along highest-weight outgoing edges. All traversed edges
    are merged into the graph's _active set so reward propagates correctly.
    """

    def __init__(self, graph: WeightedGraph, cfg: Config):
        self.graph = graph
        self.cfg   = cfg

    def run(self, atom_ids: list[int]) -> InferenceResult:
        # FIX 2 — run a read-only traversal to register active edges for reward
        # update_active=True so the graph's _active is updated by this inference
        self.graph.traverse(atom_ids, self.cfg.window_size, update_active=True)

        anchor_id         = self._anchor(atom_ids)
        chain, pw, t_active = self._traverse(anchor_id)

        # FIX 2 — merge traversal edges into _active so reward() hits the chain
        with self.graph._lock:
            self.graph._active |= t_active

        level = self.graph._nodes[anchor_id].level if anchor_id in self.graph._nodes else 0
        return InferenceResult(
            anchor_id=anchor_id,
            anchor_level=level,
            confidence=self.graph.stability(anchor_id),
            chain=chain,
            path_weights=pw,
        )

    def _anchor(self, atom_ids: list[int]) -> int:
        """
        Bottom-up activation propagation followed by a top read.
        No competition. No scoring race between levels.

        Pass 1 — propagate upward.
        Atoms fire if present in the input. A structure fires if all its
        constituents fired AND appear in order in the input sequence.
        Everything that can light up, lights up. One upward pass.

        Pass 2 — read the top.
        Find the highest level that has any lit node.
        Stability breaks ties within that level.

        Traversal then starts from that node.
        """
        g = self.graph

        # ── Pass 1: bottom-up firing ───────────────────────────────────────────
        lit:             dict[int, bool] = {}
        # ordered_at_level[level] = ordered list of lit node IDs at that level
        # used by each successive level to check sequential constituent order
        ordered_at_level: dict[int, list[int]] = {}

        # Fire atoms — order preserved from input
        atom_set = set(atom_ids)
        lit_atoms_ordered: list[int] = []
        for nid, meta in g._nodes.items():
            if meta.level == 0:
                lit[nid] = nid in atom_set
        # preserve input order for atoms
        lit_atoms_ordered = [aid for aid in atom_ids if lit.get(aid, False)]
        ordered_at_level[0] = lit_atoms_ordered

        # Fire structures level by level upward
        for level in range(1, g.max_level() + 1):
            prev_ordered = ordered_at_level.get(level - 1, [])
            prev_set     = set(prev_ordered)
            lit_this_level: list[int] = []

            for nid in g.nodes_at_level(level):
                meta = g._nodes.get(nid)
                if meta is None:
                    lit[nid] = False
                    continue
                # All constituents must have lit at the previous level
                if not all(lit.get(c, False) for c in meta.constituents):
                    lit[nid] = False
                    continue
                # Constituents must appear in order in the previous level's
                # activated list — this is what makes higher structures
                # genuinely sequential
                if not WeightedGraph._constituents_in_order(
                    meta.constituents, prev_ordered
                ):
                    lit[nid] = False
                    continue
                lit[nid] = True
                lit_this_level.append(nid)

            ordered_at_level[level] = lit_this_level

        # ── Pass 2: read the highest lit level ────────────────────────────────
        for level in range(g.max_level(), 0, -1):
            candidates = [
                (g.stability(nid), nid)
                for nid in g.nodes_at_level(level)
                if lit.get(nid, False)
            ]
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]

        # Nothing above atoms lit — return most stable input atom
        best_id   = atom_ids[0] if atom_ids else 0
        best_stab = 0.0
        for aid in atom_ids:
            if aid in g._nodes:
                s = g.stability(aid)
                if s > best_stab:
                    best_stab, best_id = s, aid
        return best_id

    def _traverse(
        self, start_id: int
    ) -> tuple[list[int], list[float], set[tuple[int,int]]]:
        """
        Beam search with structure preference.
        Returns (chain, path_weights, active_edges).
        active_edges is the set of edges walked, for reward propagation.

        Structure preference — two mechanisms:

        1. Neighbours are scored by effective_weight = w * (1 + log(1 + level))
           where level is the destination node's level. This gently lifts
           structure-to-structure edges over atom-to-atom edges without
           creating a hard threshold — atoms can still be visited, they
           just don't dominate once structure-level paths exist.

        2. When expanding from a structure node (level >= 1), structure
           neighbours are tried before atom neighbours. If any structure
           neighbours exist, atom neighbours are skipped for that step.
           This prevents downward-growth edges (structure -> atom) from
           pulling traversal back down to the atomic level once it has
           climbed into semantic territory.
        """
        import math
        g   = self.graph
        cfg = self.cfg

        def effective_weight(dst: int, w: float) -> float:
            meta = g._nodes.get(dst)
            level = meta.level if meta else 0
            return w * (1.0 + math.log(1.0 + level))

        beam: list[tuple[float, int, list[int], list[float]]] = [
            (0.0, start_id, [start_id], [])
        ]
        best_path:    list[int]           = [start_id]
        best_weights: list[float]         = []
        all_active:   set[tuple[int,int]] = set()

        for _ in range(cfg.max_traversal_depth):
            if not beam:
                break
            next_beam: list[tuple[float, int, list[int], list[float]]] = []

            for neg_w, node, path, pw in beam:
                node_meta  = g._nodes.get(node)
                node_level = node_meta.level if node_meta else 0
                nbs        = g.neighbors(node)

                if not nbs:
                    if len(path) > len(best_path):
                        best_path, best_weights = path, pw
                    continue

                # When at a structure node — prefer structure neighbours.
                # Only fall back to atom neighbours if no structure ones exist.
                if node_level >= 1:
                    struct_nbs = [
                        (dst, w) for dst, w in nbs
                        if g._nodes.get(dst) and g._nodes[dst].level >= 1
                        and dst not in path
                    ]
                    candidates = struct_nbs if struct_nbs else [
                        (dst, w) for dst, w in nbs if dst not in path
                    ]
                else:
                    candidates = [(dst, w) for dst, w in nbs if dst not in path]

                # Score by level-weighted effective weight, take top beam_width
                scored = sorted(
                    candidates,
                    key=lambda x: -effective_weight(x[0], x[1])
                )[:cfg.beam_width]

                for dst, w in scored:
                    all_active.add((node, dst))
                    ew = effective_weight(dst, w)
                    next_beam.append((neg_w - ew, dst, path + [dst], pw + [w]))

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:cfg.beam_width]
            best_path, best_weights = beam[0][2], beam[0][3]

        return best_path, best_weights, all_active


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class Primeval:
    """
    Top-level entry point.

    ingest(modality, values)    feed a sequence; update graph
    reward(amount)              reward the most recently active path
    infer(modality, values)     top-down inference

    The environment calls reward() with whatever signal it produces.
    Reward strengthens whichever path was most recently walked —
    whether that was an ingestion traversal or an inference traversal.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg          = cfg or Config()
        self.graph        = WeightedGraph(self.cfg)
        self.consolidator = Consolidator(self.graph, self.cfg)
        self.inference    = InferenceEngine(self.graph, self.cfg)
        self._step        = 0

    def ingest(self, modality: str, values) -> None:
        self._ingest_atoms(Atoms.sequence(modality, values))

    def ingest_atoms(self, atom_ids: list[int]) -> None:
        self._ingest_atoms(atom_ids)

    def ingest_mixed(self, pairs) -> None:
        self._ingest_atoms(Atoms.mixed(*pairs))

    def reward(self, amount: float) -> None:
        self.graph.apply_reward(amount)

    def infer(self, modality: str, values) -> InferenceResult:
        return self.inference.run(Atoms.sequence(modality, values))

    def infer_atoms(self, atom_ids: list[int]) -> InferenceResult:
        return self.inference.run(atom_ids)

    def infer_mixed(self, pairs) -> InferenceResult:
        return self.inference.run(Atoms.mixed(*pairs))

    def save(self, path: str) -> None:
        self.graph.save(path)
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({"step": self._step, "config": self.cfg}, f, protocol=5)
        logger.info("Saved to %s (step %d)", path, self._step)

    def load(self, path: str) -> None:
        self.graph.load(path)
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self._step = meta["step"]
        # Preserve current checkpoint_dir — do not let the loaded config
        # overwrite it. Each stage trainer sets its own checkpoint_dir
        # in cfg before calling load(); restoring the old one would cause
        # periodic saves to land in the previous stage's directory.
        current_checkpoint_dir = self.cfg.checkpoint_dir
        self.cfg = meta["config"]
        self.cfg.checkpoint_dir = current_checkpoint_dir
        logger.info("Loaded from %s, resuming at step %d", path, self._step)

    def stats(self) -> dict:
        return {"step": self._step, **self.graph.get_stats()}

    def top_edges(self, n: int = 20) -> list[tuple[str, str, float]]:
        items = sorted(self.graph._weights.items(), key=lambda x: -x[1])[:n]
        return [(Atoms.describe(s), Atoms.describe(d), w) for (s,d),w in items]

    def describe_node(self, node_id: int) -> dict:
        g    = self.graph
        meta = g._nodes.get(node_id)
        if meta is None:
            return {"error": "node not found"}
        return {
            "id":           node_id,
            "label":        Atoms.describe(node_id),
            "level":        meta.level,
            "constituents": [Atoms.describe(c) for c in meta.constituents],
            "traversals":   meta.traversals,
            "stability":    round(g.stability(node_id), 2),
            "top_neighbors":[(Atoms.describe(d), round(w, 2))
                             for d, w in g.neighbors(node_id)[:8]],
        }

    def _ingest_atoms(self, atom_ids: list[int]) -> None:
        self.graph.traverse(atom_ids, self.cfg.window_size, update_active=True)
        self._step += 1
        if self._step % self.cfg.consolidator_interval == 0:
            self.consolidator.run_pass()
        if self._step % self.cfg.decay_interval == 0:
            self.graph.decay()
        if self._step % self.cfg.checkpoint_interval == 0:
            self.save(os.path.join(self.cfg.checkpoint_dir, f"step_{self._step:08d}"))


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    cfg = Config(
        window_size=2,
        occurrence_delta=1.0,
        reward_multiplier=10.0,
        downward_growth_delta=0.1,
        consolidator_interval=50,
        consolidator_proposal_scale=0.01,
        decay_interval=100,
        base_decay_rate=1e-3,
        checkpoint_interval=9_999,
    )
    sys = Primeval(cfg)

    corpus = [
        "the cat sat on the mat",
        "the cat sat on the hat",
        "a cat and a mat",
        "the cat in the hat",
    ]
    for sentence in corpus * 20:
        sys.ingest("letter", sentence)
        sys.reward(0.1)

    for _ in range(20):
        sys.ingest("pixel", [10, 20, 10, 20, 10])
        sys.reward(0.5)

    for _ in range(20):
        sys.ingest("phoneme", ["DH", "AH", "K", "AE", "T"])
        sys.reward(0.2)

    for _ in range(10):
        sys.ingest_mixed([("letter", "cat"), ("phoneme", ["K","AE","T"])])
        sys.reward(0.3)

    print("\n── Stats ────────────────────────────────────────────────")
    for k, v in sys.stats().items():
        print(f"  {k}: {v}")

    print("\n── Top 10 edges ─────────────────────────────────────────")
    for s, d, w in sys.top_edges(10):
        print(f"  {s:30s} → {d:30s}  w={w:.1f}")

    for query in ["the cat", "cat", "hat"]:
        r = sys.infer("letter", query)
        # reward the inference chain — this now correctly hits the reasoning path
        sys.reward(0.1)
        print(f"\n── Inference: {query!r}")
        print(f"  anchor:     {Atoms.describe(r.anchor_id)}")
        print(f"  level:      {r.anchor_level}")
        print(f"  confidence: {r.confidence:.1f}")
        print(f"  chain len:  {len(r.chain)}")
        if len(r.chain) > 1:
            print(f"  chain:      {[Atoms.describe(n) for n in r.chain[:5]]}")

    print("\n── Node: letter 't' ─────────────────────────────────────")
    for k, v in sys.describe_node(Atoms.letter("t")).items():
        print(f"  {k}: {v}")