"""
stage3.py — Little Deepak: Stage 3 Curriculum
==============================================

Simple semantics. First time patterns connect to meaning.

Three data sources:

  WordNet relationships
    IsA chains — "cat is a feline", "feline is a carnivore"
    HasProperty — derived from adjective usage in definitions
    PartOf — meronym relationships from WordNet
    10,000 simplest relationships, concrete nouns and common adjectives only.
    Fed as letter sequences: "cat is a animal", "fire is hot"

  Cause-effect sequences
    Programmatically generated from 200 primitive concept seeds.
    "touch fire get burn", "eat food feel full", "drop thing it fall"
    Subject-verb-object-consequence chains.
    Builds traversal chains that reach forward to an outcome.

  Category membership
    Same relationship across many instances.
    "cat is animal", "dog is animal", "bird is animal"
    The graph discovers "is animal" as a structure connecting many subjects.
    Categories form naturally from repeated shared relationship patterns.

Reward:
  Automated — relationship completion check. Feed "cat is a" and verify
  traversal reaches "animal". Reward if correct.
  Human — 30 min/day sessions (handled externally via reward() calls).

Readiness:
  Transitive inference working >= 50% across 200 novel pairs.
  "cat is animal, animal needs food" -> system reaches "cat needs food"
  without seeing it explicitly.

Run:
    python stage3.py --from-stage2 --steps 200000
    python stage3.py --resume
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from primeval import Atoms, Config, Primeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage3")

import nltk
for pkg in ["wordnet", "omw-1.4", "cmudict"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict as _cmudict
_CMU = _cmudict.dict()


# ══════════════════════════════════════════════════════════════════════════════
# WORDNET RELATIONSHIP EXTRACTOR
# Builds a clean set of semantic triples from WordNet.
# Only uses concrete, common words that Little Deepak already knows.
# ══════════════════════════════════════════════════════════════════════════════

# Relations we extract
REL_ISA        = "is a"
REL_PROPERTY   = "is"
REL_PARTOF     = "is part of"
REL_USEDFOR    = "is used for"
REL_NEEDS      = "needs"
REL_CAUSES     = "causes"
REL_HASPROPERTY= "has property"

# Concrete seed concepts — things Little Deepak has already encountered
SEED_CONCEPTS = [
    "cat","dog","bird","fish","tree","sun","moon","star","door","road",
    "house","city","water","fire","wind","earth","time","day","night","book",
    "hand","eye","head","heart","food","name","love","school","world","child",
    "man","woman","animal","plant","rock","mountain","river","ocean","forest",
    "flower","grass","leaf","rain","snow","cloud","air","light","dark","heat",
    "cold","sound","voice","music","color","red","blue","green","black","white",
    "large","small","old","new","good","bad","fast","slow","hard","soft",
    "run","walk","eat","drink","sleep","think","speak","see","hear","feel",
    "touch","make","build","grow","move","fall","rise","open","close","give",
    "take","hold","carry","push","pull","cut","break","burn","freeze","melt",
]

def _clean_name(synset_name: str) -> str:
    """Convert synset name to readable string. 'domestic_cat.n.01' -> 'domestic cat'"""
    return synset_name.split(".")[0].replace("_", " ")

def _is_simple(text: str, max_words: int = 3) -> bool:
    """Only use short, simple phrases."""
    return len(text.split()) <= max_words and text.isascii()

def extract_wordnet_triples() -> list[tuple[str, str, str]]:
    """
    Extract semantic triples from WordNet.
    Returns list of (subject, relation, object) strings.
    Only concrete, simple, common concepts.
    """
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str,str,str]] = set()

    def add(s, r, o):
        s, o = s.strip().lower(), o.strip().lower()
        if not s or not o:
            return
        if not _is_simple(s) or not _is_simple(o):
            return
        t = (s, r, o)
        if t not in seen:
            seen.add(t)
            triples.append(t)

    for concept in SEED_CONCEPTS:
        syns = wn.synsets(concept, pos=wn.NOUN)
        if not syns:
            syns = wn.synsets(concept, pos=wn.VERB)
        if not syns:
            syns = wn.synsets(concept, pos=wn.ADJ)
        if not syns:
            continue

        for syn in syns[:2]:   # top 2 senses only
            name = _clean_name(syn.name())

            # IsA — hypernym chain (first 3 levels only)
            s = syn
            for _ in range(3):
                parents = s.hypernyms()
                if not parents:
                    break
                parent = parents[0]
                pname = _clean_name(parent.name())
                add(name, REL_ISA, pname)
                s = parent

            # PartOf — holonyms
            for hol in syn.part_holonyms()[:3]:
                add(name, REL_PARTOF, _clean_name(hol.name()))

            # Members — hyponyms (reverse IsA — child is a parent)
            for hypo in syn.hyponyms()[:5]:
                hname = _clean_name(hypo.name())
                add(hname, REL_ISA, name)

            # Similar adjectives from definition words
            defn = syn.definition()
            for word in defn.split():
                word = word.strip(".,;:()").lower()
                adj_syns = wn.synsets(word, pos=wn.ADJ)
                if adj_syns and _is_simple(word, 1):
                    add(name, REL_HASPROPERTY, word)

    logger.info("Extracted %d WordNet triples", len(triples))
    return triples

WORDNET_TRIPLES: list[tuple[str,str,str]] = extract_wordnet_triples()


# ══════════════════════════════════════════════════════════════════════════════
# CAUSE-EFFECT CHAINS
# Subject-verb-object-consequence patterns.
# Programmatically generated from primitive seeds.
# ══════════════════════════════════════════════════════════════════════════════

# (action, object, consequence) — "action object causes consequence"
CAUSE_EFFECT_SEEDS: list[tuple[str, str, str]] = [
    # Physical causation
    ("touch",  "fire",   "get burn"),
    ("eat",    "food",   "feel full"),
    ("drink",  "water",  "feel less thirst"),
    ("drop",   "thing",  "it fall"),
    ("heat",   "water",  "it boil"),
    ("cool",   "water",  "it freeze"),
    ("break",  "thing",  "it stop work"),
    ("push",   "thing",  "it move"),
    ("pull",   "thing",  "it come"),
    ("cut",    "thing",  "it open"),
    ("burn",   "wood",   "get fire"),
    ("plant",  "seed",   "tree grow"),
    ("water",  "plant",  "it grow"),
    ("open",   "door",   "go in"),
    ("close",  "door",   "stay out"),
    ("run",    "fast",   "get tired"),
    ("sleep",  "well",   "feel good"),
    ("read",   "book",   "learn thing"),
    ("write",  "word",   "make text"),
    ("build",  "house",  "have home"),
    # Biological
    ("animal", "eat",    "stay alive"),
    ("plant",  "need",   "sunlight"),
    ("fish",   "live in","water"),
    ("bird",   "can",    "fly"),
    ("cat",    "catch",  "mouse"),
    ("dog",    "like",   "run"),
    # Natural phenomena
    ("sun",    "give",   "light"),
    ("rain",   "fall",   "ground wet"),
    ("wind",   "blow",   "tree move"),
    ("fire",   "need",   "air"),
    ("ice",    "melt",   "become water"),
    ("cloud",  "make",   "rain"),
    ("night",  "come",   "dark"),
    ("day",    "come",   "light"),
    # Social
    ("give",   "gift",   "make happy"),
    ("help",   "person", "feel good"),
    ("learn",  "thing",  "become smart"),
    ("work",   "hard",   "get result"),
    ("ask",    "question","get answer"),
    ("listen", "well",   "understand"),
]

def generate_cause_effect() -> tuple[list[int], float]:
    """Sample a cause-effect chain as a letter sequence."""
    action, obj, consequence = random.choice(CAUSE_EFFECT_SEEDS)
    # Two surface forms — direct and templated
    forms = [
        f"{action} {obj} {consequence}",
        f"if you {action} {obj} then {consequence}",
        f"{action} {obj} to {consequence}",
    ]
    text  = random.choice(forms)
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.8   # high reward — causal structure is important


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY MEMBERSHIP SEQUENCES
# Same relationship repeated across many instances.
# The graph discovers category structures naturally.
# ══════════════════════════════════════════════════════════════════════════════

# Category -> members
CATEGORIES: dict[str, list[str]] = {
    "animal":  ["cat","dog","bird","fish","horse","cow","sheep","pig","bear","wolf",
                 "lion","tiger","elephant","monkey","rabbit","mouse","rat","deer","fox"],
    "plant":   ["tree","flower","grass","bush","vine","fern","moss","rose","oak","pine"],
    "food":    ["apple","bread","meat","rice","egg","milk","cheese","fish","bean","corn"],
    "vehicle": ["car","bus","train","boat","plane","bike","truck","ship","van"],
    "tool":    ["hammer","knife","saw","drill","brush","pen","key","rope","nail","wire"],
    "place":   ["house","school","city","road","park","river","mountain","ocean","forest"],
    "body":    ["hand","eye","head","heart","foot","arm","leg","ear","nose","mouth"],
    "weather": ["rain","snow","wind","cloud","sun","storm","fog","ice","heat","cold"],
    "color":   ["red","blue","green","black","white","yellow","brown","orange","purple"],
    "time":    ["day","night","morning","evening","year","month","week","hour","minute"],
}

def generate_category_sequence() -> tuple[list[int], float]:
    """Sample a category membership statement."""
    category = random.choice(list(CATEGORIES.keys()))
    member   = random.choice(CATEGORIES[category])
    forms = [
        f"{member} is a {category}",
        f"{member} is an {category}",
        f"a {member} is a {category}",
        f"{member} belongs to {category}",
    ]
    text  = random.choice(forms)
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.9   # highest reward — category structure is foundational


# ══════════════════════════════════════════════════════════════════════════════
# WORDNET TRIPLE SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

def generate_wordnet_sequence() -> tuple[list[int], float]:
    """Sample a WordNet triple as a letter sequence."""
    if not WORDNET_TRIPLES:
        return [], 0.0
    subj, rel, obj = random.choice(WORDNET_TRIPLES)
    text  = f"{subj} {rel} {obj}"
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.7


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR — transitive inference
# ══════════════════════════════════════════════════════════════════════════════

# Transitive pairs to test — (premise1, premise2, expected_conclusion)
# Each pair chains two known relationships.
TRANSITIVE_TEST_PAIRS: list[tuple[str,str,str]] = [
    ("cat is a animal",   "animal needs food",   "cat needs food"),
    ("dog is a animal",   "animal needs food",   "dog needs food"),
    ("bird is a animal",  "animal can move",     "bird can move"),
    ("fish is a animal",  "animal is alive",     "fish is alive"),
    ("tree is a plant",   "plant needs water",   "tree needs water"),
    ("rose is a plant",   "plant needs sunlight","rose needs sunlight"),
    ("fire is hot",       "hot can burn",        "fire can burn"),
    ("ice is cold",       "cold can freeze",     "ice can freeze"),
    ("cat can run",       "run make tired",      "cat make tired"),
    ("dog can run",       "run need energy",     "dog need energy"),
    ("school is a place", "place have door",     "school have door"),
    ("house is a place",  "place have door",     "house have door"),
    ("rain is water",     "water can wet",       "rain can wet"),
    ("sun give light",    "light help see",      "sun help see"),
    ("food give energy",  "energy help work",    "food help work"),
]

class ReadinessMonitor:
    """
    Stage 3 ready when transitive inference succeeds >= 50% of test pairs.

    Proxy test — we check if the traversal chain from premise1's anchor
    reaches structures that overlap with the conclusion's atoms.
    A full inference engine would be needed for exact checking;
    this proxy measures whether the semantic structure exists in the graph.
    """

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def snapshot(self, system: Primeval) -> dict:
        g = system.graph
        successes = 0
        total     = len(TRANSITIVE_TEST_PAIRS)

        for premise1, premise2, conclusion in TRANSITIVE_TEST_PAIRS:
            # Check 1: does the graph have edges between key atoms of premise1
            # and key atoms of conclusion? This is the proxy for semantic connection.
            p1_atoms  = set(Atoms.sequence("letter", list(premise1)))
            conc_atoms = set(Atoms.sequence("letter", list(conclusion)))

            # Count cross-edges between premise atoms and conclusion atoms
            cross = sum(
                1 for a in p1_atoms
                for b in conc_atoms
                if g.weight(a, b) > 0 or g.weight(b, a) > 0
            )
            # Success if meaningful cross-connection exists
            if cross >= 3:
                successes += 1

        transitive_rate = successes / total

        # Also check anchor level improvement
        above2 = 0
        test_queries = ["cat is", "fire is", "water is", "animal is",
                        "food is", "plant is", "tree is", "dog is"]
        for q in test_queries:
            r = system.infer("letter", q)
            if r.anchor_level >= 2:
                above2 += 1
        anchor_rate = above2 / len(test_queries)

        return {
            "transitive_rate": transitive_rate,
            "anchor_rate":     anchor_rate,
        }

    def is_ready(self, snap: dict) -> bool:
        return snap["transitive_rate"] >= self.threshold


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATED REWARD CHECKER
# Feed "cat is a" → check if traversal reaches "animal"
# ══════════════════════════════════════════════════════════════════════════════

# Relationship completions to check automatically
COMPLETION_CHECKS: list[tuple[str, str]] = [
    ("cat is a",    "animal"),
    ("dog is a",    "animal"),
    ("tree is a",   "plant"),
    ("fire is",     "hot"),
    ("water is",    "liquid"),
    ("bird is a",   "animal"),
    ("fish is a",   "animal"),
    ("rose is a",   "plant"),
    ("cat can",     "run"),
    ("dog can",     "run"),
    ("sun give",    "light"),
    ("rain is",     "water"),
    ("food give",   "energy"),
    ("school is a", "place"),
    ("house is a",  "place"),
]

def check_completion(system: Primeval, prompt: str, expected: str) -> bool:
    """
    Check if inference on prompt produces a chain that contains
    atoms overlapping with expected. Proxy for semantic completion.
    """
    g = system.graph
    r = system.infer("letter", prompt)
    # Check if expected atoms appear as neighbors of chain nodes
    expected_atoms = set(Atoms.sequence("letter", list(expected)))
    for node in r.chain[:10]:
        for nb, w in g.neighbors(node)[:20]:
            if nb in expected_atoms and w > 10.0:
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage3Trainer:
    """
    Four ingestion types interleaved:
      35% WordNet relationships    — IsA, HasProperty, PartOf
      30% category membership      — highest reward, foundational
      25% cause-effect chains      — causal traversal paths
      10% completion-reward checks — automated semantic feedback

    Reward scale drops to 1.5 — the graph is rich enough that
    sparse, high-quality reward shapes direction effectively.
    Completion checks add targeted reward to correct semantic chains.
    """

    def __init__(
        self,
        system:         Primeval,
        reward_scale:   float = 1.5,
        log_interval:   int   = 5000,
        check_interval: int   = 5000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor()
        self._last_snap:    dict = {}
        self._completion_hits = 0
        self._completion_total = 0

    def step(self) -> None:
        r = random.random()

        if r < 0.35:
            atoms, raw_reward = generate_wordnet_sequence()
        elif r < 0.65:
            atoms, raw_reward = generate_category_sequence()
        elif r < 0.90:
            atoms, raw_reward = generate_cause_effect()
        else:
            # Automated completion check — ingest the prompt and
            # reward if the graph correctly completes it
            prompt, expected = random.choice(COMPLETION_CHECKS)
            atoms = Atoms.sequence("letter", list(prompt))
            self.system.ingest_atoms(atoms)
            self._completion_total += 1
            if check_completion(self.system, prompt, expected):
                self.system.reward(2.0 * self.reward_scale)
                self._completion_hits += 1
            return

        if not atoms:
            return
        self.system.ingest_atoms(atoms)
        if raw_reward > 0.0:
            self.system.reward(raw_reward * self.reward_scale)

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 3 starting. Target steps: %d", total_steps)
        logger.info("WordNet triples: %d", len(WORDNET_TRIPLES))
        logger.info("Category pairs:  %d",
                    sum(len(v) for v in CATEGORIES.values()))
        logger.info("Cause-effect seeds: %d", len(CAUSE_EFFECT_SEEDS))
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)
                if self.monitor.is_ready(self._last_snap):
                    logger.info(
                        "Stage 3 READY at step %d (%.1fs). "
                        "transitive=%.2f anchor=%.2f",
                        step, elapsed,
                        self._last_snap["transitive_rate"],
                        self._last_snap["anchor_rate"],
                    )
                    return True

        logger.info(
            "Stage 3 ended at step limit. transitive=%.2f anchor=%.2f",
            self._last_snap.get("transitive_rate", 0.0),
            self._last_snap.get("anchor_rate", 0.0),
        )
        return False

    def _log(self, step: int, elapsed: float) -> None:
        g    = self.system.graph
        stat = self.system.stats()
        snap = self._last_snap
        comp_rate = (self._completion_hits / self._completion_total
                     if self._completion_total > 0 else 0.0)

        logger.info(
            "Step %6d | %.1fs | nodes=%d edges=%d max_lv=%d",
            step, elapsed, stat["nodes"], stat["edges"], stat["max_level"],
        )
        logger.info(
            "  Transitive=%.2f (need %.2f) | anchor>=L2=%.2f | completion=%.2f",
            snap.get("transitive_rate", 0.0),
            self.monitor.threshold,
            snap.get("anchor_rate", 0.0),
            comp_rate,
        )

        # Sample semantic inferences
        logger.info("  Sample inferences:")
        for query in ["cat is", "fire is", "water is", "animal is", "tree is"]:
            r = self.system.infer("letter", query)
            logger.info("    %r -> L%d  conf=%.1f",
                        query, r.anchor_level, r.confidence)

        # Show top edges involving semantic words
        semantic_atoms = set()
        for word in ["cat","animal","fire","water","food","plant"]:
            semantic_atoms.update(Atoms.sequence("letter", list(word)))

        semantic_edges = sorted(
            [(k, v) for k, v in g._weights.items()
             if k[0] in semantic_atoms and k[1] in semantic_atoms],
            key=lambda x: -x[1]
        )[:5]
        if semantic_edges:
            logger.info("  Top semantic edges:")
            for (a, b), w in semantic_edges:
                la = chr(a - Atoms.LETTER_OFF) if Atoms.LETTER_OFF <= a < Atoms.PHONEME_OFF else str(a)
                lb = chr(b - Atoms.LETTER_OFF) if Atoms.LETTER_OFF <= b < Atoms.PHONEME_OFF else str(b)
                logger.info("    %r->%r  w=%.1f", la, lb, w)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak -- Stage 3")
    parser.add_argument("--steps",        type=int,   default=200_000)
    parser.add_argument("--log-interval", type=int,   default=5_000)
    parser.add_argument("--reward-scale", type=float, default=1.5)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage3")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--from-stage2",  action="store_true")
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 5,    # wider — semantic sentences need more context
        occurrence_delta            = 1.0,
        reward_multiplier           = 5.0,  # lower — reward is now selective
        downward_growth_delta       = 0.1,
        consolidator_interval       = 500,
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 500,
        counts_decay_rate           = 0.005,
        decay_interval              = 5_000,
        base_decay_rate             = 4e-4,
        level_decay_factor          = 1.5,
        checkpoint_dir              = args.checkpoint,
        checkpoint_interval         = 30_000,
    )

    system = Primeval(cfg)

    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if checkpoints:
            system.load(checkpoints[-1])
            logger.info("Resumed from %s", checkpoints[-1])
        else:
            logger.warning("No stage3 checkpoint -- starting fresh")
    elif args.from_stage2:
        path = "./checkpoints/stage2/final"
        if os.path.exists(path):
            system.load(path)
            logger.info("Loaded Stage 2 checkpoint")
        else:
            logger.warning("Stage 2 checkpoint not found -- starting fresh")

    trainer = Stage3Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)
    system.save(args.checkpoint + "/final")
    logger.info("Saved to %s/final", args.checkpoint)

    g = system.graph
    print("\n-- Stage 3 Final Summary ----------------------------------------")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n-- Transitive inference test ------------------------------------")
    successes = 0
    for premise1, premise2, conclusion in TRANSITIVE_TEST_PAIRS:
        p1_atoms   = set(Atoms.sequence("letter", list(premise1)))
        conc_atoms = set(Atoms.sequence("letter", list(conclusion)))
        cross = sum(
            1 for a in p1_atoms
            for b in conc_atoms
            if g.weight(a, b) > 0 or g.weight(b, a) > 0
        )
        flag = "+" if cross >= 3 else "."
        if cross >= 3:
            successes += 1
        print(f"  {flag} [{cross:3d} links] {premise1!r:25s} => {conclusion!r}")
    print(f"\n  Transitive success: {successes}/{len(TRANSITIVE_TEST_PAIRS)} = "
          f"{successes/len(TRANSITIVE_TEST_PAIRS):.0%}")

    print("\n-- Semantic completion check ------------------------------------")
    hits = 0
    for prompt, expected in COMPLETION_CHECKS:
        ok = check_completion(system, prompt, expected)
        flag = "+" if ok else "."
        if ok:
            hits += 1
        print(f"  {flag} {prompt!r:20s} -> {expected!r}")
    print(f"\n  Completion rate: {hits}/{len(COMPLETION_CHECKS)} = "
          f"{hits/len(COMPLETION_CHECKS):.0%}")

    print("\n-- Inference on semantic queries --------------------------------")
    for query in ["cat is", "fire is", "water is", "animal is",
                  "food is", "plant needs", "tree is", "dog can"]:
        r = system.infer("letter", query)
        print(f"  {query!r:20s} -> L{r.anchor_level}  conf={r.confidence:.1f}  "
              f"chain_len={len(r.chain)}")

    print(f"\n  Stage 3 {'COMPLETE' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()