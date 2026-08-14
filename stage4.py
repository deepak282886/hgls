"""
stage4.py — Little Deepak: Stage 4 Curriculum
==============================================

Language structure. Grammar discovered as pattern, not taught as rules.

Three data sources:

  Brown + Gutenberg corpus sentences
    ~18,000 clean English sentences, 4-15 words, alphabetic only.
    Fiction, romance, humor, adventure categories — concrete and narrative.
    Rewarded for grammatical surface coherence (article-noun, subject-verb).
    This is the main driver — real English sentences in volume.

  Grammatical templates
    10 core English sentence patterns generated with word-class slots.
    "The [noun] [verb].", "A [adj] [noun] [verb] the [noun]."
    Each template instantiated with 50+ word combinations.
    Rewards the structural pattern itself, not just the specific words.

  Morphological families
    Word form relationships — plural, tense, derivation.
    "cat/cats", "run/runs/running/ran", "happy/happily/unhappy"
    Fed as paired sequences in the same ingestion window.
    Builds the first productive structures — templates that apply
    to any word of the right type.

Reward:
  Automated grammatical surface check — noun after article, verb after
  subject, sentence ends with punctuation. Reward correct patterns.
  Completion reward carried forward from Stage 3.
  Human still 30 min/day (handled externally via reward() calls).

Readiness:
  Grammatical template structures at level 3+.
  Inference on articles ("the", "a") anchors at level 3 or above.
  60% of sentence fragment completions grammatically correct.

Run:
    python stage4.py --from-stage3 --steps 300000
    python stage4.py --resume
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import re
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
logger = logging.getLogger("stage4")

import nltk
for pkg in ["brown", "gutenberg", "punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "cmudict"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import brown, gutenberg
from nltk.corpus import cmudict as _cmudict
_CMU = _cmudict.dict()


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS BUILDER
# Clean sentences from Brown + Gutenberg corpora.
# ══════════════════════════════════════════════════════════════════════════════

def _is_clean_word(w: str) -> bool:
    return (w.replace("'","").isalpha() or w in [".",",","!","?",";",":"])

def _build_corpus() -> list[str]:
    sents: list[str] = []

    # Brown corpus — fiction, romance, humor, adventure, lore, hobbies
    for cat in ["fiction","romance","humor","adventure","lore","hobbies"]:
        for sent in brown.sents(categories=[cat]):
            if 4 <= len(sent) <= 15 and all(_is_clean_word(w) for w in sent):
                text = " ".join(sent).lower()
                sents.append(text)

    # Gutenberg — classic novels, clean prose
    for fileid in gutenberg.fileids():
        for sent in gutenberg.sents(fileid):
            if 4 <= len(sent) <= 12 and all(_is_clean_word(w) for w in sent):
                text = " ".join(sent).lower()
                sents.append(text)

    # Deduplicate
    sents = list(dict.fromkeys(sents))
    random.shuffle(sents)
    logger.info("Corpus built: %d sentences", len(sents))
    return sents

CORPUS: list[str] = _build_corpus()


# ══════════════════════════════════════════════════════════════════════════════
# GRAMMATICAL TEMPLATES
# Core English sentence patterns with word-class slots.
# Each template instantiated across many word combinations.
# ══════════════════════════════════════════════════════════════════════════════

NOUNS = [
    "cat","dog","bird","fish","tree","sun","water","fire","wind","man",
    "woman","child","house","school","road","river","mountain","flower",
    "book","door","hand","eye","heart","food","rain","cloud","star","moon",
    "animal","plant","rock","voice","light","dark","sound","time","day",
]
VERBS = [
    "ran","walked","sat","fell","rose","moved","stopped","started","turned",
    "looked","said","asked","heard","felt","knew","thought","saw","came",
    "went","found","left","opened","closed","gave","took","made","kept",
]
VERBS_INTRANS = [
    "ran","walked","sat","fell","rose","moved","stopped","started","turned",
    "looked","slept","waited","arrived","left","returned","stayed","lived",
]
ADJS = [
    "big","small","old","new","good","bad","hot","cold","fast","slow",
    "long","short","high","low","dark","light","hard","soft","clear","deep",
    "bright","quiet","warm","cool","heavy","thin","wide","narrow","rough","smooth",
]
ADVS = [
    "quickly","slowly","quietly","loudly","carefully","gently","suddenly",
    "always","never","often","soon","still","already","again","away",
]

TEMPLATES = [
    "the {noun} {verb} .",
    "a {adj} {noun} {verb} .",
    "the {noun} {verb} the {noun} .",
    "a {noun} {verb} .",
    "the {adj} {noun} {verb} {adv} .",
    "the {noun} and the {noun} {verb} .",
    "a {noun} is {adj} .",
    "the {noun} is a {noun} .",
    "the {noun} {verb} and {verb} .",
    "a {adj} {noun} is {adj} .",
    "the {noun} {verb} the {adj} {noun} .",
    "the {noun} of the {noun} is {adj} .",
]

def generate_template_sentence() -> tuple[list[int], float]:
    """Instantiate a grammatical template with random words."""
    template = random.choice(TEMPLATES)
    text = template.format(
        noun  = random.choice(NOUNS),
        verb  = random.choice(VERBS_INTRANS if "{noun} ." in template else VERBS),
        adj   = random.choice(ADJS),
        adv   = random.choice(ADVS),
    )
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.9   # high reward — grammatical templates are foundational


# ══════════════════════════════════════════════════════════════════════════════
# MORPHOLOGICAL FAMILIES
# Word form relationships fed as paired sequences.
# ══════════════════════════════════════════════════════════════════════════════

MORPH_FAMILIES: list[list[str]] = [
    # Plurals
    ["cat","cats"],["dog","dogs"],["bird","birds"],["tree","trees"],
    ["house","houses"],["book","books"],["hand","hands"],["eye","eyes"],
    ["river","rivers"],["mountain","mountains"],["flower","flowers"],
    # Verb tenses
    ["run","runs","running","ran"],["walk","walks","walking","walked"],
    ["eat","eats","eating","ate"],["see","sees","seeing","saw"],
    ["go","goes","going","went"],["come","comes","coming","came"],
    ["make","makes","making","made"],["take","takes","taking","took"],
    ["give","gives","giving","gave"],["think","thinks","thinking","thought"],
    ["know","knows","knowing","knew"],["feel","feels","feeling","felt"],
    # Derivations
    ["happy","happily","unhappy","happiness"],
    ["quick","quickly","quicker","quickest"],
    ["dark","darkness","darker","darkest"],
    ["light","lighter","lightest","lighten"],
    ["cold","colder","coldest","coldness"],
    ["warm","warmer","warmest","warmth"],
    ["clear","clearly","clearer","clearest"],
    ["slow","slowly","slower","slowest"],
]

def generate_morph_sequence() -> tuple[list[int], float]:
    """
    Sample a morphological family. Feed two related forms together
    in the same ingestion window — the graph discovers their relationship
    through co-occurrence of their shared atomic substrate.
    """
    family = random.choice(MORPH_FAMILIES)
    # Pick 2 related forms
    if len(family) >= 2:
        pair = random.sample(family, 2)
    else:
        pair = family * 2
    # Feed as: "form1 form2" — space-separated, single ingestion
    text  = " ".join(pair)
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.6


# ══════════════════════════════════════════════════════════════════════════════
# GRAMMATICAL COHERENCE CHECKER
# Surface-level automated reward signal.
# ══════════════════════════════════════════════════════════════════════════════

ARTICLES   = {"the","a","an"}
PUNCT      = {".","!","?"}
AUX_VERBS  = {"is","are","was","were","will","would","can","could","should","did","do","does"}

def grammatical_reward(sentence: str) -> float:
    """
    Surface grammatical coherence score (0.0 to 1.0).
    Checks simple structural heuristics.
    """
    words = sentence.lower().split()
    if len(words) < 3:
        return 0.0

    score  = 0.0
    checks = 0

    # Article followed by noun or adjective
    for i, w in enumerate(words[:-1]):
        if w in ARTICLES:
            next_w = words[i+1]
            if next_w.isalpha() and next_w not in ARTICLES:
                score += 1.0
            checks += 1

    # Ends with punctuation
    if words[-1] in PUNCT:
        score  += 1.0
        checks += 1

    # Has at least one known verb form
    verb_found = any(
        w in AUX_VERBS or w.endswith("ed") or w.endswith("ing")
        for w in words
    )
    if verb_found:
        score  += 1.0
        checks += 1

    # Sentence starts with capital-ish word (article or noun)
    if words[0] in ARTICLES or (words[0].isalpha() and len(words[0]) > 2):
        score  += 0.5
        checks += 1

    return score / checks if checks > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETION CHECKS — carried forward from Stage 3 + new grammatical ones
# ══════════════════════════════════════════════════════════════════════════════

COMPLETION_CHECKS: list[tuple[str, str]] = [
    # Semantic (Stage 3)
    ("cat is a",     "animal"),
    ("fire is",      "hot"),
    ("water is",     "liquid"),
    ("tree is a",    "plant"),
    ("sun give",     "light"),
    # Grammatical (Stage 4)
    ("the cat",      "sat"),
    ("a dog",        "ran"),
    ("the bird",     "flew"),
    ("the man",      "said"),
    ("the woman",    "looked"),
    ("the child",    "ran"),
    ("the water",    "is"),
    ("a bird",       "flew"),
]

def check_completion(system: Primeval, prompt: str, expected: str) -> bool:
    """Check if inference chain neighbours contain expected word atoms."""
    g = system.graph
    r = system.infer("letter", prompt)
    expected_atoms = set(Atoms.sequence("letter", list(expected)))
    for node in r.chain[:10]:
        for nb, w in g.neighbors(node)[:20]:
            if nb in expected_atoms and w > 10.0:
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ReadinessMonitor:
    """
    Stage 4 ready when:
    1. Articles ("the", "a") anchor at level >= 3
    2. 60% of fragment completions grammatically appropriate
    3. Level-3+ structures exist with meaningful stability
    """

    def __init__(self, threshold_anchor: int = 3, threshold_completion: float = 0.60):
        self.threshold_anchor     = threshold_anchor
        self.threshold_completion = threshold_completion

    def snapshot(self, system: Primeval) -> dict:
        g = system.graph

        # Article anchor levels
        article_levels: list[int] = []
        for article in ["the", "a", "an"]:
            r = system.infer("letter", article)
            article_levels.append(r.anchor_level)
        mean_article_level = np.mean(article_levels) if article_levels else 0.0

        # Completion rate
        hits  = sum(1 for p, e in COMPLETION_CHECKS
                    if check_completion(system, p, e))
        completion_rate = hits / len(COMPLETION_CHECKS)

        # Grammatical template inference
        template_above3 = 0
        template_tests  = [
            "the cat", "a dog", "the bird", "the man", "a woman",
            "the child", "a tree", "the water",
        ]
        for query in template_tests:
            r = system.infer("letter", query)
            if r.anchor_level >= 3:
                template_above3 += 1
        template_rate = template_above3 / len(template_tests)

        # Level-3+ structure count and mean stability
        l3_structs = [
            nid for level in range(3, g.max_level() + 1)
            for nid in g.nodes_at_level(level)
        ]
        mean_l3_stab = (
            float(np.mean([g.stability(n) for n in l3_structs]))
            if l3_structs else 0.0
        )

        return {
            "mean_article_level": mean_article_level,
            "completion_rate":    completion_rate,
            "template_rate":      template_rate,
            "l3_struct_count":    len(l3_structs),
            "mean_l3_stab":       mean_l3_stab,
        }

    def is_ready(self, snap: dict) -> bool:
        return (
            snap["mean_article_level"] >= self.threshold_anchor and
            snap["completion_rate"]    >= self.threshold_completion
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage4Trainer:
    """
    Four ingestion types interleaved:
      45% corpus sentences         — real English, grammatical coherence reward
      30% grammatical templates    — pure structural patterns
      15% morphological families   — word form relationships
      10% completion checks        — automated semantic + grammatical feedback

    Reward scale drops to 1.0 — graph is mature.
    Corpus sentences get grammatical coherence reward (0.0-1.0 × scale).
    Templates get fixed high reward — structural patterns are valuable.
    """

    def __init__(
        self,
        system:         Primeval,
        reward_scale:   float = 1.0,
        log_interval:   int   = 5000,
        check_interval: int   = 5000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor()
        self._last_snap:    dict = {}
        self._corpus_idx    = 0
        self._completion_hits  = 0
        self._completion_total = 0

    def _next_corpus_sentence(self) -> str:
        if not CORPUS:
            return ""
        sent = CORPUS[self._corpus_idx % len(CORPUS)]
        self._corpus_idx += 1
        return sent

    def step(self) -> None:
        r = random.random()

        if r < 0.45:
            # Real corpus sentence with grammatical coherence reward
            sent = self._next_corpus_sentence()
            if not sent:
                return
            atoms = Atoms.sequence("letter", list(sent))
            self.system.ingest_atoms(atoms)
            gram_score = grammatical_reward(sent)
            if gram_score > 0.3:
                self.system.reward(gram_score * self.reward_scale)

        elif r < 0.75:
            # Grammatical template
            atoms, raw_reward = generate_template_sentence()
            if not atoms:
                return
            self.system.ingest_atoms(atoms)
            self.system.reward(raw_reward * self.reward_scale)

        elif r < 0.90:
            # Morphological family
            atoms, raw_reward = generate_morph_sequence()
            if not atoms:
                return
            self.system.ingest_atoms(atoms)
            if raw_reward > 0:
                self.system.reward(raw_reward * self.reward_scale)

        else:
            # Automated completion check
            prompt, expected = random.choice(COMPLETION_CHECKS)
            atoms = Atoms.sequence("letter", list(prompt))
            self.system.ingest_atoms(atoms)
            self._completion_total += 1
            if check_completion(self.system, prompt, expected):
                self.system.reward(1.5 * self.reward_scale)
                self._completion_hits += 1

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 4 starting. Target steps: %d", total_steps)
        logger.info("Corpus sentences: %d", len(CORPUS))
        logger.info("Templates: %d", len(TEMPLATES))
        logger.info("Morph families: %d", len(MORPH_FAMILIES))
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)
                if self.monitor.is_ready(self._last_snap):
                    logger.info(
                        "Stage 4 READY at step %d (%.1fs). "
                        "article_level=%.1f completion=%.2f",
                        step, elapsed,
                        self._last_snap["mean_article_level"],
                        self._last_snap["completion_rate"],
                    )
                    return True

        logger.info(
            "Stage 4 ended at step limit. article_level=%.1f completion=%.2f",
            self._last_snap.get("mean_article_level", 0.0),
            self._last_snap.get("completion_rate", 0.0),
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
            "  article_level=%.1f (need %d) | completion=%.2f (need %.2f) | "
            "template>=L3=%.2f | L3+structs=%d",
            snap.get("mean_article_level", 0.0),
            self.monitor.threshold_anchor,
            snap.get("completion_rate", 0.0),
            self.monitor.threshold_completion,
            snap.get("template_rate", 0.0),
            snap.get("l3_struct_count", 0),
        )
        logger.info(
            "  auto_completion=%.2f | mean_L3_stab=%.1f",
            comp_rate,
            snap.get("mean_l3_stab", 0.0),
        )

        # Sample inferences
        logger.info("  Sample inferences:")
        queries = ["the", "a", "the cat", "a dog", "cat is", "water is", "the cat sat"]
        for q in queries:
            r = self.system.infer("letter", q)
            logger.info("    %r -> L%d  conf=%.1f  chain=%d",
                        q, r.anchor_level, r.confidence, len(r.chain))

        # Top L2+ letter structures
        l2_structs = sorted(
            [nid for nid in g.nodes_at_level(2)
             if g._nodes.get(nid) and all(
                 Atoms.LETTER_OFF <= c < Atoms.PHONEME_OFF
                 for c in g._nodes[nid].constituents)],
            key=lambda n: -g.stability(n)
        )[:3]
        if l2_structs:
            logger.info("  Top L2 letter structures:")
            for nid in l2_structs:
                d = self.system.describe_node(nid)
                logger.info("    struct(%d) stab=%.1f trav=%d",
                            nid, d["stability"], d["traversals"])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak -- Stage 4")
    parser.add_argument("--steps",        type=int,   default=300_000)
    parser.add_argument("--log-interval", type=int,   default=5_000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage4")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--from-stage3",  action="store_true")
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 5,
        occurrence_delta            = 1.0,
        reward_multiplier           = 4.0,   # lower — reward is selective now
        downward_growth_delta       = 0.05,  # gentler downward growth
        consolidator_interval       = 500,
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 500,
        counts_decay_rate           = 0.005,
        decay_interval              = 5_000,
        base_decay_rate             = 5e-4,
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
            logger.warning("No stage4 checkpoint -- starting fresh")
    elif args.from_stage3:
        path = "./checkpoints/stage3/final"
        if os.path.exists(path):
            system.load(path)
            logger.info("Loaded Stage 3 checkpoint")
        else:
            logger.warning("Stage 3 checkpoint not found -- starting fresh")

    trainer = Stage4Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)
    system.save(args.checkpoint + "/final")
    logger.info("Saved to %s/final", args.checkpoint)

    g = system.graph
    print("\n-- Stage 4 Final Summary ----------------------------------------")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n-- Inference on grammatical fragments ---------------------------")
    test_queries = [
        "the", "a", "the cat", "a dog", "the bird flew",
        "a man", "the woman said", "water is", "fire is hot",
        "the cat sat on", "a small bird",
    ]
    for q in test_queries:
        r = system.infer("letter", q)
        print(f"  {q!r:25s} -> L{r.anchor_level}  conf={r.confidence:.1f}  "
              f"chain={len(r.chain)}")

    print("\n-- Completion check ---------------------------------------------")
    hits = 0
    for prompt, expected in COMPLETION_CHECKS:
        ok = check_completion(system, prompt, expected)
        flag = "+" if ok else "."
        if ok: hits += 1
        print(f"  {flag} {prompt!r:20s} -> {expected!r}")
    print(f"\n  Completion rate: {hits}/{len(COMPLETION_CHECKS)} = "
          f"{hits/len(COMPLETION_CHECKS):.0%}")

    print("\n-- Morphological family test ------------------------------------")
    for family in MORPH_FAMILIES[:8]:
        # Feed first form, check if related forms' atoms are reachable
        r = system.infer("letter", family[0])
        chain_atoms = set()
        for node in r.chain:
            for nb, w in g.neighbors(node)[:10]:
                chain_atoms.add(nb)
        related = [
            f for f in family[1:]
            if any(a in chain_atoms
                   for a in Atoms.sequence("letter", list(f)))
        ]
        flag = "+" if related else "."
        print(f"  {flag} {family[0]:12s} -> related: {related or 'none found'}")

    snap = trainer._last_snap
    print(f"\n  Article anchor level: {snap.get('mean_article_level',0):.1f} "
          f"(need {trainer.monitor.threshold_anchor})")
    print(f"  Completion rate:      {snap.get('completion_rate',0):.2f} "
          f"(need {trainer.monitor.threshold_completion:.2f})")
    print(f"\n  Stage 4 {'COMPLETE' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()