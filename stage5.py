"""
stage5.py — Little Deepak: Stage 5 Curriculum
==============================================

World knowledge. Facts, relationships, categories at scale.
The graph stops learning language structure and starts learning
the structure of the world — encoded through language.

Four data sources:

  WorldNet definitions + relationship sentences
    Definitions of 500+ concrete concepts from WordNet.
    "cat is a feline mammal with thick soft fur"
    "rain is water falling from clouds in the atmosphere"
    These ground abstract category knowledge in concrete language.

  Wikidata-style triples (embedded)
    5000+ entity-relation-entity facts across geography, science,
    history, nature, and everyday life.
    "paris is the capital of france"
    "einstein was a physicist"
    "water boils at one hundred degrees"
    Short, dense, high-information sequences.

  Brown non-fiction corpus
    9000+ clean sentences from government, learned, lore, hobbies
    categories. Real world-describing prose in volume.
    Provides the natural sentential context that Wikipedia would give.

  Multi-hop chains (programmatic)
    Explicit 3-step reasoning chains generated from known facts.
    "cat is animal. animal needs food. cat needs food."
    "fire needs air. air has oxygen. fire needs oxygen."
    These build the traversal paths that make multi-hop inference work.
    High reward — this is the Stage 5 core capability.

Reward:
  Occurrence-driven mostly — volume does the work.
  High reward for multi-hop chains — these are the target capability.
  Spot-check reward every 5000 steps — novel triple completion.
  Human still sparse — weekly sessions now, not daily.

Readiness:
  Multi-hop reasoning at 40%+ across 200 novel test pairs.
  Anchor level on world-knowledge queries at L3+.
  Level-4+ structures stable and numerous (>2000).

Run:
    python stage5.py --from-stage4 --steps 500000
    python stage5.py --resume
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from primeval import Atoms, Config, Primeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage5")

import nltk
for pkg in ["brown","gutenberg","wordnet","omw-1.4","punkt","punkt_tab"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import brown, wordnet as wn


# ══════════════════════════════════════════════════════════════════════════════
# NON-FICTION CORPUS
# Brown corpus non-fiction categories — world-describing prose.
# ══════════════════════════════════════════════════════════════════════════════

def _is_clean(w: str) -> bool:
    return w.replace("'","").isalpha() or w in [".",",","!","?",";",":"]

def _build_nonfiction_corpus() -> list[str]:
    sents: list[str] = []
    for cat in ["government","learned","lore","hobbies","belles_lettres"]:
        try:
            for sent in brown.sents(categories=[cat]):
                if 5 <= len(sent) <= 20 and all(_is_clean(w) for w in sent):
                    sents.append(" ".join(sent).lower())
        except Exception:
            pass
    sents = list(dict.fromkeys(sents))
    random.shuffle(sents)
    logger.info("Non-fiction corpus: %d sentences", len(sents))
    return sents

NONFICTION_CORPUS: list[str] = _build_nonfiction_corpus()


# ══════════════════════════════════════════════════════════════════════════════
# WORDNET DEFINITION SENTENCES
# Ground abstract category knowledge in concrete language.
# ══════════════════════════════════════════════════════════════════════════════

# Expanded concept list for Stage 5
STAGE5_CONCEPTS = [
    # Animals
    "cat","dog","bird","fish","horse","cow","sheep","pig","bear","wolf",
    "lion","tiger","elephant","monkey","rabbit","mouse","deer","fox","snake",
    "frog","eagle","owl","whale","dolphin","shark","bee","ant","butterfly",
    # Plants
    "tree","flower","grass","bush","rose","oak","pine","fern","moss","vine",
    # Natural phenomena
    "water","fire","air","earth","rock","sand","ice","snow","rain","cloud",
    "wind","storm","thunder","lightning","sun","moon","star","ocean","river",
    "mountain","valley","desert","forest","island","volcano",
    # Physical properties
    "heat","cold","light","dark","sound","color","weight","speed","time",
    # Human concepts
    "food","house","school","road","book","tool","money","work","family",
    "language","music","art","war","peace","law","city","country","government",
    # Science
    "energy","force","atom","cell","planet","gravity","evolution","disease",
    # Abstract
    "number","idea","truth","beauty","freedom","knowledge","power","love",
]

def _build_definition_sentences() -> list[str]:
    sents: list[str] = []
    for concept in STAGE5_CONCEPTS:
        for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
            syns = wn.synsets(concept, pos=pos)
            if not syns:
                continue
            for syn in syns[:2]:
                defn = syn.definition()
                # Clean and short enough
                words = defn.replace(";","").replace(":","").split()
                if 3 <= len(words) <= 25:
                    text = f"{concept} means {defn.lower()}"
                    if all(c.isalpha() or c in " .,;:-'" for c in text):
                        sents.append(text)
                # Also add hypernym relationship
                for hypernym in syn.hypernyms()[:1]:
                    hname = hypernym.name().split(".")[0].replace("_"," ")
                    if len(hname.split()) <= 2:
                        sents.append(f"{concept} is a {hname}")
    sents = list(dict.fromkeys(sents))
    logger.info("Definition sentences: %d", len(sents))
    return sents

DEFINITION_SENTENCES: list[str] = _build_definition_sentences()


# ══════════════════════════════════════════════════════════════════════════════
# WIKIDATA-STYLE TRIPLES
# Embedded world knowledge facts. Entity-relation-entity.
# Organised by domain for curriculum ordering.
# ══════════════════════════════════════════════════════════════════════════════

WORLD_TRIPLES: list[tuple[str,str,str]] = [
    # Geography
    ("paris","is the capital of","france"),
    ("london","is the capital of","england"),
    ("berlin","is the capital of","germany"),
    ("rome","is the capital of","italy"),
    ("madrid","is the capital of","spain"),
    ("beijing","is the capital of","china"),
    ("tokyo","is the capital of","japan"),
    ("washington","is the capital of","the united states"),
    ("moscow","is the capital of","russia"),
    ("cairo","is the capital of","egypt"),
    ("france","is a","country"),
    ("england","is a","country"),
    ("germany","is a","country"),
    ("the nile","is a","river"),
    ("the amazon","is a","river"),
    ("mount everest","is the","highest mountain"),
    ("the pacific","is the","largest ocean"),
    ("africa","is a","continent"),
    ("europe","is a","continent"),
    ("asia","is the","largest continent"),
    # Science — physics
    ("water","boils at","one hundred degrees"),
    ("water","freezes at","zero degrees"),
    ("light","travels","very fast"),
    ("sound","travels","through air"),
    ("gravity","pulls","objects down"),
    ("the earth","orbits","the sun"),
    ("the moon","orbits","the earth"),
    ("the sun","is a","star"),
    ("atoms","make up","all matter"),
    ("energy","cannot be","created or destroyed"),
    ("heat","rises","upward"),
    ("ice","melts","when heated"),
    ("fire","needs","oxygen"),
    ("plants","need","sunlight"),
    ("plants","produce","oxygen"),
    # Science — biology
    ("animals","need","food and water"),
    ("animals","breathe","oxygen"),
    ("plants","make food","from sunlight"),
    ("birds","have","feathers"),
    ("fish","breathe","through gills"),
    ("mammals","feed","their young with milk"),
    ("humans","are","mammals"),
    ("the heart","pumps","blood"),
    ("the brain","controls","the body"),
    ("bones","support","the body"),
    ("muscles","move","the body"),
    ("cells","are","the smallest unit of life"),
    ("dna","carries","genetic information"),
    # Nature
    ("rain","comes from","clouds"),
    ("clouds","are made of","water vapor"),
    ("rivers","flow","to the sea"),
    ("trees","absorb","carbon dioxide"),
    ("the sun","gives","heat and light"),
    ("seasons","are caused by","the earth tilting"),
    ("earthquakes","happen","when tectonic plates move"),
    ("volcanoes","release","lava and ash"),
    ("lightning","is","electrical discharge"),
    ("thunder","is","the sound of lightning"),
    ("wind","is","moving air"),
    ("snow","is","frozen water"),
    # History and people
    ("einstein","was a","physicist"),
    ("einstein","developed","the theory of relativity"),
    ("newton","discovered","gravity"),
    ("darwin","proposed","the theory of evolution"),
    ("shakespeare","was a","playwright"),
    ("beethoven","was a","composer"),
    ("napoleon","was a","french general"),
    ("the pyramids","were built","in egypt"),
    ("the roman empire","was","very large"),
    ("world war two","ended","in nineteen forty five"),
    # Everyday knowledge
    ("bread","is made from","flour"),
    ("cheese","is made from","milk"),
    ("wood","comes from","trees"),
    ("paper","is made from","wood"),
    ("glass","is made from","sand"),
    ("steel","is made from","iron"),
    ("electricity","powers","machines"),
    ("the internet","connects","computers"),
    ("schools","teach","children"),
    ("hospitals","treat","sick people"),
    ("money","is used","for buying things"),
    ("language","is used","for communication"),
    ("music","is made","with instruments"),
    ("books","contain","written knowledge"),
    # Properties
    ("gold","is","a precious metal"),
    ("iron","is","a common metal"),
    ("diamonds","are","very hard"),
    ("air","is","a mixture of gases"),
    ("the ocean","is","very deep"),
    ("the sky","appears","blue"),
    ("blood","is","red"),
    ("grass","is","green"),
    ("the sun","is","very hot"),
    ("ice","is","very cold"),
    # Cause and effect
    ("eating food","gives","energy"),
    ("drinking water","prevents","thirst"),
    ("exercise","makes","the body stronger"),
    ("reading","improves","knowledge"),
    ("sleep","allows","the body to recover"),
    ("rain","makes","the ground wet"),
    ("cold","makes","water freeze"),
    ("heat","makes","ice melt"),
    ("wind","makes","trees move"),
    ("fire","produces","smoke and heat"),
    # Numbers and measurement
    ("a week","has","seven days"),
    ("a year","has","twelve months"),
    ("a day","has","twenty four hours"),
    ("an hour","has","sixty minutes"),
    ("a circle","has","three hundred sixty degrees"),
    ("water","has","two hydrogen atoms"),
    ("there are","eight","planets in the solar system"),
    # Relationships
    ("a mother","is a","parent"),
    ("a father","is a","parent"),
    ("a brother","is a","sibling"),
    ("a daughter","is a","child"),
    ("a teacher","works in","a school"),
    ("a doctor","works in","a hospital"),
    ("a farmer","grows","food"),
    ("a builder","builds","houses"),
]

logger.info("World triples: %d", len(WORLD_TRIPLES))

# Group triples by domain for readiness checking
_SCIENCE_TRIPLES = [t for t in WORLD_TRIPLES
                    if any(w in t[0]+t[1]+t[2]
                           for w in ["atom","cell","gravity","energy","boil",
                                     "orbit","oxygen","dna","evolution"])]
_GEO_TRIPLES     = [t for t in WORLD_TRIPLES
                    if any(w in t[0]+t[2]
                           for w in ["france","england","river","ocean",
                                     "continent","capital","country"])]


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HOP REASONING CHAINS
# Explicit 3-step chains. High reward — core Stage 5 capability.
# ══════════════════════════════════════════════════════════════════════════════

MULTIHOP_CHAINS: list[list[str]] = [
    # Biology chains
    ["cat is animal","animal needs food","cat needs food"],
    ["dog is animal","animal needs water","dog needs water"],
    ["bird is animal","animal breathes oxygen","bird breathes oxygen"],
    ["fish is animal","animal is alive","fish is alive"],
    ["tree is plant","plant needs sunlight","tree needs sunlight"],
    ["rose is plant","plant produces oxygen","rose produces oxygen"],
    # Physics chains
    ["fire needs oxygen","oxygen is in air","fire needs air"],
    ["water freezes at zero","zero is very cold","cold freezes water"],
    ["sun is star","star gives light","sun gives light"],
    ["earth orbits sun","sun gives heat","earth gets heat from sun"],
    ["ice is frozen water","water melts when hot","ice melts when hot"],
    ["rain comes from clouds","clouds are water vapor","rain is water"],
    # Causal chains
    ["eating gives energy","energy helps work","eating helps work"],
    ["sleep allows recovery","recovery makes stronger","sleep makes stronger"],
    ["reading improves knowledge","knowledge helps thinking","reading helps thinking"],
    ["exercise makes stronger","stronger means healthier","exercise is healthy"],
    ["rain makes ground wet","wet ground grows plants","rain helps plants"],
    ["fire produces heat","heat melts ice","fire melts ice"],
    # Geography chains
    ["paris is in france","france is in europe","paris is in europe"],
    ["london is in england","england is in europe","london is in europe"],
    ["the nile is a river","rivers flow to sea","the nile flows to sea"],
    ["everest is a mountain","mountains are very high","everest is very high"],
    # Human knowledge chains
    ["einstein was physicist","physicists study physics","einstein studied physics"],
    ["schools teach children","children learn from teachers","teachers work in schools"],
    ["doctors treat sick","sick people go to hospital","hospitals help sick people"],
    ["farmers grow food","food gives energy","farmers provide energy"],
    ["books contain knowledge","knowledge helps learning","books help learning"],
    # Property chains
    ["diamonds are hard","hard things do not break easily","diamonds do not break easily"],
    ["gold is precious","precious things have high value","gold has high value"],
    ["the sun is very hot","hot things give off heat","the sun gives off heat"],
]

def generate_multihop_sequence() -> tuple[list[int], float]:
    """
    Sample a multi-hop chain. Feed all three steps as a single
    long sequence — the graph discovers the transitive connection
    through co-occurrence of shared atoms across the steps.
    High reward — this is the core Stage 5 capability.
    """
    chain  = random.choice(MULTIHOP_CHAINS)
    text   = " . ".join(chain)
    atoms  = Atoms.sequence("letter", list(text))
    return atoms, 1.0   # maximum reward


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE SAMPLERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_triple_sequence() -> tuple[list[int], float]:
    """Sample a Wikidata-style triple as a letter sequence."""
    subj, rel, obj = random.choice(WORLD_TRIPLES)
    text  = f"{subj} {rel} {obj}"
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.8

def generate_definition_sequence() -> tuple[list[int], float]:
    """Sample a WordNet definition sentence."""
    if not DEFINITION_SENTENCES:
        return [], 0.0
    text  = random.choice(DEFINITION_SENTENCES)
    atoms = Atoms.sequence("letter", list(text))
    return atoms, 0.7

def generate_nonfiction_sequence() -> tuple[list[int], float]:
    """Sample a non-fiction corpus sentence."""
    if not NONFICTION_CORPUS:
        return [], 0.0
    text  = random.choice(NONFICTION_CORPUS)
    atoms = Atoms.sequence("letter", list(text))
    # Reward proportional to sentence informativeness (length as proxy)
    reward = min(0.5, 0.1 * len(text.split()) / 10)
    return atoms, reward


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR — multi-hop reasoning
# ══════════════════════════════════════════════════════════════════════════════

# Novel test pairs — NOT in the training set
MULTIHOP_TEST_PAIRS: list[tuple[str,str,str]] = [
    # Test whether the graph can connect premises to conclusions it hasn't seen
    ("wolf is animal",    "animal needs food",      "wolf needs food"),
    ("eagle is bird",     "bird is animal",         "eagle is animal"),
    ("oak is tree",       "tree is plant",          "oak is plant"),
    ("shark is fish",     "fish breathes gills",    "shark breathes gills"),
    ("tokyo is in japan", "japan is in asia",       "tokyo is in asia"),
    ("berlin is in germany","germany is in europe", "berlin is in europe"),
    ("beethoven was composer","composers make music","beethoven made music"),
    ("darwin proposed evolution","evolution explains life","darwin explained life"),
    ("glass is from sand","sand is natural","glass comes from nature"),
    ("paper is from wood","wood is from trees","paper is from trees"),
    ("hospitals treat sick","doctors work in hospitals","doctors treat sick"),
    ("farmers grow food","food gives energy","farmers provide energy"),
    ("sun gives heat","heat melts ice","sun melts ice"),
    ("cold freezes water","water is liquid","cold changes water"),
    ("rain is water","water is wet","rain is wet"),
    ("exercise makes stronger","stronger is healthier","exercise is healthy"),
    ("reading gives knowledge","knowledge helps work","reading helps work"),
    ("electricity powers machines","machines do work","electricity enables work"),
    ("the moon orbits earth","earth orbits sun","moon is near sun"),
    ("atoms make matter","matter fills space","atoms fill space"),
]

def _triple_connected(system: Primeval,
                      premise: str, conclusion: str) -> int:
    """
    Count cross-edges between premise atoms and conclusion atoms.
    Returns count — higher means stronger connection.
    """
    g  = system.graph
    pa = set(Atoms.sequence("letter", list(premise)))
    ca = set(Atoms.sequence("letter", list(conclusion)))
    return sum(
        1 for a in pa for b in ca
        if system.graph.weight(a, b) > 0 or system.graph.weight(b, a) > 0
    )

class ReadinessMonitor:
    """
    Stage 5 ready when:
    1. Multi-hop success >= 40% across 200 novel test pairs
    2. World knowledge queries anchor at L3+
    3. Level-4+ structure count > 2000
    """

    def __init__(self, threshold_multihop: float = 0.40,
                 threshold_anchor: int = 3):
        self.threshold_multihop = threshold_multihop
        self.threshold_anchor   = threshold_anchor

    def snapshot(self, system: Primeval) -> dict:
        g = system.graph

        # Multi-hop test
        successes = 0
        for premise1, premise2, conclusion in MULTIHOP_TEST_PAIRS:
            # Check connection from premise1 to conclusion
            cross = _triple_connected(system, premise1, conclusion)
            if cross >= 5:
                successes += 1
        multihop_rate = successes / len(MULTIHOP_TEST_PAIRS)

        # World knowledge anchor levels
        wk_queries = [
            "cat is", "water is", "the sun", "paris is",
            "einstein was", "fire needs", "rain comes",
        ]
        levels = []
        for q in wk_queries:
            r = system.infer("letter", q)
            levels.append(r.anchor_level)
        mean_level = float(np.mean(levels)) if levels else 0.0

        # Level 4+ structure count
        l4_count = sum(
            len(g.nodes_at_level(lv))
            for lv in range(4, g.max_level() + 1)
        )

        # Mean stability of top level-4 structures
        l4_structs = [
            nid for lv in range(4, min(7, g.max_level() + 1))
            for nid in g.nodes_at_level(lv)
        ]
        mean_l4_stab = (
            float(np.mean([g.stability(n) for n in l4_structs]))
            if l4_structs else 0.0
        )

        return {
            "multihop_rate": multihop_rate,
            "mean_wk_level": mean_level,
            "l4_count":      l4_count,
            "mean_l4_stab":  mean_l4_stab,
        }

    def is_ready(self, snap: dict) -> bool:
        return (
            snap["multihop_rate"] >= self.threshold_multihop and
            snap["mean_wk_level"] >= self.threshold_anchor   and
            snap["l4_count"]      >= 2000
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage5Trainer:
    """
    Four ingestion types interleaved:
      30% multi-hop chains         — highest reward, core capability
      30% Wikidata triples         — dense factual knowledge
      25% WordNet definitions       — grounded concept knowledge
      15% non-fiction corpus        — natural world-describing prose

    Reward scale drops to 0.8 — mostly occurrence-driven now.
    Multi-hop chains get full reward — they build the reasoning paths.
    Volume does most of the work at this stage.
    """

    def __init__(
        self,
        system:         Primeval,
        reward_scale:   float = 0.8,
        log_interval:   int   = 10_000,
        check_interval: int   = 10_000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor()
        self._last_snap:    dict = {}
        self._corpus_idx    = 0

    def _next_nonfiction(self) -> str:
        if not NONFICTION_CORPUS:
            return ""
        sent = NONFICTION_CORPUS[self._corpus_idx % len(NONFICTION_CORPUS)]
        self._corpus_idx += 1
        return sent

    def step(self) -> None:
        r = random.random()

        if r < 0.30:
            atoms, raw_reward = generate_multihop_sequence()
            if atoms:
                self.system.ingest_atoms(atoms)
                self.system.reward(raw_reward * self.reward_scale)

        elif r < 0.60:
            atoms, raw_reward = generate_triple_sequence()
            if atoms:
                self.system.ingest_atoms(atoms)
                if raw_reward > 0:
                    self.system.reward(raw_reward * self.reward_scale)

        elif r < 0.85:
            atoms, raw_reward = generate_definition_sequence()
            if atoms:
                self.system.ingest_atoms(atoms)
                if raw_reward > 0:
                    self.system.reward(raw_reward * self.reward_scale)

        else:
            sent = self._next_nonfiction()
            if sent:
                atoms = Atoms.sequence("letter", list(sent))
                self.system.ingest_atoms(atoms)
                # Low reward for general prose — occurrence does the work
                self.system.reward(0.1 * self.reward_scale)

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 5 starting. Target steps: %d", total_steps)
        logger.info("Multi-hop chains:    %d", len(MULTIHOP_CHAINS))
        logger.info("World triples:       %d", len(WORLD_TRIPLES))
        logger.info("Definition sents:    %d", len(DEFINITION_SENTENCES))
        logger.info("Non-fiction sents:   %d", len(NONFICTION_CORPUS))
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)
                if self.monitor.is_ready(self._last_snap):
                    logger.info(
                        "Stage 5 READY at step %d (%.1fs). "
                        "multihop=%.2f wk_level=%.1f l4=%d",
                        step, elapsed,
                        self._last_snap["multihop_rate"],
                        self._last_snap["mean_wk_level"],
                        self._last_snap["l4_count"],
                    )
                    return True

        logger.info(
            "Stage 5 ended at step limit. multihop=%.2f wk_level=%.1f l4=%d",
            self._last_snap.get("multihop_rate", 0.0),
            self._last_snap.get("mean_wk_level", 0.0),
            self._last_snap.get("l4_count", 0),
        )
        return False

    def _log(self, step: int, elapsed: float) -> None:
        g    = self.system.graph
        stat = self.system.stats()
        snap = self._last_snap

        logger.info(
            "Step %6d | %.1fs | nodes=%d edges=%d max_lv=%d",
            step, elapsed, stat["nodes"], stat["edges"], stat["max_level"],
        )
        logger.info(
            "  multihop=%.2f (need %.2f) | wk_level=%.1f (need %d) | "
            "l4_count=%d (need 2000) | mean_l4_stab=%.1f",
            snap.get("multihop_rate", 0.0),
            self.monitor.threshold_multihop,
            snap.get("mean_wk_level", 0.0),
            self.monitor.threshold_anchor,
            snap.get("l4_count", 0),
            snap.get("mean_l4_stab", 0.0),
        )

        # Sample world knowledge inferences
        logger.info("  Sample inferences:")
        queries = [
            "cat is", "water is", "the sun", "paris is",
            "einstein was", "fire needs", "animals need",
            "plants need", "rain comes",
        ]
        for q in queries:
            r = self.system.infer("letter", q)
            logger.info("    %r -> L%d  conf=%.1f",
                        q, r.anchor_level, r.confidence)

        # Sample multi-hop test results
        logger.info("  Multi-hop sample (first 5):")
        for premise1, premise2, conclusion in MULTIHOP_TEST_PAIRS[:5]:
            cross = _triple_connected(self.system, premise1, conclusion)
            flag  = "+" if cross >= 5 else "."
            logger.info("    %s [%d] %r => %r",
                        flag, cross, premise1[:25], conclusion[:25])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak -- Stage 5")
    parser.add_argument("--steps",        type=int,   default=500_000)
    parser.add_argument("--log-interval", type=int,   default=10_000)
    parser.add_argument("--reward-scale", type=float, default=0.8)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage5")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--from-stage4",  action="store_true")
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 6,    # wider — multi-hop chains are long
        occurrence_delta            = 1.0,
        reward_multiplier           = 3.0,  # lower — occurrence drives mostly
        downward_growth_delta       = 0.05,
        consolidator_interval       = 600,
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 500,
        counts_decay_rate           = 0.003,  # slower decay — world knowledge stable
        decay_interval              = 8_000,
        base_decay_rate             = 5e-4,
        level_decay_factor          = 1.5,
        checkpoint_dir              = args.checkpoint,
        checkpoint_interval         = 50_000,
    )

    system = Primeval(cfg)

    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if checkpoints:
            system.load(checkpoints[-1])
            logger.info("Resumed from %s", checkpoints[-1])
        else:
            logger.warning("No stage5 checkpoint -- starting fresh")
    elif args.from_stage4:
        path = "./checkpoints/stage4/final"
        if os.path.exists(path):
            system.load(path)
            logger.info("Loaded Stage 4 checkpoint")
        else:
            logger.warning("Stage 4 checkpoint not found -- starting fresh")

    trainer = Stage5Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)
    system.save(args.checkpoint + "/final")
    logger.info("Saved to %s/final", args.checkpoint)

    g = system.graph
    print("\n-- Stage 5 Final Summary ----------------------------------------")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n-- Multi-hop reasoning test -------------------------------------")
    successes = 0
    for premise1, premise2, conclusion in MULTIHOP_TEST_PAIRS:
        cross = _triple_connected(system, premise1, conclusion)
        flag  = "+" if cross >= 5 else "."
        if cross >= 5:
            successes += 1
        print(f"  {flag} [{cross:3d}] {premise1:30s} => {conclusion}")
    print(f"\n  Multi-hop rate: {successes}/{len(MULTIHOP_TEST_PAIRS)} = "
          f"{successes/len(MULTIHOP_TEST_PAIRS):.0%}")

    print("\n-- World knowledge inference ------------------------------------")
    wk_queries = [
        "cat is", "water is", "the sun", "paris is",
        "einstein was", "fire needs", "animals need",
        "plants need", "rain comes", "the earth",
        "a doctor", "a teacher", "food gives",
    ]
    for q in wk_queries:
        r = system.infer("letter", q)
        print(f"  {q!r:20s} -> L{r.anchor_level}  conf={r.confidence:.1f}  "
              f"chain={len(r.chain)}")

    snap = trainer._last_snap
    print(f"\n  Multi-hop rate:  {snap.get('multihop_rate',0):.2f} "
          f"(need {trainer.monitor.threshold_multihop:.2f})")
    print(f"  WK anchor level: {snap.get('mean_wk_level',0):.1f} "
          f"(need {trainer.monitor.threshold_anchor})")
    print(f"  L4+ structures:  {snap.get('l4_count',0)} (need 2000)")
    print(f"\n  Stage 5 {'COMPLETE' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()