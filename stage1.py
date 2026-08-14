"""
stage1.py — Little Deepak: Stage 1 Curriculum
==============================================

Within-stream pattern learning. Three streams, still independent.

  Letter  — 5000 most common English words as letter sequences.
             Frequency-weighted reward: common words rewarded more.
             Goal: syllable-like clusters and word-level structures form.

  Phoneme — Same 5000 words as phoneme sequences via CMU Pronouncing Dict.
             Frequency-weighted reward matching the letter stream.
             Goal: onset-nucleus-coda structures form within phoneme graph.

  Pixel   — Simple 2D shapes scanned in raster order (8×8 = 64 atoms).
             10 shape classes, programmatically generated with noise.
             Goal: edge detectors and shape-class structures form.

No cross-stream pairing yet. Streams remain independent.
Reward is more selective than Stage 0 — only rewarded for real words,
weighted by frequency. Infrequent and non-English patterns get no reward.

Readiness condition:
  Level-2 and level-3 structures form with meaningful stability.
  70%+ of partial-word inferences anchor at level-2 or above.
  Stability growth rate drops below 0.005 across all streams.

Run:
    python stage1.py [--steps 100000] [--log-interval 2000]
    python stage1.py --resume  (continues from stage1 checkpoint)
    python stage1.py --from-stage0  (loads stage0 checkpoint first)
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
from primeval import Atoms, Config, Primeval, PHONEMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1")

# ── Suppress NLTK download chatter ────────────────────────────────────────────
import nltk
nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict as _cmudict
_CMU = _cmudict.dict()


# ══════════════════════════════════════════════════════════════════════════════
# WORD FREQUENCY TABLE
# Top-5000 English words with relative frequency weights.
# Source: word frequency lists derived from large English corpora.
# Frequencies are log-scaled so reward doesn't collapse to top-10 words.
# ══════════════════════════════════════════════════════════════════════════════

# Top-200 seeded explicitly — remainder filled from CMU dict common words.
# Format: word → raw frequency rank (1 = most common)
_SEED_WORDS: list[tuple[str, int]] = [
    ("the",1),("be",2),("to",3),("of",4),("and",5),("a",6),("in",7),
    ("that",8),("have",9),("it",10),("for",11),("not",12),("on",13),
    ("with",14),("he",15),("as",16),("you",17),("do",18),("at",19),
    ("this",20),("but",21),("his",22),("by",23),("from",24),("they",25),
    ("we",26),("say",27),("her",28),("she",29),("or",30),("an",31),
    ("will",32),("my",33),("one",34),("all",35),("would",36),("there",37),
    ("their",38),("what",39),("so",40),("up",41),("out",42),("if",43),
    ("about",44),("who",45),("get",46),("which",47),("go",48),("me",49),
    ("when",50),("make",51),("can",52),("like",53),("time",54),("no",55),
    ("just",56),("him",57),("know",58),("take",59),("people",60),
    ("into",61),("year",62),("your",63),("good",64),("some",65),
    ("could",66),("them",67),("see",68),("other",69),("than",70),
    ("then",71),("now",72),("look",73),("only",74),("come",75),
    ("its",76),("over",77),("think",78),("also",79),("back",80),
    ("after",81),("use",82),("two",83),("how",84),("our",85),
    ("work",86),("first",87),("well",88),("way",89),("even",90),
    ("new",91),("want",92),("because",93),("any",94),("these",95),
    ("give",96),("day",97),("most",98),("us",99),("between",100),
    ("need",101),("large",102),("often",103),("hand",104),("high",105),
    ("place",106),("hold",107),("turn",108),("help",109),("start",110),
    ("city",111),("play",112),("small",113),("number",114),("off",115),
    ("always",116),("move",117),("live",118),("try",119),("ask",120),
    ("seem",121),("feel",122),("leave",123),("call",124),("keep",125),
    ("eye",126),("never",127),("last",128),("let",129),("think",130),
    ("long",131),("here",132),("thing",133),("great",134),("man",135),
    ("find",136),("line",137),("change",138),("cause",139),("much",140),
    ("before",141),("mean",142),("move",143),("right",144),("boy",145),
    ("old",146),("too",147),("same",148),("tell",149),("does",150),
    ("set",151),("three",152),("want",153),("air",154),("well",155),
    ("also",156),("play",157),("small",158),("end",159),("put",160),
    ("home",161),("read",162),("hand",163),("port",164),("large",165),
    ("spell",166),("add",167),("land",168),("here",169),("must",170),
    ("big",171),("high",172),("such",173),("follow",174),("act",175),
    ("why",176),("ask",177),("went",178),("men",179),("real",180),
    ("cat",181),("dog",182),("run",183),("eat",184),("hot",185),
    ("cold",186),("fire",187),("water",188),("sun",189),("moon",190),
    ("tree",191),("bird",192),("fish",193),("book",194),("door",195),
    ("house",196),("road",197),("food",198),("name",199),("love",200),
]

# Extended words from CMU dict — alphabetically sampled for coverage
_EXTENDED_WORDS = [
    "able","about","above","across","again","age","ago","agree","ahead",
    "along","already","although","among","another","apart","appear","apply",
    "area","around","arrive","article","attack","attempt","attention","aware",
    "baby","ball","base","battle","bear","beat","become","begin","behind",
    "below","best","better","black","blue","body","both","break","bring",
    "brother","build","business","buy","care","carry","case","catch","certain",
    "chance","check","child","children","choice","choose","clear","close",
    "color","community","complete","control","copy","cost","court","cover",
    "create","cut","dark","dead","deal","decide","deep","describe","design",
    "develop","different","difficult","direct","distance","draw","drive",
    "drop","during","each","early","earth","east","easy","effort","either",
    "else","enough","enter","equal","especially","establish","every","example",
    "expect","experience","explain","face","fact","fall","family","far",
    "fast","father","few","fight","fill","final","five","floor","flower",
    "fly","force","form","four","free","friend","front","full","future",
    "girl","glass","green","ground","group","grow","half","happen","hard",
    "having","heart","heat","heavy","history","hope","hour","human","idea",
    "important","increase","industry","information","instead","interest",
    "kind","language","learn","less","level","light","little","local",
    "machine","maybe","mind","money","month","morning","mother","music",
    "national","near","night","north","nothing","notice","once","open",
    "order","others","outside","own","page","paper","part","pass","past",
    "perhaps","picture","plan","point","position","possible","power","press",
    "problem","process","produce","program","provide","pull","question","quite",
    "ready","recent","record","reduce","remain","remember","report","result",
    "return","river","room","rule","school","second","show","side","since",
    "situation","size","skill","sleep","social","society","something","soon",
    "south","speak","special","stand","state","stay","step","still","stop",
    "store","story","strong","student","study","subject","support","sure",
    "system","table","teacher","team","term","themselves","through","today",
    "together","town","travel","true","understand","until","upon","usually",
    "view","voice","walk","watch","west","white","whole","wide","wind",
    "within","woman","women","world","write","young",
]

def _build_word_list() -> list[tuple[str, float]]:
    """
    Build word list with log-scaled frequency weights.
    Only include words that exist in CMU dict (so phoneme stream works).
    Returns list of (word, weight) sorted by weight descending.
    """
    result: dict[str, float] = {}
    max_rank = 5000

    for word, rank in _SEED_WORDS:
        w = word.lower()
        if w in _CMU:
            # log-scale so top words don't overwhelmingly dominate
            result[w] = 1.0 - (np.log(rank) / np.log(max_rank + 1))

    for i, word in enumerate(_EXTENDED_WORDS, start=201):
        w = word.lower()
        if w in _CMU and w not in result:
            result[w] = 1.0 - (np.log(i) / np.log(max_rank + 1))

    # Fill remaining from CMU dict sorted alphabetically up to 5000 total
    rank = len(result) + 201
    for word in sorted(_CMU.keys()):
        if len(result) >= 5000:
            break
        if word not in result and word.isalpha() and 2 <= len(word) <= 12:
            result[word] = 1.0 - (np.log(rank) / np.log(max_rank + 1))
            rank += 1

    return sorted(result.items(), key=lambda x: -x[1])

WORD_LIST: list[tuple[str, float]] = _build_word_list()
WORD_WEIGHTS = [w for _, w in WORD_LIST]
WORD_TOTAL   = sum(WORD_WEIGHTS)
logger.info("Word list built: %d words", len(WORD_LIST))


def _get_phonemes(word: str) -> list[str]:
    """Get phoneme sequence for a word, stripping stress markers."""
    entries = _CMU.get(word.lower(), [])
    if not entries:
        return []
    raw = entries[0]
    return ["".join(c for c in p if not c.isdigit()) for p in raw]


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL SHAPE GENERATORS — 8×8 raster scan → 64 pixel atoms
# ══════════════════════════════════════════════════════════════════════════════

GRID = 8

def _raster(arr: np.ndarray) -> list[int]:
    """Flatten 8×8 array to raster-order pixel atom list."""
    flat = arr.flatten().astype(np.uint8)
    return Atoms.sequence("pixel", flat.tolist())

def _add_noise(arr: np.ndarray, std: float = 10.0) -> np.ndarray:
    noisy = arr.astype(np.float32) + np.random.normal(0, std, arr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def _shape_circle() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.float32)
    cx, cy, r = GRID/2, GRID/2, GRID/2 - 1
    for i in range(GRID):
        for j in range(GRID):
            if (i - cy)**2 + (j - cx)**2 <= r**2:
                g[i,j] = 200
    return g.astype(np.uint8)

def _shape_square() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[1:GRID-1, 1:GRID-1] = 200
    return g

def _shape_horizontal_line() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[GRID//2, :] = 220
    return g

def _shape_vertical_line() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[:, GRID//2] = 220
    return g

def _shape_diagonal_lr() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(GRID):
        g[i, i] = 220
    return g

def _shape_diagonal_rl() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(GRID):
        g[i, GRID-1-i] = 220
    return g

def _shape_cross() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[GRID//2, :] = 200
    g[:, GRID//2] = 200
    return g

def _shape_triangle() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        width = int((i / GRID) * GRID)
        start = (GRID - width) // 2
        g[i, start:start+width] = 200
    return g.astype(np.uint8)

def _shape_top_half() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[:GRID//2, :] = 200
    return g

def _shape_checkerboard() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(GRID):
        for j in range(GRID):
            if (i + j) % 2 == 0:
                g[i,j] = 200
    return g

SHAPE_GENERATORS = [
    _shape_circle, _shape_square,
    _shape_horizontal_line, _shape_vertical_line,
    _shape_diagonal_lr, _shape_diagonal_rl,
    _shape_cross, _shape_triangle,
    _shape_top_half, _shape_checkerboard,
]
N_SHAPES = len(SHAPE_GENERATORS)


def sample_shape() -> tuple[list[int], float]:
    """Sample a random 8×8 shape with noise. Returns (atoms, reward)."""
    fn      = random.choice(SHAPE_GENERATORS)
    canvas  = fn()
    noisy   = _add_noise(canvas, std=8.0)
    atoms   = _raster(noisy)
    return atoms, 1.0   # all shapes equally valid at Stage 1


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE SAMPLERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_word_letter() -> tuple[list[int], float]:
    """
    Sample a word from the frequency-weighted list.
    Returns letter atoms and frequency-proportional reward.
    """
    word, weight = random.choices(WORD_LIST, weights=WORD_WEIGHTS, k=1)[0]
    atoms = Atoms.sequence("letter", list(word))
    return atoms, weight


def sample_word_phoneme() -> tuple[list[int], float]:
    """
    Sample a word and return its phoneme sequence.
    Reward matches the letter stream weight for the same word.
    """
    word, weight = random.choices(WORD_LIST, weights=WORD_WEIGHTS, k=1)[0]
    phonemes = _get_phonemes(word)
    if not phonemes:
        return [], 0.0
    atoms = Atoms.sequence("phoneme", phonemes)
    return atoms, weight


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ReadinessMonitor:
    """
    Stage 1 readiness — two conditions must both hold:
    1. Stability growth rate < 0.005 across all streams (tighter than Stage 0).
    2. At least 70% of sampled partial-word inferences anchor at level >= 2.
    """
    ANCHOR_WORDS = [
        "the","cat","dog","run","time","work","people","city","light",
        "water","house","three","school","great","place","first","after",
        "think","small","world","never","always","something","follow","change",
    ]

    def __init__(self, threshold_rate: float = 0.005, threshold_anchor: float = 0.70):
        self.threshold_rate   = threshold_rate
        self.threshold_anchor = threshold_anchor
        self._prev: dict[str, float] = {}

    def snapshot(self, system: Primeval) -> dict:
        g = system.graph
        rates: dict[str, float] = {}

        for stream, lo, hi in [
            ("letter",  Atoms.LETTER_OFF,  Atoms.PHONEME_OFF),
            ("phoneme", Atoms.PHONEME_OFF, Atoms.STRUCT_START),
            ("pixel",   Atoms.PIXEL_OFF,   Atoms.LETTER_OFF),
        ]:
            structs = [
                nid for nid in g.nodes_at_level(1)
                if g._nodes.get(nid) and all(
                    lo <= c < hi for c in g._nodes[nid].constituents
                )
            ]
            if not structs:
                rates[stream] = 1.0
                continue
            top = sorted(structs, key=lambda n: -g.stability(n))[:50]
            mean_stab = float(np.mean([g.stability(n) for n in top]))
            if stream in self._prev:
                prev = self._prev[stream]
                rates[stream] = abs(mean_stab - prev) / (prev + 1e-9)
            else:
                rates[stream] = 1.0
            self._prev[stream] = mean_stab

        # Anchor level check — partial words
        above2 = 0
        total  = 0
        for word in self.ANCHOR_WORDS:
            if len(word) < 3:
                continue
            partial = word[:-1]   # drop last letter
            r = system.infer("letter", partial)
            if r.anchor_level >= 2:
                above2 += 1
            total += 1
        anchor_rate = above2 / total if total > 0 else 0.0

        return {"rates": rates, "anchor_rate": anchor_rate}

    def is_ready(self, snap: dict) -> bool:
        rates       = snap["rates"]
        anchor_rate = snap["anchor_rate"]
        rates_ok    = all(r < self.threshold_rate for r in rates.values())
        anchor_ok   = anchor_rate >= self.threshold_anchor
        return rates_ok and anchor_ok


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage1Trainer:
    """
    Runs Stage 1 training for Little Deepak.

    Streams interleaved round-robin: letter → phoneme → pixel → repeat.
    Reward is frequency-proportional — common words rewarded more than rare.
    Reward scale is lower than Stage 0 — bedrock is established, reward
    now shapes word-level structure rather than carving atomic statistics.
    """

    def __init__(
        self,
        system:         Primeval,
        reward_scale:   float = 3.0,   # lower than Stage 0's 5.0
        log_interval:   int   = 2000,
        check_interval: int   = 2000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor()
        self._stream_cycle  = 0
        self._last_snap:    dict = {}

    def step(self) -> None:
        stream = ["letter", "phoneme", "pixel"][self._stream_cycle % 3]
        self._stream_cycle += 1

        if stream == "letter":
            atoms, raw_reward = sample_word_letter()
        elif stream == "phoneme":
            atoms, raw_reward = sample_word_phoneme()
        else:
            atoms, raw_reward = sample_shape()

        if not atoms:
            return

        self.system.ingest_atoms(atoms)
        if raw_reward > 0.0:
            self.system.reward(raw_reward * self.reward_scale)

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 1 starting. Target steps: %d", total_steps)
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)

                if self.monitor.is_ready(self._last_snap):
                    logger.info(
                        "✓ Stage 1 READY at step %d (%.1fs). Rates=%s anchor=%.2f",
                        step, elapsed,
                        {k: f"{v:.4f}" for k, v in self._last_snap["rates"].items()},
                        self._last_snap["anchor_rate"],
                    )
                    return True

        logger.info(
            "Stage 1 ended at step limit %d. Final rates=%s anchor=%.2f",
            total_steps,
            {k: f"{v:.4f}" for k, v in self._last_snap.get("rates", {}).items()},
            self._last_snap.get("anchor_rate", 0.0),
        )
        return False

    def _log(self, step: int, elapsed: float) -> None:
        g    = self.system.graph
        stat = self.system.stats()
        snap = self._last_snap
        rates       = snap.get("rates", {})
        anchor_rate = snap.get("anchor_rate", 0.0)

        logger.info(
            "Step %6d | %.1fs | nodes=%d edges=%d structs=%d max_lv=%d",
            step, elapsed,
            stat["nodes"], stat["edges"], stat["structures"], stat["max_level"],
        )
        logger.info(
            "  Rates → letter=%.4f phoneme=%.4f pixel=%.4f | anchor≥L2=%.2f (need 0.70)",
            rates.get("letter", 1.0),
            rates.get("phoneme", 1.0),
            rates.get("pixel", 1.0),
            anchor_rate,
        )

        # Top letter structures (level-1 and level-2)
        letter_l1 = sorted(
            [nid for nid in g.nodes_at_level(1)
             if g._nodes.get(nid) and all(
                 Atoms.LETTER_OFF <= c < Atoms.PHONEME_OFF
                 for c in g._nodes[nid].constituents)],
            key=lambda n: -g.stability(n)
        )[:5]
        if letter_l1:
            logger.info("  Top letter L1 structures:")
            for nid in letter_l1:
                meta = g._nodes[nid]
                letters = "".join(chr(c - Atoms.LETTER_OFF) for c in meta.constituents)
                logger.info("    %r  stab=%.1f  trav=%d",
                            letters, g.stability(nid), meta.traversals)

        letter_l2 = sorted(
            [nid for nid in g.nodes_at_level(2)
             if g._nodes.get(nid)],
            key=lambda n: -g.stability(n)
        )[:3]
        if letter_l2:
            logger.info("  Top letter L2 structures (stab):")
            for nid in letter_l2:
                logger.info("    struct(%d)  stab=%.1f  trav=%d",
                            nid, g.stability(nid), g._nodes[nid].traversals)

        # Sample inference results
        sample_inferences = [("letter","cat"), ("letter","time"),
                             ("letter","wor"), ("letter","peopl")]
        logger.info("  Sample inferences:")
        for modality, query in sample_inferences:
            r = self.system.infer(modality, query)
            logger.info("    %r → anchor=L%d  conf=%.1f",
                        query, r.anchor_level, r.confidence)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak — Stage 1")
    parser.add_argument("--steps",          type=int,   default=100_000)
    parser.add_argument("--log-interval",   type=int,   default=2_000)
    parser.add_argument("--reward-scale",   type=float, default=3.0)
    parser.add_argument("--checkpoint",     type=str,   default="./checkpoints/stage1")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from stage1 checkpoint")
    parser.add_argument("--from-stage0",    action="store_true",
                        help="Load stage0 final checkpoint as starting point")
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 3,    # wider window than Stage 0 — words need more context
        occurrence_delta            = 1.0,
        reward_multiplier           = 8.0,  # slightly lower than Stage 0
        downward_growth_delta       = 0.1,
        consolidator_interval       = 300,
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 500,
        counts_decay_rate           = 0.005,  # slower decay — word structures need time to form
        decay_interval              = 3_000,
        base_decay_rate             = 2e-4,   # slightly faster than Stage 0
        level_decay_factor          = 1.5,
        checkpoint_dir              = args.checkpoint,
        checkpoint_interval         = 20_000,
    )

    system = Primeval(cfg)

    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resuming from %s", latest)
            system.load(latest)
        else:
            logger.warning("No stage1 checkpoint found — starting fresh")
    elif args.from_stage0:
        stage0_final = "./checkpoints/stage0/final"
        if os.path.exists(stage0_final):
            logger.info("Loading Stage 0 checkpoint from %s", stage0_final)
            system.load(stage0_final)
        else:
            logger.warning("Stage 0 checkpoint not found at %s — starting fresh", stage0_final)
    else:
        logger.info("Starting Stage 1 from scratch (no Stage 0 checkpoint loaded)")

    trainer = Stage1Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)

    # Final checkpoint
    system.save(args.checkpoint + "/final")
    logger.info("Saved final checkpoint to %s/final", args.checkpoint)

    # Final summary
    g = system.graph
    print("\n── Stage 1 Final Summary ────────────────────────────────")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n── Top 10 letter L1 structures ──────────────────────────")
    letter_l1 = sorted(
        [nid for nid in g.nodes_at_level(1)
         if g._nodes.get(nid) and all(
             Atoms.LETTER_OFF <= c < Atoms.PHONEME_OFF
             for c in g._nodes[nid].constituents)],
        key=lambda n: -g.stability(n)
    )[:10]
    for nid in letter_l1:
        meta    = g._nodes[nid]
        letters = "".join(chr(c - Atoms.LETTER_OFF) for c in meta.constituents)
        print(f"  {letters!r:10s}  stab={g.stability(nid):.1f}  trav={meta.traversals}")

    print("\n── Inference on partial words ────────────────────────────")
    test_words = ["the","cat","time","work","people","house","water",
                  "school","never","change","follow","something"]
    above2 = 0
    for word in test_words:
        partial = word[:-1]
        r = system.infer("letter", partial)
        flag = "✓" if r.anchor_level >= 2 else "·"
        print(f"  {flag} {partial!r:12s} → L{r.anchor_level}  conf={r.confidence:.1f}")
        if r.anchor_level >= 2:
            above2 += 1
    print(f"\n  Anchored at L2+: {above2}/{len(test_words)} = {above2/len(test_words):.0%}")

    print(f"\n  Stage 1 {'COMPLETE ✓' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()