"""
stage2.py — Little Deepak: Stage 2 Curriculum
==============================================

Cross-stream grounding. For the first time streams are paired.

Three pairing types:

  Letter + Phoneme
    The same word fed as a mixed atom sequence — letters first, then
    phonemes, in a single ingestion window. The graph discovers that
    letter-sequence "cat" and phoneme-sequence K-AE-T always co-occur
    with the same reward signal and builds cross-stream bridges.
    All 5000 words from Stage 1, frequency-weighted reward.

  Letter + Pixel
    500 concrete nouns paired with their synthetic 8x8 shape.
    "circle" arrives with a circle image in the same ingestion.
    "line" with a line. "square" with a square.
    This is literal grounding — words getting connected to percepts.

  Phrases (letter only, but richer context)
    Short 2-4 word phrases feeding each word in context.
    "the cat sat", "cold water", "a big house".
    Gives short common words the sequential co-activation they need
    to form level-2 structures — fixing the Stage 1 short-word gap.

Reward is still automated. Cross-stream co-activation earns reward.
Human not yet in the loop.

Readiness:
  Feed only letter sequence for a word -> phoneme structures light up.
  Feed only phoneme sequence -> letter structures light up.
  Bidirectional cross-stream activation >= 60% of test words.

Run:
    python stage2.py --from-stage1 --steps 150000
    python stage2.py --resume
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
from primeval import Atoms, Config, Primeval, PHONEMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage2")

import nltk
nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict as _cmudict
_CMU = _cmudict.dict()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_phonemes(word: str) -> list[str]:
    entries = _CMU.get(word.lower(), [])
    if not entries:
        return []
    return ["".join(c for c in p if not c.isdigit()) for p in entries[0]]


# ══════════════════════════════════════════════════════════════════════════════
# WORD LIST
# ══════════════════════════════════════════════════════════════════════════════

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
    ("cat",101),("dog",102),("run",103),("eat",104),("hot",105),
    ("cold",106),("fire",107),("water",108),("sun",109),("moon",110),
    ("tree",111),("bird",112),("fish",113),("book",114),("door",115),
    ("house",116),("road",117),("food",118),("name",119),("love",120),
    ("need",121),("large",122),("hand",123),("high",124),("place",125),
    ("city",126),("play",127),("small",128),("number",129),("move",130),
    ("live",131),("ask",132),("feel",133),("call",134),("keep",135),
    ("never",136),("last",137),("long",138),("thing",139),("great",140),
    ("find",141),("change",142),("much",143),("before",144),("mean",145),
    ("right",146),("old",147),("same",148),("tell",149),("set",150),
    ("three",151),("air",152),("read",153),("land",154),("must",155),
    ("big",156),("such",157),("follow",158),("act",159),("why",160),
    ("real",161),("home",162),("school",163),("world",164),
]

def _build_word_list() -> list[tuple[str, float]]:
    result: dict[str, float] = {}
    max_rank = 5000
    for word, rank in _SEED_WORDS:
        w = word.lower()
        if w in _CMU:
            result[w] = 1.0 - (np.log(rank) / np.log(max_rank + 1))
    rank = len(result) + 201
    for word in sorted(_CMU.keys()):
        if len(result) >= 5000:
            break
        if word not in result and word.isalpha() and 2 <= len(word) <= 12:
            result[word] = 1.0 - (np.log(rank) / np.log(max_rank + 1))
            rank += 1
    return sorted(result.items(), key=lambda x: -x[1])

WORD_LIST    = _build_word_list()
WORD_WEIGHTS = [w for _, w in WORD_LIST]
logger.info("Word list: %d words", len(WORD_LIST))


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL SHAPES
# ══════════════════════════════════════════════════════════════════════════════

GRID = 8

def _raster(arr: np.ndarray) -> list[int]:
    return Atoms.sequence("pixel", arr.flatten().astype(np.uint8).tolist())

def _noise(arr: np.ndarray, std: float = 8.0) -> np.ndarray:
    return np.clip(
        arr.astype(np.float32) + np.random.normal(0, std, arr.shape),
        0, 255
    ).astype(np.uint8)

def _circle() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.float32)
    cx, cy, r = GRID/2, GRID/2, GRID/2 - 1
    for i in range(GRID):
        for j in range(GRID):
            if (i - cy)**2 + (j - cx)**2 <= r**2:
                g[i, j] = 200
    return g.astype(np.uint8)

def _square() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[1:GRID-1, 1:GRID-1] = 200
    return g

def _hline() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[GRID//2, :] = 220
    return g

def _vline() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[:, GRID//2] = 220
    return g

def _diag_lr() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(GRID):
        g[i, i] = 220
    return g

def _cross() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[GRID//2, :] = 200
    g[:, GRID//2] = 200
    return g

def _triangle() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        w = int((i / GRID) * GRID)
        s = (GRID - w) // 2
        g[i, s:s+w] = 200
    return g.astype(np.uint8)

def _dot() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    g[GRID//2, GRID//2] = 255
    return g

def _wave() -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for j in range(GRID):
        i = int((np.sin(j * np.pi / GRID * 2) * 0.5 + 0.5) * (GRID - 1))
        g[min(i, GRID-1), j] = 220
    return g

SHAPE_MAP: dict[str, object] = {
    "circle":   _circle,
    "square":   _square,
    "line":     _hline,
    "vline":    _vline,
    "cross":    _cross,
    "triangle": _triangle,
    "dot":      _dot,
    "wave":     _wave,
    "box":      _square,
    "ring":     _circle,
    "arrow":    _diag_lr,
}

_NOUN_SHAPE_SEED: dict[str, str] = {
    "circle":"circle","ball":"circle","ring":"circle","sun":"circle",
    "moon":"circle","eye":"circle","wheel":"circle","coin":"circle",
    "head":"circle","cup":"circle","bowl":"circle","egg":"circle",
    "hole":"circle","button":"circle","globe":"circle","bubble":"circle",
    "square":"square","box":"square","book":"square","door":"square",
    "house":"square","window":"square","screen":"square","table":"square",
    "wall":"square","floor":"square","board":"square","card":"square",
    "page":"square","frame":"square","block":"square","room":"square",
    "line":"line","road":"line","river":"line","wire":"line","rope":"line",
    "path":"line","rail":"line","pipe":"line","stick":"line","bar":"line",
    "pole":"line","edge":"line","border":"line","stripe":"line",
    "cross":"cross","plus":"cross","star":"cross",
    "triangle":"triangle","mountain":"triangle","roof":"triangle",
    "cone":"triangle","peak":"triangle","hill":"triangle",
    "dot":"dot","point":"dot","spot":"dot","mark":"dot","seed":"dot",
    "wave":"wave","curve":"wave","arc":"wave","bend":"wave",
}

def _build_noun_shape_list() -> list[tuple[str, str]]:
    result: dict[str, str] = dict(_NOUN_SHAPE_SEED)
    shapes_cycle = list(SHAPE_MAP.keys())
    idx = 0
    for word in sorted(_CMU.keys()):
        if len(result) >= 500:
            break
        if word not in result and word.isalpha() and 3 <= len(word) <= 10:
            result[word] = shapes_cycle[idx % len(shapes_cycle)]
            idx += 1
    return list(result.items())

NOUN_SHAPE_LIST = _build_noun_shape_list()
logger.info("Noun-shape pairs: %d", len(NOUN_SHAPE_LIST))


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

PHRASE_TEMPLATES = [
    "the {noun} is {adj}",
    "a {adj} {noun}",
    "the {noun} {verb}",
    "{noun} and {noun}",
    "the {adj} {noun} {verb}",
    "a {noun} in the {noun}",
    "the {noun} of the {noun}",
    "{verb} the {noun}",
    "a {noun} on the {noun}",
    "the {noun} {verb} the {noun}",
]

NOUNS = [
    "cat","dog","bird","fish","tree","sun","moon","star","door","road",
    "house","city","water","fire","wind","earth","time","day","night","book",
    "hand","eye","head","heart","word","man","woman","child","school","world",
]
VERBS = [
    "run","see","know","think","say","go","come","look","walk","find",
    "give","take","make","get","move","keep","turn","start","follow","change",
]
ADJS = [
    "big","small","old","new","good","bad","hot","cold","fast","slow",
    "long","short","high","low","dark","light","hard","soft","clear","deep",
]

def _sample_phrase() -> str:
    template = random.choice(PHRASE_TEMPLATES)
    return template.format(
        noun=random.choice(NOUNS),
        verb=random.choice(VERBS),
        adj=random.choice(ADJS),
    )


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION SAMPLERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_letter_phoneme() -> tuple[list[int], float]:
    word, weight = random.choices(WORD_LIST, weights=WORD_WEIGHTS, k=1)[0]
    phonemes = _get_phonemes(word)
    if not phonemes:
        return [], 0.0
    letter_atoms  = Atoms.sequence("letter",  list(word))
    phoneme_atoms = Atoms.sequence("phoneme", phonemes)
    # Randomly alternate ordering — 50% letter-first, 50% phoneme-first.
    # Fixed letter-first ordering structurally prevents phoneme->letter
    # edges from forming within the window. Alternating creates both
    # directions naturally, building true bidirectional grounding.
    if random.random() < 0.5:
        return letter_atoms + phoneme_atoms, weight   # letter -> phoneme
    else:
        return phoneme_atoms + letter_atoms, weight   # phoneme -> letter


def sample_noun_shape() -> tuple[list[int], float]:
    noun, shape_name = random.choice(NOUN_SHAPE_LIST)
    if not _get_phonemes(noun):
        return [], 0.0
    shape_fn     = SHAPE_MAP.get(shape_name, _circle)
    noisy        = _noise(shape_fn())
    letter_atoms = Atoms.sequence("letter", list(noun))
    pixel_atoms  = _raster(noisy)
    return letter_atoms + pixel_atoms, 0.6


def sample_phrase_context() -> tuple[list[int], float]:
    phrase = _sample_phrase()
    return Atoms.sequence("letter", list(phrase)), 0.4


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR
# ══════════════════════════════════════════════════════════════════════════════

CROSS_STREAM_TEST_WORDS = [
    "cat","dog","time","work","house","water","school","never",
    "change","follow","people","three","light","small","world",
    "bird","tree","book","door","road","fire","moon","hand","food","name",
]

class ReadinessMonitor:

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def snapshot(self, system: Primeval) -> dict:
        g = system.graph
        letter_to_phoneme = 0
        phoneme_to_letter = 0
        total = 0

        for word in CROSS_STREAM_TEST_WORDS:
            phonemes = _get_phonemes(word)
            if not phonemes:
                continue
            total += 1
            letter_atoms  = Atoms.sequence("letter",  list(word))
            phoneme_atoms = Atoms.sequence("phoneme", phonemes)

            # letter -> phoneme: any cross-stream edge from letter to phoneme atoms
            cross_lp = any(
                g.weight(la, pa) > 0
                for la in letter_atoms
                for pa in phoneme_atoms[:1]
            )
            if cross_lp:
                letter_to_phoneme += 1

            # phoneme -> letter: any cross-stream edge from phoneme to letter atoms
            cross_pl = any(
                g.weight(pa, la) > 0
                for pa in phoneme_atoms
                for la in letter_atoms[:1]
            )
            if cross_pl:
                phoneme_to_letter += 1

        lp_rate = letter_to_phoneme / total if total > 0 else 0.0
        pl_rate = phoneme_to_letter / total if total > 0 else 0.0

        above2 = 0
        anchor_total = 0
        for word in CROSS_STREAM_TEST_WORDS[:12]:
            if len(word) < 3:
                continue
            r = system.infer("letter", word[:-1])
            if r.anchor_level >= 2:
                above2 += 1
            anchor_total += 1
        anchor_rate = above2 / anchor_total if anchor_total > 0 else 0.0

        return {
            "letter_to_phoneme": lp_rate,
            "phoneme_to_letter": pl_rate,
            "anchor_rate":       anchor_rate,
        }

    def is_ready(self, snap: dict) -> bool:
        return (
            snap["letter_to_phoneme"] >= self.threshold and
            snap["phoneme_to_letter"] >= self.threshold
        )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage2Trainer:
    """
    Three ingestion types interleaved:
      50% letter+phoneme pairs  — core cross-stream grounding
      25% noun+shape pairs      — visual grounding
      25% phrase context        — short-word structural enrichment

    Reward scale drops to 2.0 — graph is mature, reward shapes direction.
    """

    def __init__(
        self,
        system:         Primeval,
        reward_scale:   float = 2.0,
        log_interval:   int   = 3000,
        check_interval: int   = 3000,
    ):
        self.system         = system
        self.reward_scale   = reward_scale
        self.log_interval   = log_interval
        self.check_interval = check_interval
        self.monitor        = ReadinessMonitor()
        self._last_snap:    dict = {}

    def step(self) -> None:
        r = random.random()
        if r < 0.50:
            atoms, raw_reward = sample_letter_phoneme()
        elif r < 0.75:
            atoms, raw_reward = sample_noun_shape()
        else:
            atoms, raw_reward = sample_phrase_context()

        if not atoms:
            return

        self.system.ingest_atoms(atoms)
        if raw_reward > 0.0:
            self.system.reward(raw_reward * self.reward_scale)

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 2 starting. Target steps: %d", total_steps)
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.check_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                self._log(step, elapsed)
                if self.monitor.is_ready(self._last_snap):
                    logger.info(
                        "Stage 2 READY at step %d (%.1fs). L->P=%.2f P->L=%.2f anchor=%.2f",
                        step, elapsed,
                        self._last_snap["letter_to_phoneme"],
                        self._last_snap["phoneme_to_letter"],
                        self._last_snap["anchor_rate"],
                    )
                    return True

        logger.info(
            "Stage 2 ended at step limit. L->P=%.2f P->L=%.2f anchor=%.2f",
            self._last_snap.get("letter_to_phoneme", 0.0),
            self._last_snap.get("phoneme_to_letter", 0.0),
            self._last_snap.get("anchor_rate", 0.0),
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
            "  Cross-stream: L->P=%.2f  P->L=%.2f (need %.2f) | anchor>=L2=%.2f",
            snap.get("letter_to_phoneme", 0.0),
            snap.get("phoneme_to_letter", 0.0),
            self.monitor.threshold,
            snap.get("anchor_rate", 0.0),
        )

        # Top cross-stream edges
        cross_edges = sorted(
            [(k, v) for k, v in g._weights.items()
             if (Atoms.LETTER_OFF  <= k[0] < Atoms.PHONEME_OFF and
                 Atoms.PHONEME_OFF <= k[1] < Atoms.STRUCT_START)],
            key=lambda x: -x[1]
        )[:5]
        if cross_edges:
            logger.info("  Top letter->phoneme edges:")
            for (a, b), w in cross_edges:
                la = chr(a - Atoms.LETTER_OFF)
                pb_idx = b - Atoms.PHONEME_OFF
                pb = PHONEMES[pb_idx] if pb_idx < len(PHONEMES) else "?"
                logger.info("    letter(%r) -> phoneme(%s)  w=%.1f", la, pb, w)

        logger.info("  Sample inferences:")
        for query in ["ca", "tim", "the", "wor", "peopl", "hous"]:
            r = self.system.infer("letter", query)
            logger.info("    %r -> L%d  conf=%.1f", query, r.anchor_level, r.confidence)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak -- Stage 2")
    parser.add_argument("--steps",        type=int,   default=150_000)
    parser.add_argument("--log-interval", type=int,   default=3_000)
    parser.add_argument("--reward-scale", type=float, default=2.0)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage2")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--from-stage1",  action="store_true")
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 4,
        occurrence_delta            = 1.0,
        reward_multiplier           = 6.0,
        downward_growth_delta       = 0.1,
        consolidator_interval       = 400,
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 500,
        counts_decay_rate           = 0.005,
        decay_interval              = 4_000,
        base_decay_rate             = 3e-4,
        level_decay_factor          = 1.5,
        checkpoint_dir              = args.checkpoint,
        checkpoint_interval         = 25_000,
    )

    system = Primeval(cfg)

    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if checkpoints:
            system.load(checkpoints[-1])
            logger.info("Resumed from %s", checkpoints[-1])
        else:
            logger.warning("No stage2 checkpoint found -- starting fresh")
    elif args.from_stage1:
        path = "./checkpoints/stage1/final"
        if os.path.exists(path):
            system.load(path)
            logger.info("Loaded Stage 1 checkpoint")
        else:
            logger.warning("Stage 1 checkpoint not found -- starting fresh")

    trainer = Stage2Trainer(
        system        = system,
        reward_scale  = args.reward_scale,
        log_interval  = args.log_interval,
        check_interval= args.log_interval,
    )

    ready = trainer.run(args.steps)
    system.save(args.checkpoint + "/final")
    logger.info("Saved to %s/final", args.checkpoint)

    g = system.graph
    print("\n-- Stage 2 Final Summary ----------------------------------------")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n-- Top cross-stream edges (letter -> phoneme) -------------------")
    cross = sorted(
        [(k, v) for k, v in g._weights.items()
         if Atoms.LETTER_OFF <= k[0] < Atoms.PHONEME_OFF
         and Atoms.PHONEME_OFF <= k[1] < Atoms.STRUCT_START],
        key=lambda x: -x[1]
    )[:10]
    for (a, b), w in cross:
        la    = chr(a - Atoms.LETTER_OFF)
        pb_i  = b - Atoms.PHONEME_OFF
        pb    = PHONEMES[pb_i] if pb_i < len(PHONEMES) else "?"
        print(f"  letter({la!r}) -> phoneme({pb})  w={w:.1f}")

    print("\n-- Top cross-stream edges (letter -> pixel) ---------------------")
    lp = sorted(
        [(k, v) for k, v in g._weights.items()
         if Atoms.LETTER_OFF <= k[0] < Atoms.PHONEME_OFF
         and k[1] < Atoms.LETTER_OFF],
        key=lambda x: -x[1]
    )[:5]
    for (a, b), w in lp:
        print(f"  letter({chr(a - Atoms.LETTER_OFF)!r}) -> pixel({b})  w={w:.1f}")

    print("\n-- Inference on partial words ------------------------------------")
    test_words = ["the","cat","time","work","people","house","water",
                  "school","never","change","follow","something"]
    above2 = 0
    for word in test_words:
        partial = word[:-1]
        r = system.infer("letter", partial)
        flag = "+" if r.anchor_level >= 2 else "."
        print(f"  {flag} {partial!r:12s} -> L{r.anchor_level}  conf={r.confidence:.1f}")
        if r.anchor_level >= 2:
            above2 += 1
    print(f"\n  Anchored at L2+: {above2}/{len(test_words)} = {above2/len(test_words):.0%}")

    snap = trainer._last_snap
    print(f"\n  Cross-stream L->P: {snap.get('letter_to_phoneme', 0):.2f}")
    print(f"  Cross-stream P->L: {snap.get('phoneme_to_letter', 0):.2f}")
    print(f"\n  Stage 2 {'COMPLETE' if ready else 'INCOMPLETE (step limit reached)'}")


if __name__ == "__main__":
    main()