"""
stage6.py — Little Deepak: Stage 6 Curriculum
==============================================

Dialogue. Little Deepak learns to respond.

Core insight:
  Generation is atom-level traversal from the last prompt atom forward.
  The graph already has all the structure from Stages 0-5.
  Stage 6 is purely about strengthening specific question->answer paths.

Training loop per QA pair (three steps):

  Step 1 — Build the answer path first.
    Ingest just the answer many times and reward strongly.
    The answer sequence becomes a thick stable path in the graph
    BEFORE being connected to any question.

  Step 2 — Build the bridge.
    Ingest question + answer together and reward strongly.
    The bridge edges from question-end atoms to answer-start atoms
    form on top of an already-solid answer foundation.

  Step 3 — Confirm the path.
    Ingest just the question. Check if atom-level traversal from
    the last question atom reaches answer atoms. Reward if yes.
    Penalise if no — then re-do steps 1 and 2.

Generation:
  Start from the last atom of the prompt.
  Follow outgoing edges greedily at the atom level.
  Collect letter atoms. Decode to text.
  The graph earns the right response through reward — no tricks.

Consolidator:
  Parked at interval 50000. Stage 6 is not about discovering
  new structure — it is about strengthening known paths.
  All structure was built in Stages 0-5.

Run:
    python stage6.py --from-stage5 --steps 500000
    python stage6.py --resume --interactive
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, ".")
from primeval import Atoms, Config, Primeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage6")

import nltk
for pkg in ["nps_chat", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import nps_chat


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION — atom-level traversal from prompt end
# ══════════════════════════════════════════════════════════════════════════════

def generate(system: Primeval, prompt: str,
             decay_factor: float = 0.5) -> str:
    """
    Generate by following atom-level edges forward from the last
    prompt atom, with a per-step dynamic threshold.

    At each step threshold = current_edge_weight * decay_factor.
    The next edge must exceed this to continue.

    Termination happens naturally when the path weakens sharply —
    which happens after '.' once answer paths are well trained.
    No visited set. No step limit. No hardcoded stop.
    The graph earns the stopping point through training volume.
    Give it all the answers — the traversal paths become dominant
    and generation follows them cleanly to their natural end.
    """
    system.ingest("letter", prompt)

    prompt_atoms = Atoms.sequence("letter", list(prompt))
    if not prompt_atoms:
        return "[no response]"

    current = prompt_atoms[-1]
    g       = system.graph
    chars:  list[str] = []

    nbs = [
        (nb, w) for nb, w in g.neighbors(current)
        if Atoms.LETTER_OFF <= nb < Atoms.PHONEME_OFF
    ]
    if not nbs:
        return "[no response]"

    current, current_w = nbs[0]
    chars.append(chr(current - Atoms.LETTER_OFF))

    while True:
        threshold = current_w * decay_factor
        nbs = [
            (nb, w) for nb, w in g.neighbors(current)
            if Atoms.LETTER_OFF <= nb < Atoms.PHONEME_OFF
            and w >= threshold
        ]
        if not nbs:
            break
        current, current_w = nbs[0]
        chars.append(chr(current - Atoms.LETTER_OFF))

    return "".join(chars).strip() if chars else "[no response]"


def response_overlap(response: str, expected: str) -> float:
    """
    Fraction of expected content words found in response.
    Content words: length > 2, not stop words.
    """
    STOP = {"the","a","an","is","are","was","were","in","on","at",
            "to","of","and","or","for","with","as","it","by","do","be"}
    content = [w for w in expected.lower().split()
               if len(w) > 2 and w not in STOP]
    if not content:
        return 0.5
    hits = sum(1 for w in content if w in response.lower())
    return hits / len(content)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

SQUAD_URL   = ("https://raw.githubusercontent.com/rajpurkar/"
               "SQuAD-explorer/master/dataset/train-v2.0.json")
SQUAD_CACHE = "./checkpoints/stage6/squad_cache.json"

def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_squad(max_pairs: int = 50_000) -> list[tuple[str,str]]:
    os.makedirs(os.path.dirname(SQUAD_CACHE), exist_ok=True)
    if os.path.exists(SQUAD_CACHE):
        logger.info("Loading SQuAD from cache")
        with open(SQUAD_CACHE) as f:
            pairs = json.load(f)
        logger.info("SQuAD: %d pairs", len(pairs))
        return pairs
    logger.info("Downloading SQuAD v2...")
    with urllib.request.urlopen(SQUAD_URL, timeout=60) as r:
        data = json.loads(r.read())
    pairs: list[tuple[str,str]] = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                q = _clean(qa["question"])
                a = _clean(answers[0]["text"])
                if q and a and 1 <= len(a.split()) <= 15:
                    pairs.append((q, a))
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    random.shuffle(pairs)
    with open(SQUAD_CACHE, "w") as f:
        json.dump(pairs, f)
    logger.info("SQuAD: %d pairs cached", len(pairs))
    return pairs

def load_nps_chat() -> list[tuple[str,str]]:
    posts = nps_chat.posts()
    pairs: list[tuple[str,str]] = []
    for i in range(len(posts) - 1):
        a = _clean(" ".join(posts[i]))
        b = _clean(" ".join(posts[i+1]))
        if a and b and 2 <= len(a.split()) <= 15 and 2 <= len(b.split()) <= 15:
            pairs.append((a, b))
    logger.info("NPS chat: %d pairs", len(pairs))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# THREE-STEP TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_pair(system: Primeval,
               question: str,
               answer:   str,
               reward_scale: float,
               answer_reps:  int = 5,
               bridge_reps:  int = 3) -> float:
    """
    Three-step training for one QA pair.

    Step 1 — Build answer path.
      Ingest answer alone, reward strongly (answer_reps times).
      Answer sequence becomes thick BEFORE being tied to question.

    Step 2 — Build bridge.
      Ingest question + answer together, reward strongly (bridge_reps times).
      Bridge edges form on top of solid answer foundation.

    Step 3 — Confirm path.
      Ingest question alone. Generate response.
      If response overlaps with answer — reward the path.
      If not — penalise bad path, re-strengthen answer and bridge.

    Returns overlap score (0-1).
    """
    # Full sequence: question + answer + full stop
    # The full stop teaches the graph where sequences end.
    # Over training the graph learns '.' terminates the answer naturally.
    q_atoms = Atoms.sequence("letter", list(question))
    a_atoms = Atoms.sequence("letter", list(" " + answer + "."))

    # Step 1 — answer path first (with full stop)
    for _ in range(answer_reps):
        system.ingest_atoms(a_atoms)
        system.reward(reward_scale * 3.0)

    # Step 2 — bridge: full sequence question + answer + stop
    for _ in range(bridge_reps):
        system.ingest_atoms(q_atoms + a_atoms)
        system.reward(reward_scale * 2.0)

    # Step 3 — confirm
    response = generate(system, question)
    score    = response_overlap(response, answer)

    if score > 0.2:
        # Good path — reward it
        system.ingest("letter", question)
        system.reward(reward_scale * score * 2.0)
    else:
        # Bad path — penalise it, re-strengthen answer and bridge
        system.ingest("letter", question)
        system.reward(-reward_scale * 0.5)
        # Re-do answer and bridge with extra strength
        for _ in range(answer_reps):
            system.ingest_atoms(a_atoms)
            system.reward(reward_scale * 4.0)
        system.ingest_atoms(q_atoms + a_atoms)
        system.reward(reward_scale * 3.0)

    return score


# ══════════════════════════════════════════════════════════════════════════════
# READINESS MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ReadinessMonitor:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self._holdout: list[tuple[str,str]] = []

    def set_holdout(self, pairs: list[tuple[str,str]], n: int = 200) -> None:
        self._holdout = random.sample(pairs, min(n, len(pairs)))

    def snapshot(self, system: Primeval) -> dict:
        if not self._holdout:
            return {"qa_rate": 0.0}
        sample = random.sample(self._holdout, min(50, len(self._holdout)))
        scores = []
        for q, a in sample:
            response = generate(system, q)
            scores.append(response_overlap(response, a))
        return {"qa_rate": float(np.mean(scores))}

    def is_ready(self, snap: dict) -> bool:
        return snap["qa_rate"] >= self.threshold


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Stage6Trainer:
    """
    Works through QA pairs using the three-step training loop.
    Pairs that fail confirmation get re-trained more aggressively.
    Consolidator is parked — no new structure discovery at this stage.
    """

    def __init__(
        self,
        system:       Primeval,
        squad_pairs:  list[tuple[str,str]],
        chat_pairs:   list[tuple[str,str]],
        reward_scale: float = 1.0,
        log_interval: int   = 2_000,
    ):
        self.system       = system
        self.all_pairs    = squad_pairs + chat_pairs
        self.reward_scale = reward_scale
        self.log_interval = log_interval
        self.monitor      = ReadinessMonitor()
        self._last_snap:  dict  = {}
        self._score_sum   = 0.0
        self._score_count = 0
        self._pair_idx    = 0

    def step(self) -> None:
        q, a = self.all_pairs[self._pair_idx % len(self.all_pairs)]
        self._pair_idx += 1
        score = train_pair(
            self.system, q, a,
            reward_scale = self.reward_scale,
            answer_reps  = 5,
            bridge_reps  = 3,
        )
        self._score_sum   += score
        self._score_count += 1

    def run(self, total_steps: int) -> bool:
        logger.info("Stage 6 starting. Steps=%d  Pairs=%d",
                    total_steps, len(self.all_pairs))
        t0 = time.monotonic()

        for step in range(1, total_steps + 1):
            self.step()

            if step % self.log_interval == 0:
                self._last_snap = self.monitor.snapshot(self.system)
                elapsed = time.monotonic() - t0
                mean_q  = (self._score_sum / self._score_count
                           if self._score_count else 0.0)
                stat    = self.system.stats()

                logger.info(
                    "Step %6d | %.1fs | nodes=%d edges=%d max_lv=%d",
                    step, elapsed,
                    stat["nodes"], stat["edges"], stat["max_level"],
                )
                logger.info(
                    "  holdout_qa=%.2f (need %.2f) | train_quality=%.2f",
                    self._last_snap.get("qa_rate", 0.0),
                    self.monitor.threshold,
                    mean_q,
                )

                # Sample responses
                logger.info("  Sample responses (holdout):")
                if self.monitor._holdout:
                    for q, a in random.sample(
                            self.monitor._holdout,
                            min(5, len(self.monitor._holdout))):
                        resp  = generate(self.system, q)
                        score = response_overlap(resp, a)
                        logger.info("    [%.2f] Q: %r", score, q[:40])
                        logger.info("           A: %r", resp[:60])
                        logger.info("           X: %r", a[:60])

                if self.monitor.is_ready(self._last_snap):
                    logger.info("Stage 6 READY at step %d", step)
                    return True

        logger.info("Stage 6 ended. holdout_qa=%.2f",
                    self._last_snap.get("qa_rate", 0.0))
        return False


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive(system: Primeval) -> None:
    print("\n" + "="*60)
    print("  Little Deepak — Interactive Session")
    print("="*60)
    print("  <prompt>      get a response")
    print("  +             reward last exchange")
    print("  -             penalise last exchange")
    print("  c: <text>     teach correct answer")
    print("  s             graph stats")
    print("  q             quit and save")
    print("="*60 + "\n")

    last_q  = ""
    last_a  = ""
    rewards = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if not user_input:
            continue

        if user_input.lower() == "q":
            break

        elif user_input == "s":
            s = system.stats()
            print(f"  nodes={s['nodes']}  edges={s['edges']}  "
                  f"max_lv={s['max_level']}  step={s['step']}")

        elif user_input == "+":
            if last_q and last_a:
                # Reward the path that produced last_a from last_q
                a_atoms = Atoms.sequence("letter", list(last_a))
                q_atoms = Atoms.sequence("letter", list(last_q))
                system.ingest_atoms(a_atoms)
                system.reward(3.0)
                system.ingest_atoms(q_atoms + a_atoms)
                system.reward(2.0)
                system.ingest("letter", last_q)
                system.reward(2.0)
                rewards += 1
                print(f"  [rewarded — total: {rewards}]")

        elif user_input == "-":
            if last_q:
                system.ingest("letter", last_q)
                system.reward(-1.0)
                print("  [penalised]")

        elif user_input.startswith("c:"):
            correction = user_input[2:].strip()
            if correction and last_q:
                # Three-step teach with correction
                score = train_pair(
                    system, last_q, correction,
                    reward_scale=2.0,
                    answer_reps=10,
                    bridge_reps=5,
                )
                rewards += 1
                print(f"  [learned: {correction!r}  score={score:.2f}]")
            else:
                print("  [usage: c: <correct answer>]")

        else:
            last_q = user_input
            last_a = generate(system, user_input)
            result = system.infer("letter", user_input)
            print(f"  Little Deepak: {last_a}")
            print(f"  [L{result.anchor_level}  conf={result.confidence:.0f}]")

    print(f"\n  Session done. Rewards given: {rewards}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Little Deepak -- Stage 6")
    parser.add_argument("--steps",        type=int,   default=500_000)
    parser.add_argument("--log-interval", type=int,   default=2_000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint",   type=str,   default="./checkpoints/stage6")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--from-stage5",  action="store_true")
    parser.add_argument("--interactive",  action="store_true")
    parser.add_argument("--squad-pairs",  type=int,   default=50_000)
    args = parser.parse_args()

    cfg = Config(
        window_size                 = 6,
        occurrence_delta            = 1.0,
        reward_multiplier           = 3.0,
        downward_growth_delta       = 0.02,   # gentle — structure is built
        consolidator_interval       = 50_000, # parked — no new discovery
        consolidator_proposal_scale = 0.01,
        consolidator_budget         = 50,     # minimal
        counts_decay_rate           = 0.001,  # slow — preserve QA paths
        decay_interval              = 20_000, # slow decay
        base_decay_rate             = 2e-4,
        level_decay_factor          = 1.3,    # gentler level penalty
        checkpoint_dir              = args.checkpoint,
        checkpoint_interval         = 25_000,
    )

    system = Primeval(cfg)

    if args.resume:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint, "step_*")))
        if ckpts:
            system.load(ckpts[-1])
            logger.info("Resumed from %s", ckpts[-1])
        else:
            logger.warning("No stage6 checkpoint -- starting fresh")
    elif args.from_stage5:
        path = "./checkpoints/stage5/final"
        if os.path.exists(path):
            system.load(path)
            logger.info("Loaded Stage 5 checkpoint")
        else:
            logger.warning("Stage 5 not found -- starting fresh")

    if args.interactive:
        run_interactive(system)
        system.save(args.checkpoint + "/final")
        logger.info("Saved to %s/final", args.checkpoint)
        return

    # Load datasets
    squad_pairs = load_squad(max_pairs=args.squad_pairs)
    chat_pairs  = load_nps_chat()

    random.shuffle(squad_pairs)
    holdout_n   = min(500, len(squad_pairs) // 10)
    holdout     = squad_pairs[:holdout_n]
    squad_train = squad_pairs[holdout_n:]

    trainer = Stage6Trainer(
        system       = system,
        squad_pairs  = squad_train,
        chat_pairs   = chat_pairs,
        reward_scale = args.reward_scale,
        log_interval = args.log_interval,
    )
    trainer.monitor.set_holdout(holdout)

    ready = trainer.run(args.steps)
    system.save(args.checkpoint + "/final")
    logger.info("Saved to %s/final", args.checkpoint)

    print("\n-- Stage 6 Final Summary ----------------------------------------")
    for k, v in system.stats().items():
        print(f"  {k}: {v}")

    print("\n-- Holdout QA sample --------------------------------------------")
    if trainer.monitor._holdout:
        scores = []
        for q, a in random.sample(trainer.monitor._holdout,
                                   min(15, len(trainer.monitor._holdout))):
            resp  = generate(system, q)
            score = response_overlap(resp, a)
            scores.append(score)
            flag  = "+" if score > 0.2 else "."
            print(f"  {flag} [{score:.2f}] Q: {q[:45]!r}")
            print(f"         X: {a[:55]!r}")
            print(f"         A: {resp[:55]!r}")
        print(f"\n  Mean holdout score: {np.mean(scores):.2f} (need 0.35)")

    print(f"\n  Stage 6 {'COMPLETE' if ready else 'INCOMPLETE (step limit reached)'}")
    print(f"\n  python stage6.py --resume --interactive")


if __name__ == "__main__":
    main()