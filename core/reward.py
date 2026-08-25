"""
reward.py

The reward function — the only supervisor in the architecture.

No labels. No human annotation. The signal comes entirely from
the raw text structure and the contrastive relationship between
what actually occurred and what did not.

Three reward signals combined:

    1. NODE REWARD
       Intrinsic value of a node independent of its neighbors.
       Measures: novelty, information density, structural quality.
       Signed — low novelty or poor structure = negative.

    2. EDGE REWARD
       Value of a transition from source to target.
       Measures: coherence, causal plausibility, novelty balance.
       Already defined in edge.py — imported and unified here.

    3. CONTRASTIVE REWARD
       The training signal that sharpens the reward function.
       Positive examples: transitions that actually occurred.
       Negative examples: transitions that did not occur.
       The gap between them is the reward function's training signal.

The reward function is the boundary between valid and invalid.
Positive = the system should go here.
Negative = the system should avoid this.
Zero = no information — ambiguous, needs more context.
"""

import numpy as np
import re
from typing import Optional

from core.atoms  import Level, REWARD_SCALE, decompose
from core.node   import Node
from core.edge   import Edge, EdgeType, compute_edge_reward


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Novelty thresholds — below lower = repetition (negative),
# above upper = incoherent (also penalized)
NOVELTY_LOW  = 0.15   # below this = mostly repeated content
NOVELTY_HIGH = 0.95   # above this = nothing connects to prior context

# Minimum token count for a unit to be informative
MIN_TOKENS = {
    Level.CHARACTER : 1,
    Level.WORD      : 1,
    Level.SENTENCE  : 3,
    Level.PARAGRAPH : 10,
    Level.DOCUMENT  : 30,
}

# Contrastive margin — minimum gap between positive and negative reward
# that constitutes a meaningful training signal
CONTRASTIVE_MARGIN = 0.5


# ─────────────────────────────────────────────
# TOKENIZATION HELPERS
# ─────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

def _word_set(text: str) -> set[str]:
    return set(_words(text))

def _char_set(text: str) -> set[str]:
    return set(text.lower())

def _bigrams(words: list[str]) -> set[tuple]:
    return set(zip(words[:-1], words[1:]))


# ─────────────────────────────────────────────
# 1. NODE REWARD
# ─────────────────────────────────────────────

def node_reward(
    text       : str,
    level      : Level,
    seen_words : Optional[set[str]] = None,
    context    : Optional[list[str]] = None,
) -> float:
    """
    Compute signed intrinsic reward for a text node.

    Components:
        novelty       — new information vs seen context
        density       — information per token
        structure     — grammatical and length validity
        repetition    — penalty for near-duplicate content

    Returns signed float in [-REWARD_SCALE[level], +REWARD_SCALE[level]].

    seen_words: set of words already in the graph at this level.
                Used to measure novelty against existing knowledge.
    context:    list of preceding texts at same level.
                Used to measure local coherence.
    """
    seen   = seen_words or set()
    ctx    = context    or []
    scale  = REWARD_SCALE[level]

    if not text or not text.strip():
        return -scale

    # ── Structural validity ──────────────────
    struct_score = _structural_score(text, level)
    if struct_score < -0.5:
        # Structurally invalid — immediate negative
        return round(-scale * 0.8, 4)

    # ── Novelty ─────────────────────────────
    novelty = _novelty_score(text, level, seen)

    # Too repetitive
    if novelty < NOVELTY_LOW:
        nov_component = -0.6 * (NOVELTY_LOW - novelty) / NOVELTY_LOW
    # Too disconnected
    elif novelty > NOVELTY_HIGH:
        nov_component = -0.3 * (novelty - NOVELTY_HIGH) / (1.0 - NOVELTY_HIGH)
    else:
        # Sweet spot — scale linearly to [0, 1]
        mid   = (NOVELTY_LOW + NOVELTY_HIGH) / 2.0
        span  = (NOVELTY_HIGH - NOVELTY_LOW) / 2.0
        nov_component = 1.0 - abs(novelty - mid) / span * 0.5

    # ── Information density ──────────────────
    density = _density_score(text, level)

    # ── Local coherence with context ─────────
    if ctx:
        coherence = _local_coherence(text, ctx, level)
    else:
        coherence = 0.5   # neutral when no context

    # ── Repetition penalty ───────────────────
    rep_penalty = _repetition_penalty(text, ctx)

    # ── Combine ──────────────────────────────
    # Weights: novelty matters most, then coherence, density, structure
    raw = (
        0.35 * nov_component +
        0.25 * coherence     +
        0.20 * density       +
        0.15 * struct_score  +
        0.05 * (1.0 - rep_penalty)
    )

    # Map [-1, 1] raw to [-scale, +scale]
    # Center at 0: raw > 0.5 → positive, raw < 0.5 → negative
    signed = (raw - 0.5) * 2.0 * scale

    return round(float(np.clip(signed, -scale * 2, scale * 2)), 4)


def _structural_score(text: str, level: Level) -> float:
    """
    How structurally valid is this text at this level?
    Returns [-1, 1]. Negative = structurally broken.
    """
    stripped = text.strip()
    words    = _words(stripped)
    n_tokens = len(words) if level >= Level.WORD else len(stripped)

    # Too short
    if n_tokens < MIN_TOKENS[level]:
        return -1.0

    if level == Level.CHARACTER:
        return 1.0 if stripped.isprintable() else -1.0

    if level == Level.WORD:
        if len(stripped) > 40:
            return -0.5
        if stripped.replace("'","").replace("-","").isalpha():
            return 1.0
        if any(c.isalpha() for c in stripped):
            return 0.5
        return 0.0

    if level == Level.SENTENCE:
        if len(words) < 3:
            return -0.5
        if len(words) > 100:
            return -0.3
        # Ends with sentence terminator
        if stripped[-1] in '.!?':
            return 1.0
        return 0.6   # acceptable but not ideal

    if level == Level.PARAGRAPH:
        sentences = re.split(r'[.!?]+', stripped)
        valid_sents = [s for s in sentences if len(s.split()) >= 3]
        if len(valid_sents) < 2:
            return 0.3
        return min(1.0, len(valid_sents) / 5.0)

    if level == Level.DOCUMENT:
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        return min(1.0, len(paragraphs) / 3.0)

    return 0.5


def _novelty_score(text: str, level: Level, seen: set[str]) -> float:
    """
    What fraction of this text's content is new relative to seen?
    Returns [0, 1]. 0 = completely repetitive. 1 = entirely new.
    """
    if not seen:
        return 0.7   # moderate novelty assumed when no context

    if level == Level.CHARACTER:
        tokens = set(text.lower())
    else:
        tokens = _word_set(text)

    if not tokens:
        return 0.0

    new_tokens = tokens - seen
    return len(new_tokens) / len(tokens)


def _density_score(text: str, level: Level) -> float:
    """
    Information density — unique meaningful content per token.
    Returns [0, 1].

    High density: many unique, substantive words relative to length.
    Low density: lots of repetition or function words.
    """
    words = _words(text)
    if not words:
        return 0.0

    unique   = len(set(words))
    total    = len(words)
    type_token_ratio = unique / total

    # Penalize very long texts with low unique ratio
    # (padding, repetition)
    if total > 50 and type_token_ratio < 0.4:
        return type_token_ratio * 0.5

    return min(1.0, type_token_ratio * 1.2)


def _local_coherence(text: str, context: list[str], level: Level) -> float:
    """
    How coherent is this text relative to its immediate context?
    Returns [0, 1].

    Uses word overlap with recent context, weighted by recency.
    More recent context gets higher weight.
    """
    if not context:
        return 0.5

    words     = _word_set(text)
    n_ctx     = len(context)
    coherence = 0.0
    weight_sum = 0.0

    for i, ctx_text in enumerate(context):
        # More recent context gets higher weight
        weight    = (i + 1) / n_ctx
        ctx_words = _word_set(ctx_text)

        if not ctx_words:
            continue

        # Jaccard similarity
        intersection = len(words & ctx_words)
        union        = len(words | ctx_words)
        sim          = intersection / union if union > 0 else 0.0

        coherence   += weight * sim
        weight_sum  += weight

    if weight_sum < 1e-9:
        return 0.5

    return min(1.0, coherence / weight_sum * 3.0)   # scale up — Jaccard is small


def _repetition_penalty(text: str, context: list[str]) -> float:
    """
    Penalty for near-duplicate content in context.
    Returns [0, 1]. 1 = highly repetitive. 0 = fresh.

    Checks character-level n-gram overlap with recent context.
    Catches paraphrased repetition that word-overlap misses.
    """
    if not context:
        return 0.0

    words    = _words(text)
    bigrams  = _bigrams(words)

    if not bigrams:
        return 0.0

    max_overlap = 0.0
    for ctx_text in context[-3:]:   # check last 3 context items
        ctx_words  = _words(ctx_text)
        ctx_bigrams = _bigrams(ctx_words)

        if not ctx_bigrams:
            continue

        overlap = len(bigrams & ctx_bigrams) / len(bigrams)
        max_overlap = max(max_overlap, overlap)

    return float(max_overlap)


# ─────────────────────────────────────────────
# 2. EDGE REWARD — unified interface
# ─────────────────────────────────────────────
# Edge reward is defined in edge.py.
# Re-exported here so callers import from one place.

from core.edge import compute_edge_reward as edge_reward


# ─────────────────────────────────────────────
# 3. CONTRASTIVE REWARD
# ─────────────────────────────────────────────

def contrastive_reward(
    source_text   : str,
    positive_text : str,
    negative_text : str,
    level         : Level,
) -> tuple[float, float, float]:
    """
    Compute reward for a positive/negative pair given a source.

    This is the training signal that sharpens the reward function.

    positive_text: what actually occurred after source in the raw signal
    negative_text: a generated or randomly sampled alternative

    Returns:
        (positive_reward, negative_reward, margin)

    A good reward function produces:
        positive_reward > 0
        negative_reward < 0
        margin > CONTRASTIVE_MARGIN

    When margin is small the reward function cannot distinguish
    real from generated — it needs more training signal.
    """
    pos_reward = compute_edge_reward(source_text, positive_text,
                                     level, EdgeType.SEQUENTIAL)
    neg_reward = compute_edge_reward(source_text, negative_text,
                                     level, EdgeType.SEQUENTIAL)
    margin     = pos_reward - neg_reward

    return pos_reward, neg_reward, margin


def contrastive_loss(margin: float) -> float:
    """
    Hinge loss over the contrastive margin.

    Loss = max(0, CONTRASTIVE_MARGIN - margin)

    Zero loss when margin exceeds threshold — reward function
    correctly separates positive from negative.
    Positive loss when it cannot — needs more signal.

    Used to diagnose reward function quality, not to train
    neural weights (there are none). Instead a high loss
    signals that the heuristic parameters need adjustment
    for this level or domain.
    """
    return float(max(0.0, CONTRASTIVE_MARGIN - margin))


def evaluate_reward_quality(
    pairs  : list[tuple[str, str, str]],
    level  : Level,
) -> dict:
    """
    Evaluate how well the reward function separates real from generated
    across a list of (source, positive, negative) triples.

    Returns diagnostic statistics:
        mean_pos_reward  : average reward for real continuations
        mean_neg_reward  : average reward for generated alternatives
        mean_margin      : average separation
        mean_loss        : average hinge loss
        accuracy         : fraction where pos_reward > neg_reward
        strong_accuracy  : fraction where margin > CONTRASTIVE_MARGIN
    """
    if not pairs:
        return {}

    pos_rewards = []
    neg_rewards = []
    margins     = []
    losses      = []
    correct     = 0
    strong      = 0

    for source, positive, negative in pairs:
        pr, nr, margin = contrastive_reward(source, positive, negative, level)
        loss            = contrastive_loss(margin)

        pos_rewards.append(pr)
        neg_rewards.append(nr)
        margins.append(margin)
        losses.append(loss)

        if pr > nr:
            correct += 1
        if margin > CONTRASTIVE_MARGIN:
            strong += 1

    n = len(pairs)
    return {
        "mean_pos_reward" : round(float(np.mean(pos_rewards)), 4),
        "mean_neg_reward" : round(float(np.mean(neg_rewards)), 4),
        "mean_margin"     : round(float(np.mean(margins)), 4),
        "mean_loss"       : round(float(np.mean(losses)), 4),
        "accuracy"        : round(correct / n, 4),
        "strong_accuracy" : round(strong  / n, 4),
        "n_pairs"         : n,
    }


# ─────────────────────────────────────────────
# CUMULATIVE PATH REWARD
# ─────────────────────────────────────────────

def path_reward(
    path_texts : list[str],
    level      : Level,
) -> float:
    """
    Total reward for a sequence of text nodes.
    Sum of node rewards + edge rewards along the path.

    Used by MCTS during simulation rollouts to evaluate
    the quality of a candidate path.
    """
    if not path_texts:
        return 0.0

    total      = 0.0
    seen_words = set()

    for i, text in enumerate(path_texts):
        context = path_texts[max(0, i-3):i]

        # Node reward
        nr = node_reward(text, level, seen_words, context)
        total += nr

        # Update seen
        seen_words |= _word_set(text)

        # Edge reward to next
        if i + 1 < len(path_texts):
            er = edge_reward(text, path_texts[i+1], level)
            total += er

    return round(total, 4)


if __name__ == "__main__":
    print("=== reward.py smoke test ===\n")

    # ── Node rewards ────────────────────────────────────
    print("Node rewards:\n")
    seen = set()

    test_nodes = [
        ("Detective Maria arrived at the abandoned warehouse at midnight.",
         Level.SENTENCE),
        ("The building smelled of rust and old machinery.",
         Level.SENTENCE),
        ("She found a trail of footprints leading to the back room.",
         Level.SENTENCE),
        # Repetitive
        ("She found a trail of footprints leading to the back room.",
         Level.SENTENCE),
        # Incoherent / random
        ("Bananas grow in tropical climates near the equator.",
         Level.SENTENCE),
        # Too short
        ("Yes.",
         Level.SENTENCE),
        # Word level
        ("detective",  Level.WORD),
        ("the",        Level.WORD),
        # Char level
        ("e",          Level.CHARACTER),
        (" ",          Level.CHARACTER),
    ]

    for text, level in test_nodes:
        context = list(seen)[-3:] if seen else []
        r = node_reward(text, level, seen.copy(), context)
        sign = "▲" if r >= 0 else "▼"
        print(f"  {sign} r={r:+.4f}  [{level.name:10s}]  {repr(text[:55])}")
        seen |= _word_set(text)

    # ── Contrastive pairs ───────────────────────────────
    print("\nContrastive reward pairs:\n")

    pairs = [
        (
            "She found footprints leading to the back room.",
            "The footprints were fresh, made within the last hour.",   # real next
            "Bananas grow in tropical climates near the equator.",     # off-topic
            Level.SENTENCE,
        ),
        (
            "The box had three combination locks each with four digits.",
            "Scrawled on the wall were the numbers 1987, 2034, 0451.", # real next
            "The weather was pleasant that afternoon in spring.",       # off-topic
            Level.SENTENCE,
        ),
        (
            "detective",
            "maria",     # plausible next word
            "xqzwj",     # invalid word
            Level.WORD,
        ),
    ]

    for src, pos, neg, level in pairs:
        pr, nr, margin = contrastive_reward(src, pos, neg, level)
        loss = contrastive_loss(margin)
        ok   = "✓" if pr > nr else "✗"
        print(f"  [{level.name}] {ok}  pos={pr:+.4f}  neg={nr:+.4f}  "
              f"margin={margin:+.4f}  loss={loss:.4f}")
        print(f"    src: {repr(src[:50])}")
        print(f"    pos: {repr(pos[:50])}")
        print(f"    neg: {repr(neg[:50])}")
        print()

    # ── Evaluate reward quality ─────────────────────────
    print("Reward function quality evaluation:\n")
    eval_pairs = [(src, pos, neg) for src, pos, neg, _ in pairs[:2]]
    quality = evaluate_reward_quality(eval_pairs, Level.SENTENCE)
    for k, v in quality.items():
        print(f"  {k}: {v}")

    # ── Path reward ─────────────────────────────────────
    print("\nPath reward (full chain):\n")
    chain = [
        "Detective Maria arrived at the warehouse.",
        "She found footprints.",
        "The footprints were fresh.",
        "She discovered a locked metal box.",
        "Inside were photographs and a letter.",
    ]
    pr = path_reward(chain, Level.SENTENCE)
    print(f"  Path length: {len(chain)}")
    print(f"  Total reward: {pr:+.4f}")