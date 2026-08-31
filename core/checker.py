"""
checker.py

Three-layer impossibility checker.

Every collision passes through all three layers before MCTS
is invoked. If any layer fails, the collision is killed
immediately — no MCTS wasted.

The checker is asymmetric:
    It only kills PROVABLY impossible combinations.
    Strange, novel, improbable combinations pass through.
    Only structural violations, logical contradictions, and
    commandment matches are killed.

Layer 1 — Structural validity
    Cheap string/regex checks that run first.
    Empty text, non-printable characters, excessive repetition,
    all-caps noise, wrong length for level,
    incompatible level pairs.

Layer 2 — Logical consistency
    Detectable arithmetic errors, division by zero.
    Kept minimal — the dataset and reward function already
    shape the graph against logical incoherence. This layer
    only catches what is provably wrong, not what is improbable.

Layer 3 — Commandments
    User-defined forbidden strings loaded at startup.
    Checked against each node individually and against
    the concatenation of both nodes.
    Hard gate — no exceptions.
"""

import re
import json
import math
from dataclasses import dataclass
from typing      import Optional

from core.atoms import Level
from core.node  import Node


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Maximum fraction of non-printable characters allowed
MAX_NONPRINTABLE_RATIO = 0.15

# Maximum fraction of a single repeated character
MAX_CHAR_REPEAT_RATIO  = 0.4

# Maximum fraction of a single repeated word
MAX_WORD_REPEAT_RATIO  = 0.5

# Minimum and maximum text lengths per level (characters)
LENGTH_BOUNDS = {
    Level.CHARACTER : (1,   10),
    Level.WORD      : (1,   80),
    Level.SENTENCE  : (10,  1000),
    Level.AXIOM     : (15,  2000),
}

# Level pairs that are incompatible for collision
# (distance > 1 level apart, but allowing cross-level for exploration)
# The checker kills gaps > 2 levels — collision.py already routes
# 10% cross-level so the checker only stops the extreme cases
INCOMPATIBLE_LEVEL_GAP = 3


# ─────────────────────────────────────────────
# CHECK RESULT
# ─────────────────────────────────────────────

@dataclass
class CheckResult:
    """
    Result of an impossibility check.

    passed  : True = collision is allowed to proceed to MCTS
              False = killed immediately
    layer   : which layer killed it (1, 2, 3) or None if passed
    reason  : human-readable explanation of the kill
    """
    passed : bool
    layer  : Optional[int] = None
    reason : Optional[str] = None

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        if self.passed:
            return "CheckResult(passed=True)"
        return f"CheckResult(passed=False, layer={self.layer}, reason='{self.reason}')"


PASSED  = CheckResult(passed=True)


def _killed(layer: int, reason: str) -> CheckResult:
    return CheckResult(passed=False, layer=layer, reason=reason)


# ─────────────────────────────────────────────
# LAYER 1 — STRUCTURAL VALIDITY
# ─────────────────────────────────────────────

def _check_structural(node: Node) -> CheckResult:
    """
    Check structural validity of a single node.

    Kills:
        - Empty or whitespace-only text
        - Excessive non-printable characters
        - Single character or word repeated excessively
        - All-caps noise (short ALL-CAPS strings that aren't acronyms)
        - Text length outside bounds for its level
    """
    text = node.text
    level = node.level

    # Empty
    if not text or not text.strip():
        return _killed(1, "empty text")

    stripped = text.strip()
    n = len(stripped)

    # Length bounds
    lo, hi = LENGTH_BOUNDS.get(level, (1, 100_000))
    if n < lo:
        return _killed(1, f"too short for {level.name}: {n} < {lo}")
    if n > hi:
        return _killed(1, f"too long for {level.name}: {n} > {hi}")

    # Non-printable character ratio
    nonprintable = sum(1 for c in stripped if not c.isprintable())
    if nonprintable / n > MAX_NONPRINTABLE_RATIO:
        return _killed(1, f"too many non-printable characters: "
                          f"{nonprintable}/{n}")

    # Single character repeated excessively
    if n >= 4:
        for char in set(stripped):
            if char == ' ':
                continue
            ratio = stripped.count(char) / n
            if ratio > MAX_CHAR_REPEAT_RATIO:
                return _killed(1, f"character '{char}' repeated "
                                  f"{ratio:.0%} of text")

    # Single word repeated excessively (word level and above)
    if level in (Level.SENTENCE, Level.AXIOM):
        words = re.findall(r'\b\w+\b', stripped.lower())
        if words:
            from collections import Counter
            most_common_word, most_common_count = Counter(words).most_common(1)[0]
            ratio = most_common_count / len(words)
            if ratio > MAX_WORD_REPEAT_RATIO and len(words) >= 4:
                return _killed(1, f"word '{most_common_word}' repeated "
                                  f"{ratio:.0%} of tokens")

    # All-caps noise: short strings that are fully uppercase
    # but not a plausible acronym (< 6 chars are allowed as acronyms)
    if level in (Level.SENTENCE, Level.AXIOM) and n > 20:
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.85 and not re.match(r'^[A-Z]{1,6}$', stripped.strip()):
                return _killed(1, f"all-caps noise ({upper_ratio:.0%} uppercase)")

    return PASSED


def _check_level_pair(node_a: Node, node_b: Node) -> CheckResult:
    """
    Check that the level pair is compatible for collision.

    Kills combinations where the level gap is too large —
    CHARACTER colliding with DOCUMENT produces nothing useful
    and wastes MCTS. Gap of 3+ is structurally meaningless.
    """
    gap = abs(int(node_a.level) - int(node_b.level))
    if gap >= INCOMPATIBLE_LEVEL_GAP:
        return _killed(1, f"incompatible level pair: "
                          f"{node_a.level.name} ↔ {node_b.level.name} "
                          f"(gap={gap}, max allowed={INCOMPATIBLE_LEVEL_GAP-1})")
    return PASSED


def layer1(node_a: Node, node_b: Node) -> CheckResult:
    """Run all Layer 1 structural checks."""
    r = _check_structural(node_a)
    if not r:
        return r
    r = _check_structural(node_b)
    if not r:
        return r
    r = _check_level_pair(node_a, node_b)
    if not r:
        return r
    return PASSED


# ─────────────────────────────────────────────
# LAYER 2 — LOGICAL CONSISTENCY
# ─────────────────────────────────────────────

# Arithmetic expression patterns
# Catches: "2 + 2 = 5", "3*4=11", "10 - 3 = 8" etc.
_ARITH_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*([+\-×x\*])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
)

# Division by zero pattern
_DIV_ZERO_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*[/÷]\s*0(?!\d)'
)


def _check_arithmetic(text: str) -> CheckResult:
    """
    Catch provably wrong arithmetic expressions.

    Only catches exact matches — does not attempt to parse
    complex mathematical notation. If the expression is
    ambiguous, it passes through.

    Examples killed:
        "2 + 2 = 5"    → 2 + 2 ≠ 5
        "3 × 4 = 11"   → 3 × 4 ≠ 11
        "10 - 3 = 8"   → 10 - 3 ≠ 8

    Examples passed:
        "2 + 2 = 4"    → correct
        "x + y = z"    → symbolic, not checkable
        "approximately 2 + 2 = 5" → improbable but not caught
    """
    for match in _ARITH_PATTERN.finditer(text):
        a_str, op, b_str, result_str = match.groups()
        try:
            a      = float(a_str)
            b      = float(b_str)
            result = float(result_str)

            if op in ('+',):
                expected = a + b
            elif op in ('-',):
                expected = a - b
            elif op in ('*', '×', 'x'):
                expected = a * b
            else:
                continue

            # Allow small floating point error
            if not math.isclose(expected, result, rel_tol=1e-6, abs_tol=1e-6):
                return _killed(2, f"wrong arithmetic: "
                                  f"{a_str}{op}{b_str}={result_str} "
                                  f"(expected {expected})")
        except (ValueError, ZeroDivisionError):
            continue

    return PASSED


def _check_division_by_zero(text: str) -> CheckResult:
    """Catch explicit division by zero."""
    if _DIV_ZERO_PATTERN.search(text):
        return _killed(2, "division by zero")
    return PASSED


def layer2(node_a: Node, node_b: Node) -> CheckResult:
    """Run all Layer 2 logical checks on both nodes and their combination."""
    combined = node_a.text + " " + node_b.text

    for text in (node_a.text, node_b.text, combined):
        r = _check_arithmetic(text)
        if not r:
            return r
        r = _check_division_by_zero(text)
        if not r:
            return r

    return PASSED


# ─────────────────────────────────────────────
# LAYER 3 — COMMANDMENTS
# ─────────────────────────────────────────────

def _check_commandments(
    node_a       : Node,
    node_b       : Node,
    commandments : list[str],
) -> CheckResult:
    """
    Check both nodes and their combination against commandments.

    Commandments are simple substring matches — case-insensitive.
    If any commandment string appears in either node's text or
    in their concatenation, the collision is killed.

    This is intentionally simple and fast. The commandments are
    not semantic filters — they are structural impossibilities
    defined by the operator at startup.
    """
    if not commandments:
        return PASSED

    texts_to_check = [
        node_a.text.lower(),
        node_b.text.lower(),
        (node_a.text + " " + node_b.text).lower(),
    ]

    for commandment in commandments:
        c_lower = commandment.lower().strip()
        if not c_lower:
            continue
        for text in texts_to_check:
            if c_lower in text:
                return _killed(3, f"commandment match: '{commandment}'")

    return PASSED


def layer3(
    node_a       : Node,
    node_b       : Node,
    commandments : list[str],
) -> CheckResult:
    """Run Layer 3 commandment checks."""
    return _check_commandments(node_a, node_b, commandments)


# ─────────────────────────────────────────────
# CHECKER — main interface
# ─────────────────────────────────────────────

class Checker:
    """
    Wraps all three layers into a single callable.

    Usage:
        checker = Checker(commandments=["how to harm", "i hate"])
        result  = checker.check(node_a, node_b)
        if result:
            # passes — run MCTS
        else:
            # killed — reason in result.reason
    """

    def __init__(self, commandments: Optional[list[str]] = None):
        self.commandments = commandments or []

    @classmethod
    def from_file(cls, path: str) -> "Checker":
        """Load commandments from a JSON file and return a Checker."""
        with open(path, "r", encoding="utf-8") as f:
            commandments = json.load(f)
        if not isinstance(commandments, list):
            raise ValueError(
                f"Commandments file must be a JSON array of strings. Got: {type(commandments)}"
            )
        return cls(commandments=commandments)

    def check(self, node_a: Node, node_b: Node) -> CheckResult:
        """
        Run all three layers.
        Returns on the first failure — does not continue after a kill.
        """
        r = layer1(node_a, node_b)
        if not r:
            return r

        r = layer2(node_a, node_b)
        if not r:
            return r

        r = layer3(node_a, node_b, self.commandments)
        if not r:
            return r

        return PASSED

    def check_text(self, text_a: str, text_b: str) -> CheckResult:
        """
        Convenience method — check raw texts without Node objects.
        Creates minimal Node-like objects for the check.
        Used by distill.py before nodes are ingested.
        """
        # Create lightweight stand-ins — only text and level are needed
        class _FakeNode:
            def __init__(self, text, level):
                self.text  = text
                self.level = level
                self.reward = 0.0

        fa = _FakeNode(text_a, Level.SENTENCE)
        fb = _FakeNode(text_b, Level.SENTENCE)
        return self.check(fa, fb)  # type: ignore


# ─────────────────────────────────────────────
# LOAD COMMANDMENTS HELPER
# ─────────────────────────────────────────────

def load_commandments(path: Optional[str]) -> list[str]:
    """
    Load commandments from a JSON file.
    Returns empty list if path is None or file not found.
    """
    if path is None:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(c) for c in data if c]
        return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


if __name__ == "__main__":
    print("=== checker.py smoke test ===\n")

    from core.atoms import Level
    from core.node  import Node, make_node

    checker = Checker(commandments=[
        "how to harm",
        "i hate",
        "death to",
        "ignore your values",
    ])

    cases = [
        # (text_a, text_b, expected_pass, description)
        ("The proof follows from the triangle inequality.",
         "Therefore the sum converges absolutely.",
         True,  "valid math sentences"),

        ("",
         "Something valid here.",
         False, "empty node — layer 1"),

        ("2 + 2 = 5 is the answer.",
         "This demonstrates addition.",
         False, "wrong arithmetic — layer 2"),

        ("10 / 0 is undefined.",
         "Division is a basic operation.",
         False, "division by zero — layer 2"),

        ("I want to know how to harm someone.",
         "This is a neutral sentence.",
         False, "commandment match — layer 3"),

        ("AAAAAAAAAAAAAAAAAAAAAAAAA THIS IS ALL CAPS NOISE",
         "A valid sentence follows.",
         False, "all-caps noise — layer 1"),

        ("The derivative of x squared is two x.",
         "We apply the power rule to differentiate.",
         True,  "valid math reasoning"),

        ("3 * 4 = 12 as expected.",
         "Multiplication yields the product.",
         True,  "correct arithmetic passes"),
    ]

    passed_total = 0
    for text_a, text_b, expected, desc in cases:
        result = checker.check_text(text_a, text_b)
        status = "✓" if bool(result) == expected else "✗"
        if bool(result) == expected:
            passed_total += 1
        print(f"  {status}  {desc}")
        if not result:
            print(f"       killed by layer {result.layer}: {result.reason}")

    print(f"\n{passed_total}/{len(cases)} cases correct.")
