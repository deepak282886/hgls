"""
curriculum.py — Curriculum Controller

Developmental stages from characters to meta-reasoning.
The same algorithm runs at every level — only the content changes.
All training content is sourced from hgls/persona.py.

Stages:
  0  Characters       — a–z, the atomic building blocks
  1  Combinations     — roots of key vocabulary
  2  Words            — world, values, family, health
  3  Phrases          — good habits stated simply
  4  Schemas          — cause and effect
  5  Reasoning        — how to think, not just what to know
  6  Meta-Reasoning   — reasoning about reasoning (no ceiling)
"""

from enum import IntEnum
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library

from hgls.structures import GenerativeStructure
import hgls.persona as persona


class Stage(IntEnum):
    CHARACTERS     = 0
    COMBINATIONS   = 1
    WORDS          = 2
    PHRASES        = 3
    SCHEMAS        = 4
    REASONING      = 5
    META_REASONING = 6


STAGE_NAMES: Dict[int, str] = {
    0: "Characters",
    1: "Combinations",
    2: "Words — Little Deepak's World",
    3: "Phrases — Good Habits",
    4: "Schemas — Cause and Effect",
    5: "Reasoning Patterns",
    6: "Meta-Reasoning",
}

# Successes needed at level N before advancing to N+1
ADVANCEMENT_THRESHOLDS: Dict[Stage, int] = {
    Stage.CHARACTERS:     30,
    Stage.COMBINATIONS:   100,
    Stage.WORDS:          200,
    Stage.PHRASES:        400,
    Stage.SCHEMAS:        400,
    Stage.REASONING:      200,
    # META_REASONING: final stage — no threshold
}

_CHARS = list("abcdefghijklmnopqrstuvwxyz.,!?'-")
_ALL   = _CHARS + [' ']

# Seed content per stage — sourced entirely from persona
_STAGE_INPUTS = {
    Stage.CHARACTERS:     persona.LEVEL_1_CHARS,
    Stage.COMBINATIONS:   persona.LEVEL_2_COMBINATIONS,
    Stage.WORDS:          persona.LEVEL_3_WORDS,
    Stage.PHRASES:        persona.LEVEL_4_PHRASES,
    Stage.SCHEMAS:        persona.LEVEL_5_SCHEMAS,
    Stage.REASONING:      persona.REASONING_SEED_SCHEMAS,
    Stage.META_REASONING: persona.META_SEED_SCHEMAS,
}


class CurriculumController:

    def __init__(self):
        self.current_stage = Stage.CHARACTERS
        self.stage_history = [Stage.CHARACTERS]
        self._cycles_at: Dict[int, int] = {0: 0}

    # ── Primitives and inputs ─────────────────────────────────────

    def get_primitives(self) -> List[str]:
        """Space available from word level onward."""
        return _ALL if self.current_stage >= Stage.WORDS else _CHARS

    def get_active_level(self) -> int:
        return int(self.current_stage)

    def get_training_inputs(self) -> List[str]:
        return list(_STAGE_INPUTS.get(self.current_stage, []))

    # ── Progression ───────────────────────────────────────────────

    def tick(self) -> None:
        lvl = int(self.current_stage)
        self._cycles_at[lvl] = self._cycles_at.get(lvl, 0) + 1

    def should_advance(self, library: 'Library') -> bool:
        if self.current_stage >= Stage.META_REASONING:
            return False
        threshold = ADVANCEMENT_THRESHOLDS.get(self.current_stage, 9999)
        return library.success_count_at_level(int(self.current_stage)) >= threshold

    def advance(self) -> Stage:
        if self.current_stage < Stage.META_REASONING:
            self.current_stage = Stage(int(self.current_stage) + 1)
            self.stage_history.append(self.current_stage)
            self._cycles_at[int(self.current_stage)] = 0
        return self.current_stage

    # ── Bootstrap ─────────────────────────────────────────────────

    def bootstrap_char_structures(self) -> List[GenerativeStructure]:
        """Seed library with atomic character primitives."""
        return [
            GenerativeStructure(
                level=0,
                elements=[ch],
                source='bootstrap',
                fitness=1.0,
                description=f"primitive '{ch}'",
            )
            for ch in _ALL
        ]

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'persona':       persona.NAME,
            'current_stage': STAGE_NAMES[int(self.current_stage)],
            'stage_level':   int(self.current_stage),
            'cycles_at':     dict(self._cycles_at),
            'stage_history': [STAGE_NAMES[int(s)] for s in self.stage_history],
        }