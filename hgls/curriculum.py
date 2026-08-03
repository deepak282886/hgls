"""
curriculum.py — Curriculum Controller (Little Deepak Edition)

Developmental stages mirror human literacy acquisition,
structured entirely around Little Deepak's world and values.
All training content is sourced from hgls/persona.py.

Stages:
  0  Characters          — a–z, the atomic building blocks
  1  Combinations        — roots of Little Deepak's key vocabulary
  2  Words               — his world: health, family, values, learning
  3  Phrases             — good habits stated simply and clearly
  4  Schemas             — cause and effect, values become reasoning
"""

from enum import IntEnum
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library

from hgls.structures import GenerativeStructure
import hgls.persona as persona


class Stage(IntEnum):
    CHARACTERS   = 0
    COMBINATIONS = 1
    WORDS        = 2
    PHRASES      = 3
    SCHEMAS      = 4


STAGE_NAMES: Dict[int, str] = {
    0: "Characters",
    1: "Combinations",
    2: "Words — Little Deepak's World",
    3: "Phrases — Good Habits",
    4: "Schemas — Cause and Effect",
}

# Library successes needed at level N before advancing to N+1.
# Thresholds are deliberately high so each level consolidates
# before structures at the next level are built on top of it.
ADVANCEMENT_THRESHOLDS: Dict[Stage, int] = {
    Stage.CHARACTERS:   30,    # all 26 chars + common punctuation solid
    Stage.COMBINATIONS: 100,   # enough syllable patterns to support words
    Stage.WORDS:        200,   # rich word vocabulary before phrases
    Stage.PHRASES:      400,   # solid phrase library before schemas
}

# Space introduced only at word stage (as composition token)
_CHARS = list("abcdefghijklmnopqrstuvwxyz.,!?'-")
_ALL   = _CHARS + [' ']

# Curriculum content sourced entirely from persona
_STAGE_INPUTS = {
    Stage.CHARACTERS:   persona.LEVEL_1_CHARS,
    Stage.COMBINATIONS: persona.LEVEL_2_COMBINATIONS,
    Stage.WORDS:        persona.LEVEL_3_WORDS,
    Stage.PHRASES:      persona.LEVEL_4_PHRASES,
    Stage.SCHEMAS:      persona.LEVEL_5_SCHEMAS,
}


class CurriculumController:

    def __init__(self):
        self.current_stage = Stage.CHARACTERS
        self.stage_history = [Stage.CHARACTERS]
        self._cycles_at: Dict[int, int] = {0: 0}

    # ── Primitives and inputs ─────────────────────────────────────

    def get_primitives(self) -> List[str]:
        """Space is introduced only once words are stable."""
        return _ALL if self.current_stage >= Stage.WORDS else _CHARS

    def get_active_level(self) -> int:
        return int(self.current_stage)

    def get_training_inputs(self) -> List[str]:
        """Return the curriculum inputs appropriate for the current stage."""
        return list(_STAGE_INPUTS.get(self.current_stage, []))

    # ── Progression ───────────────────────────────────────────────

    def tick(self) -> None:
        lvl = int(self.current_stage)
        self._cycles_at[lvl] = self._cycles_at.get(lvl, 0) + 1

    def should_advance(self, library: 'Library') -> bool:
        if self.current_stage >= Stage.SCHEMAS:
            return False
        threshold = ADVANCEMENT_THRESHOLDS.get(self.current_stage, 9999)
        return library.success_count_at_level(int(self.current_stage)) >= threshold

    def advance(self) -> Stage:
        if self.current_stage < Stage.SCHEMAS:
            self.current_stage = Stage(int(self.current_stage) + 1)
            self.stage_history.append(self.current_stage)
            self._cycles_at[int(self.current_stage)] = 0
        return self.current_stage

    # ── Bootstrap ─────────────────────────────────────────────────

    def bootstrap_char_structures(self) -> List[GenerativeStructure]:
        """
        Seed the library with atomic character primitives before learning begins.
        These are the building blocks from which everything else grows.
        """
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
            'persona':         persona.NAME,
            'current_stage':   STAGE_NAMES[int(self.current_stage)],
            'stage_level':     int(self.current_stage),
            'cycles_at':       dict(self._cycles_at),
            'stage_history':   [STAGE_NAMES[int(s)] for s in self.stage_history],
        }