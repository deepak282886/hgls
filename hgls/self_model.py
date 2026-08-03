"""
self_model.py — Self-Model / Agency Marker.

Distinguishes self-generated structures from externally-provided ones
(bootstrapped primitives, LLM proposals).
Agency ratio tracks how autonomous the system is becoming.
"""

from typing import Set
from hgls.structures import GenerativeStructure


class SelfModel:

    def __init__(self):
        self._self_ids:     Set[str] = set()
        self._external_ids: Set[str] = set()

    def mark_self_generated(self, structure: GenerativeStructure) -> None:
        self._self_ids.add(structure.id)

    def mark_external(self, structure: GenerativeStructure) -> None:
        self._external_ids.add(structure.id)

    def is_self_generated(self, struct_id: str) -> bool:
        return struct_id in self._self_ids

    def agency_ratio(self) -> float:
        total = len(self._self_ids) + len(self._external_ids)
        return len(self._self_ids) / total if total else 0.0

    def stats(self) -> dict:
        return {
            'self_generated': len(self._self_ids),
            'external':       len(self._external_ids),
            'agency_ratio':   round(self.agency_ratio(), 3),
        }