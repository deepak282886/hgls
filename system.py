"""
system.py — The System.

Wires everything together. Single entry point for all interaction.

  graph   — the knowledge
  eval    — keeps it coherent
  tinkerer— grows it
  algo    — runs the process

Three interaction modes:

  learn(atoms, modality, reward)
    — single modality input with reward signal
    — used when you are teaching one modality at a time

  learn_multi(text, voice, vision, reward)
    — all three modalities at once
    — used when presenting a concept across modalities simultaneously
    — cross-modal edges form on positive reward

  query(atoms, modality)
    — no reward signal, just propagate and return path + score
    — used to check what the system knows
    — caller decides if output is good enough to reward

Save and load are atomic. One file. No backups needed beyond
the single .bak the OS rename gives you.
"""

import os
from typing import List, Dict, Optional, Tuple

from memory   import Graph
from atoms    import bootstrap, encode_text, encode_phonemes, encode_patches
from eval     import Eval
from tinkerer import Tinkerer
from algo     import Algo


class System:

    def __init__(self, path: str = 'knowledge.json'):
        self.path  = path
        self.graph = Graph()
        self.ev    = Eval(self.graph)
        self.tk    = Tinkerer(self.graph)
        self.al    = Algo(self.graph, self.ev, self.tk)

        if os.path.exists(path):
            self.graph.load(path)
            print(f"[System] Loaded — {len(self.graph)} nodes, "
                  f"{self.graph.edge_count()} edges")
        else:
            bootstrap(self.graph)
            self.save()

    # ── Learning ──────────────────────────────────────────────────

    def learn(
        self,
        atoms:     List[str],
        modality:  str,
        reward:    bool,
    ) -> Tuple[List[str], float, float]:
        """
        Single modality learning interaction.

        atoms    — atom node ids (from encode_text / encode_phonemes / encode_patches)
        modality — 'text' | 'voice' | 'vision'
        reward   — did this interaction deserve reward?

        Returns (path, coherence_score, intensity):
          path            — nodes traversed
          coherence_score — how strong the path was before reward
          intensity       — emotional intensity of this interaction
        """
        path, score = self.al.process(atoms, modality)
        intensity   = self.al.reward(path, reward)
        return path, score, intensity

    def learn_multi(
        self,
        text_atoms:   List[str] = None,
        voice_atoms:  List[str] = None,
        vision_atoms: List[str] = None,
        reward:       bool = True,
    ) -> Tuple[Dict[str, List[str]], float, float]:
        """
        Multi-modal learning interaction.
        All three modalities processed simultaneously.
        Cross-modal edges form on positive reward.

        Returns (paths, cross_modal_score, avg_intensity):
          paths             — dict of modality → path
          cross_modal_score — coherence across modalities
          avg_intensity     — average emotional intensity across modalities
        """
        paths, cross_score = self.al.integrate(
            text_atoms   or [],
            voice_atoms  or [],
            vision_atoms or [],
        )
        self.al.reward_integration(paths, reward)
        intensities = [
            self.ev.score(p) for p in paths.values() if p
        ]
        avg_intensity = sum(intensities) / max(len(intensities), 1)
        return paths, cross_score, avg_intensity

    # ── Query ─────────────────────────────────────────────────────

    def query(
        self,
        atoms:    List[str],
        modality: str,
    ) -> Tuple[List[str], float]:
        """
        Query what the system knows about this input.
        No reward signal — just propagation and scoring.
        Returns (path, score).
        Score tells you how confident the system is.
        High score = strong familiar path.
        Low score  = weak or novel territory.
        """
        path, score = self.al.process(atoms, modality)
        return path, score

    # ── Convenience encoders ──────────────────────────────────────

    def text(self, s: str) -> List[str]:
        """Encode a string to text atom ids."""
        return encode_text(s)

    def phonemes(self, sequence: List[str]) -> List[str]:
        """Encode a phoneme sequence to voice atom ids."""
        return encode_phonemes(sequence)

    def patches(self, patch_list: List[dict]) -> List[str]:
        """Encode a list of patch feature dicts to vision atom ids."""
        return encode_patches(patch_list)

    # ── State ─────────────────────────────────────────────────────

    def state(self) -> dict:
        """
        Current state of the system.
        Nodes and edges by level give a picture of
        how much has been learned at each abstraction level.
        """
        by_level: Dict[int, Dict[str, int]] = {}
        for node in self.graph._nodes.values():
            lvl = node.level
            if lvl not in by_level:
                by_level[lvl] = {'nodes': 0, 'modalities': set()}
            by_level[lvl]['nodes'] += 1
            by_level[lvl]['modalities'].add(node.modality)

        return {
            'total_nodes': len(self.graph),
            'total_edges': self.graph.edge_count(),
            'by_level':    {
                lvl: {
                    'nodes':      d['nodes'],
                    'modalities': list(d['modalities']),
                }
                for lvl, d in sorted(by_level.items())
            },
        }

    # ── Persistence ───────────────────────────────────────────────

    def save(self) -> None:
        self.graph.save(self.path)

    def load(self) -> bool:
        return self.graph.load(self.path)