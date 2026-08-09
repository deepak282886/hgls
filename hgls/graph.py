"""
graph.py — Memory Graph: Edge Store for HGLS.

Nodes are GenerativeStructure IDs already in the Library.
Edges are connections between nodes.

Three edge types:
  compositional — parent structure contains child
  co_occurrence — two structures appear together frequently
  novel         — proposed by tinkering engine, validated by LLM

Each edge has:
  strength        — float 0-1, grows with recurrence or validation
  occurrence_count— how many times this pair co-occurred
  validated       — True if confirmed (by recurrence or LLM)
  level_span      — (source_level, target_level)

Serialises alongside deepak_memory.json as deepak_graph.json.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

EDGE_TYPES = ('compositional', 'co_occurrence', 'novel')

CO_OCCURRENCE_STRENGTH_INCREMENT = 0.01
STRENGTH_DECAY                   = 0.0001
NOVEL_INITIAL_STRENGTH           = 0.7


@dataclass
class Edge:
    source_id:        str
    target_id:        str
    edge_type:        str
    strength:         float = 0.0
    occurrence_count: int   = 0
    validated:        bool  = False
    level_span:       Tuple[int, int] = (0, 0)
    created_at:       float = field(default_factory=time.time)
    last_seen:        float = field(default_factory=time.time)
    metadata:         dict  = field(default_factory=dict)

    def reinforce(self, amount: float = CO_OCCURRENCE_STRENGTH_INCREMENT):
        self.strength         = min(1.0, self.strength + amount)
        self.occurrence_count += 1
        self.last_seen        = time.time()

    def decay(self, amount: float = STRENGTH_DECAY):
        self.strength = max(0.0, self.strength - amount)

    def to_dict(self) -> dict:
        return {
            'source_id':        self.source_id,
            'target_id':        self.target_id,
            'edge_type':        self.edge_type,
            'strength':         self.strength,
            'occurrence_count': self.occurrence_count,
            'validated':        self.validated,
            'level_span':       list(self.level_span),
            'created_at':       self.created_at,
            'last_seen':        self.last_seen,
            'metadata':         self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Edge':
        return cls(
            source_id        = d['source_id'],
            target_id        = d['target_id'],
            edge_type        = d['edge_type'],
            strength         = d['strength'],
            occurrence_count = d['occurrence_count'],
            validated        = d['validated'],
            level_span       = tuple(d['level_span']),
            created_at       = d.get('created_at', 0.0),
            last_seen        = d.get('last_seen', 0.0),
            metadata         = d.get('metadata', {}),
        )


class MemoryGraph:
    """
    The graph that grows over the library.
    Nodes  = structure IDs (from Library)
    Edges  = typed, weighted connections
    """

    def __init__(self):
        self._edges: Dict[Tuple[str, str], Edge]           = {}
        self._node_index: Dict[str, Set[Tuple[str, str]]]  = defaultdict(set)
        self._pending_co_occ: Dict[Tuple[str, str], int]   = defaultdict(int)
        self._total_edges_added  = 0
        self._total_co_occ_obs   = 0
        self._total_novel_added  = 0

    # ── Edge helpers ──────────────────────────────────────────────

    def has_edge(self, key: Tuple[str, str]) -> bool:
        return key in self._edges

    def reinforce_edge(self, key: Tuple[str, str]) -> None:
        if key in self._edges:
            self._edges[key].reinforce()

    # ── Edge addition ─────────────────────────────────────────────

    def add_compositional(
        self,
        parent_id: str,
        child_id: str,
        parent_level: int,
        child_level: int,
    ) -> None:
        key = (parent_id, child_id)
        if key in self._edges:
            self._edges[key].reinforce(0.05)
            return
        edge = Edge(
            source_id  = parent_id,
            target_id  = child_id,
            edge_type  = 'compositional',
            strength   = 1.0,
            validated  = True,
            level_span = (parent_level, child_level),
        )
        self._store(key, edge)

    def observe_co_occurrence(
        self,
        id_a: str,
        id_b: str,
        level: int,
        threshold: int = 50,
    ) -> bool:
        """
        Record one co-occurrence. Returns True if edge became permanent.
        Threshold is passed in (adaptive) rather than hardcoded.
        """
        self._total_co_occ_obs += 1
        key = (min(id_a, id_b), max(id_a, id_b))

        if key in self._edges:
            self._edges[key].reinforce()
            return False

        self._pending_co_occ[key] += 1

        if self._pending_co_occ[key] >= threshold:
            del self._pending_co_occ[key]
            edge = Edge(
                source_id        = key[0],
                target_id        = key[1],
                edge_type        = 'co_occurrence',
                strength         = min(1.0, threshold * CO_OCCURRENCE_STRENGTH_INCREMENT),
                occurrence_count = threshold,
                validated        = True,
                level_span       = (level, level),
            )
            self._store(key, edge)
            return True

        return False

    def add_novel(
        self,
        source_id: str,
        target_id: str,
        source_level: int,
        target_level: int,
        metadata: dict = None,
    ) -> None:
        key  = (source_id, target_id)
        edge = Edge(
            source_id  = source_id,
            target_id  = target_id,
            edge_type  = 'novel',
            strength   = NOVEL_INITIAL_STRENGTH,
            validated  = True,
            level_span = (source_level, target_level),
            metadata   = metadata or {},
        )
        self._store(key, edge)
        self._total_novel_added += 1

    def add_novel_failure(self, source_id: str, target_id: str) -> None:
        key  = (source_id, target_id)
        edge = Edge(
            source_id = source_id,
            target_id = target_id,
            edge_type = 'novel',
            strength  = 0.0,
            validated = False,
            metadata  = {'rejected': True},
        )
        self._store(key, edge)

    # ── Traversal ─────────────────────────────────────────────────

    def get_neighbours(
        self,
        node_id: str,
        min_strength: float = 0.1,
        edge_types: tuple = EDGE_TYPES,
    ) -> List[Edge]:
        result = []
        for key in self._node_index.get(node_id, set()):
            edge = self._edges.get(key)
            if edge and edge.strength >= min_strength and edge.edge_type in edge_types:
                result.append(edge)
        return sorted(result, key=lambda e: e.strength, reverse=True)

    def get_region(
        self,
        node_id: str,
        depth: int = 2,
        min_strength: float = 0.2,
    ) -> Dict[str, List[Edge]]:
        visited  = {}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                edges = self.get_neighbours(nid, min_strength=min_strength)
                visited[nid] = edges
                for edge in edges:
                    other = edge.target_id if edge.source_id == nid else edge.source_id
                    if other not in visited:
                        next_frontier.add(other)
            frontier = next_frontier
        return visited

    def density(self, node_id: str) -> float:
        neighbours = self.get_neighbours(node_id)
        if not neighbours:
            return 0.0
        n           = len(neighbours)
        avg_strength = sum(e.strength for e in neighbours) / n
        return min(1.0, (n / 20.0) * avg_strength)

    def is_rejected_novel(self, source_id: str, target_id: str) -> bool:
        key  = (source_id, target_id)
        edge = self._edges.get(key)
        return edge is not None and edge.metadata.get('rejected', False)

    def strongest_edges(
        self, edge_type: str = None, top_k: int = 20
    ) -> List[Edge]:
        edges = list(self._edges.values())
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return sorted(edges, key=lambda e: e.strength, reverse=True)[:top_k]

    # ── Stats and decay ───────────────────────────────────────────

    def decay_all(self) -> None:
        for edge in self._edges.values():
            if edge.edge_type == 'co_occurrence':
                edge.decay()

    def stats(self) -> dict:
        by_type = defaultdict(int)
        for e in self._edges.values():
            by_type[e.edge_type] += 1
        return {
            'total_edges':       len(self._edges),
            'pending_co_occ':    len(self._pending_co_occ),
            'by_type':           dict(by_type),
            'total_co_occ_obs':  self._total_co_occ_obs,
            'total_novel_added': self._total_novel_added,
        }

    def __len__(self):
        return len(self._edges)

    # ── Internal ──────────────────────────────────────────────────

    def _store(self, key: Tuple[str, str], edge: Edge) -> None:
        self._edges[key] = edge
        self._node_index[key[0]].add(key)
        self._node_index[key[1]].add(key)
        self._total_edges_added += 1

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str = 'deepak_graph.json') -> None:
        data = {
            'version': '0.6',
            'edges':   {
                f"{k[0]}|{k[1]}": e.to_dict()
                for k, e in self._edges.items()
            },
            'pending': {
                f"{k[0]}|{k[1]}": v
                for k, v in self._pending_co_occ.items()
            },
        }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        if os.path.exists(path):
            backup = path + '.bak'
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
        os.rename(tmp, path)

    def load(self, path: str = 'deepak_graph.json') -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            for k_str, ed in data['edges'].items():
                src, tgt = k_str.split('|', 1)
                edge     = Edge.from_dict(ed)
                key      = (src, tgt)
                self._edges[key] = edge
                self._node_index[src].add(key)
                self._node_index[tgt].add(key)
            for k_str, count in data.get('pending', {}).items():
                src, tgt = k_str.split('|', 1)
                self._pending_co_occ[(src, tgt)] = count
            print(f"[Graph] Loaded {len(self._edges)} edges.")
            return True
        except Exception as e:
            print(f"[Graph] Load failed: {e}")
            return False