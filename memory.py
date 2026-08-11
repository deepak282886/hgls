"""
memory.py — The Graph.

The graph IS the knowledge. One structure, one file.

Nodes  — GenerativeStructure (atom or abstraction)
Edges  — weighted connections between nodes

Edge strength is the only state that matters.
Tinkerer grows the graph.
Eval shapes the strengths.
Algo reads and writes both.

No thresholds. No decay. No deletion.
Strength moves up on reward, down on no reward.
Adjustment is inversely proportional to current strength —
strong edges are stable, weak edges are still finding their place.
"""

import uuid
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


# ── Node ──────────────────────────────────────────────────────────

@dataclass
class Node:
    id:       str
    level:    int
    modality: str          # 'text' | 'voice' | 'vision'
    elements: List[str]    # ids of child nodes, or raw atom value if level 0
    strength: float = 0.0  # cumulative reinforcement this node has received

    def is_atom(self) -> bool:
        return self.level == 0

    def to_dict(self) -> dict:
        return {
            'id':       self.id,
            'level':    self.level,
            'modality': self.modality,
            'elements': self.elements,
            'strength': self.strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Node':
        return cls(
            id       = d['id'],
            level    = d['level'],
            modality = d['modality'],
            elements = d['elements'],
            strength = d['strength'],
        )

    def __repr__(self):
        return f"Node(id={self.id}, lvl={self.level}, mod={self.modality}, str={self.strength:.3f})"


# ── Edge ──────────────────────────────────────────────────────────

@dataclass
class Edge:
    source:   str
    target:   str
    strength: float = 0.01   # starts very weak, grows with reinforcement

    def reinforce(self, amount: float) -> None:
        self.strength += amount

    def weaken(self, amount: float) -> None:
        self.strength = max(0.0, self.strength - amount)

    def to_dict(self) -> dict:
        return {
            'source':   self.source,
            'target':   self.target,
            'strength': self.strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Edge':
        return cls(
            source   = d['source'],
            target   = d['target'],
            strength = d['strength'],
        )


# ── Graph ─────────────────────────────────────────────────────────

class Graph:
    """
    The single knowledge structure.

    Nodes are added when new structures are learned or abstracted.
    Edges are added by tinkerer when two activated nodes are unconnected.
    Strengths are adjusted by eval based on reward signal.

    Selection from neighbours is weighted by strength —
    stronger edges are more likely to be followed,
    but all edges remain eligible. No cutoffs.
    """

    def __init__(self):
        self._nodes: Dict[str, Node]              = {}
        self._edges: Dict[Tuple[str,str], Edge]   = {}
        self._index: Dict[str, Set[str]]          = {}  # node_id → connected node_ids

    # ── Nodes ─────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        if node.id not in self._index:
            self._index[node.id] = set()

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def nodes_at_level(self, level: int, modality: str = None) -> List[Node]:
        return [
            n for n in self._nodes.values()
            if n.level == level and (modality is None or n.modality == modality)
        ]

    # ── Edges ─────────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str) -> Edge:
        """
        Add edge if it doesn't exist. Return existing edge if it does.
        Every interaction that activates two nodes together
        should call this — first call creates, subsequent calls
        let eval adjust strength.
        """
        key = (source_id, target_id)
        if key not in self._edges:
            edge = Edge(source=source_id, target=target_id)
            self._edges[key] = edge
            self._index.setdefault(source_id, set()).add(target_id)
            self._index.setdefault(target_id, set()).add(source_id)
        return self._edges[key]

    def has_edge(self, source_id: str, target_id: str) -> bool:
        return (source_id, target_id) in self._edges or \
               (target_id, source_id) in self._edges

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        return self._edges.get((source_id, target_id)) or \
               self._edges.get((target_id, source_id))

    def neighbours(self, node_id: str) -> List[Tuple[Node, Edge]]:
        """
        All neighbours of a node with their connecting edges.
        Ordered by edge strength descending —
        strongest connections first, but all returned.
        """
        result = []
        for neighbour_id in self._index.get(node_id, set()):
            node = self._nodes.get(neighbour_id)
            edge = self.get_edge(node_id, neighbour_id)
            if node and edge:
                result.append((node, edge))
        result.sort(key=lambda x: x[1].strength, reverse=True)
        return result

    def activated_path(self, node_ids: List[str]) -> List[Edge]:
        """
        Return all edges between a set of activated nodes.
        Used by eval to know which edges to adjust after an interaction.
        """
        edges = []
        seen  = set()
        for i, a in enumerate(node_ids):
            for b in node_ids[i+1:]:
                edge = self.get_edge(a, b)
                if edge:
                    key = (min(a,b), max(a,b))
                    if key not in seen:
                        edges.append(edge)
                        seen.add(key)
        return edges

    # ── Strength adjustment ───────────────────────────────────────

    def reinforce_path(self, node_ids: List[str]) -> None:
        """
        Reward arrived. Strengthen all edges on this path.
        Adjustment inversely proportional to current strength —
        strong edges move less, weak edges move more.
        Also strengthen the nodes themselves.
        """
        edges = self.activated_path(node_ids)
        for edge in edges:
            amount = 1.0 / (1.0 + edge.strength)
            edge.reinforce(amount)

        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                amount = 1.0 / (1.0 + node.strength)
                node.strength += amount

    def weaken_path(self, node_ids: List[str]) -> None:
        """
        No reward. Weaken all edges on this path.
        Same inverse proportionality — stable edges resist weakening.
        """
        edges = self.activated_path(node_ids)
        for edge in edges:
            amount = 1.0 / (1.0 + edge.strength)
            edge.weaken(amount)

        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                amount = 1.0 / (1.0 + node.strength)
                node.strength = max(0.0, node.strength - amount)

    # ── Abstraction ───────────────────────────────────────────────

    def abstract(self, node_ids: List[str], modality: str, level: int) -> Node:
        """
        A cluster of strongly connected nodes becomes a new node
        at the next level up. Called by algo when it detects
        a stable cluster worth abstracting.
        Returns the new abstract node.
        """
        new_id   = str(uuid.uuid4())[:8]
        new_node = Node(
            id       = new_id,
            level    = level,
            modality = modality,
            elements = node_ids,
            strength = sum(
                self._nodes[nid].strength
                for nid in node_ids
                if nid in self._nodes
            ) / max(len(node_ids), 1),
        )
        self.add_node(new_node)

        # Connect new abstract node to all its constituents
        for nid in node_ids:
            self.add_edge(new_id, nid)

        return new_node

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            'nodes': {nid: n.to_dict() for nid, n in self._nodes.items()},
            'edges': {
                f"{k[0]}|{k[1]}": e.to_dict()
                for k, e in self._edges.items()
            },
        }
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        if os.path.exists(path):
            os.replace(path, path + '.bak')
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            for nid, nd in data['nodes'].items():
                self._nodes[nid] = Node.from_dict(nd)
                self._index.setdefault(nid, set())
            for k_str, ed in data['edges'].items():
                src, tgt = k_str.split('|', 1)
                key = (src, tgt)
                self._edges[key] = Edge.from_dict(ed)
                self._index.setdefault(src, set()).add(tgt)
                self._index.setdefault(tgt, set()).add(src)
            return True
        except Exception as e:
            print(f"[Graph] Load failed: {e}")
            return False

    def __len__(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self._edges)