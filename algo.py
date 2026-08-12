"""
algo.py — The One Algorithm.

Same process at every level, every modality:

  1. Input arrives as atom ids
  2. Activate those atoms in the graph
  3. Propagate forward through existing edges
     — follow neighbours weighted by strength
     — eval scores the path as it forms
  4. If path is strong → output what the path leads to
  5. If path is weak or missing → tinkerer engages
     — proposes connections between activated nodes
  6. External reward arrives
     — eval adjusts all edge strengths on the path
     — tinkerer updates strategy edges
  7. If dense cluster detected → abstract to next level

One function. One loop. No special cases per modality or level.
The graph structure does the work. Algo just runs the process.
"""

from typing import List, Optional, Tuple, Dict
from memory import Graph, Node
from eval import Eval
from tinkerer import Tinkerer


# How weak a path needs to be before tinkerer engages
# Not a threshold — tinkerer always engages on missing edges,
# this just determines when propagation is considered "uncertain"
_WEAK = 0.1


class Algo:

    def __init__(self, graph: Graph, ev: Eval, tk: Tinkerer):
        self.graph = graph
        self.ev    = ev
        self.tk    = tk

    # ── Primary: process one input ────────────────────────────────

    def process(
        self,
        atom_ids: List[str],
        modality: str,
    ) -> Tuple[List[str], float]:
        """
        Process one input. Core loop.

        atom_ids : atom node ids activated by this input
        modality : 'text' | 'voice' | 'vision'

        Returns (path, coherence_score):
          path             — ordered list of node ids traversed
          coherence_score  — eval's score of the path
                             caller uses this to decide if output is confident
        """
        if not atom_ids:
            return [], 0.0

        # Step 1: activate input atoms — ensure they exist in graph
        for aid in atom_ids:
            if not self.graph.has_node(aid):
                return [], 0.0

        # Step 2: propagate forward from activated atoms
        path, score = self._propagate(atom_ids)

        # Step 3: tinkerer always checks input atoms for unconnected pairs.
        # Even when path score is high (due to existing connections),
        # new input atoms may not yet be connected to each other.
        # tk.engage() is cheap — returns empty if all pairs already connected.
        proposals = self.tk.engage(atom_ids)
        if proposals:
            path, score = self._propagate(atom_ids)

        return path, score

    def reward(
        self,
        path:      List[str],
        got_reward: bool,
        proposals: List[Tuple[str, str, str]] = None,
    ) -> float:
        """
        External reward signal arrives.
        Adjust graph via eval. Update tinkerer strategy edges.
        Check for abstraction opportunities.

        Returns emotional intensity of this interaction.
        """
        intensity = self.ev.signal(path, got_reward)

        if proposals:
            self.tk.outcome(proposals, got_reward)

        # Check if any dense clusters are ready to abstract
        if got_reward:
            self._try_abstract(path)

        return intensity

    # ── Propagation ───────────────────────────────────────────────

    def _propagate(
        self,
        start_ids: List[str],
        max_depth: int = 6,
    ) -> Tuple[List[str], float]:
        """
        Propagate activation forward through the graph from start nodes.

        At each step: look at all neighbours of currently active nodes.
        The neighbour with the strongest cumulative edge strength
        from the active set gets added to the path.
        Continue until no stronger next step exists or max depth reached.

        This is not random. The graph structure guides propagation.
        Strong paths are followed. Weak paths are not.
        All edges remain in the graph — only the current traversal
        is deterministic.
        """
        path    = list(start_ids)
        visited = set(start_ids)

        for _ in range(max_depth):
            # Collect all candidate next nodes with their
            # total incoming strength from current active set
            candidates: Dict[str, float] = {}

            for nid in path:
                for neighbour, edge in self.graph.neighbours(nid):
                    if neighbour.id not in visited:
                        candidates[neighbour.id] = (
                            candidates.get(neighbour.id, 0.0) + edge.strength
                        )

            if not candidates:
                break

            # Follow the strongest signal
            best_id    = max(candidates, key=candidates.get)
            best_score = candidates[best_id]

            if best_score <= 0:
                break

            path.append(best_id)
            visited.add(best_id)

        score = self.ev.score(path)
        return path, score

    # ── Abstraction ───────────────────────────────────────────────

    def _try_abstract(self, path: List[str]) -> Optional[Node]:
        """
        After a rewarded interaction, check if any dense cluster
        in the path should be abstracted to the next level up.

        A cluster becomes a new node when:
        - tinkerer identifies it as dense (all pairs connected)
        - all members are at the same level
        - no abstract node already represents this cluster

        The new node inherits the average strength of its members
        and connects back to all of them.
        """
        candidates = self.tk.compression_candidates(path, min_cluster=3)

        for cluster in candidates:
            # All must be same level and modality
            nodes    = [self.graph.get_node(nid) for nid in cluster]
            nodes    = [n for n in nodes if n is not None]
            if len(nodes) != len(cluster):
                continue

            levels    = {n.level    for n in nodes}
            modalities = {n.modality for n in nodes}

            if len(levels) > 1 or len(modalities) > 1:
                continue

            level    = levels.pop()
            modality = modalities.pop()

            # Check if this exact cluster already has an abstract node
            if self._already_abstracted(cluster):
                continue

            # Create the abstract node one level up
            abstract = self.graph.abstract(cluster, modality, level + 1)
            return abstract

        return None

    def _already_abstracted(self, cluster: List[str]) -> bool:
        """
        Check if a node already exists that has exactly
        these cluster members as its elements.
        Prevents duplicate abstraction of the same cluster.
        """
        cluster_set = set(cluster)
        for node in self.graph.nodes_at_level(
            self.graph.get_node(cluster[0]).level + 1
            if self.graph.get_node(cluster[0]) else 1
        ):
            if set(node.elements) == cluster_set:
                return True
        return False

    # ── Bottom-up propagation ─────────────────────────────────────

    def propagate_up(self, atom_ids: List[str], modality: str = 'text') -> List:
        """
        Given activated input atoms, propagate activation UPWARD
        through the graph to find the best matching concept.

        For each concept node, score = active_edge_strength / total_edge_strength
        weighted by input coverage.

        Only counts level-0 atoms of the same modality as the input
        — prevents voice atoms and strategy nodes (added by tinkerer)
        from distorting text-based concept matching.

        Returns list of (concept_node, score) sorted descending.
        """
        input_set     = set(atom_ids)
        input_modality = modality
        scores        = {}

        input_count = len(input_set)

        for node in self.graph._nodes.values():
            if node.modality != 'concept':
                continue

            active_strength = 0.0
            total_strength  = 0.0
            active_count    = 0

            for neighbour, edge in self.graph.neighbours(node.id):
                # Only count level-0 atoms of the same modality as the input.
                # Concept nodes can have edges to voice atoms, strategy nodes etc.
                # from tinkerer activity — those must not interfere with text recognition.
                if neighbour.level == 0 and neighbour.modality == input_modality:
                    total_strength += edge.strength
                    if neighbour.id in input_set:
                        active_strength += edge.strength
                        active_count    += 1

            if total_strength > 0 and active_strength > 0:
                # How much of this concept is activated by the input
                concept_coverage = active_strength / total_strength
                # How much of the input is explained by this concept
                input_coverage   = active_count / max(input_count, 1)
                # Base score: both coverages must be high
                base = concept_coverage * input_coverage

                # Substring bonus: if the input text appears as a substring
                # inside the concept's stored text, it is a stronger match.
                # This uses the concept node's own element — still graph-based.
                concept_text  = node.elements[0] if node.elements else ''
                input_text_reconstructed = ''.join(
                    self.graph.get_node(aid).elements[0]
                    for aid in atom_ids
                    if self.graph.get_node(aid)
                    and self.graph.get_node(aid).modality == modality
                )
                if (input_text_reconstructed and concept_text and
                        input_text_reconstructed in concept_text):
                    bonus = len(input_text_reconstructed) / max(len(concept_text), 1)
                else:
                    bonus = 0.0

                scores[node.id] = base + bonus

        results = sorted(
            [
                (self.graph.get_node(nid), score)
                for nid, score in scores.items()
                if self.graph.get_node(nid)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return results

    # ── Cross-modal integration ───────────────────────────────────

    def integrate(
        self,
        text_atoms:   List[str],
        voice_atoms:  List[str],
        vision_atoms: List[str],
    ) -> Tuple[Dict[str, List[str]], float]:
        """
        Process input across all three modalities simultaneously.
        Each modality propagates independently first.
        Then cross-modal coherence is scored.

        Returns:
          paths  — dict of modality → path
          score  — cross-modal coherence score

        High cross-modal score means all three modalities are
        converging on the same higher-level nodes — a concept
        is being grounded across modalities.
        """
        paths = {}

        if text_atoms:
            paths['text'],   _ = self.process(text_atoms,   'text')
        if voice_atoms:
            paths['voice'],  _ = self.process(voice_atoms,  'voice')
        if vision_atoms:
            paths['vision'], _ = self.process(vision_atoms, 'vision')

        cross_score = self.ev.cross_modal_score(
            paths.get('text',   []),
            paths.get('voice',  []),
            paths.get('vision', []),
        )

        return paths, cross_score

    def reward_integration(
        self,
        paths:      Dict[str, List[str]],
        got_reward: bool,
    ) -> None:
        """
        Reward signal for a cross-modal interaction.
        Each modality's path gets adjusted.
        Tinkerer also gets a chance to connect across modalities
        if the reward was positive — cross-modal edges form
        when the same concept is seen correctly in multiple modalities.
        """
        all_nodes = []
        for path in paths.values():
            all_nodes.extend(path)
            self.ev.signal(path, got_reward)

        # Check abstraction across all activated nodes
        if got_reward and all_nodes:
            self._try_abstract(all_nodes)

        # On positive reward, let tinkerer try cross-modal connections
        # between the highest-level nodes reached in each modality
        if got_reward and len(paths) > 1:
            top_nodes = []
            for path in paths.values():
                if path:
                    # Highest level node reached in this modality
                    top = max(
                        (self.graph.get_node(nid) for nid in path
                         if self.graph.get_node(nid)),
                        key=lambda n: n.level,
                        default=None,
                    )
                    if top:
                        top_nodes.append(top.id)

            if len(top_nodes) > 1:
                self.tk.engage(top_nodes)