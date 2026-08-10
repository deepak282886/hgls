"""
co_occurrence.py — Co-occurrence Detector.

Every co-occurrence immediately creates or strengthens an edge.
No threshold. No minimum count. Frequency IS the strength.

A pair seen once gets strength 0.01.
A pair seen 100 times gets strength 1.0.
The graph grows continuously from the very first sentence.
"""

from itertools import combinations
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library
    from hgls.graph   import MemoryGraph

_STOP_WORDS = {
    'the','a','an','is','are','was','were','be','been','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall',
    'can','to','of','in','on','at','by','for','with','about','from','into',
    'and','or','but','not','no','so','if','then','that','this','it','he',
    'she','they','we','i','you','my','your','its','his','her','their','our',
    'who','which','what','when','where','as','than','one','two','three',
    'also','more','most','such','after','before','during','both','each',
    'other','only','just','very','well','up','out','all','any','some',
}

# Strength added per co-occurrence observation
CO_OCC_INCREMENT = 0.01


class CoOccurrenceDetector:

    def __init__(self, library: 'Library', graph: 'MemoryGraph'):
        self.library     = library
        self.graph       = graph
        self._obs_count  = 0
        self._edge_count = 0
        self._sentences  = 0

    def observe_phrase(self, phrase_struct_id: str) -> int:
        """
        Observe co-occurrences between all content words in a phrase.
        Every pair immediately creates or strengthens an edge.
        No threshold. No minimum.
        Returns number of new edges created.
        """
        self._sentences += 1
        phrase = self.library.get(phrase_struct_id)
        if phrase is None:
            return 0

        # Compositional edges: phrase → direct elements
        for elem in phrase.elements:
            if isinstance(elem, str) and len(elem) > 1 and self.library.has(elem):
                child = self.library.get(elem)
                self.graph.add_compositional(
                    parent_id    = phrase.id,
                    child_id     = elem,
                    parent_level = phrase.level,
                    child_level  = child.level,
                )

        # Co-occurrence edges between content words
        word_ids = self._extract_content_word_ids(phrase_struct_id)
        if len(word_ids) < 2:
            return 0

        new_edges = 0
        for id_a, id_b in combinations(word_ids, 2):
            new_edges += self._observe_pair(id_a, id_b, level=2)
            self._obs_count += 1

        return new_edges

    def _observe_pair(self, id_a: str, id_b: str, level: int) -> int:
        """
        Observe one co-occurrence pair.
        Immediately creates edge if new, or strengthens if existing.
        Returns 1 if new edge created, 0 if existing edge strengthened.
        """
        from hgls.graph import Edge

        key = (min(id_a, id_b), max(id_a, id_b))

        if self.graph.has_edge(key):
            self.graph.reinforce_edge(key)
            return 0

        # Create edge immediately — no threshold
        edge = Edge(
            source_id        = key[0],
            target_id        = key[1],
            edge_type        = 'co_occurrence',
            strength         = CO_OCC_INCREMENT,
            occurrence_count = 1,
            validated        = True,
            level_span       = (level, level),
        )
        self.graph._store(key, edge)
        self._edge_count += 1
        return 1

    def _extract_content_word_ids(
        self, struct_id: str, _depth: int = 0
    ) -> Set[str]:
        """Recursively collect level-2 content word structure IDs."""
        if _depth > 6:
            return set()

        struct = self.library.get(struct_id)
        if struct is None:
            return set()

        result = set()
        if struct.level == 2:
            word = struct.generate(self.library).strip().lower()
            if word and word not in _STOP_WORDS and len(word) > 2:
                result.add(struct_id)
            return result

        for elem in struct.elements:
            if isinstance(elem, str) and len(elem) > 1 and self.library.has(elem):
                result |= self._extract_content_word_ids(elem, _depth + 1)

        return result

    def scan_library(
        self,
        sentences_processed: int = None,
        verbose: bool = True,
    ) -> dict:
        """
        Scan all phrase-level structures and build co-occurrence edges.
        No threshold — all pairs immediately get edges.
        """
        if verbose:
            print('[CoOccurrence] Scanning library (no threshold — all pairs get edges)...')

        total_phrases = 0
        total_new     = 0

        for level in range(3, 7):
            for struct in self.library.get_at_level(level, kind='success'):
                # Compositional edges
                for elem in struct.elements:
                    if isinstance(elem, str) and len(elem) > 1 and self.library.has(elem):
                        child = self.library.get(elem)
                        self.graph.add_compositional(
                            parent_id    = struct.id,
                            child_id     = elem,
                            parent_level = struct.level,
                            child_level  = child.level,
                        )

                # Co-occurrence
                word_ids = self._extract_content_word_ids(struct.id)
                for id_a, id_b in combinations(word_ids, 2):
                    new = self._observe_pair(id_a, id_b, level=2)
                    total_new    += new
                    self._obs_count += 1

                total_phrases += 1
                if verbose and total_phrases % 10000 == 0:
                    print(f'  phrases={total_phrases:,} | edges={len(self.graph):,}')

        if verbose:
            print(
                f'[CoOccurrence] Complete. '
                f'phrases={total_phrases:,} | '
                f'new_edges={total_new:,} | '
                f'total_edges={len(self.graph):,}'
            )

        return {
            'phrases_scanned': total_phrases,
            'new_edges':       total_new,
            'graph_total':     len(self.graph),
        }

    def stats(self) -> dict:
        return {
            'observations':  self._obs_count,
            'edges_created': self._edge_count,
            'sentences_seen': self._sentences,
        }