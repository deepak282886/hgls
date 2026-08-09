"""
co_occurrence.py — Co-occurrence Detector.

Two modes:

  Passive  — runs during ingestion. Every time a phrase structure is learned,
             extract contributing word-level structures and observe all pairs.

  Batch    — runs once on existing library after ingestion. Scans every
             phrase structure, extracts word pairs, builds base graph edges.
             Run this after Wikipedia ingestion completes.

Threshold scales with sentences processed — not hardcoded.
Function words filtered out before observing co-occurrence.
"""

from itertools import combinations
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.library import Library
    from hgls.graph   import MemoryGraph

# Words that carry no meaningful co-occurrence signal
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
    'at', 'by', 'for', 'with', 'about', 'from', 'into', 'through', 'as',
    'and', 'or', 'but', 'not', 'no', 'so', 'if', 'then', 'that', 'this',
    'it', 'he', 'she', 'they', 'we', 'i', 'you', 'my', 'your', 'its',
    'his', 'her', 'their', 'our', 'who', 'which', 'what', 'when', 'where',
    'also', 'more', 'most', 'such', 'than', 'one', 'two', 'three', 'new',
    'first', 'last', 'after', 'before', 'between', 'during', 'while',
    'however', 'although', 'though', 'because', 'since', 'until', 'both',
    'each', 'other', 'only', 'just', 'very', 'well', 'been', 'up', 'out',
}


def _adaptive_threshold(sentences_processed: int) -> int:
    """
    Threshold scales with how much data has been seen.
    Small corpus → low threshold (edges form easily).
    Large corpus → higher threshold (only truly recurrent pairs earn edges).

    sentences_processed  threshold
    < 10,000             3
    10k - 50k            5
    50k - 200k           10
    200k - 1M            20
    > 1M                 50
    """
    if sentences_processed < 10_000:
        return 3
    elif sentences_processed < 50_000:
        return 5
    elif sentences_processed < 200_000:
        return 10
    elif sentences_processed < 1_000_000:
        return 20
    else:
        return 50


class CoOccurrenceDetector:

    def __init__(self, library: 'Library', graph: 'MemoryGraph'):
        self.library            = library
        self.graph              = graph
        self._obs_count         = 0
        self._edge_count        = 0
        self._sentences_seen    = 0

    # ── Passive: called during ingestion ──────────────────────────

    def observe_phrase(self, phrase_struct_id: str) -> int:
        """
        Given a newly learned phrase structure, observe co-occurrences
        between all contributing content word-level (level 2) structures.
        Skips stop words. Uses adaptive threshold.
        Returns number of new permanent edges created.
        """
        self._sentences_seen += 1
        phrase = self.library.get(phrase_struct_id)
        if phrase is None:
            return 0

        # Add compositional edges: phrase → each direct element
        for elem in phrase.elements:
            if isinstance(elem, str) and len(elem) > 1 and self.library.has(elem):
                child = self.library.get(elem)
                self.graph.add_compositional(
                    parent_id    = phrase.id,
                    child_id     = elem,
                    parent_level = phrase.level,
                    child_level  = child.level,
                )

        # Collect content word structures contributing to this phrase
        word_ids = self._extract_content_word_ids(phrase_struct_id)
        if len(word_ids) < 2:
            return 0

        threshold = _adaptive_threshold(self._sentences_seen)
        new_edges = 0

        for id_a, id_b in combinations(word_ids, 2):
            became_permanent = self._observe(id_a, id_b, level=2, threshold=threshold)
            self._obs_count += 1
            if became_permanent:
                new_edges += 1
                self._edge_count += 1

        return new_edges

    def _observe(
        self,
        id_a: str,
        id_b: str,
        level: int,
        threshold: int,
    ) -> bool:
        """
        Observe one co-occurrence. Returns True if edge became permanent.
        Overrides graph's fixed threshold with our adaptive one.
        """
        from hgls.graph import Edge
        import time

        key = (min(id_a, id_b), max(id_a, id_b))

        # Already permanent — reinforce
        if self.graph.has_edge(key):
            self.graph.reinforce_edge(key)
            return False

        # Increment pending
        self.graph._pending_co_occ[key] = self.graph._pending_co_occ.get(key, 0) + 1

        if self.graph._pending_co_occ[key] >= threshold:
            del self.graph._pending_co_occ[key]
            edge = Edge(
                source_id        = key[0],
                target_id        = key[1],
                edge_type        = 'co_occurrence',
                strength         = min(1.0, threshold * 0.01),
                occurrence_count = threshold,
                validated        = True,
                level_span       = (level, level),
            )
            self.graph._store(key, edge)
            return True

        return False

    def _extract_content_word_ids(
        self, struct_id: str, _depth: int = 0
    ) -> Set[str]:
        """
        Recursively collect level-2 structure IDs contributing to this structure.
        Filters out stop words — only content words form meaningful edges.
        """
        if _depth > 6:
            return set()

        struct = self.library.get(struct_id)
        if struct is None:
            return set()

        result = set()

        if struct.level == 2:
            # Only include if it's a content word
            word = struct.generate(self.library).strip().lower()
            if word and word not in _STOP_WORDS and len(word) > 2:
                result.add(struct_id)
            return result

        for elem in struct.elements:
            if isinstance(elem, str) and len(elem) > 1 and self.library.has(elem):
                result |= self._extract_content_word_ids(elem, _depth + 1)

        return result

    # ── Batch: run on existing library ────────────────────────────

    def scan_library(
        self,
        sentences_processed: int = None,
        verbose: bool = True,
    ) -> dict:
        """
        Scan all phrase-level (3+) structures in the library and build
        co-occurrence edges from what already exists.

        sentences_processed — pass actual count for accurate threshold.
                              If None, estimated from library size.
        """
        if sentences_processed is None:
            # Rough estimate: library has ~10 structures per sentence on average
            sentences_processed = len(self.library) // 10

        threshold = _adaptive_threshold(sentences_processed)

        if verbose:
            print(
                f'[CoOccurrence] Scanning library...\n'
                f'  sentences_estimated={sentences_processed:,} | '
                f'threshold={threshold}'
            )

        total_phrases = 0
        total_edges   = 0

        for level in range(3, 7):
            structs = self.library.get_at_level(level, kind='success')
            for struct in structs:
                # Add compositional edges
                for elem in struct.elements:
                    if (isinstance(elem, str) and len(elem) > 1
                            and self.library.has(elem)):
                        child = self.library.get(elem)
                        self.graph.add_compositional(
                            parent_id    = struct.id,
                            child_id     = elem,
                            parent_level = struct.level,
                            child_level  = child.level,
                        )

                # Co-occurrence on content words
                word_ids = self._extract_content_word_ids(struct.id)
                if len(word_ids) >= 2:
                    for id_a, id_b in combinations(word_ids, 2):
                        became = self._observe(id_a, id_b, level=2, threshold=threshold)
                        self._obs_count += 1
                        if became:
                            total_edges += 1
                            self._edge_count += 1

                total_phrases += 1

                if verbose and total_phrases % 10000 == 0:
                    print(
                        f'  phrases_scanned={total_phrases:,} | '
                        f'graph_edges={len(self.graph):,}'
                    )

        if verbose:
            print(
                f'[CoOccurrence] Scan complete. '
                f'phrases={total_phrases:,} | '
                f'new_edges={total_edges:,} | '
                f'graph_total={len(self.graph):,}'
            )

        return {
            'phrases_scanned': total_phrases,
            'new_edges':       total_edges,
            'graph_total':     len(self.graph),
            'threshold_used':  threshold,
        }

    def stats(self) -> dict:
        return {
            'observations':     self._obs_count,
            'edges_created':    self._edge_count,
            'sentences_seen':   self._sentences_seen,
            'current_threshold': _adaptive_threshold(
                max(self._sentences_seen, 1)
            ),
        }