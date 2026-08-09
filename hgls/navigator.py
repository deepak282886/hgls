"""
navigator.py — Graph Navigator and Traversal Strategy Selector.

The PFC equivalent. Doesn't store memories — navigates them.

When a new input arrives:
  1. Locate it in the graph — which nodes does it activate?
  2. Read local topology — dense familiar or sparse frontier?
  3. Select traversal strategy — exploit or explore?
  4. Return strategy weights for hypothesis generation.

Traversal strategies:
  exploit    — mine known edges, mutate nearby structures (dense regions)
  decompose  — break input using validated lower-level structures
  extend     — follow existing edges outward from activated nodes
  cross      — bridge to a distant region with structural similarity
  compress   — propose this input as a new primitive for the next level

Starts with primitive topology reading (density threshold).
Learns which strategies work for which topology over time.
Same algorithm — strategy selection is itself learned.
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hgls.graph              import MemoryGraph, Edge
    from hgls.library            import Library
    from hgls.emotional_evaluator import EmotionalEvaluator

# Strategy weight vector keys
STRATEGIES = ('exploit', 'decompose', 'extend', 'cross', 'compress')

# Default weights when graph is empty or maturity is low
_DEFAULT_WEIGHTS = {
    'exploit':    0.4,
    'decompose':  0.35,
    'extend':     0.15,
    'cross':      0.05,
    'compress':   0.05,
}


class MetaStructure:
    """
    What the navigator knows about an input before hypothesis generation.
    """
    def __init__(
        self,
        input_text:       str,
        activated_nodes:  List[str],
        topology:         str,           # dense | moderate | sparse | isolated
        avg_density:      float,
        strategy_weights: Dict[str, float],
        region_type:      str,           # known | frontier | bridge | unknown
    ):
        self.input_text       = input_text
        self.activated_nodes  = activated_nodes
        self.topology         = topology
        self.avg_density      = avg_density
        self.strategy_weights = strategy_weights
        self.region_type      = region_type

    def __repr__(self):
        return (
            f"MetaStructure(topology={self.topology}, "
            f"region={self.region_type}, "
            f"nodes={len(self.activated_nodes)}, "
            f"top_strategy={max(self.strategy_weights, key=self.strategy_weights.get)})"
        )


class GraphNavigator:
    """
    Reads graph topology and selects traversal strategies.
    Learns which strategies work for which topology over time.
    """

    def __init__(
        self,
        graph:    Optional['MemoryGraph']       = None,
        library:  Optional['Library']           = None,
        evaluator: Optional['EmotionalEvaluator'] = None,
    ):
        self.graph     = graph
        self.library   = library
        self.evaluator = evaluator

        # Strategy performance history: strategy → [outcome_scores]
        self._strategy_history: Dict[str, List[float]] = {s: [] for s in STRATEGIES}
        self._nav_count = 0

    # ── Main: analyse input ───────────────────────────────────────

    def analyse(self, input_text: str, level: int = 3) -> MetaStructure:
        """
        Analyse an input and return MetaStructure with traversal strategy.
        This is called by hypothesis engine before generating hypotheses.
        """
        self._nav_count += 1

        # Step 1: find which library structures this input activates
        activated = self._activate_nodes(input_text, level)

        # Step 2: read topology around activated nodes
        avg_density, topology = self._read_topology(activated)

        # Step 3: classify the region
        region_type = self._classify_region(activated, avg_density)

        # Step 4: select strategy weights
        weights = self._select_strategy(topology, region_type, avg_density)

        return MetaStructure(
            input_text       = input_text,
            activated_nodes  = activated,
            topology         = topology,
            avg_density      = avg_density,
            strategy_weights = weights,
            region_type      = region_type,
        )

    # ── Node activation ───────────────────────────────────────────

    def _activate_nodes(self, input_text: str, level: int) -> List[str]:
        """
        Find library structures that this input overlaps with.
        A structure is activated if its generated text shares words with input.
        """
        if not self.library:
            return []

        input_words = set(input_text.lower().split())
        activated   = []

        # Check structures at this level and one below
        for lvl in range(max(0, level - 1), level + 1):
            for struct in self.library.get_at_level(lvl, kind='success'):
                text       = struct.generate(self.library)
                text_words = set(text.lower().split())
                overlap    = len(input_words & text_words)
                if overlap >= 2:
                    activated.append(struct.id)
                if len(activated) >= 50:  # cap for speed
                    break

        return activated

    # ── Topology reading ──────────────────────────────────────────

    def _read_topology(
        self, activated_nodes: List[str]
    ) -> Tuple[float, str]:
        """
        Measure average density of activated region.
        Returns (avg_density, topology_label).
        """
        if not self.graph or not activated_nodes:
            return 0.0, 'isolated'

        densities = [
            self.graph.density(nid) for nid in activated_nodes[:20]
        ]
        avg = sum(densities) / max(len(densities), 1)

        if avg >= 0.6:
            topology = 'dense'
        elif avg >= 0.3:
            topology = 'moderate'
        elif avg > 0.0:
            topology = 'sparse'
        else:
            topology = 'isolated'

        return avg, topology

    # ── Region classification ─────────────────────────────────────

    def _classify_region(
        self, activated_nodes: List[str], avg_density: float
    ) -> str:
        """
        Classify what kind of territory this input is in.

        known    — dense, well-connected, well-explored
        frontier — moderate density, edges being formed
        bridge   — input activates nodes from two disconnected dense regions
        unknown  — sparse or isolated, rarely visited
        """
        if not self.graph or not activated_nodes:
            return 'unknown'

        if avg_density >= 0.6:
            return 'known'

        if avg_density >= 0.3:
            # Check if activated nodes span disconnected regions
            if len(activated_nodes) >= 4:
                region_a = set(self.graph.get_region(activated_nodes[0], depth=2).keys())
                for nid in activated_nodes[2:]:
                    if nid not in region_a:
                        return 'bridge'
            return 'frontier'

        return 'unknown'

    # ── Strategy selection ────────────────────────────────────────

    def _select_strategy(
        self,
        topology:    str,
        region_type: str,
        avg_density: float,
    ) -> Dict[str, float]:
        """
        Select traversal strategy weights based on topology.

        Dense + known     → exploit heavily, light exploration
        Moderate frontier → balanced exploit + decompose + extend
        Bridge            → cross-region strategy gets high weight
        Sparse + unknown  → decompose + extend, minimal exploit
        Isolated          → decompose only (rebuild from lower levels)

        These weights are starting points. They drift based on
        what actually works (_update_from_outcome).
        """
        # Start from defaults
        w = dict(_DEFAULT_WEIGHTS)

        if topology == 'dense' and region_type == 'known':
            w['exploit']   = 0.55
            w['decompose'] = 0.25
            w['extend']    = 0.10
            w['cross']     = 0.05
            w['compress']  = 0.05

        elif region_type == 'bridge':
            w['exploit']   = 0.15
            w['decompose'] = 0.25
            w['extend']    = 0.20
            w['cross']     = 0.35  # bridge → cross-region connection
            w['compress']  = 0.05

        elif topology == 'moderate' and region_type == 'frontier':
            w['exploit']   = 0.30
            w['decompose'] = 0.35
            w['extend']    = 0.25
            w['cross']     = 0.05
            w['compress']  = 0.05

        elif topology in ('sparse', 'isolated'):
            w['exploit']   = 0.15
            w['decompose'] = 0.55  # rebuild from known lower-level pieces
            w['extend']    = 0.20
            w['cross']     = 0.05
            w['compress']  = 0.05

        # Blend with learned performance history
        w = self._blend_with_history(w)

        # Normalise
        total = sum(w.values())
        return {s: round(v / total, 3) for s, v in w.items()}

    def _blend_with_history(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Blend topology-derived weights with learned strategy performance.
        Strategies that consistently produce successes get higher weight.
        """
        for strategy in STRATEGIES:
            hist = self._strategy_history[strategy]
            if len(hist) < 20:
                continue
            avg_outcome = sum(hist[-20:]) / 20
            # Shift weight toward strategies with avg_outcome > 0.5
            delta = (avg_outcome - 0.5) * 0.1
            weights[strategy] = max(0.01, weights[strategy] + delta)
        return weights

    # ── Learning from outcomes ────────────────────────────────────

    def update_from_outcome(self, strategy: str, outcome_score: float) -> None:
        """
        Record that using a strategy produced a given outcome score.
        This is how the navigator learns which strategies work where.
        """
        if strategy not in self._strategy_history:
            return
        hist = self._strategy_history[strategy]
        hist.append(outcome_score)
        if len(hist) > 200:
            hist.pop(0)

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        avg_outcomes = {}
        for s, hist in self._strategy_history.items():
            if hist:
                avg_outcomes[s] = round(sum(hist[-20:]) / len(hist[-20:]), 3)
        return {
            'nav_count':    self._nav_count,
            'avg_outcomes': avg_outcomes,
        }
