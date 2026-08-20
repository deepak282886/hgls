"""
main.py

Entry point for the reward graph architecture.

Wires all modules together and exposes a CLI with four commands:

    ingest   — read a dataset and build the graph
    query    — activate + traverse from a text query
    traverse — traverse from a known node ID
    stats    — show graph and training statistics
    sweep    — force a full Bellman sweep
    repl     — interactive query loop

Usage:
    python -m core.main ingest  <dataset_path> [options]
    python -m core.main query   "<text>"       [options]
    python -m core.main traverse <node_id>     [options]
    python -m core.main stats
    python -m core.main sweep
    python -m core.main repl

State persists across runs via:
    graph.db        — SQLite graph (nodes, edges, V values)
    normalizer.json — VNormalizer running statistics
"""

import argparse
import json
import os
import sys
import time

from core.atoms      import Level
from core.graph      import Graph
from core.normalizer import Normalizer
from core.bellman    import BellmanManager
from core.ingestion  import IngestionConfig, ingest, ingest_text
from core.traversal  import traverse, traverse_from_node, replay
from core.activation import activate, activate_from_node


# ─────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────

DEFAULT_DB_PATH   = "graph.db"
DEFAULT_NORM_PATH = "normalizer.json"


# ─────────────────────────────────────────────
# STATE PERSISTENCE
# ─────────────────────────────────────────────

def save_normalizer(normalizer: Normalizer, path: str):
    """Save VNormalizer running statistics to disk."""
    with open(path, "w") as f:
        json.dump(normalizer.to_dict(), f, indent=2)


def load_normalizer(path: str) -> Normalizer:
    """Load VNormalizer from disk, or return fresh if not found."""
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return Normalizer.from_dict(d)
    return Normalizer()


# ─────────────────────────────────────────────
# SYSTEM INIT
# ─────────────────────────────────────────────

def init_system(
    db_path   : str = DEFAULT_DB_PATH,
    norm_path : str = DEFAULT_NORM_PATH,
) -> tuple[Graph, Normalizer, BellmanManager]:
    """
    Open or create the graph, load normalizer, init BellmanManager.
    Called at the start of every command.
    """
    graph      = Graph(db_path)
    normalizer = load_normalizer(norm_path)
    bm         = BellmanManager(graph, normalizer)
    return graph, normalizer, bm


def shutdown_system(
    graph      : Graph,
    normalizer : Normalizer,
    norm_path  : str = DEFAULT_NORM_PATH,
):
    """Persist state and close connections."""
    save_normalizer(normalizer, norm_path)
    graph.close()


# ─────────────────────────────────────────────
# LEVEL PARSER
# ─────────────────────────────────────────────

LEVEL_MAP = {
    "char"      : Level.CHARACTER,
    "character" : Level.CHARACTER,
    "word"      : Level.WORD,
    "sentence"  : Level.SENTENCE,
    "paragraph" : Level.PARAGRAPH,
    "doc"       : Level.DOCUMENT,
    "document"  : Level.DOCUMENT,
}

def parse_level(s: str) -> Level:
    key = s.lower().strip()
    if key not in LEVEL_MAP:
        raise ValueError(
            f"Unknown level '{s}'. "
            f"Choose from: {', '.join(LEVEL_MAP)}"
        )
    return LEVEL_MAP[key]


# ─────────────────────────────────────────────
# COMMAND: INGEST
# ─────────────────────────────────────────────

def cmd_ingest(args):
    """Ingest a dataset file or directory into the graph."""
    graph, norm, bm = init_system(args.db, args.norm)

    level = parse_level(args.level)

    config = IngestionConfig(
        max_level          = level,
        batch_size         = args.batch_size,
        max_nodes          = args.max_nodes,
        min_sentence_words = args.min_words,
        verbose            = True,
        bellman_every      = args.bellman_every,
    )

    print(f"\nIngesting: {args.dataset_path}")
    print(f"Graph DB:  {args.db}")
    print(f"Level:     {level.name}\n")

    try:
        stats = ingest(args.dataset_path, graph, norm, bm, config)
    finally:
        shutdown_system(graph, norm, args.norm)

    return stats


# ─────────────────────────────────────────────
# COMMAND: QUERY
# ─────────────────────────────────────────────

def cmd_query(args):
    """Activate + traverse from a text query."""
    graph, norm, bm = init_system(args.db, args.norm)

    level = parse_level(args.level)
    query = args.query

    gs = graph.stats()
    if gs["node_count"] == 0:
        print("Graph is empty. Run 'ingest' first.")
        shutdown_system(graph, norm, args.norm)
        return

    print(f"\nQuery:  '{query}'")
    print(f"Mode:   {args.mode}")
    print(f"Level:  {level.name}")
    print(f"Graph:  {gs['node_count']} nodes, {gs['edge_count']} edges\n")

    t0     = time.time()
    result = traverse(
        query      = query,
        graph      = graph,
        normalizer = norm,
        level      = level,
        mode       = args.mode,
        max_depth  = args.max_depth,
        max_iter   = args.max_iter,
        max_hops   = args.max_hops,
    )
    elapsed = time.time() - t0

    if not result.path:
        print("No path found. Query did not activate any nodes.")
        print("Try a different query or ingest more data.")
        shutdown_system(graph, norm, args.norm)
        return

    print(f"Found path: {len(result.path)} nodes  "
          f"reward={result.total_reward:.3f}  "
          f"elapsed={elapsed:.4f}s\n")

    print("─" * 60)
    for i, (nid, text) in enumerate(zip(result.path, result.texts)):
        node = graph.get_node(nid)
        v    = node.V if node else 0.0
        print(f"[{i:02d}] id={nid}  V={v:+.3f}")
        print(f"     {text}\n")
    print("─" * 60)

    if args.mode == "mcts" and result.mcts_result:
        mr = result.mcts_result
        print(f"\nMCTS: {mr.iterations} iterations  "
              f"depth={mr.depth}  "
              f"committed_early={mr.committed_early}  "
              f"confidence={mr.confidence:.3f}")

    if args.chain:
        print(f"\nFull text chain:\n")
        print(result.text_chain(separator="\n\n"))

    shutdown_system(graph, norm, args.norm)
    return result


# ─────────────────────────────────────────────
# COMMAND: TRAVERSE
# ─────────────────────────────────────────────

def cmd_traverse(args):
    """Traverse from a known node ID."""
    graph, norm, bm = init_system(args.db, args.norm)

    node = graph.get_node(args.node_id)
    if node is None:
        print(f"Node {args.node_id} not found in graph.")
        shutdown_system(graph, norm, args.norm)
        return

    level = node.level if args.level == "auto" else parse_level(args.level)

    print(f"\nTraversing from node {args.node_id}:")
    print(f"  Text:  {node.text[:70]}")
    print(f"  Level: {level.name}")
    print(f"  V:     {node.V:.3f}")
    print(f"  Mode:  {args.mode}\n")

    result = traverse_from_node(
        node_id    = args.node_id,
        graph      = graph,
        normalizer = norm,
        level      = level,
        mode       = args.mode,
        max_depth  = args.max_depth,
        max_iter   = args.max_iter,
        max_hops   = args.max_hops,
    )

    print(f"Path: {len(result.path)} nodes  "
          f"reward={result.total_reward:.3f}\n")
    print("─" * 60)
    for i, (nid, text) in enumerate(zip(result.path, result.texts)):
        n = graph.get_node(nid)
        v = n.V if n else 0.0
        print(f"[{i:02d}] id={nid}  V={v:+.3f}")
        print(f"     {text}\n")
    print("─" * 60)

    shutdown_system(graph, norm, args.norm)
    return result


# ─────────────────────────────────────────────
# COMMAND: STATS
# ─────────────────────────────────────────────

def cmd_stats(args):
    """Show graph statistics."""
    graph, norm, bm = init_system(args.db, args.norm)

    gs = graph.stats()
    print(f"\nGraph Statistics")
    print("─" * 40)
    print(f"  DB path       : {gs['db_path']}")
    print(f"  Total nodes   : {gs['node_count']}")
    print(f"  Total edges   : {gs['edge_count']}")
    print(f"  Terminal nodes: {gs['terminal_count']}")
    print(f"  Cache size    : {gs['cache_size']}")
    print(f"  Avg V         : {gs['avg_V']:.4f}")
    print(f"  Max V         : {gs['max_V']:.4f}")
    print(f"\n  Level breakdown:")
    for level, count in gs["level_counts"].items():
        if count > 0:
            bar = "█" * min(40, count // max(1, gs["node_count"] // 40))
            print(f"    {level:12s}: {count:8d}  {bar}")

    print(f"\nNormalizer")
    print("─" * 40)
    print(f"  {norm.vnorm}")
    for lv in Level:
        mean = graph.level_mean(lv)
        std  = graph.level_std(lv)
        cnt  = graph._level_stats[lv]["count"]
        if cnt > 0:
            print(f"  {lv.name:12s}: "
                  f"mean={mean:.3f}  std={std:.3f}  n={cnt}")

    print(f"\nTop 5 nodes by V:")
    top = graph.top_nodes(5)
    for n in top:
        print(f"  [{n.node_id:5d}] {n.level.name:10s}  "
              f"V={n.V:+.3f}  {n.text[:50]}")

    print(f"\nBellman manager status:")
    bm_s = bm.status()
    for k, v in bm_s.items():
        print(f"  {k}: {v}")

    shutdown_system(graph, norm, args.norm)


# ─────────────────────────────────────────────
# COMMAND: SWEEP
# ─────────────────────────────────────────────

def cmd_sweep(args):
    """Force a full Bellman sweep."""
    graph, norm, bm = init_system(args.db, args.norm)

    print(f"\nRunning full Bellman sweep on {graph.stats()['node_count']} nodes...")
    stats = bm.sweep(verbose=True)
    print(f"\nSweep complete: {stats}")
    shutdown_system(graph, norm, args.norm)


# ─────────────────────────────────────────────
# COMMAND: REPL
# ─────────────────────────────────────────────

def cmd_repl(args):
    """Interactive query loop."""
    graph, norm, bm = init_system(args.db, args.norm)

    gs = graph.stats()
    print(f"\nReward Graph REPL")
    print(f"  Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")
    print(f"  Mode:  {args.mode}  |  Level: {args.level}")
    print(f"\nCommands:")
    print(f"  <text>         — query the graph")
    print(f"  :node <id>     — traverse from node ID")
    print(f"  :stats         — show stats")
    print(f"  :sweep         — run Bellman sweep")
    print(f"  :mode <m>      — switch mode (greedy/mcts)")
    print(f"  :level <l>     — switch level (sentence/word/...)")
    print(f"  :ingest <path> — ingest a file")
    print(f"  :top           — show top 10 nodes by V")
    print(f"  :quit          — exit")
    print()

    mode  = args.mode
    level = parse_level(args.level)

    while True:
        try:
            line = input("▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        # ── Commands ──────────────────────────
        if line.startswith(":quit") or line.startswith(":exit"):
            print("Exiting.")
            break

        elif line.startswith(":mode "):
            mode = line.split()[1].lower()
            print(f"  Mode → {mode}")

        elif line.startswith(":level "):
            try:
                level = parse_level(line.split()[1])
                print(f"  Level → {level.name}")
            except ValueError as e:
                print(f"  {e}")

        elif line.startswith(":stats"):
            gs = graph.stats()
            print(f"  Nodes: {gs['node_count']}  "
                  f"Edges: {gs['edge_count']}  "
                  f"Avg V: {gs['avg_V']:.3f}  "
                  f"Max V: {gs['max_V']:.3f}")

        elif line.startswith(":sweep"):
            print("  Sweeping...")
            stats = bm.sweep(verbose=False)
            print(f"  Done: {stats}")
            save_normalizer(norm, args.norm)

        elif line.startswith(":top"):
            top = graph.top_nodes(10, level=level)
            if not top:
                top = graph.top_nodes(10)
            print(f"  Top nodes by V ({level.name}):")
            for n in top:
                print(f"    [{n.node_id:5d}] V={n.V:+.3f}  {n.text[:55]}")

        elif line.startswith(":node "):
            try:
                nid = int(line.split()[1])
                result = traverse_from_node(
                    nid, graph, norm,
                    level     = level,
                    mode      = mode,
                    max_depth = args.max_depth,
                    max_iter  = args.max_iter,
                    max_hops  = args.max_hops,
                )
                _print_result(result, graph)
            except (ValueError, IndexError):
                print("  Usage: :node <integer_id>")

        elif line.startswith(":ingest "):
            path = line[8:].strip()
            if not os.path.exists(path):
                print(f"  File not found: {path}")
            else:
                cfg = IngestionConfig(
                    max_level = level, verbose=True, batch_size=50
                )
                ingest(path, graph, norm, bm, cfg)
                save_normalizer(norm, args.norm)

        # ── Text query ────────────────────────
        else:
            gs = graph.stats()
            if gs["node_count"] == 0:
                print("  Graph is empty. Use :ingest <path> to load data.")
                continue

            result = traverse(
                query      = line,
                graph      = graph,
                normalizer = norm,
                level      = level,
                mode       = mode,
                max_depth  = args.max_depth,
                max_iter   = args.max_iter,
                max_hops   = args.max_hops,
            )
            _print_result(result, graph)

    shutdown_system(graph, norm, args.norm)


def _print_result(result, graph):
    """Print a TraversalResult in REPL format."""
    if not result.path:
        print("  No path found.")
        return

    print(f"\n  Path ({result.mode})  "
          f"steps={len(result.path)}  "
          f"reward={result.total_reward:.3f}  "
          f"elapsed={result.elapsed:.4f}s\n")

    for i, (nid, text) in enumerate(zip(result.path, result.texts)):
        node = graph.get_node(nid)
        v    = node.V if node else 0.0
        lvl  = node.level.name[:4] if node else "????"
        print(f"  [{i:02d}] {lvl}  id={nid}  V={v:+.3f}")
        # Wrap text at 70 chars
        words = text.split()
        line  = "       "
        for w in words:
            if len(line) + len(w) > 70:
                print(line)
                line = "       "
            line += w + " "
        if line.strip():
            print(line)
        print()


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m core.main",
        description = "Reward Graph Architecture — continuous learning graph",
    )

    # Global options
    parser.add_argument("--db",   default=DEFAULT_DB_PATH,
                        help=f"Graph DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--norm", default=DEFAULT_NORM_PATH,
                        help=f"Normalizer state path (default: {DEFAULT_NORM_PATH})")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest a dataset")
    p_ingest.add_argument("dataset_path",
                          help="Path to .txt, .jsonl file, or directory")
    p_ingest.add_argument("--level",        default="sentence",
                          help="Max ingestion level (default: sentence)")
    p_ingest.add_argument("--batch-size",   type=int, default=100,
                          help="Docs per Bellman sweep (default: 100)")
    p_ingest.add_argument("--max-nodes",    type=int, default=0,
                          help="Stop after N nodes (0=unlimited)")
    p_ingest.add_argument("--min-words",    type=int, default=3,
                          help="Min words per sentence (default: 3)")
    p_ingest.add_argument("--bellman-every",type=int, default=10,
                          help="Delta propagation every N edges (default: 10)")

    # ── query ───────────────────────────────
    p_query = sub.add_parser("query", help="Query the graph")
    p_query.add_argument("query",      help="Text query")
    p_query.add_argument("--mode",     default="mcts",
                         choices=["mcts","greedy"])
    p_query.add_argument("--level",    default="sentence")
    p_query.add_argument("--max-depth",type=int, default=50)
    p_query.add_argument("--max-iter", type=int, default=100)
    p_query.add_argument("--max-hops", type=int, default=6)
    p_query.add_argument("--chain",    action="store_true",
                         help="Print full text chain")

    # ── traverse ────────────────────────────
    p_trav = sub.add_parser("traverse", help="Traverse from node ID")
    p_trav.add_argument("node_id",    type=int, help="Starting node ID")
    p_trav.add_argument("--mode",     default="mcts",
                        choices=["mcts","greedy"])
    p_trav.add_argument("--level",    default="auto",
                        help="Level (default: auto from node)")
    p_trav.add_argument("--max-depth",type=int, default=50)
    p_trav.add_argument("--max-iter", type=int, default=100)
    p_trav.add_argument("--max-hops", type=int, default=6)

    # ── stats ───────────────────────────────
    sub.add_parser("stats", help="Show graph statistics")

    # ── sweep ───────────────────────────────
    sub.add_parser("sweep", help="Force Bellman sweep")

    # ── repl ────────────────────────────────
    p_repl = sub.add_parser("repl", help="Interactive query loop")
    p_repl.add_argument("--mode",     default="mcts",
                        choices=["mcts","greedy"])
    p_repl.add_argument("--level",    default="sentence")
    p_repl.add_argument("--max-depth",type=int, default=50)
    p_repl.add_argument("--max-iter", type=int, default=100)
    p_repl.add_argument("--max-hops", type=int, default=6)

    return parser


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    cmd_map = {
        "ingest"  : cmd_ingest,
        "query"   : cmd_query,
        "traverse": cmd_traverse,
        "stats"   : cmd_stats,
        "sweep"   : cmd_sweep,
        "repl"    : cmd_repl,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()