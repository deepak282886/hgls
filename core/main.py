"""
main.py

Entry point for the reward graph architecture.

Commands:

    bootstrap  — ingest complete characters + full dictionary (run first)
    ingest     — ingest a text dataset into the graph
    axiom      — extract and ingest axiom corpus from frontier model
    curriculum — run reasoning agent to generate sentences
    distill    — run Ollama agent distillation loop
    collide    — run continuous collision loop
    query      — activate + traverse from a text query
    traverse   — traverse from a known node ID
    stats      — show graph statistics
    sweep      — force a full Bellman sweep
    repl       — interactive query loop

Recommended startup sequence:
    1. python -m core.main bootstrap
    2. python -m core.main axiom extract --together-key KEY
    3. python -m core.main axiom ingest
    4. python -m core.main curriculum run --mode guided
    5. python -m core.main ingest <dataset>
    6. python -m core.main collide --commandments commandments.json
    7. python -m core.main query "<question>"
"""

import argparse
import json
import os
import sys
import time

from core.atoms      import Level, level_from_name
from core.graph      import Graph
from core.normalizer import Normalizer
from core.bellman    import BellmanManager
from core.ingestion  import IngestionConfig, ingest, ingest_text, ingest_distillation_batch
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
    try:
        with open(path, "w") as f:
            json.dump(normalizer.to_dict(), f, indent=2)
    except Exception:
        pass


def load_normalizer(path: str) -> Normalizer:
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            return Normalizer.from_dict(d)
        except Exception:
            pass
    return Normalizer()


# ─────────────────────────────────────────────
# SYSTEM INIT / SHUTDOWN
# ─────────────────────────────────────────────

def init_system(
    db_path   : str = DEFAULT_DB_PATH,
    norm_path : str = DEFAULT_NORM_PATH,
) -> tuple[Graph, Normalizer, BellmanManager]:
    graph      = Graph(db_path)
    normalizer = load_normalizer(norm_path)
    bm         = BellmanManager(graph, normalizer)
    return graph, normalizer, bm


def shutdown_system(
    graph      : Graph,
    normalizer : Normalizer,
    norm_path  : str = DEFAULT_NORM_PATH,
):
    save_normalizer(normalizer, norm_path)
    graph.close()


# ─────────────────────────────────────────────
# LEVEL PARSER
# ─────────────────────────────────────────────

def parse_level(s: str) -> Level:
    """Parse a level name string. Supports short forms."""
    aliases = {
        "char"  : "character",
        "sent"  : "sentence",
        "word"  : "word",
        "ax"    : "axiom",
    }
    key = s.lower().strip()
    key = aliases.get(key, key)
    try:
        return level_from_name(key)
    except ValueError:
        raise ValueError(
            f"Unknown level '{s}'. "
            f"Choose from: character, word, sentence, axiom"
        )


# ─────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────

def _print_result(result, graph):
    """Print a TraversalResult."""
    if not result.path:
        print("  No path found.")
        return

    print(f"\n  Path ({result.mode})  "
          f"steps={len(result.path)}  "
          f"reward={result.total_reward:.3f}  "
          f"elapsed={result.elapsed:.4f}s\n")

    for i, (nid, text) in enumerate(zip(result.path, result.texts)):
        node = graph.get_node(nid)
        v    = node.V    if node else 0.0
        lvl  = node.level.name[:4] if node else "????"
        print(f"  [{i:02d}] {lvl}  id={nid}  V={v:+.3f}")
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

    if result.mcts_result:
        mr = result.mcts_result
        print(f"  MCTS: {mr.iterations} iter  "
              f"depth={mr.depth}  "
              f"confidence={mr.confidence:.3f}")


# ─────────────────────────────────────────────
# COMMAND: BOOTSTRAP
# ─────────────────────────────────────────────

def cmd_bootstrap(args):
    """Bootstrap the graph with complete characters and dictionary words."""
    from core.bootstrap import run_bootstrap

    run_bootstrap(
        db_path        = args.db,
        norm_path      = args.norm,
        word_list_path = args.words,
        skip_chars     = args.skip_chars,
        skip_words     = args.skip_words,
        verify_only    = args.verify_only,
        verbose        = True,
    )


# ─────────────────────────────────────────────
# COMMAND: AXIOM
# ─────────────────────────────────────────────

def cmd_axiom(args):
    """Extract and manage the axiom corpus."""
    import core.axiom as axiom_mod
    old_argv = sys.argv
    sys.argv  = ["axiom"] + (args.axiom_args or [])
    try:
        axiom_mod.main()
    finally:
        sys.argv = old_argv


# ─────────────────────────────────────────────
# COMMAND: CURRICULUM
# ─────────────────────────────────────────────

def cmd_curriculum(args):
    """Run the curriculum agent to generate reasoning sentences."""
    import core.curriculum as curr_mod
    old_argv = sys.argv
    sys.argv  = ["curriculum"] + (args.curriculum_args or [])
    try:
        curr_mod.main()
    finally:
        sys.argv = old_argv


# ─────────────────────────────────────────────
# COMMAND: INGEST
# ─────────────────────────────────────────────

def cmd_ingest(args):
    """Ingest a text dataset into the graph."""
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
    """Activate + single MCTS traverse from a text query."""
    graph, norm, bm = init_system(args.db, args.norm)

    level = parse_level(args.level)
    query = args.query

    gs = graph.stats()
    if gs["node_count"] == 0:
        print("Graph is empty. Run 'bootstrap' then 'ingest' first.")
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
        print("Try a different query or run bootstrap/axiom/curriculum first.")
        shutdown_system(graph, norm, args.norm)
        return

    print(f"Path: {len(result.path)} steps  "
          f"reward={result.total_reward:.3f}  "
          f"elapsed={elapsed:.4f}s\n")

    print("─" * 60)
    for i, (nid, text) in enumerate(zip(result.path, result.texts)):
        node = graph.get_node(nid)
        v    = node.V    if node else 0.0
        lvl  = node.level.name[:4] if node else "????"
        print(f"[{i:02d}] {lvl}  id={nid}  V={v:+.3f}")
        print(f"     {text}\n")
    print("─" * 60)

    if args.mode == "mcts" and result.mcts_result:
        mr = result.mcts_result
        print(f"\nMCTS: {mr.iterations} iter  "
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
    print(f"  Avg V         : {gs['avg_V']:.4f}")
    print(f"  Max V         : {gs['max_V']:.4f}")
    print(f"\n  Level breakdown:")
    for level, count in gs["level_counts"].items():
        if count > 0:
            bar = "█" * min(40, count // max(1, gs["node_count"] // 40))
            print(f"    {level:12s}: {count:8d}  {bar}")

    print(f"\nTop 5 nodes by V:")
    top = graph.top_nodes(5)
    for n in top:
        print(f"  [{n.node_id:5d}] {n.level.name:10s}  "
              f"V={n.V:+.3f}  {n.text[:50]}")

    print(f"\nBellman manager status:")
    for k, v in bm.status().items():
        print(f"  {k}: {v}")

    shutdown_system(graph, norm, args.norm)


# ─────────────────────────────────────────────
# COMMAND: SWEEP
# ─────────────────────────────────────────────

def cmd_sweep(args):
    """Force a full Bellman sweep."""
    graph, norm, bm = init_system(args.db, args.norm)

    n = graph.stats()["node_count"]
    print(f"\nRunning full Bellman sweep on {n} nodes...")
    stats = bm.sweep(verbose=True)
    print(f"\nSweep complete: {stats}")

    shutdown_system(graph, norm, args.norm)


# ─────────────────────────────────────────────
# COMMAND: COLLIDE
# ─────────────────────────────────────────────

def cmd_collide(args):
    """Run the continuous collision loop."""
    from core.collision import run_collision_loop

    graph, norm, bm = init_system(args.db, args.norm)

    gs = graph.stats()
    if gs["node_count"] == 0:
        print("Graph is empty. Run 'bootstrap' first.")
        shutdown_system(graph, norm, args.norm)
        return

    try:
        stats = run_collision_loop(
            graph             = graph,
            normalizer        = norm,
            bm                = bm,
            commandments_path = args.commandments,
            max_iter          = args.max_iter,
            verbose           = True,
        )
    finally:
        shutdown_system(graph, norm, args.norm)

    return stats


# ─────────────────────────────────────────────
# COMMAND: DISTILL
# ─────────────────────────────────────────────

def cmd_distill(args):
    """Run the Ollama reasoning agent distillation loop."""
    from core.distill import Distiller, DistillConfig

    graph, norm, bm = init_system(args.db, args.norm)

    config = DistillConfig(
        model_name       = args.model,
        prompts_path     = args.prompts,
        reward_threshold = args.threshold,
        batch_size       = args.batch_size,
        max_new_tokens   = args.max_tokens,
        temperature      = args.temperature,
        verbose          = True,
    )

    level = parse_level(args.level)

    ingest_config = IngestionConfig(
        max_level  = level,
        verbose    = False,
        batch_size = 100,
    )

    print(f"\nDistillation agent")
    print(f"  Model:     {args.model}")
    print(f"  Prompts:   {args.prompts or '(built-in curriculum)'}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Level:     {level.name}\n")

    distiller = Distiller(config)
    total_pos = 0
    total_neg = 0
    batch_num = 0

    try:
        for pos_texts, neg_texts in distiller.generate_batches():
            batch_num += 1
            pos_stats, neg_stats = ingest_distillation_batch(
                pos_texts, neg_texts, graph, norm, bm,
                ingest_config, verbose=False,
            )
            total_pos += pos_stats.nodes_created
            total_neg += neg_stats.nodes_created

            gs = graph.stats()
            print(f"  Batch {batch_num:4d}  "
                  f"+{pos_stats.nodes_created} pos  "
                  f"+{neg_stats.nodes_created} neg  "
                  f"graph={gs['node_count']} nodes")

            save_normalizer(norm, args.norm)

    except KeyboardInterrupt:
        print("\nDistillation interrupted.")

    print(f"\nDistillation complete.")
    print(f"  Positive nodes: {total_pos}")
    print(f"  Contrastive nodes: {total_neg}")

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
    print(f"  :top           — show top 10 nodes by V")
    print(f"  :stats         — show graph stats")
    print(f"  :sweep         — run Bellman sweep")
    print(f"  :mode <m>      — switch mode (greedy/mcts)")
    print(f"  :level <l>     — switch level (character/word/sentence/axiom)")
    print(f"  :ingest <path> — ingest a file")
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
            print(f"  Levels: " + "  ".join(
                f"{lv}={cnt}" for lv, cnt in gs["level_counts"].items()
                if cnt > 0
            ))

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
                lvl = n.level.name[:4]
                print(f"    [{n.node_id:5d}] {lvl}  V={n.V:+.3f}  {n.text[:55]}")

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

        else:
            gs = graph.stats()
            if gs["node_count"] == 0:
                print("  Graph is empty. Use :ingest <path> or run bootstrap.")
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


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m core.main",
        description = "Reward Graph Architecture — axiom-structured reasoning graph",
    )

    parser.add_argument("--db",   default=DEFAULT_DB_PATH,
                        help=f"Graph DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--norm", default=DEFAULT_NORM_PATH,
                        help=f"Normalizer path (default: {DEFAULT_NORM_PATH})")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── bootstrap ───────────────────────────
    p_boot = sub.add_parser("bootstrap",
                            help="Ingest complete character + word layers (run first)")
    p_boot.add_argument("--words",       default=None,
                        help="Path to word list file (auto-detected if not given)")
    p_boot.add_argument("--skip-chars",  action="store_true",
                        help="Skip character ingestion")
    p_boot.add_argument("--skip-words",  action="store_true",
                        help="Skip word ingestion")
    p_boot.add_argument("--verify-only", action="store_true",
                        help="Only verify existing bootstrap")

    # ── axiom ───────────────────────────────
    p_axiom = sub.add_parser("axiom",
                             help="Extract/ingest/validate axiom corpus")
    p_axiom.add_argument("axiom_args", nargs=argparse.REMAINDER,
                         help="Arguments passed to axiom.py "
                              "(extract/ingest/list/validate)")

    # ── curriculum ──────────────────────────
    p_curr = sub.add_parser("curriculum",
                            help="Run reasoning agent sentence generation")
    p_curr.add_argument("curriculum_args", nargs=argparse.REMAINDER,
                        help="Arguments passed to curriculum.py "
                             "(run/ingest/list)")

    # ── ingest ──────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest a text dataset")
    p_ingest.add_argument("dataset_path",
                          help="Path to .txt, .jsonl, or directory")
    p_ingest.add_argument("--level",         default="sentence",
                          choices=["character","word","sentence","axiom"],
                          help="Max ingestion level (default: sentence)")
    p_ingest.add_argument("--batch-size",    type=int, default=100)
    p_ingest.add_argument("--max-nodes",     type=int, default=0)
    p_ingest.add_argument("--min-words",     type=int, default=3)
    p_ingest.add_argument("--bellman-every", type=int, default=10)

    # ── query ───────────────────────────────
    p_query = sub.add_parser("query", help="Query the graph")
    p_query.add_argument("query",      help="Text query")
    p_query.add_argument("--mode",     default="mcts",
                         choices=["mcts","greedy"])
    p_query.add_argument("--level",    default="sentence",
                         choices=["character","word","sentence","axiom"])
    p_query.add_argument("--max-depth",type=int, default=50)
    p_query.add_argument("--max-iter", type=int, default=100)
    p_query.add_argument("--max-hops", type=int, default=6)
    p_query.add_argument("--chain",    action="store_true",
                         help="Print full text chain at end")

    # ── traverse ────────────────────────────
    p_trav = sub.add_parser("traverse", help="Traverse from a node ID")
    p_trav.add_argument("node_id",    type=int)
    p_trav.add_argument("--mode",     default="mcts",
                        choices=["mcts","greedy"])
    p_trav.add_argument("--level",    default="auto")
    p_trav.add_argument("--max-depth",type=int, default=50)
    p_trav.add_argument("--max-iter", type=int, default=100)
    p_trav.add_argument("--max-hops", type=int, default=6)

    # ── collide ─────────────────────────────
    p_collide = sub.add_parser("collide", help="Run collision loop")
    p_collide.add_argument("--max-iter",     type=int, default=0,
                           help="Stop after N iterations (0=forever)")
    p_collide.add_argument("--commandments", default=None,
                           help="Path to commandments JSON file")

    # ── distill ─────────────────────────────
    p_distill = sub.add_parser("distill", help="Run Ollama reasoning agent")
    p_distill.add_argument("--model",       default="deepseek-r1:1.5b",
                           help="Ollama model (default: deepseek-r1:1.5b)")
    p_distill.add_argument("--prompts",     default="",
                           help="JSONL prompts file (empty = built-in curriculum)")
    p_distill.add_argument("--threshold",   type=float, default=0.0)
    p_distill.add_argument("--batch-size",  type=int,   default=4)
    p_distill.add_argument("--max-tokens",  type=int,   default=512)
    p_distill.add_argument("--temperature", type=float, default=0.8)
    p_distill.add_argument("--level",       default="sentence",
                           choices=["character","word","sentence","axiom"])

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
        "bootstrap"  : cmd_bootstrap,
        "axiom"      : cmd_axiom,
        "curriculum" : cmd_curriculum,
        "ingest"     : cmd_ingest,
        "query"      : cmd_query,
        "traverse"   : cmd_traverse,
        "collide"    : cmd_collide,
        "distill"    : cmd_distill,
        "stats"      : cmd_stats,
        "sweep"      : cmd_sweep,
        "repl"       : cmd_repl,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
