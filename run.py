"""
run.py

Entry point for the living system.

Commands:
    python run.py train  <path>     — load CHILDES data and train
    python run.py chat              — interactive conversation
    python run.py status            — show graph health
    python run.py train-chat <path> — load data then go straight to chat

Examples:
    # Load Brown corpus and chat
    python run.py train-chat Brown.zip

    # Load all of Eng-NA collection
    python run.py train /path/to/Eng-NA/

    # Resume conversation with existing graph
    python run.py chat

    # Check graph health
    python run.py status

The graph persists in graph.db — every session builds on the last.
Stop and restart anytime. The graph keeps growing.
"""

import sys
import argparse
import time

from core.graph      import Graph
from core.loop       import LivingLoop, LoopConfig
from core.interface  import Interface
from childes_loader  import CHILDESLoader, load_childes


# ─────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────

DEFAULT_DB = "graph.db"


# ─────────────────────────────────────────────
# SYSTEM INIT
# ─────────────────────────────────────────────

def init_system(
    db_path : str  = DEFAULT_DB,
    verbose : bool = True,
) -> tuple[Graph, LivingLoop, Interface]:
    """Initialize graph, loop, and interface."""
    graph  = Graph(db_path)
    config = LoopConfig(
        compress_every   = 50,
        project_every    = 3,
        decay_every_sec  = 30.0,
        verbose          = verbose,
    )
    loop      = LivingLoop(graph, config)
    interface = Interface(loop)
    return graph, loop, interface


# ─────────────────────────────────────────────
# COMMAND: TRAIN
# ─────────────────────────────────────────────

def cmd_train(args):
    """Load CHILDES data into the graph."""
    graph, loop, interface = init_system(
        db_path = args.db,
        verbose = not args.quiet,
    )

    loader = CHILDESLoader(
        loop,
        min_utt_words = args.min_words,
        max_utt_words = args.max_words,
    )

    loop.start()

    print(f"\nLoading: {args.path}")
    print(f"Graph DB: {args.db}")
    if args.limit:
        print(f"Limit: {args.limit} utterances")
    print()

    try:
        if args.recursive:
            stats = loader.load_all(
                args.path,
                verbose = not args.quiet,
                limit   = args.limit,
            )
        else:
            stats = loader.load(
                args.path,
                verbose = not args.quiet,
                limit   = args.limit,
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    loop.stop()
    graph.close()


# ─────────────────────────────────────────────
# COMMAND: CHAT
# ─────────────────────────────────────────────

def cmd_chat(args):
    """Start interactive conversation."""
    graph, loop, interface = init_system(
        db_path = args.db,
        verbose = args.verbose,
    )

    gs = graph.stats()
    print(f"\nGraph loaded: {gs['nodes']} nodes  "
          f"{gs['edges']} edges  "
          f"max_level={gs['max_level']}")

    if gs["nodes"] == 0:
        print("\nGraph is empty. Run 'python run.py train <path>' first.")
        print("Or start chatting — the system will learn from your input.\n")

    interface.conversation()
    graph.close()


# ─────────────────────────────────────────────
# COMMAND: TRAIN-CHAT
# Load data then immediately start chatting
# ─────────────────────────────────────────────

def cmd_train_chat(args):
    """Load CHILDES data then start interactive conversation."""
    graph, loop, interface = init_system(
        db_path = args.db,
        verbose = not args.quiet,
    )

    loader = CHILDESLoader(
        loop,
        min_utt_words = args.min_words,
        max_utt_words = args.max_words,
    )

    loop.start()

    print(f"\nLoading: {args.path}")
    print(f"Graph DB: {args.db}\n")

    try:
        stats = loader.load(
            args.path,
            verbose = not args.quiet,
            limit   = args.limit,
        )
    except KeyboardInterrupt:
        print("\nLoading interrupted. Starting chat with partial graph...\n")

    # Go straight to conversation
    interface.conversation()
    graph.close()


# ─────────────────────────────────────────────
# COMMAND: STATUS
# ─────────────────────────────────────────────

def cmd_status(args):
    """Show graph health and statistics."""
    graph  = Graph(args.db)
    loop   = LivingLoop(graph)

    gs     = graph.stats()
    health = loop.decay.health()

    print(f"\nGraph: {args.db}")
    print("─" * 40)
    print(f"  Nodes        : {gs['nodes']}")
    print(f"  Edges        : {gs['edges']}")
    print(f"  Hyperedges   : {gs['hyperedges']}")
    print(f"  Max level    : {gs['max_level']}")
    print(f"  Avg weight   : {gs['avg_weight']:.4f}")
    print()
    print(f"  State breakdown:")
    for state, count in gs["state_counts"].items():
        bar = "█" * min(30, int(count / max(gs["nodes"], 1) * 30))
        print(f"    {state:12s}: {count:6d}  {bar}")
    print()
    print(f"  Health:")
    print(f"    stability   : {health['stability_ratio']:.3f}")
    print(f"    churn       : {health['churn_ratio']:.3f}")
    print(f"    hypothetical: {health['hypothesis_ratio']:.3f}")
    print(f"    is_healthy  : {health['is_healthy']}")

    # Show top nodes by weight
    rows = graph._conn.execute(
        "SELECT id, level, raw, weight, state FROM nodes "
        "WHERE state='stable' ORDER BY weight DESC LIMIT 10"
    ).fetchall()

    if rows:
        print(f"\n  Top 10 nodes by weight:")
        for row in rows:
            surface = graph.surface(row["id"])
            print(f"    [{row['id']:4d}] L{row['level']}  "
                  f"w={row['weight']:.3f}  '{surface[:40]}'")

    # Show schemas
    if gs["hyperedges"] > 0:
        from core.atoms import SLOT
        import json
        schemas = graph._conn.execute(
            "SELECT id, level, structure, instances, weight FROM hyperedges "
            "WHERE state='stable' ORDER BY weight DESC LIMIT 5"
        ).fetchall()

        if schemas:
            print(f"\n  Top schemas:")
            for s in schemas:
                structure = json.loads(s["structure"])
                instances = json.loads(s["instances"])
                parts = []
                for pos in structure:
                    if pos == SLOT:
                        parts.append("___")
                    elif isinstance(pos, int):
                        parts.append(f"'{graph.surface(pos)}'")
                    else:
                        parts.append(str(pos))
                print(f"    [{s['id']}] L{s['level']}  "
                      f"[{', '.join(parts)}]  "
                      f"instances={len(instances)}  "
                      f"w={s['weight']:.3f}")

    graph.close()


# ─────────────────────────────────────────────
# COMMAND: SWEEP
# Force a full compression + projection cycle
# ─────────────────────────────────────────────

def cmd_sweep(args):
    """Force a full compression and projection sweep."""
    from core.compression import Compression
    from core.projection  import Projection
    from core.decay       import Decay
    from core.normalizer  import Normalizer

    graph = Graph(args.db)
    loop  = LivingLoop(graph)

    print(f"\nRunning full sweep on {args.db}...")
    print(f"Graph: {graph.stats()['nodes']} nodes\n")

    t0 = time.time()

    print("Compression...")
    comp = loop.compression
    r1   = comp.run()
    print(f"  {r1}\n")

    print("Projection...")
    proj = loop.projection
    r2   = proj.run()
    print(f"  {r2}\n")

    print("Decay...")
    dec = loop.decay
    r3  = dec.run()
    print(f"  {r3}\n")

    elapsed = time.time() - t0
    gs      = graph.stats()
    print(f"Done in {elapsed:.2f}s")
    print(f"Graph: {gs['nodes']} nodes  "
          f"{gs['edges']} edges  "
          f"max_level={gs['max_level']}")

    graph.close()


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python run.py",
        description = "Living System — learns from conversation",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Graph database path (default: {DEFAULT_DB})"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ───────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Load CHILDES data into the graph"
    )
    p_train.add_argument("path", help="Path to .zip, .cha, or directory")
    p_train.add_argument("--limit",     type=int, default=0,
                         help="Max utterances to load (0=unlimited)")
    p_train.add_argument("--min-words", type=int, default=2,
                         dest="min_words")
    p_train.add_argument("--max-words", type=int, default=50,
                         dest="max_words")
    p_train.add_argument("--recursive", action="store_true",
                         help="Recursively load all corpora in directory")
    p_train.add_argument("--quiet",     action="store_true")

    # ── chat ────────────────────────────────
    p_chat = sub.add_parser(
        "chat",
        help="Interactive conversation"
    )
    p_chat.add_argument("--verbose", action="store_true",
                        help="Show compression activity during chat")

    # ── train-chat ──────────────────────────
    p_tc = sub.add_parser(
        "train-chat",
        help="Load data then start chatting"
    )
    p_tc.add_argument("path", help="Path to .zip, .cha, or directory")
    p_tc.add_argument("--limit",     type=int, default=0)
    p_tc.add_argument("--min-words", type=int, default=2, dest="min_words")
    p_tc.add_argument("--max-words", type=int, default=50, dest="max_words")
    p_tc.add_argument("--quiet",     action="store_true")

    # ── status ──────────────────────────────
    sub.add_parser("status", help="Show graph health")

    # ── sweep ───────────────────────────────
    sub.add_parser("sweep", help="Force compression + projection cycle")

    return parser


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    cmd_map = {
        "train"      : cmd_train,
        "chat"       : cmd_chat,
        "train-chat" : cmd_train_chat,
        "status"     : cmd_status,
        "sweep"      : cmd_sweep,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
