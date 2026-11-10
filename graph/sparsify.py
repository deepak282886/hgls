"""
Standalone sparsifier for hypergraph-lm.

Useful if you want to re-prune the graph with different thresholds
without rebuilding from scratch.

Run:
    python graph/sparsify.py                     # all levels
    python graph/sparsify.py --level 1           # single level
    python graph/sparsify.py --min-count 10      # stricter threshold
    python graph/sparsify.py --min-prob 0.02     # stricter prob
    python graph/sparsify.py --top-k 30          # fewer edges per context
"""

import argparse
import duckdb
from pathlib import Path
from tqdm import tqdm

DB_PATH    = Path(__file__).parent / "hypergraph.db"
MAX_LEVELS = 12

# Defaults — can be overridden via CLI
DEFAULT_MIN_COUNT = 5
DEFAULT_MIN_PROB  = 0.01
DEFAULT_TOP_K     = 50


def sparsify_level(
    con: duckdb.DuckDBPyConnection,
    level: int,
    min_count: int,
    min_prob: float,
    top_k: int,
):
    before = con.execute(
        "SELECT COUNT(*) FROM edges WHERE level = ?", [level]
    ).fetchone()[0]

    if before == 0:
        print(f"  Level {level}: no edges found, skipping")
        return

    # Drop low count + low prob
    con.execute(f"""
        DELETE FROM edges
        WHERE level = {level}
          AND (cnt < {min_count} OR prob < {min_prob})
    """)

    # Keep top K per context
    con.execute(f"""
        DELETE FROM edges
        WHERE level = {level}
          AND rowid NOT IN (
              SELECT rowid FROM (
                  SELECT
                      rowid,
                      ROW_NUMBER() OVER (
                          PARTITION BY level, context
                          ORDER BY prob DESC
                      ) AS rn
                  FROM edges
                  WHERE level = {level}
              ) ranked
              WHERE rn <= {top_k}
          )
    """)

    after = con.execute(
        "SELECT COUNT(*) FROM edges WHERE level = ?", [level]
    ).fetchone()[0]

    dropped = before - after
    print(f"  Level {level}: {before:,} → {after:,} edges  "
          f"(dropped {dropped:,}, {dropped/max(before,1)*100:.1f}%)")


def print_stats(con: duckdb.DuckDBPyConnection):
    print("\nCurrent graph stats:")
    print(f"{'Level':>6}  {'Context Len':>12}  {'Contexts':>12}  {'Edges':>12}")
    print("-" * 50)
    rows = con.execute("""
        SELECT level, context_len, unique_contexts, total_edges
        FROM level_stats
        ORDER BY level
    """).fetchall()
    for row in rows:
        print(f"  {row[0]:>4}  {row[1]:>12,}  {row[2]:>12,}  {row[3]:>12,}")

    total = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"\n  Total edges in graph: {total:,}")

    size_bytes = Path(DB_PATH).stat().st_size
    print(f"  DB size on disk:      {size_bytes/1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",     type=int,   default=None)
    parser.add_argument("--min-count", type=int,   default=DEFAULT_MIN_COUNT)
    parser.add_argument("--min-prob",  type=float, default=DEFAULT_MIN_PROB)
    parser.add_argument("--top-k",     type=int,   default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run: python graph/build.py first")
        return

    con = duckdb.connect(str(DB_PATH))

    print("=" * 50)
    print("Hypergraph-LM Sparsifier")
    print("=" * 50)
    print(f"min_count : {args.min_count}")
    print(f"min_prob  : {args.min_prob}")
    print(f"top_k     : {args.top_k}")
    print()

    levels = [args.level] if args.level is not None else list(range(MAX_LEVELS))

    for level in tqdm(levels, desc="Sparsifying"):
        sparsify_level(con, level, args.min_count, args.min_prob, args.top_k)

    # Update level_stats after sparsification
    print("\nUpdating level_stats...")
    for level in levels:
        row = con.execute(f"""
            SELECT
                COUNT(DISTINCT context) AS unique_contexts,
                COUNT(*)                AS total_edges
            FROM edges
            WHERE level = {level}
        """).fetchone()
        con.execute("""
            UPDATE level_stats
            SET unique_contexts = ?, total_edges = ?
            WHERE level = ?
        """, [row[0], row[1], level])

    print_stats(con)
    con.close()


if __name__ == "__main__":
    main()