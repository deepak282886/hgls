"""
Graph builder for hypergraph-lm.

Architecture:
  For each level:
    1. Stream corpus.bin in chunks
    2. numpy sliding window -> np.unique per chunk -> write parquet shard
    3. DuckDB reads all shards -> GROUP BY SUM -> INSERT INTO edges
    4. Normalize probabilities in DuckDB
    5. Sparsify in DuckDB

Context stored as VARCHAR: comma-separated token IDs e.g. "42,1337"
No binary packing — works with any DuckDB version.

Run:
    python graph/build.py              # all levels
    python graph/build.py --level 0   # single level
    python graph/build.py --resume    # skip already-built levels
"""

import time
import argparse
import shutil
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_BIN  = Path(__file__).parent.parent / "data" / "corpus.bin"
DB_PATH     = Path(__file__).parent / "hypergraph.db"
TMP_DIR     = Path(__file__).parent / "tmp_shards"
MAX_LEVELS  = 12
CHUNK_SIZE  = 10_000_000
MIN_COUNT   = 5
MIN_PROB    = 0.01
TOP_K       = 50
# ──────────────────────────────────────────────────────────────────────────────


def count_windows_numpy(windows: np.ndarray):
    """
    Count unique rows in 2D uint32 array via numpy void-view.
    Pure C — no Python loops. Returns (unique_windows, counts).
    """
    void_view = np.ascontiguousarray(windows).view(
        np.dtype((np.void, windows.dtype.itemsize * windows.shape[1]))
    ).ravel()
    unique_void, counts = np.unique(void_view, return_counts=True)
    unique_windows = unique_void.view(np.uint32).reshape(-1, windows.shape[1])
    return unique_windows, counts.astype(np.uint32)


def write_shard(unique_windows: np.ndarray, counts: np.ndarray,
                context_len: int, shard_path: Path):
    """
    Write parquet shard with one column per context token + next_tok + cnt.
    e.g. level 1: columns = c0, c1, next_tok, cnt
    """
    arrays = {}
    for i in range(context_len):
        arrays[f"c{i}"] = pa.array(unique_windows[:, i], type=pa.uint32())
    arrays["next_tok"] = pa.array(unique_windows[:, -1], type=pa.uint32())
    arrays["cnt"]      = pa.array(counts,                type=pa.uint32())

    pq.write_table(
        pa.table(arrays),
        shard_path,
        compression="snappy",
    )


def build_context_str_expr(context_len: int) -> str:
    """
    DuckDB SQL expression that concatenates context token columns
    into a comma-separated VARCHAR: e.g. c0||','||c1||','||c2
    """
    parts = " || ',' || ".join(
        f"CAST(c{i} AS VARCHAR)" for i in range(context_len)
    )
    return parts


def aggregate_shards(con: duckdb.DuckDBPyConnection, level: int,
                     shard_dir: Path, context_len: int):
    """
    DuckDB reads all parquet shards, GROUP BY SUM, INSERT INTO edges.
    Context packed as comma-separated VARCHAR in SQL — no Python row handling.
    """
    shard_pattern = str(shard_dir / f"level_{level}_*.parquet").replace("\\", "/")
    ctx_cols      = ", ".join(f"c{i}" for i in range(context_len))
    ctx_str_expr  = build_context_str_expr(context_len)

    print(f"  DuckDB aggregating shards...")
    con.execute(f"""
        INSERT INTO edges (level, context, next_tok, cnt, prob)
        SELECT
            {level}             AS level,
            {ctx_str_expr}      AS context,
            next_tok,
            SUM(cnt)::UINTEGER  AS cnt,
            0.0::FLOAT          AS prob
        FROM read_parquet('{shard_pattern}')
        GROUP BY {ctx_cols}, next_tok
        HAVING SUM(cnt) >= {MIN_COUNT}
    """)


def normalize(con: duckdb.DuckDBPyConnection, level: int):
    print(f"  Normalizing...")

    # Step 1: compute per-context totals into a temp table
    con.execute(f"""
        CREATE TEMP TABLE IF NOT EXISTS ctx_totals_{level} AS
        SELECT context, SUM(cnt)::FLOAT AS total
        FROM edges
        WHERE level = {level}
        GROUP BY context
    """)

    # Step 2: update prob using the totals table — no window function
    con.execute(f"""
        UPDATE edges
        SET prob = edges.cnt::FLOAT / t.total
        FROM ctx_totals_{level} t
        WHERE edges.context = t.context
          AND edges.level = {level}
    """)

    con.execute(f"DROP TABLE IF EXISTS ctx_totals_{level}")


def sparsify(con: duckdb.DuckDBPyConnection, level: int):
    print(f"  Sparsifying step 1: drop low count/prob...")

    # Step 1 — simple filter, no window function, very low memory
    con.execute(f"""
        DELETE FROM edges
        WHERE level = {level}
          AND (cnt < {MIN_COUNT} OR prob < {MIN_PROB})
    """)

    print(f"  Sparsifying step 2: top-K per context...")

    # Step 2 — find contexts that have more than TOP_K edges
    # then delete the lowest prob ones in small batches
    # Avoid window function over full table — do per-context threshold instead

    # Get the minimum prob threshold per context using a temp table
    con.execute(f"""
        CREATE TEMP TABLE IF NOT EXISTS topk_thresh_{level} AS
        SELECT context,
               MIN(prob) AS min_keep_prob
        FROM (
            SELECT context, prob,
                   ROW_NUMBER() OVER (
                       PARTITION BY context ORDER BY prob DESC
                   ) AS rn
            FROM edges
            WHERE level = {level}
        ) ranked
        WHERE rn <= {TOP_K}
        GROUP BY context
    """)

    con.execute(f"""
        DELETE FROM edges
        WHERE level = {level}
          AND prob < (
              SELECT min_keep_prob
              FROM topk_thresh_{level} t
              WHERE t.context = edges.context
          )
    """)

    con.execute(f"DROP TABLE IF EXISTS topk_thresh_{level}")


def record_stats(con: duckdb.DuckDBPyConnection, level: int, elapsed: float):
    row = con.execute(f"""
        SELECT COUNT(DISTINCT context), COUNT(*)
        FROM edges WHERE level = {level}
    """).fetchone()
    unique_contexts, total_edges = row
    con.execute("""
        INSERT OR REPLACE INTO level_stats
            (level, context_len, unique_contexts, total_edges, build_time_sec)
        VALUES (?, ?, ?, ?, ?)
    """, [level, 2**level, unique_contexts, total_edges, elapsed])
    print(f"  Done: {unique_contexts:,} contexts, "
          f"{total_edges:,} edges, {elapsed:.0f}s elapsed")


def level_already_built(con: duckdb.DuckDBPyConnection, level: int) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM level_stats WHERE level = ?", [level]
    ).fetchone()
    return row[0] > 0


def build_level(con: duckdb.DuckDBPyConnection, level: int):
    context_len  = 2 ** level
    window_len   = context_len + 1
    total_tokens = int(CORPUS_BIN.stat().st_size // 4)

    print(f"\n{'='*55}")
    print(f"Level {level}  |  context_len={context_len}  |  tokens={total_tokens:,}")
    print(f"{'='*55}")

    t0 = time.time()

    shard_dir = TMP_DIR / f"level_{level}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: stream corpus → numpy unique → parquet shards ───────────────
    total_chunks = len(range(0, total_tokens - context_len, CHUNK_SIZE))
    print(f"  Phase 1: writing {total_chunks} parquet shards...")

    fp = open(str(CORPUS_BIN), "rb")
    for chunk_idx, start in enumerate(tqdm(
            range(0, total_tokens - context_len, CHUNK_SIZE),
            desc=f"L{level} shards", total=total_chunks)):

        read_count = min(CHUNK_SIZE + context_len, total_tokens - start)
        fp.seek(start * 4)
        chunk = np.frombuffer(fp.read(read_count * 4), dtype=np.uint32).copy()

        if len(chunk) < window_len:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(chunk, window_len).copy()
        del chunk

        unique_windows, counts = count_windows_numpy(windows)
        del windows

        shard_path = shard_dir / f"level_{level}_{chunk_idx:04d}.parquet"
        write_shard(unique_windows, counts, context_len, shard_path)
        del unique_windows, counts

    fp.close()

    shard_size_gb = sum(f.stat().st_size for f in shard_dir.glob("*.parquet")) / 1e9
    print(f"  Shards written: {shard_size_gb:.2f} GB")

    # ── Phase 2: DuckDB aggregates all shards → edges ─────────────────────────
    print(f"  Phase 2: aggregating in DuckDB...")
    aggregate_shards(con, level, shard_dir, context_len)

    # ── Phase 3: normalize + sparsify ─────────────────────────────────────────
    normalize(con, level)
    sparsify(con, level)

    elapsed = time.time() - t0
    record_stats(con, level, elapsed)

    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"  Shards cleaned up.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",  type=int,  default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip levels already in level_stats")
    args = parser.parse_args()

    if not CORPUS_BIN.exists():
        print(f"corpus.bin not found. Run: python data/tokenizer.py")
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    from schema import create_schema
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    create_schema(con)

    levels = [args.level] if args.level is not None else list(range(MAX_LEVELS))

    for level in levels:
        if args.resume and level_already_built(con, level):
            print(f"Level {level} already built, skipping.")
            continue
        build_level(con, level)

    print("\nAll levels done.")
    con.close()


if __name__ == "__main__":
    main()