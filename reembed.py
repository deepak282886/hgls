"""
reembed.py

Migration script — re-embed all SENTENCE-level nodes using the
new word feature hashing embedding function.

Run once after updating node.py's compute_embedding function.
All other levels (CHARACTER, WORD, PARAGRAPH, DOCUMENT) are
unchanged — only SENTENCE embeddings are replaced because that
is the level used for query activation and collision sampling.

Usage:
    python reembed.py
    python reembed.py --db path/to/graph.db
    python reembed.py --batch-size 500 --verbose

The script is safe to interrupt and resume — it processes nodes
in batches and commits after each batch. If interrupted, re-run
and it will skip already-updated nodes... actually it just
re-embeds everything from scratch on each run, which is fine
since re-embedding is idempotent and fast (pure CPU, no model).

Estimated time: ~2-3 minutes for 100k sentence nodes on CPU.
"""

import sys
import time
import sqlite3
import argparse
import numpy as np
from pathlib import Path


def reembed_sentences(
    db_path    : str  = "graph.db",
    batch_size : int  = 200,
    verbose    : bool = True,
) -> dict:
    """
    Re-embed all SENTENCE-level nodes in the database.

    Reads each sentence node's text, recomputes the embedding
    using the current compute_embedding function (word hashing),
    and writes the new embedding bytes back to the DB.

    Returns stats dict.
    """
    # Import here so the script uses the updated node.py
    from core.atoms import Level
    from core.node  import compute_embedding

    SENTENCE_LEVEL = int(Level.SENTENCE)

    t0 = time.time()

    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")

    # Count sentence nodes
    total = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE level = ?",
        (SENTENCE_LEVEL,)
    ).fetchone()[0]

    if verbose:
        print(f"\nRe-embedding {total} sentence nodes")
        print(f"  DB:         {db_path}")
        print(f"  Batch size: {batch_size}")
        print(f"  New method: word feature hashing (128 dims)\n")

    # Fetch all sentence node IDs and texts
    rows = conn.execute(
        "SELECT node_id, text FROM nodes WHERE level = ? ORDER BY node_id",
        (SENTENCE_LEVEL,)
    ).fetchall()

    updated   = 0
    errors    = 0
    batch_num = 0

    for batch_start in range(0, len(rows), batch_size):
        batch     = rows[batch_start : batch_start + batch_size]
        batch_num += 1

        updates = []
        for node_id, text in batch:
            try:
                new_emb = compute_embedding(text, Level.SENTENCE)
                emb_bytes = new_emb.astype(np.float32).tobytes()
                updates.append((emb_bytes, node_id))
                updated += 1
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  Error on node {node_id}: {e}")

        # Batch update
        conn.executemany(
            "UPDATE nodes SET embedding = ? WHERE node_id = ?",
            updates
        )
        conn.commit()

        if verbose and batch_num % 10 == 0:
            elapsed = time.time() - t0
            rate    = updated / (elapsed + 1e-9)
            pct     = updated / (total + 1e-9) * 100
            eta     = (total - updated) / (rate + 1e-9)
            print(
                f"  [{pct:5.1f}%]  updated={updated}  "
                f"rate={rate:.0f}/s  "
                f"eta={eta:.0f}s"
            )

    conn.close()
    elapsed = time.time() - t0

    stats = {
        "total"    : total,
        "updated"  : updated,
        "errors"   : errors,
        "elapsed"  : round(elapsed, 1),
        "rate"     : round(updated / (elapsed + 1e-9), 1),
    }

    if verbose:
        print(f"\nDone.")
        print(f"  Updated : {updated}/{total} nodes")
        print(f"  Errors  : {errors}")
        print(f"  Elapsed : {elapsed:.1f}s")
        print(f"  Rate    : {stats['rate']:.0f} nodes/s")

    return stats


def verify_embeddings(db_path: str = "graph.db", n_samples: int = 5):
    """
    Spot-check that the new embeddings give meaningful similarity.
    Prints similarity between query and a few known-relevant nodes.
    """
    from core.atoms import Level
    from core.node  import compute_embedding

    SENTENCE_LEVEL = int(Level.SENTENCE)

    conn = sqlite3.connect(db_path)

    print("\nVerification — cosine similarity spot check:")
    print("(higher = more similar, should be > 0.3 for relevant pairs)\n")

    query = "prove that the derivative of x squared is 2x"
    q_emb = compute_embedding(query, Level.SENTENCE)

    # Sample some nodes to check similarity distribution
    rows = conn.execute(
        "SELECT node_id, text, embedding FROM nodes "
        "WHERE level = ? ORDER BY RANDOM() LIMIT 20",
        (SENTENCE_LEVEL,)
    ).fetchall()

    sims = []
    for node_id, text, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        if emb.shape == q_emb.shape:
            sim = float(np.dot(q_emb, emb))
            sims.append((sim, text[:70]))

    sims.sort(reverse=True)

    print(f"Query: '{query}'\n")
    print("Top 5 most similar from random sample:")
    for sim, text in sims[:5]:
        print(f"  sim={sim:+.4f}  {repr(text)}")

    print("\nBottom 5 least similar from random sample:")
    for sim, text in sims[-5:]:
        print(f"  sim={sim:+.4f}  {repr(text)}")

    # Manually check a known-relevant sentence
    print("\nManual check — known relevant sentence:")
    manual = "The derivative of x squared is 2x by the power rule."
    m_emb  = compute_embedding(manual, Level.SENTENCE)
    sim    = float(np.dot(q_emb, m_emb))
    print(f"  sim={sim:+.4f}  '{manual}'")

    manual2 = "and Alonso, Alfonso and Baltzer, Jennifer L."
    m_emb2  = compute_embedding(manual2, Level.SENTENCE)
    sim2    = float(np.dot(q_emb, m_emb2))
    print(f"  sim={sim2:+.4f}  '{manual2}'")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Re-embed sentence nodes with word hashing")
    parser.add_argument("--db",         default="graph.db", help="Database path")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--verify",     action="store_true",  help="Run verification after re-embedding")
    parser.add_argument("--verify-only",action="store_true",  help="Only run verification, skip re-embedding")
    args = parser.parse_args()

    if not args.verify_only:
        reembed_sentences(
            db_path    = args.db,
            batch_size = args.batch_size,
            verbose    = True,
        )

    if args.verify or args.verify_only:
        verify_embeddings(db_path=args.db)


if __name__ == "__main__":
    main()