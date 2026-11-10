"""
Schema creation for hypergraph-lm.
Context stored as VARCHAR (comma-separated token IDs) for portability.
"""

import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "hypergraph.db"


def create_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            level    UTINYINT  NOT NULL,
            context  VARCHAR   NOT NULL,
            next_tok UINTEGER  NOT NULL,
            cnt      UINTEGER  NOT NULL,
            prob     FLOAT     NOT NULL DEFAULT 0.0
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS level_stats (
            level            UTINYINT PRIMARY KEY,
            context_len      UINTEGER,
            unique_contexts  UBIGINT,
            total_edges      UBIGINT,
            build_time_sec   FLOAT
        )
    """)

    print(f"Schema created at {DB_PATH}")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    create_schema(con)
    con.close()


if __name__ == "__main__":
    main()