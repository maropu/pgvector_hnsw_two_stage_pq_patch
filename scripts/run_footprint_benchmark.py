#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sift1m_footprint_benchmark.py - run benchmark to monitor DB block footprint

  - Connect to a specified DB (local UNIX socket; port 5432)
  - For each ef_search in --ef, run the ANN query --runs times after SET hnsw.ef_search = ef
  - Query: SELECT * FROM <data_table> ORDER BY <embedding_col> <-> '<vector>'::vector LIMIT K
  - Store per-run buffer stats into a TEMP table qbuf_sample_{ts}
  - Export per-run rows to CSV: {output_prefix}_{ts}.csv
  - Render a line chart (x=ef_search, y=blocks) for selected series with error bars (stddev);
    save as {output_prefix}_{ts}.svg

Notes:
  - Assumes pgvector is installed and a table like 'sift1m(embedding vector, ...)' exists.
  - "total_refs" = shared_hit + shared_read + local_hit + local_read + temp_read

Usage:
  python run_sift1m_footprint_benchmark.py \
      --dataset sift1m \
      --dbname postgres \
      --user maropu \
      --table sift1m \
      --embedding-col embedding \
      --ef 10,40,120,400,800 \
      --runs 100 \
      --series total_refs,shared_read,temp_read \
      --output_prefix sift1m_hnsw_l2 \
      --verbose
"""

from __future__ import annotations
import argparse
import random
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime

import psycopg
from psycopg import sql as psql
from psycopg.rows import tuple_row

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt


# --- Dataset presets for query vector generation (dimension, value range) ---
DATASET_PRESETS = {
    "sift1m": {"dim": 128, "min": 0, "max": 218},
    "gist1m": {"dim": 960, "min": 0, "max": 255},
    "deep1m": {"dim": 96, "min": 0, "max": 1},
    "fashion-mnist": {"dim": 784, "min": 0, "max": 255},
}

# DDL for the TEMP stats table (ts suffix will be applied to the name)
STAT_TABLE_DDL = """
DROP TABLE IF EXISTS {table_ident};
CREATE TEMP TABLE {table_ident} (
  id                bigserial PRIMARY KEY,
  ef_search         integer,
  query             text,
  shared_hit        bigint,
  shared_read       bigint,
  shared_dirtied    bigint,
  shared_written    bigint,
  local_hit         bigint,
  local_read        bigint,
  local_dirtied     bigint,
  local_written     bigint,
  temp_read         bigint,
  temp_written      bigint,
  total_refs        bigint,
  plan_time_ms      double precision,
  exec_time_ms      double precision,
  block_size_bytes  integer
) ON COMMIT PRESERVE ROWS;
"""

# INSERT template for the statistics table
STAT_INSERT_SQL = """
INSERT INTO {table_ident} (
  ef_search, query,
  shared_hit, shared_read, shared_dirtied, shared_written,
  local_hit,  local_read,  local_dirtied,  local_written,
  temp_read,  temp_written,
  total_refs, plan_time_ms, exec_time_ms, block_size_bytes
)
VALUES (%(ef)s, %(query)s,
        %(sh)s, %(sr)s, %(sd)s, %(sw)s,
        %(lh)s, %(lr)s, %(ld)s, %(lw)s,
        %(tr)s, %(tw)s,
        %(total_refs)s, %(plan_time_ms)s, %(exec_time_ms)s, %(block_size)s)
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark pgvector block refs via EXPLAIN JSON across ef_search values."
    )
    ap.add_argument("--dbname", required=True, help="Target database name (pgvector+sift1m loaded)")
    ap.add_argument("--user", default="postgres", help="User name (default: postgres)")
    ap.add_argument("--table", default="sift1m", help="Data table name (default: sift1m)")
    ap.add_argument("--embedding-col", default="embedding", help="Vector column name (default: embedding)")
    ap.add_argument("--k", type=int, default=3, help="LIMIT K for ANN query (default: 3)")
    ap.add_argument("--runs", type=int, default=100, help="Runs per ef_search (default: 10)")
    ap.add_argument("--ef", default="10,40,120,400,800",
                    help="Comma-separated ef_search values (default: 10,40,120,400,800)")
    ap.add_argument("--output_prefix", default="result",
                    help="Output prefix -> {prefix}_{ts}.csv and {prefix}_{ts}.svg")

    ap.add_argument(
        "--dataset",
        type=str,
        choices=sorted(list(DATASET_PRESETS.keys())),
        default="sift1m",
        help="Dataset preset for query vector generation (dimension and value range).",
    )
    ap.add_argument(
        "--series",
        default="total_refs,shared_hit,shared_read",
        help=("Comma-separated columns to plot. Allowed: "
              "total_refs,shared_hit,shared_read,shared_dirtied,shared_written,"
              "local_hit,local_read,local_dirtied,local_written,temp_read,temp_written")
    )
    ap.add_argument("--seed", type=int, default=None, help="PRNG seed for reproducibility")
    ap.add_argument("--verbose", action="store_true", help="Print per-run summary")
    return ap.parse_args()


def ident_qualified(name: str) -> psql.Composed:
    parts = [p.strip() for p in name.split(".")]
    return psql.SQL(".").join(psql.Identifier(p) for p in parts)


def ensure_stat_table(conn: psycopg.Connection, stat_table_name: str) -> None:
    ident = psql.Identifier(stat_table_name)
    ddl = psql.SQL(STAT_TABLE_DDL).format(table_ident=ident)
    with conn.cursor() as cur:
        cur.execute(ddl)


def drop_stat_table(conn: psycopg.Connection, stat_table_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(psql.SQL("DROP TABLE IF EXISTS {}").format(psql.Identifier(stat_table_name)))


def fetch_block_size(conn: psycopg.Connection) -> int:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SHOW block_size")
        (bs,) = cur.fetchone()
        return int(bs)


def sum_buffer_counters(plan_node: Dict[str, Any]) -> Dict[str, int]:
    keys = [
        "Shared Hit Blocks", "Shared Read Blocks", "Shared Dirtied Blocks", "Shared Written Blocks",
        "Local Hit Blocks",  "Local Read Blocks",  "Local Dirtied Blocks",  "Local Written Blocks",
        "Temp Read Blocks",  "Temp Written Blocks",
    ]
    acc = {k: 0 for k in keys}
    stack = [plan_node]
    while stack:
        n = stack.pop()
        for k in keys:
            if k in n and n[k] is not None:
                acc[k] += int(n[k])
        for child in (n.get("Plans") or []):
            stack.append(child)
    return acc


def explain_analyze_buffers(cur: psycopg.Cursor, data_table: str, emb_col: str, k: int, vector_literal: str) -> Dict[str, Any]:
    stmt = psql.SQL(
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
        "SELECT * FROM {} ORDER BY {} <-> %s::vector LIMIT %s"
    ).format(ident_qualified(data_table), psql.Identifier(emb_col))
    cur.execute(stmt, (vector_literal, k))
    (json_obj,) = cur.fetchone()
    if not isinstance(json_obj, list) or not json_obj or not isinstance(json_obj[0], dict):
        raise TypeError(f"Unexpected EXPLAIN JSON shape: {type(json_obj)}: {json_obj!r}")
    root = json_obj[0]
    plan = root["Plan"]
    return {
        "plan_time_ms": float(root.get("Planning Time", 0.0)),
        "exec_time_ms": float(root.get("Execution Time", 0.0)),
        "counters": sum_buffer_counters(plan),
        "executed_sql": "SELECT ... (omitted in CSV to keep it compact)",
    }


def make_random_vector_literal(dim: int, low: int, high: int, rng: random.Random) -> str:
    values = [rng.randint(low, high) for _ in range(dim)]
    return "[" + ",".join(map(str, values)) + "]"


def export_csv(conn: psycopg.Connection, table: str, output_csv_path: str, verbose: bool) -> None:
    sql = psql.SQL("COPY (SELECT * FROM {} ORDER BY id) TO STDOUT WITH CSV HEADER").format(psql.Identifier(table))
    if verbose:
        print(f"[export] Writing CSV -> {output_csv_path}")
    with conn.cursor() as cur, cur.copy(sql) as copy, open(output_csv_path, "wb") as f:
        while True:
            chunk = copy.read()
            if not chunk:
                break
            f.write(chunk)


def group_stats_for_plot(conn: psycopg.Connection, table: str,
                         series: List[str]) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Returns:
      - ef_values: sorted unique ef_search values
      - means:  dict series->list of means aligned with ef_values
      - vars_:  dict series->list of (sample) variances aligned with ef_values
    """
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(psql.SQL("SELECT DISTINCT ef_search FROM {} ORDER BY ef_search").format(psql.Identifier(table)))
        ef_values = [int(r[0]) for r in cur.fetchall()]

        means: Dict[str, List[float]] = {s: [] for s in series}
        vars_: Dict[str, List[float]] = {s: [] for s in series}

        for ef in ef_values:
            cur.execute(
                psql.SQL("SELECT {} FROM {} WHERE ef_search = %s").format(
                    psql.SQL(", ").join(psql.Identifier(s) for s in series),
                    psql.Identifier(table),
                ),
                (ef,),
            )
            rows = cur.fetchall()
            cols_data: List[List[float]] = [[] for _ in series]
            for row in rows:
                for i, v in enumerate(row):
                    cols_data[i].append(float(v or 0))

            for s_idx, s in enumerate(series):
                data = cols_data[s_idx]
                n = len(data)
                if n == 0:
                    means[s].append(0.0)
                    vars_[s].append(0.0)
                else:
                    m = sum(data) / n
                    var = (sum((x - m) ** 2 for x in data) / (n - 1)) if n >= 2 else 0.0
                    means[s].append(m)
                    vars_[s].append(var)

    return ef_values, means, vars_


def plot_lines_with_error(ef_values: List[int], means: Dict[str, List[float]], vars_: Dict[str, List[float]],
                          out_svg_path: str, verbose: bool) -> None:
    """
    Line chart:
      - x-axis: ef_search values
      - for each series: a line with markers and error bars (stddev = sqrt(variance))
      - no variance axis/bars
    """
    if not ef_values:
        print(f"[warn] No ef_search values found; skip plot ({out_svg_path})", file=sys.stderr)
        return

    if verbose:
        print("[ef-wise means]")
        for ser, ys in means.items():
            for ef, m in zip(ef_values, ys):
                print(f"  series={ser:>12} ef={ef:>4} mean={m:.3f}")

    series = list(means.keys())
    xs = ef_values  # plot using numeric ef values to maintain natural spacing

    fig, ax = plt.subplots()
    for ser in series:
        y = means[ser]
        std = [v ** 0.5 for v in vars_[ser]]
        ax.errorbar(xs, y, yerr=std, marker="o", linestyle="-", capsize=3, label=ser)

    # Improve x tick label readability
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs], rotation=30, ha="right")
    ax.margins(x=0.03)

    ax.set_xlabel("hnsw.ef_search")
    ax.set_ylabel("blocks (8 KiB each)")
    ax.set_title("pgvector block counts vs ef_search (mean ± stdev)")
    ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)  # extra room for rotated labels
    fig.savefig(out_svg_path, format="svg")
    plt.close(fig)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Timestamp token (YYYYMMDD_HHMMSS) used for table and file names
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"{args.output_prefix}_{ts}.csv"
    output_svg = f"{args.output_prefix}_{ts}.svg"
    stat_table_name = f"qbuf_sample_{ts}"

    # Parse ef list
    try:
        ef_list = [int(x.strip()) for x in args.ef.split(",") if x.strip()]
        if not ef_list:
            raise ValueError
    except Exception:
        print("[error] --ef must be a comma-separated list of integers (e.g., 10,20,40)", file=sys.stderr)
        sys.exit(2)

    # Validate selected series
    allowed = {
        "total_refs", "shared_hit", "shared_read", "shared_dirtied", "shared_written",
        "local_hit", "local_read", "local_dirtied", "local_written",
        "temp_read", "temp_written",
    }
    series_cols = [s.strip() for s in args.series.split(",") if s.strip()]
    invalid = [c for c in series_cols if c not in allowed]
    if invalid:
        print(f"[error] Unknown series: {', '.join(invalid)}", file=sys.stderr)
        print(f"        Allowed: {', '.join(sorted(allowed))}", file=sys.stderr)
        sys.exit(2)

    # Connect via local UNIX socket (host unspecified), port fixed
    conn = psycopg.connect(dbname=args.dbname, user=args.user, port=5432, autocommit=True)


    # Resolve dataset preset for query vector
    preset = DATASET_PRESETS.get(args.dataset)
    if preset is None:
        raise ValueError(f"Unknown dataset preset: {args.dataset}")

    qdim, qmin, qmax = int(preset["dim"]), int(preset["min"]), int(preset["max"])

    try:
        ensure_stat_table(conn, stat_table_name)
        block_size = fetch_block_size(conn)

        inserted = 0
        with conn.cursor() as cur:
            for ef in ef_list:
                cur.execute(f"SET hnsw.ef_search = {ef}")
                for i in range(args.runs):
                    vec = make_random_vector_literal(qdim, qmin, qmax, random)
                    res = explain_analyze_buffers(cur, args.table, args.embedding_col, args.k, vec)
                    c = res["counters"]

                    sh = c.get("Shared Hit Blocks", 0)
                    sr = c.get("Shared Read Blocks", 0)
                    sd = c.get("Shared Dirtied Blocks", 0)
                    sw = c.get("Shared Written Blocks", 0)
                    lh = c.get("Local Hit Blocks", 0)
                    lr = c.get("Local Read Blocks", 0)
                    ld = c.get("Local Dirtied Blocks", 0)
                    lw = c.get("Local Written Blocks", 0)
                    tr = c.get("Temp Read Blocks", 0)
                    tw = c.get("Temp Written Blocks", 0)

                    total_refs = sh + sr + lh + lr + tr

                    if args.verbose:
                        print(f"[ef={ef:>4}] run {i+1:>3}/{args.runs} "
                              f"refs={total_refs} (shared h={sh},r={sr}; local h={lh},r={lr}; temp r={tr}) "
                              f"plan_ms={res['plan_time_ms']:.3f} exec_ms={res['exec_time_ms']:.3f}")

                    cur.execute(
                        psql.SQL(STAT_INSERT_SQL).format(table_ident=psql.Identifier(stat_table_name)),
                        {
                            "ef": ef,
                            "query": f"ORDER BY {args.embedding_col} <-> '[..]'::vector LIMIT {args.k}",
                            "sh": sh, "sr": sr, "sd": sd, "sw": sw,
                            "lh": lh, "lr": lr, "ld": ld, "lw": lw,
                            "tr": tr, "tw": tw,
                            "total_refs": total_refs,
                            "plan_time_ms": res["plan_time_ms"],
                            "exec_time_ms": res["exec_time_ms"],
                            "block_size": block_size,
                        },
                    )
                    inserted += 1

        # Export CSV (per-run rows)
        export_csv(conn, stat_table_name, output_csv, args.verbose)

        # Compute ef-wise means/variances and plot lines with error bars (stddev)
        ef_values, means, vars_ = group_stats_for_plot(conn, stat_table_name, series_cols)
        plot_lines_with_error(ef_values, means, vars_, output_svg, args.verbose)

        print(f"[done] rows: {inserted} | CSV: {output_csv} | SVG: {output_svg}")

    finally:
        try:
            drop_stat_table(conn, stat_table_name)
        finally:
            conn.close()


if __name__ == "__main__":
    main()

