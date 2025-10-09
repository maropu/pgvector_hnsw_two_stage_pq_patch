#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_sift1m_to_postgres.py

 - Download and cache the TexMex SIFT archive from: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
 - Reuse the cached tarball and extracted files on subsequent runs
 - Stream a chosen split (.fvecs) into a Postgres table with pgvector via COPY
 - Always CREATE TABLE (fail if it already exists)
 - No index creation

Usage:
  python load_sift_from_ftp_to_pgvector.py \
    --dbname postgres \
    --user postgres \
    --subset base \
    --table public.sift1m \
    --column embedding \
    --batch-rows 20000 \
    --verbose
"""

from __future__ import annotations
import argparse
import io
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Generator, Iterable, Tuple

import psycopg
from psycopg import sql as psql


TEXMEX_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

DEFAULT_CACHE_DIR = Path("/tmp/sift_texmex")

ARCHIVE_NAME = "sift.tar.gz"

EXTRACTED_DIRNAME = "sift"  # inside the tarball

SPLIT_TO_FILENAME = {
    "base":  "sift_base.fvecs",
    "learn": "sift_learn.fvecs",
    "query": "sift_query.fvecs",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download/cache TexMex SIFT (.tar.gz) and load a split into Postgres w/pgvector."
    )

    # Connection (Unix socket; fixed port 5432)
    ap.add_argument("--dbname", required=True, help="Target database name")
    ap.add_argument("--user", default="postgres", help="User name (default: postgres)")

    # Dataset split & cache
    ap.add_argument("--subset", choices=["base", "learn", "query"], default="base",
                    help="Which SIFT split to load (default: base)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Cache directory under /tmp (default: /tmp/sift_texmex)")

    # Destination
    ap.add_argument("--table", default="sift1m", help="Destination table name (schema-qualified allowed)")
    ap.add_argument("--column", default="embedding", help="Vector column name (default: embedding)")

    # Misc
    ap.add_argument("--batch-rows", type=int, default=10000, help="Progress interval (rows) (default: 10000)")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress output")
    return ap.parse_args()


def ensure_cache_dirs(cache_dir: Path) -> None:
    """Create cache directory if it does not exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, dst: Path, verbose: bool) -> None:
    """
    Download the .tar.gz only if it's not present already.
    If present, reuse it as-is.
    """
    if dst.exists():
        if verbose:
            print(f"[cache] Using cached archive: {dst}")
        return
    if verbose:
        print(f"[download] Fetching: {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # supports FTP
        tmp.replace(dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    if verbose:
        print(f"[download] Completed: {dst}")


def extract_tar_gz_if_missing(archive: Path, out_dir: Path, verbose: bool) -> Path:
    """
    Extract the .tar.gz if the extracted root doesn't already contain expected .fvecs.
    Returns the extracted root directory path (e.g., /tmp/sift_texmex/sift).
    """
    if archive.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError(f"Only .tar.gz is supported: {archive}")

    extracted_root = out_dir / EXTRACTED_DIRNAME
    expected = {filename for filename in SPLIT_TO_FILENAME.values()}
    if extracted_root.exists():
        have = {p.name for p in extracted_root.glob("*.fvecs")}
        if expected.issubset(have):
            if verbose:
                print(f"[cache] Using extracted data in: {extracted_root}")
            return extracted_root

    if verbose:
        print(f"[extract] Extracting: {archive} -> {out_dir}")
    with tarfile.open(archive, mode="r:gz") as tf:
        tf.extractall(out_dir)
    if verbose:
        print("[extract] Done")
    return extracted_root


def get_split_path(extracted_root: Path, subset: str) -> Path:
    """Resolve the target .fvecs path for the requested split."""
    fname = SPLIT_TO_FILENAME[subset]
    fpath = extracted_root / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Split file not found: {fpath}")
    return fpath


def ident_qualified(name: str) -> psql.Composed:
    """Return a safe schema-qualified Identifier composition."""
    parts = [p.strip() for p in name.split(".")]
    return psql.SQL(".").join(psql.Identifier(p) for p in parts)


def ensure_extension_vector(cur: psycopg.Cursor) -> None:
    """CREATE EXTENSION vector if missing."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")


def create_table_fail(cur: psycopg.Cursor, table: str, column: str, dim: int) -> None:
    """
    Always CREATE TABLE. Fail if it already exists.
    This mirrors the requested 'always fail' semantics.
    """
    tbl_ident = ident_qualified(table)
    col_ident = psql.Identifier(column)
    cur.execute(
        psql.SQL(
            "CREATE TABLE {} ("
            "  id bigserial PRIMARY KEY,"
            "  {} vector({})"
            ")"
        ).format(tbl_ident, col_ident, psql.Literal(dim))
    )


def iter_fvecs(path: Path) -> Generator[Tuple[int, Tuple[float, ...]], None, None]:
    """
    Stream records from .fvecs:
      yields (dim, tuple(float32,..))
    """
    with path.open("rb") as f:
        rd = f.read
        unpack_i32 = struct.Struct("<i").unpack
        while True:
            hdr = rd(4)
            if not hdr:
                break
            if len(hdr) != 4:
                raise IOError("Corrupt .fvecs header (short read)")
            (d,) = unpack_i32(hdr)
            payload = rd(4 * d)
            if len(payload) != 4 * d:
                raise IOError(f"Unexpected EOF: expected {4*d} bytes, got {len(payload)}")
            vec = struct.unpack("<" + "f" * d, payload)
            yield d, vec


def copy_vectors(conn: psycopg.Connection,
                 table: str,
                 column: str,
                 dim: int,
                 rows: Iterable[Tuple[int, Tuple[float, ...]]],
                 batch_rows: int,
                 verbose: bool) -> int:
    """
    Stream rows into COPY. Expects each row as (dim, tuple(float,...)).
    Returns total inserted row count.
    """
    tbl_ident = ident_qualified(table)
    col_ident = psql.Identifier(column)
    total = 0

    stmt = psql.SQL("COPY {} ({}) FROM STDIN").format(tbl_ident, col_ident)

    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for rec_dim, vec in rows:
            if rec_dim != dim:
                raise ValueError(f"Dimension mismatch: file has {rec_dim}, expected {dim}")
            lit = "[" + ",".join(f"{v:.6g}" for v in vec) + "]"  # pgvector text literal
            copy.write_row((lit,))
            total += 1
            if verbose and (total % batch_rows == 0):
                print(f"[copy] {total} rows streamed...")

    if verbose:
        print(f"[copy] finished: {total} rows")
    return total


def main():
    args = parse_args()
    cache_dir: Path = args.cache_dir
    ensure_cache_dirs(cache_dir)

    archive_path = cache_dir / ARCHIVE_NAME

    # Download .tar.gz if missing; reuse otherwise
    download_if_missing(TEXMEX_URL, archive_path, args.verbose)

    # Extract if needed; reuse otherwise (.tar.gz only)
    extracted_root = extract_tar_gz_if_missing(archive_path, cache_dir, args.verbose)

    # Resolve the path to the requested split (.fvecs)
    fvecs_path = get_split_path(extracted_root, args.subset)
    if args.verbose:
        print(f"[cache] Using split file: {fvecs_path}")

    # Connect via local UNIX socket (host unspecified), port fixed to 5432
    conn = psycopg.connect(
        dbname=args.dbname,
        user=args.user,
        port=5432,
        autocommit=True,
    )

    try:
        # Peek the first record to determine dim
        stream = iter_fvecs(fvecs_path)
        try:
            first_dim, first_vec = next(stream)
        except StopIteration:
            raise RuntimeError(f"Empty .fvecs file: {fvecs_path}")

        dim = first_dim

        with conn.cursor() as cur:
            ensure_extension_vector(cur)
            # Always CREATE TABLE; will raise if it already exists
            create_table_fail(cur, args.table, args.column, dim)

        # Chain the first record back into the stream
        def chained_rows() -> Generator[Tuple[int, Tuple[float, ...]], None, None]:
            yield first_dim, first_vec
            for rec in stream:
                yield rec

        # COPY streaming (no --limit; load all)
        inserted = copy_vectors(
            conn=conn,
            table=args.table,
            column=args.column,
            dim=dim,
            rows=chained_rows(),
            batch_rows=args.batch_rows,
            verbose=args.verbose,
        )

        if args.verbose:
            print(f"[done] inserted rows: {inserted} (subset={args.subset}, dim={dim})")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

