from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List
import pandas as pd


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # Improve write concurrency a bit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _infer_sqlite_type(series: pd.Series) -> str:
    dt = str(series.dtype)
    if dt.startswith("int"):
        return "INTEGER"
    if dt.startswith("float"):
        return "REAL"
    if dt == "bool":
        return "INTEGER"
    # Fallback: TEXT
    return "TEXT"


def create_table_with_schema(conn: sqlite3.Connection, table: str, df: pd.DataFrame, keys: List[str]) -> None:
    cols = []
    for c in df.columns:
        coltype = _infer_sqlite_type(df[c])
        cols.append(f"{c} {coltype}")
    pk = ", PRIMARY KEY (" + ",".join(keys) + ")" if keys else ""
    sql = f"CREATE TABLE IF NOT EXISTS {table} (" + ", ".join(cols) + pk + ")"
    conn.execute(sql)


def _get_existing_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        return [r[1] for r in rows] if rows else []
    except Exception:
        return []


def ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Ensure all DataFrame columns exist in the SQLite table; add missing ones via ALTER TABLE.

    Note: This does not modify primary keys or types of existing columns. New columns are added as NULLable.
    """
    existing = set(_get_existing_columns(conn, table))
    if not existing:
        # Table may not exist yet; creation will handle full schema
        return
    for c in df.columns:
        if c not in existing:
            coltype = _infer_sqlite_type(df[c])
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {c} {coltype}")
            except Exception:
                # Ignore if another process added it in the meantime
                pass


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, keys: List[str]) -> int:
    if df.empty:
        return 0
    # Ensure table exists with a compatible schema
    create_table_with_schema(conn, table, df, keys)
    # Evolve schema to include any new columns
    ensure_columns(conn, table, df)
    cols = list(df.columns)
    placeholders = ",".join(["?"] * len(cols))
    insert_cols = ",".join(cols)
    non_key_cols = [c for c in cols if c not in keys]
    if keys and non_key_cols:
        set_clause = ", ".join([f"{c}=excluded.{c}" for c in non_key_cols])
        conflict = f" ON CONFLICT (" + ",".join(keys) + ") DO UPDATE SET " + set_clause
    else:
        conflict = ""
    sql = f"INSERT INTO {table} ({insert_cols}) VALUES ({placeholders}){conflict}"
    data = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False, name=None)]
    with conn:
        conn.executemany(sql, data)
    return len(data)


def ingest_csv(conn: sqlite3.Connection, table: str, csv_path: Path, keys: List[str]) -> int:
    df = pd.read_csv(csv_path)
    # Normalize column names to valid SQLite identifiers (simple pass-through; assumes clean names)
    return upsert_df(conn, table, df, keys)
