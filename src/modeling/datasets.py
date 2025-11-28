from __future__ import annotations

from typing import Optional, List
import pandas as pd
from pathlib import Path
import re


def _to_dt(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        return pd.to_datetime([None] * len(s))


def _numeric(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([float("nan")] * len(s))


def find_daily_result_files(outputs_dir: str | Path) -> List[Path]:
    out = Path(outputs_dir)
    daily_dir = out / "daily_results"
    if not daily_dir.exists():
        return []
    files = sorted(daily_dir.glob("results_*.csv"))
    return [p for p in files if p.is_file()]


def load_training_data(outputs_dir: str | Path = "outputs", date_start: Optional[str] = None, date_end: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate daily_results files into one DataFrame and derive targets:
    - home_win (1/0)
    - ats_home_cover (1/0) if spread available
    - ou_over (1/0) if total available
    Also adds helper columns: actual_total, date (datetime64[ns]).
    """
    files = find_daily_result_files(outputs_dir)
    if not files:
        return pd.DataFrame()

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if df is None or df.empty:
                continue
            df["_source_file"] = str(p.name)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalize date
    if "date" in df.columns:
        df["date"] = _to_dt(df["date"]).dt.tz_localize(None) if str(df["date"].dtype) != "datetime64[ns]" else df["date"]
    else:
        # Try infer date from filename if present
        try:
            df["date"] = pd.to_datetime(df["_source_file"].str.extract(r"results_(\d{4}-\d{2}-\d{2})")[0], errors="coerce")
        except Exception:
            df["date"] = pd.NaT

    # Filter by date range if specified
    if date_start:
        ds = pd.to_datetime(date_start, errors="coerce")
        if pd.notna(ds):
            df = df[df["date"] >= ds]
    if date_end:
        de = pd.to_datetime(date_end, errors="coerce")
        if pd.notna(de):
            df = df[df["date"] <= de]

    # Derive basic numeric helpers
    for c in ["home_score", "away_score", "closing_spread_home", "closing_total", "spread_home", "market_total"]:
        if c in df.columns:
            df[c] = _numeric(df[c])

    if {"home_score", "away_score"}.issubset(df.columns):
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    else:
        df["actual_total"] = pd.NA
        df["home_win"] = pd.NA

    # ATS cover (home)
    if "ats_home" in df.columns:
        # Normalize textual to numeric if present
        # Accept values like True/False, 1/0, or strings
        df["ats_home_cover"] = df["ats_home"].astype(str).str.lower().map({"true": 1, "1": 1, "yes": 1, "y": 1, "t": 1, "false": 0, "0": 0, "no": 0, "n": 0, "f": 0})
        df.loc[~df["ats_home_cover"].isin([0, 1]), "ats_home_cover"] = pd.NA
        df["ats_home_cover"] = df["ats_home_cover"].astype("float").astype("Int64")
    elif {"home_score", "away_score", "closing_spread_home"}.issubset(df.columns):
        df["ats_home_cover"] = ((df["home_score"] + df["closing_spread_home"]) > df["away_score"]).astype(int)
    else:
        df["ats_home_cover"] = pd.NA

    # OU over
    if "ou_over" in df.columns:
        df["ou_over"] = df["ou_over"].astype(str).str.lower().map({"true": 1, "1": 1, "yes": 1, "y": 1, "t": 1, "over": 1, "false": 0, "0": 0, "no": 0, "n": 0, "f": 0, "under": 0})
        df.loc[~df["ou_over"].isin([0, 1]), "ou_over"] = pd.NA
        df["ou_over"] = df["ou_over"].astype("float").astype("Int64")
    elif {"home_score", "away_score", "closing_total"}.issubset(df.columns):
        df["ou_over"] = ((df["home_score"] + df["away_score"]) > df["closing_total"]).astype(int)
    else:
        df["ou_over"] = pd.NA

    return df
