"""Data loading utilities for model training.

Responsible for constructing a historical training frame with targets:
 - total_points (home_score + away_score)
 - margin (home_score - away_score)

We leverage features_all.csv (or fallback variants) plus historical games_YYYY.csv files.
"""
from __future__ import annotations
import pathlib
import pandas as pd
from typing import Tuple
from .utils import canon_slug

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

FEATURE_FILES_ORDER = [
    "features_all.csv",
    "features_hist.csv",
    "features_curr.csv",
]

def load_features() -> pd.DataFrame:
    for name in FEATURE_FILES_ORDER:
        p = OUT / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame()

def load_historical_games() -> pd.DataFrame:
    frames = []
    for p in OUT.glob("games_20*.csv"):
        try:
            df = pd.read_csv(p)
            # Basic sanity: require scores
            if {"home_score","away_score"}.issubset(df.columns) and not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True)
    # Keep columns of interest
    keep = [c for c in g.columns if c in {"game_id","date","home_team","away_team","home_score","away_score"}]
    g = g[keep]
    g["game_id"] = g.get("game_id").astype(str)
    return g

def build_training_frame() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feat = load_features()
    games = load_historical_games()
    if feat.empty or games.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    # Ensure game_id join key
    feat["game_id"] = feat.get("game_id").astype(str)
    games["game_id"] = games.get("game_id").astype(str)
    df = feat.merge(games, on="game_id", how="inner")
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    # Targets
    df["total_points"] = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")
    df["margin"] = pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")
    # Feature engineering
    base_cols = [
        "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
        "home_tempo_rating","away_tempo_rating","tempo_rating_sum"
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    df["off_diff"] = df["home_off_rating"] - df["away_off_rating"]
    df["def_diff"] = df["home_def_rating"] - df["away_def_rating"]
    df["tempo_avg"] = (
        pd.to_numeric(df["home_tempo_rating"], errors="coerce") + pd.to_numeric(df["away_tempo_rating"], errors="coerce")
    ) / 2.0
    # Canonical pair key (could aid later grouping)
    try:
        df["_home_slug"] = df["home_team"].astype(str).map(canon_slug)
        df["_away_slug"] = df["away_team"].astype(str).map(canon_slug)
        df["pair_key"] = df.apply(lambda r: "::".join(sorted([r["_home_slug"], r["_away_slug"]])), axis=1)
    except Exception:
        pass
    # Select modeling features
    feature_cols = [
        "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
        "off_diff","def_diff","tempo_avg","tempo_rating_sum"
    ]
    X = df[feature_cols].copy()
    y_total = pd.to_numeric(df["total_points"], errors="coerce")
    y_margin = pd.to_numeric(df["margin"], errors="coerce")
    # Drop rows with missing targets
    mask_valid = y_total.notna() & y_margin.notna()
    X = X[mask_valid]
    y_total = y_total[mask_valid]
    y_margin = y_margin[mask_valid]
    # Impute remaining NaNs in features with column medians
    X = X.apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        col = X[c]
        if col.isna().any():
            X[c] = col.fillna(col.median())
    return X, y_total, y_margin

__all__ = ["build_training_frame"]
