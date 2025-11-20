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

def _load_team_features() -> pd.DataFrame:
    p = OUT / "team_features.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if {"team_slug","date"}.issubset(df.columns) and not df.empty:
                return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_training_frame(return_dates: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    feat = load_features()
    games = load_historical_games()
    if feat.empty or games.empty:
        if return_dates:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), None
    # Ensure game_id join key
    feat["game_id"] = feat.get("game_id").astype(str)
    games["game_id"] = games.get("game_id").astype(str)
    df = feat.merge(games, on="game_id", how="inner")
    if df.empty:
        if return_dates:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), None
    # Normalize home/away team column names (merge may introduce _x/_y suffixes)
    if "home_team" not in df.columns:
        if "home_team_x" in df.columns:
            df["home_team"] = df["home_team_x"]
        elif "home_team_y" in df.columns:
            df["home_team"] = df["home_team_y"]
    if "away_team" not in df.columns:
        if "away_team_x" in df.columns:
            df["away_team"] = df["away_team_x"]
        elif "away_team_y" in df.columns:
            df["away_team"] = df["away_team_y"]
    # Targets
    df["total_points"] = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")
    df["margin"] = pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")
    # Feature engineering
    base_cols = [
        "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
        "home_tempo_rating","away_tempo_rating","tempo_rating_sum",
        # Additional contextual rating / margin columns (if present in features_hist/all)
        "home_rating_margin","away_rating_margin","rating_margin_diff"
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
    # Integrate team-level historical aggregates (if artifact exists)
    tfeat = _load_team_features()
    if not tfeat.empty and {"team_slug","date"}.issubset(tfeat.columns):
        # Prepare join keys (strings)
        tfeat["date"] = tfeat["date"].astype(str)
        # Ensure date present in df
        if "date" not in df.columns:
            if "date_x" in df.columns:
                df["date"] = df["date_x"]
            elif "date_y" in df.columns:
                df["date"] = df["date_y"]
            else:
                df["date"] = pd.NaT
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Canonical slugs if absent
        if "_home_slug" not in df.columns:
            try:
                df["_home_slug"] = df["home_team"].astype(str).map(canon_slug)
                df["_away_slug"] = df["away_team"].astype(str).map(canon_slug)
            except Exception:
                pass
        # Vectorized exact-date merge (team_features are pre-shifted to avoid leakage)
        try:
            home_tfeat = tfeat.copy()
            away_tfeat = tfeat.copy()
            # Prefix non-key columns
            home_cols_prefix = {c: f"home_team_{c}" for c in home_tfeat.columns if c not in {"team_slug","date"}}
            away_cols_prefix = {c: f"away_team_{c}" for c in away_tfeat.columns if c not in {"team_slug","date"}}
            home_tfeat = home_tfeat.rename(columns=home_cols_prefix)
            away_tfeat = away_tfeat.rename(columns=away_cols_prefix)
            df = df.merge(home_tfeat, left_on=["_home_slug","date"], right_on=["team_slug","date"], how="left")
            df = df.merge(away_tfeat, left_on=["_away_slug","date"], right_on=["team_slug","date"], how="left", suffixes=("","_awaydup"))
            # Drop duplicate key columns introduced by merges
            for drop_col in ["team_slug","team_slug_awaydup"]:
                if drop_col in df.columns:
                    df.drop(columns=[drop_col], inplace=True)
        except Exception:
            pass
        # Fallback: if no team-level columns present (e.g., exact date rows missing for many future games), use merge_asof backward to pull latest prior aggregates per team.
        if not any(c.startswith("home_team_season_off_ppg") for c in df.columns) and "_home_slug" in df.columns:
            try:
                # Prepare datetime versions
                df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
                tfeat_asof = tfeat.copy()
                tfeat_asof["date_dt"] = pd.to_datetime(tfeat_asof["date"], errors="coerce")
                # Sort for asof
                df_home_sorted = df.sort_values(["_home_slug","date_dt"]).copy()
                df_away_sorted = df.sort_values(["_away_slug","date_dt"]).copy()
                tfeat_sorted = tfeat_asof.sort_values(["team_slug","date_dt"]).copy()
                import pandas as _pd  # local import safety
                # Home side asof merge
                home_asof = _pd.merge_asof(
                    df_home_sorted,
                    tfeat_sorted,
                    left_on="date_dt",
                    right_on="date_dt",
                    left_by="_home_slug",
                    right_by="team_slug",
                    direction="backward",
                    allow_exact_matches=True,
                )
                # Away side asof merge
                away_asof = _pd.merge_asof(
                    df_away_sorted,
                    tfeat_sorted,
                    left_on="date_dt",
                    right_on="date_dt",
                    left_by="_away_slug",
                    right_by="team_slug",
                    direction="backward",
                    allow_exact_matches=True,
                )
                # Reindex to original order
                home_asof = home_asof.sort_index()
                away_asof = away_asof.sort_index()
                # Select team feature columns (exclude keys)
                team_cols = [c for c in tfeat_sorted.columns if c not in {"team_slug","date","date_dt"}]
                for c in team_cols:
                    hc = f"home_team_{c}"; ac = f"away_team_{c}"
                    if hc not in df.columns:
                        df[hc] = home_asof[c]
                    if ac not in df.columns:
                        df[ac] = away_asof[c]
                # Clean temp columns
                for temp in ["date_dt"]:
                    if temp in df.columns:
                        df.drop(columns=[temp], inplace=True)
            except Exception:
                pass
    # Select modeling features (base + aggregated if present)
    feature_cols = [
        "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
        "off_diff","def_diff","tempo_avg","tempo_rating_sum",
        "home_rating_margin","away_rating_margin","rating_margin_diff"
    ]
    # Optional situational rest / schedule intensity differentials
    for situ in ["rest_diff","home_rest_adv","b2b_diff"]:
        if situ in df.columns and situ not in feature_cols:
            feature_cols.append(situ)
    # Extend with selected team-level aggregate deltas
    agg_pairs = [
        ("season_off_ppg","season_def_ppg"),
        ("last5_off_ppg","last5_def_ppg"),
        ("last10_off_ppg","last10_def_ppg"),
        ("rolling15_off_ppg","rolling15_def_ppg"),
    ]
    for a_for, a_against in agg_pairs:
        hf = f"home_team_{a_for}"; af = f"away_team_{a_for}"; hd = f"home_team_{a_against}"; ad = f"away_team_{a_against}"
        for c in [hf, af, hd, ad]:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)
        # Add differential features if both sides present
        if hf in df.columns and af in df.columns:
            df[f"diff_{a_for}"] = pd.to_numeric(df[hf], errors="coerce") - pd.to_numeric(df[af], errors="coerce")
            feature_cols.append(f"diff_{a_for}")
        if hd in df.columns and ad in df.columns:
            df[f"diff_{a_against}"] = pd.to_numeric(df[hd], errors="coerce") - pd.to_numeric(df[ad], errors="coerce")
            feature_cols.append(f"diff_{a_against}")
    # Additional optional team-level feature differentials (rest / volatility / ewm recent form)
    extra_pairs = [
        ("rest_days",),
        ("season_margin_std",),
        ("season_total_std",),
        ("ewm_off_ppg",),
        ("ewm_def_ppg",),
        ("ewm_margin_avg",),
    ]
    for (ename,) in extra_pairs:
        hc = f"home_team_{ename}"; ac = f"away_team_{ename}"
        if hc in df.columns and hc not in feature_cols:
            feature_cols.append(hc)
        if ac in df.columns and ac not in feature_cols:
            feature_cols.append(ac)
        if hc in df.columns and ac in df.columns:
            diff_col = f"diff_{ename}"
            df[diff_col] = pd.to_numeric(df[hc], errors="coerce") - pd.to_numeric(df[ac], errors="coerce")
            feature_cols.append(diff_col)
    # Back-to-back flags encoded as int difference (home minus away) if present
    for flag in ["back_to_back"]:
        hc = f"home_team_{flag}"; ac = f"away_team_{flag}"
        if hc in df.columns and ac in df.columns:
            # Cast to int (True=1 False=0)
            try:
                df[hc] = df[hc].astype(int)
                df[ac] = df[ac].astype(int)
            except Exception:
                pass
            diff_col = f"diff_{flag}"
            df[diff_col] = pd.to_numeric(df[hc], errors="coerce") - pd.to_numeric(df[ac], errors="coerce")
            for c in [hc, ac, diff_col]:
                if c not in feature_cols and c in df.columns:
                    feature_cols.append(c)
    X = df[[c for c in feature_cols if c in df.columns]].copy()
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
    dates_series: pd.Series | None = None
    if return_dates:
        # Attempt to recover dates aligned to X rows
        if "date" in df.columns:
            dates_series = pd.to_datetime(df.loc[mask_valid, "date"], errors="coerce")
        else:
            dates_series = pd.Series([pd.NaT] * len(X), dtype="datetime64[ns]")
    return X, y_total, y_margin, dates_series

__all__ = ["build_training_frame"]
