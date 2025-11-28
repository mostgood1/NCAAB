from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

# Lightweight feature engineering. Keeps everything Pandas-only for portability.
# Assumes aggregated historical game-level data lives in outputs/daily_results/results_*.csv.


def _load_daily_results(outputs_dir: Path) -> pd.DataFrame:
    dr = outputs_dir / "daily_results"
    if not dr.exists():
        return pd.DataFrame()
    files = sorted(dr.glob("results_*.csv"))
    frames: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if df is None or df.empty:
                continue
            df["_source_file"] = p.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    return out


def _team_game_index(hist: pd.DataFrame) -> pd.DataFrame:
    # Build per-team chronological ordering
    if {"home_team","away_team","date"}.issubset(hist.columns):
        rows_h = hist[["home_team","date","home_score","away_score"]].rename(columns={"home_team":"team","home_score":"team_score","away_score":"opp_score"})
        rows_a = hist[["away_team","date","home_score","away_score"]].rename(columns={"away_team":"team","home_score":"opp_score","away_score":"team_score"})
        all_rows = pd.concat([rows_h, rows_a], ignore_index=True)
        all_rows = all_rows.dropna(subset=["team","date"])  # type: ignore[arg-type]
        all_rows = all_rows.sort_values("date")
        return all_rows
    return pd.DataFrame()


def build_team_history_features(outputs_dir: str | Path) -> pd.DataFrame:
    """Return per-game engineered features merged on game_id when possible.

    Features:
      - rest_days_home / rest_days_away
      - games_last7_home / games_last7_away (schedule density)
      - avg_total_last3_home / avg_total_last3_away (recent pace/scoring proxy)
      - home_recent_margin_mean / away_recent_margin_mean
    """
    outputs_dir = Path(outputs_dir)
    hist = _load_daily_results(outputs_dir)
    if hist.empty:
        return pd.DataFrame()
    # Need scores to compute totals/margins
    for col in ["home_score","away_score"]:
        if col not in hist.columns:
            return pd.DataFrame()
    hist["actual_total"] = pd.to_numeric(hist["home_score"], errors="coerce") + pd.to_numeric(hist["away_score"], errors="coerce")
    hist["home_margin"] = pd.to_numeric(hist["home_score"], errors="coerce") - pd.to_numeric(hist["away_score"], errors="coerce")

    tgi = _team_game_index(hist)
    if tgi.empty:
        return pd.DataFrame()

    # Compute rolling contexts
    tgi["team_score"] = pd.to_numeric(tgi["team_score"], errors="coerce")
    tgi["opp_score"] = pd.to_numeric(tgi["opp_score"], errors="coerce")
    tgi["total"] = tgi["team_score"] + tgi["opp_score"]
    tgi["margin"] = tgi["team_score"] - tgi["opp_score"]

    # For each team, compute rest days vs previous game
    tgi["prev_date"] = tgi.groupby("team")["date"].shift(1)
    tgi["rest_days"] = (tgi["date"] - tgi["prev_date"]).dt.days

    # Rolling counts: games in last 7 days
    # We'll approximate by counting prior games with date >= current_date - 7
    def _games_last7(sub: pd.DataFrame) -> pd.Series:
        dates = sub["date"].values
        out = []
        for i, d in enumerate(dates):
            if pd.isna(d):
                out.append(np.nan)
                continue
            thresh = d - np.timedelta64(7, 'D')
            ct = np.sum((dates[:i] >= thresh) & (dates[:i] < d))
            out.append(float(ct))
        return pd.Series(out, index=sub.index)
    tgi["games_last7"] = tgi.groupby("team", group_keys=False).apply(_games_last7)

    # Rolling averages last 3 (excluding current)
    for col in ["total","margin"]:
        tgi[f"{col}_last3_mean"] = tgi.groupby("team")[col].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Reduce to feature rows at game level (home/away)
    # First map features back to original hist by joining on (team,date)
    feat_cols = ["rest_days","games_last7","total_last3_mean","margin_last3_mean"]

    tgi_feat = tgi[["team","date"] + feat_cols].copy()
    # Build per-game rows
    hist_feat = hist[["game_id","date","home_team","away_team"]].copy() if "game_id" in hist.columns else hist[["date","home_team","away_team"]].copy()
    # Merge home features
    hist_feat = hist_feat.merge(tgi_feat.rename(columns={c: f"home_{c}" for c in feat_cols, "team":"home_team"}), on=["home_team","date"], how="left")
    # Merge away features
    hist_feat = hist_feat.merge(tgi_feat.rename(columns={c: f"away_{c}" for c in feat_cols, "team":"away_team"}), on=["away_team","date"], how="left")

    return hist_feat


def attach_team_history_features(df: pd.DataFrame, outputs_dir: str | Path) -> pd.DataFrame:
    try:
        feats = build_team_history_features(outputs_dir)
        if feats.empty:
            return df
        # Date normalization
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        if "date" in feats.columns:
            feats["date"] = pd.to_datetime(feats["date"], errors="coerce").dt.tz_localize(None)
        keys = ["date","home_team","away_team"]
        merge_keys = [k for k in keys if k in df.columns and k in feats.columns]
        if not merge_keys:
            return df
        # If game_id available on both, include it to tighten match
        if "game_id" in df.columns and "game_id" in feats.columns:
            merge_keys = ["game_id"]
        df = df.merge(feats, on=merge_keys, how="left")
    except Exception:
        return df
    return df
