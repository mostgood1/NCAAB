"""Team-level historical feature aggregation.

Generates per-team, per-date rolling & season aggregates derived from historical
games_* CSVs in outputs/. These features improve model uniformity by supplying
consistent team context (offense/defense strength, recent form, volatility)
across seasons.

Output: outputs/team_features.csv with columns (subset):
  date, season, team, team_slug,
  pts_for, pts_against, margin,
  season_games,
  season_off_ppg, season_def_ppg, season_margin_avg,
  last5_off_ppg, last5_def_ppg, last5_margin_avg,
  last10_off_ppg, last10_def_ppg, last10_margin_avg,
  rolling15_off_ppg, rolling15_def_ppg,
  season_total_std, season_margin_std

Notes:
  - Uses only completed games (requires home_score & away_score present).
  - For a given date, aggregates are computed using games strictly BEFORE that date
    to avoid target leakage for training & potential future inference reuse.
  - Early-season rows (< window size) fall back to season-to-date or simple mean.
"""
from __future__ import annotations
import argparse
import datetime as dt
import pathlib
from typing import List
import pandas as pd

from .utils import canon_slug

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def _load_all_games() -> pd.DataFrame:
    frames = []
    for p in OUT.glob("games_20*.csv"):
        try:
            df = pd.read_csv(p)
            if {"home_team","away_team","home_score","away_score","date"}.issubset(df.columns):
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True)
    # Keep completed rows (scores not null)
    g = g[pd.to_numeric(g["home_score"], errors="coerce").notna() & pd.to_numeric(g["away_score"], errors="coerce").notna()]
    if g.empty:
        return pd.DataFrame()
    # Normalize types
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g[g["date"].notna()].copy()
    g["home_team"] = g["home_team"].astype(str)
    g["away_team"] = g["away_team"].astype(str)
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
    return g


def _expand_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    # Convert each game into two team-centric rows (perspective: team vs opponent)
    rows = []
    for _, r in games.iterrows():
        rows.append({
            "date": r["date"],
            "team": r["home_team"],
            "opp": r["away_team"],
            "pts_for": r["home_score"],
            "pts_against": r["away_score"],
            "is_home": 1,
        })
        rows.append({
            "date": r["date"],
            "team": r["away_team"],
            "opp": r["home_team"],
            "pts_for": r["away_score"],
            "pts_against": r["home_score"],
            "is_home": 0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["team","date"], inplace=True)
    df["team_slug"] = df["team"].map(canon_slug)
    df["opp_slug"] = df["opp"].map(canon_slug)
    df["margin"] = df["pts_for"] - df["pts_against"]
    df["season"] = df["date"].dt.year
    return df


def _compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    # Group by team for rolling calcs
    for team, sub in df.groupby("team_slug"):
        sub = sub.sort_values("date")
        # Precompute expanding stats for efficiency
        exp_pts_for = sub["pts_for"].expanding().mean()
        exp_pts_against = sub["pts_against"].expanding().mean()
        exp_margin = sub["margin"].expanding().mean()
        exp_margin_std = sub["margin"].expanding().std().fillna(0.0)
        exp_total_std = (sub["pts_for"] + sub["pts_against"]).expanding().std().fillna(0.0)
        # Rest days between games (days since previous game)
        rest_days = sub["date"].diff().dt.days.fillna(999).astype(int)
        b2b_flag = rest_days == 1
        # Exponential weighted recent form (avoid leakage by shifting later)
        ewm_off = sub["pts_for"].ewm(alpha=0.30, adjust=False).mean()
        ewm_def = sub["pts_against"].ewm(alpha=0.30, adjust=False).mean()
        ewm_margin = sub["margin"].ewm(alpha=0.30, adjust=False).mean()
        # Rolling windows
        roll5_for = sub["pts_for"].rolling(5).mean()
        roll5_against = sub["pts_against"].rolling(5).mean()
        roll5_margin = sub["margin"].rolling(5).mean()
        roll10_for = sub["pts_for"].rolling(10).mean()
        roll10_against = sub["pts_against"].rolling(10).mean()
        roll10_margin = sub["margin"].rolling(10).mean()
        roll15_for = sub["pts_for"].rolling(15).mean()
        roll15_against = sub["pts_against"].rolling(15).mean()
        for idx, r in sub.iterrows():
            # Use statistics strictly before current row date (shift by 1 to avoid leakage)
            pos = sub.index.get_loc(idx)
            if pos == 0:
                # No prior history: skip producing a row (insufficient context)
                continue
            prior_slice = sub.iloc[:pos]
            season = int(r["season"]) if "season" in r else int(r["date"].year)
            out_rows.append({
                "date": r["date"],
                "season": season,
                "team": r["team"],
                "team_slug": r["team_slug"],
                "season_games": pos,  # number of prior games
                "season_off_ppg": float(exp_pts_for.iloc[pos-1]),
                "season_def_ppg": float(exp_pts_against.iloc[pos-1]),
                "season_margin_avg": float(exp_margin.iloc[pos-1]),
                "season_margin_std": float(exp_margin_std.iloc[pos-1]),
                "season_total_std": float(exp_total_std.iloc[pos-1]),
                "last5_off_ppg": float(roll5_for.iloc[pos-1]) if pos >= 5 else float(exp_pts_for.iloc[pos-1]),
                "last5_def_ppg": float(roll5_against.iloc[pos-1]) if pos >= 5 else float(exp_pts_against.iloc[pos-1]),
                "last5_margin_avg": float(roll5_margin.iloc[pos-1]) if pos >= 5 else float(exp_margin.iloc[pos-1]),
                "last10_off_ppg": float(roll10_for.iloc[pos-1]) if pos >= 10 else float(exp_pts_for.iloc[pos-1]),
                "last10_def_ppg": float(roll10_against.iloc[pos-1]) if pos >= 10 else float(exp_pts_against.iloc[pos-1]),
                "last10_margin_avg": float(roll10_margin.iloc[pos-1]) if pos >= 10 else float(exp_margin.iloc[pos-1]),
                "rolling15_off_ppg": float(roll15_for.iloc[pos-1]) if pos >= 15 else float(exp_pts_for.iloc[pos-1]),
                "rolling15_def_ppg": float(roll15_against.iloc[pos-1]) if pos >= 15 else float(exp_pts_against.iloc[pos-1]),
                "rest_days": int(rest_days.iloc[pos]),
                "back_to_back": bool(b2b_flag.iloc[pos]),
                "ewm_off_ppg": float(ewm_off.iloc[pos-1]),
                "ewm_def_ppg": float(ewm_def.iloc[pos-1]),
                "ewm_margin_avg": float(ewm_margin.iloc[pos-1]),
            })
    return pd.DataFrame(out_rows)


def build_team_features() -> pd.DataFrame:
    games = _load_all_games()
    if games.empty:
        return pd.DataFrame()
    team_df = _expand_team_rows(games)
    if team_df.empty:
        return pd.DataFrame()
    feat = _compute_rolling_features(team_df)
    # Final ordering & types
    if feat.empty:
        return feat
    feat.sort_values(["team_slug","date"], inplace=True)
    feat["date"] = feat["date"].dt.strftime("%Y-%m-%d")
    return feat


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "team_features.csv"), help="Output CSV path")
    args = ap.parse_args(argv)
    df = build_team_features()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} team feature rows -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
