from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentRetuneReport5MinConfig:
    backtest_csv: Path
    games_csv: Path
    out_dir: Path
    start: str | None = None
    end: str | None = None
    min_games_team: int = 10
    min_games_conference: int = 30
    shrink_k: float = 20.0


def _load_conference_map(paths: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        # Some conference files contain header-ish comment lines starting with '#'
        df = pd.read_csv(p, comment="#")
        if df.empty:
            continue
        if "team" not in df.columns or "conference" not in df.columns:
            continue
        for _, r in df.iterrows():
            team = str(r.get("team", "")).strip()
            conf = str(r.get("conference", "")).strip()
            if team and conf and team not in mapping:
                mapping[team] = conf
    return mapping


def _with_filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df


def build_segment_retune_report_5min(cfg: SegmentRetuneReport5MinConfig) -> dict:
    df = pd.read_csv(cfg.backtest_csv)
    if df.empty:
        raise ValueError(f"Empty backtest CSV: {cfg.backtest_csv}")

    required = {
        "date",
        "game_id",
        "end_min",
        "actual_total",
        "pred_q50",
        "pred_q10",
        "pred_q90",
        "err_q50",
        "abs_err_q50",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Backtest CSV missing required columns: {sorted(missing)}")

    df["date"] = df["date"].astype(str)
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64").astype(str)
    df = df[df["game_id"] != "<NA>"]
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    df = df.dropna(subset=["end_min"])

    df = _with_filter_dates(df, cfg.start, cfg.end)
    df = df.copy()

    # 80% interval coverage using q10/q90
    df["covered_80"] = (
        pd.to_numeric(df["actual_total"], errors="coerce") >= pd.to_numeric(df["pred_q10"], errors="coerce")
    ) & (pd.to_numeric(df["actual_total"], errors="coerce") <= pd.to_numeric(df["pred_q90"], errors="coerce"))

    games = pd.read_csv(cfg.games_csv)
    if games.empty:
        raise ValueError(f"Empty games CSV: {cfg.games_csv}")

    if "game_id" not in games.columns:
        raise ValueError(f"Games CSV missing game_id: {cfg.games_csv}")

    games = games.copy()
    games["game_id"] = pd.to_numeric(games["game_id"], errors="coerce").astype("Int64").astype(str)
    games = games[games["game_id"] != "<NA>"]
    if "date" in games.columns:
        games["date"] = games["date"].astype(str)

    keep_cols = [
        c
        for c in [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_score_1h",
            "away_score_1h",
            "neutral_site",
        ]
        if c in games.columns
    ]
    games = games[keep_cols]

    merged = df.merge(games, how="left", on="game_id", suffixes=("", "_g"))

    # If we accidentally merged the wrong date (should be unique per game_id), keep backtest date.
    merged["date"] = merged["date"].astype(str)

    # Two-sided team view (each game counts for both teams)
    team_rows = []
    for side, col in (("home", "home_team"), ("away", "away_team")):
        if col not in merged.columns:
            continue
        tmp = merged.copy()
        tmp["team"] = tmp[col]
        tmp["side"] = side
        team_rows.append(tmp)

    if not team_rows:
        raise ValueError("Games CSV missing both home_team and away_team; cannot build team report")

    team_df = pd.concat(team_rows, ignore_index=True)
    team_df["team"] = team_df["team"].astype(str).str.strip()
    team_df = team_df[
        (team_df["team"].str.len() > 0)
        & (~team_df["team"].str.lower().isin(["nan", "none", "null", "<na>"]))
    ]

    # Attach conferences
    repo_root = Path(__file__).resolve().parents[2]
    conf_map = _load_conference_map(
        [
            repo_root / "data" / "d1_conferences.csv",
            repo_root / "data" / "conferences.csv",
        ]
    )
    team_df["conference"] = team_df["team"].map(conf_map).fillna("")

    def agg_group(g: pd.DataFrame) -> pd.Series:
        game_n = int(g["game_id"].nunique())
        mean_err = float(pd.to_numeric(g["err_q50"], errors="coerce").mean())
        mae = float(pd.to_numeric(g["abs_err_q50"], errors="coerce").mean())
        cov80 = float(g["covered_80"].mean())
        std = float(pd.to_numeric(g["err_q50"], errors="coerce").std(ddof=1)) if game_n > 1 else float("nan")
        se = float(std / np.sqrt(game_n)) if game_n > 1 and np.isfinite(std) else float("nan")
        shrink = float(game_n / (game_n + float(cfg.shrink_k)))
        suggested = -mean_err
        suggested_shrunk = suggested * shrink
        return pd.Series(
            {
                "n_games": game_n,
                "mean_err_q50": mean_err,
                "mae_q50": mae,
                "cov80": cov80,
                "std_err_q50": std,
                "se_err_q50": se,
                "suggested_adjustment": float(suggested),
                "suggested_adjustment_shrunk": float(suggested_shrunk),
                "shrink_factor": shrink,
            }
        )

    def apply_agg(gb: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:  # type: ignore[name-defined]
        try:
            return gb.apply(agg_group, include_groups=False)
        except TypeError:
            return gb.apply(agg_group)

    team_report = (
        apply_agg(team_df.groupby(["team", "conference", "end_min"], dropna=False))
        .reset_index()
        .sort_values(["end_min", "n_games", "mae_q50"], ascending=[True, False, False])
    )
    team_report = team_report[team_report["n_games"] >= int(cfg.min_games_team)].copy()

    conf_report = (
        apply_agg(
            team_df[team_df["conference"].astype(str).str.len() > 0].groupby(["conference", "end_min"], dropna=False)
        )
        .reset_index()
        .sort_values(["end_min", "n_games", "mae_q50"], ascending=[True, False, False])
    )
    conf_report = conf_report[conf_report["n_games"] >= int(cfg.min_games_conference)].copy()

    # Predicted total bucket archetypes (based on full-game q50 at end_min=40)
    full = merged[merged["end_min"] == 40].copy()
    if not full.empty:
        full = full[["game_id", "pred_q50", "actual_total"]].copy()
        full = full.rename(columns={"pred_q50": "pred_full_q50", "actual_total": "actual_full_total"})
        merged2 = merged.merge(full, how="left", on="game_id")
        merged2["pred_full_q50"] = pd.to_numeric(merged2["pred_full_q50"], errors="coerce")
        # 5 buckets (quintiles) with stable labels
        try:
            merged2["pred_full_bucket"] = pd.qcut(merged2["pred_full_q50"], q=5, labels=False, duplicates="drop")
        except Exception:
            merged2["pred_full_bucket"] = np.nan

        bucket_report = (
            apply_agg(merged2.dropna(subset=["pred_full_bucket"]).groupby(["pred_full_bucket", "end_min"], dropna=False))
            .reset_index()
            .sort_values(["end_min", "pred_full_bucket"], ascending=[True, True])
        )
    else:
        bucket_report = pd.DataFrame()

    # Worst games by absolute error at key endpoints
    worst = merged.copy()
    worst["abs_err_q50"] = pd.to_numeric(worst["abs_err_q50"], errors="coerce")
    worst = worst.dropna(subset=["abs_err_q50"])
    worst = worst[worst["end_min"].isin([20, 40])]
    worst = worst.sort_values("abs_err_q50", ascending=False).head(50)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{cfg.start or 'ALL'}_to_{cfg.end or 'ALL'}"

    team_path = out_dir / f"segment_retune_team_{tag}.csv"
    conf_path = out_dir / f"segment_retune_conference_{tag}.csv"
    bucket_path = out_dir / f"segment_retune_buckets_{tag}.csv"
    worst_path = out_dir / f"segment_retune_worst_games_{tag}.csv"
    summary_path = out_dir / f"segment_retune_summary_{tag}.json"

    team_report.to_csv(team_path, index=False)
    conf_report.to_csv(conf_path, index=False)
    if not bucket_report.empty:
        bucket_report.to_csv(bucket_path, index=False)
    worst.to_csv(worst_path, index=False)

    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "source_backtest_csv": str(cfg.backtest_csv),
        "source_games_csv": str(cfg.games_csv),
        "start": cfg.start,
        "end": cfg.end,
        "min_games_team": int(cfg.min_games_team),
        "min_games_conference": int(cfg.min_games_conference),
        "shrink_k": float(cfg.shrink_k),
        "rows_used": int(len(df)),
        "games_used": int(df["game_id"].nunique()),
        "outputs": {
            "team": str(team_path),
            "conference": str(conf_path),
            "buckets": str(bucket_path) if bucket_report is not None and not bucket_report.empty else None,
            "worst_games": str(worst_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["out_path"] = str(summary_path)
    return summary
