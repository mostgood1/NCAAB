from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, jsonify, request
import logging
from flask import send_file
import pandas as pd
import datetime as dt
import numpy as np
from zoneinfo import ZoneInfo

# Ensure src/ is importable
import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncaab_model.config import settings  # type: ignore
from ncaab_model.data.merge_odds import normalize_name  # type: ignore
try:
    # Import CLI routines we can safely invoke programmatically
    from ncaab_model.cli import finalize_day as cli_finalize_day  # type: ignore
    from ncaab_model.cli import daily_run as cli_daily_run  # type: ignore
    import typer  # type: ignore
except Exception:
    cli_finalize_day = None  # type: ignore
    cli_daily_run = None  # type: ignore
    typer = None  # type: ignore

app = Flask(__name__)
# Basic logging setup (Render captures stdout/stderr)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ncaab_app")
def _load_eval_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Calibration metrics
    calib = OUT / "calibration" / "metrics.json"
    if calib.exists():
        try:
            out["calibration"] = json.loads(calib.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Closing backtest summary
    closing = OUT / "eval_closing" / "overall_summary.json"
    if closing.exists():
        try:
            out["closing_backtest"] = json.loads(closing.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Edge persistence summary
    edgep = OUT / "edge_persistence_summary.json"
    if edgep.exists():
        try:
            out["edge_persistence"] = json.loads(edgep.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out

OUT = settings.outputs_dir


def _today_local() -> dt.date:
    """Return 'today' in the configured schedule timezone (defaults America/New_York).

    Render dynos run UTC; college basketball slates are anchored to US/Eastern. Without
    this adjustment early-morning UTC can cause us to still see yesterday's date and undercount games.
    """
    tz_name = os.getenv("NCAAB_SCHEDULE_TZ", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
        return dt.datetime.now(tz).date()
    except Exception:
        return dt.date.today()


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _load_predictions_current() -> pd.DataFrame:
    # Prefer current-week predictions; fallback to predictions.csv/predictions_all.csv
    for name in ("predictions_week.csv", "predictions.csv", "predictions_all.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_csv_first(paths: list[Path]) -> pd.DataFrame:
    """Utility: return first existing CSV as DataFrame else empty."""
    for p in paths:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame()


def _load_stake_sheet(kind: str, date_str: str | None = None) -> pd.DataFrame:
    """Load stake sheet variant.

    kind: one of 'orig','cal','compare'.
    File conventions (today-only for now):
      - stake_sheet_today.csv (original)
      - stake_sheet_today_cal.csv (calibrated)
      - stake_sheet_today_compare.csv (side-by-side deltas)
    """
    # If a date is provided, prefer dated filenames (stake_sheet_YYYY-MM-DD*.csv) with fallbacks to today files.
    candidates: list[Path] = []
    if date_str:
        if kind == "orig":
            candidates.append(OUT / f"stake_sheet_{date_str}.csv")
            candidates.append(OUT / "stake_sheet_today.csv")
        elif kind == "cal":
            candidates.append(OUT / f"stake_sheet_{date_str}_cal.csv")
            candidates.append(OUT / "stake_sheet_today_cal.csv")
        elif kind == "compare":
            candidates.append(OUT / f"stake_sheet_{date_str}_compare.csv")
            candidates.append(OUT / "stake_sheet_today_compare.csv")
    else:
        if kind == "orig":
            candidates.append(OUT / "stake_sheet_today.csv")
        elif kind == "cal":
            candidates.append(OUT / "stake_sheet_today_cal.csv")
        elif kind == "compare":
            candidates.append(OUT / "stake_sheet_today_compare.csv")
    return _load_csv_first(candidates)


def _summarize_stake_sheet(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(df)}
    # Common columns: stake, kelly_fraction, prob, ev
    for col in ["stake","kelly_fraction","prob","ev","price","line","delta_prob","delta_kelly","delta_ev","delta_stake"]:
        if col in df.columns:
            try:
                ser = pd.to_numeric(df[col], errors="coerce")
                out[f"sum_{col}"] = float(ser.dropna().sum())
                out[f"mean_{col}"] = float(ser.dropna().mean())
            except Exception:
                pass
    # Aggregate stake by book if present
    if "book" in df.columns:
        try:
            book_stakes = (
                df.groupby("book")["stake"].sum().sort_values(ascending=False).to_dict()
                if "stake" in df.columns else {}
            )
            out["book_stakes"] = book_stakes
        except Exception:
            pass
    return out


def _load_calibration_artifact() -> dict[str, Any] | None:
    # Artifact name (totals) introduced earlier; keep flexible for future variants
    candidates = [OUT / "calibration_totals.json", OUT / "calibration" / "artifact_totals.json"]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def _compute_coverage_snapshot() -> dict[str, Any]:
    """Compute basic coverage counts & intersections for games/predictions/odds/picks.

    Returns dict with counts and unmatched ID lists (truncated) for surfacing diagnostics.
    """
    games = _safe_read_csv(OUT / "games_curr.csv")
    preds = _load_predictions_current()
    odds = _safe_read_csv(OUT / "games_with_last.csv")
    picks = _safe_read_csv(OUT / "picks_raw.csv")
    def id_set(df: pd.DataFrame) -> set[str]:
        if df.empty or "game_id" not in df.columns:
            return set()
        try:
            return set(df["game_id"].astype(str).dropna())
        except Exception:
            return set()
    g_ids = id_set(games)
    p_ids = id_set(preds)
    o_ids = id_set(odds)
    k_ids = id_set(picks)
    universe = g_ids | p_ids | o_ids | k_ids
    snap: dict[str, Any] = {
        "n_games": len(g_ids),
        "n_preds": len(p_ids),
        "n_odds": len(o_ids),
        "n_picks": len(k_ids),
        "n_universe": len(universe),
        "intersect_game_pred": len(g_ids & p_ids),
        "intersect_game_odds": len(g_ids & o_ids),
        "intersect_pred_odds": len(p_ids & o_ids),
        "intersect_all": len(g_ids & p_ids & o_ids),
    }
    # Unmatched lists (truncate for UI)
    def trunc(s: set[str]) -> list[str]:
        lim = 40
        arr = sorted(list(s))
        return arr[:lim]
    snap["games_without_preds"] = trunc(g_ids - p_ids)
    snap["preds_without_games"] = trunc(p_ids - g_ids)
    snap["games_without_odds"] = trunc(g_ids - o_ids)
    snap["odds_without_games"] = trunc(o_ids - g_ids)
    snap["preds_without_odds"] = trunc(p_ids - o_ids)
    snap["picks_without_odds"] = trunc(k_ids - o_ids)
    # Basic ratios
    def ratio(a: int, b: int) -> float | None:
        try:
            return round(a / b, 3) if b else None
        except Exception:
            return None
    snap["coverage_pred_vs_games"] = ratio(len(g_ids & p_ids), len(g_ids))
    snap["coverage_odds_vs_games"] = ratio(len(g_ids & o_ids), len(g_ids))
    snap["coverage_picks_vs_preds"] = ratio(len(k_ids & p_ids), len(p_ids))
    return snap


def _load_games_current() -> pd.DataFrame:
    # Prefer fused current day, then seasonal, then historical full sets
    for name in ("games_curr.csv", "games_fused.parquet", "games_fused.csv", "games_hist_fused.csv", "games_all.csv", "games.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_picks() -> pd.DataFrame:
    for name in ("picks_clean.csv", "picks_today.csv", "picks.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_odds_joined(date_str: str | None = None) -> pd.DataFrame:
    """Load per-game odds joined to ESPN game_id when available.

    Preference order (most reliable first):
      - games_with_odds_today_edges.csv (has game_id + per-book quotes + enriched edges)
      - merged_odds_predictions_today.csv (joined pairs with game_id and odds)
      - games_with_odds_today.csv (may be stale or partial but joined)
      - games_with_last.csv (historical last odds join)
      - games_with_closing.csv (heuristic/closing-only join)
    """
    candidate_files: list[Path] = []
    # If viewing a past date, try date-specific odds joins first
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None
    if date_str and today_str and date_str != today_str:
        for base in (
            f"games_with_odds_{date_str}_edges.csv",
            f"merged_odds_predictions_{date_str}.csv",
            f"games_with_odds_{date_str}.csv",
        ):
            candidate_files.append(OUT / base)
        # Pattern expansion: handle enriched/dist/blend/cal variants produced by pipeline
        try:
            # Glob all files starting with games_with_odds_{date_str}
            pattern = f"games_with_odds_{date_str}*.csv"
            for p in sorted(OUT.glob(pattern)):
                if p not in candidate_files:
                    candidate_files.append(p)
        except Exception:
            pass
    # Prefer different priority depending on whether date_str targets today or a past/future date
    if date_str and today_str and date_str != today_str:
        # Past/future date: prefer historical baseline joins before considering today's temp files
        for base in ("games_with_last.csv", "games_with_closing.csv"):
            candidate_files.append(OUT / base)
        # Only consider today's temp joins as a last resort (may not intersect the requested date)
        for base in (
            "games_with_odds_today_edges.csv",
            "merged_odds_predictions_today.csv",
            "games_with_odds_today.csv",
        ):
            candidate_files.append(OUT / base)
    else:
        # Today or unspecified date: try today's temp joins first, then historical baselines
        for base in (
            "games_with_odds_today_edges.csv",
            "merged_odds_predictions_today.csv",
            "games_with_odds_today.csv",
        ):
            candidate_files.append(OUT / base)
        for base in ("games_with_last.csv", "games_with_closing.csv"):
            candidate_files.append(OUT / base)
    for path in candidate_files:
        df = _safe_read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()

def _load_edges() -> pd.DataFrame:
    """Load per-book edge metrics if present (games_with_last_edges.csv)."""
    for name in ("games_with_last_edges.csv", "edges.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty:
            return df
    return pd.DataFrame()

def _aggregate_full_game_totals(odds: pd.DataFrame) -> pd.DataFrame:
    """Return per-game aggregated full-game totals odds (median total + list of quotes).

    Output columns: game_id, market_total, quotes (list[dict]), commence_time (earliest)
    """
    if odds.empty or "game_id" not in odds.columns:
        return pd.DataFrame()
    o = odds.copy()
    try:
        o["game_id"] = o["game_id"].astype(str)
    except Exception:
        pass
    # Drop rows with missing/placeholder game_id to avoid spurious 'nan' entries
    try:
        bad_keys = {"nan", "none", "", "null"}
        mask_bad = o["game_id"].astype(str).str.strip().str.lower().isin(bad_keys)
        o = o[~mask_bad]
    except Exception:
        pass
    # Filter to totals + full game periods
    if "market" in o.columns:
        o = o[o["market"].astype(str).str.lower() == "totals"]
    if "period" in o.columns:
        o = o[o["period"].astype(str).str.lower().isin(["full_game", "fg", "full game"])]
    if o.empty:
        return pd.DataFrame()
    # Build quote lists & median totals
    quotes_map: dict[str, list[dict[str, Any]]] = {}
    commence_map: dict[str, Any] = {}
    if "commence_time" in o.columns:
        try:
            o["_commence"] = pd.to_datetime(o["commence_time"], errors="coerce")
        except Exception:
            o["_commence"] = pd.NaT
    else:
        o["_commence"] = pd.NaT
    for gid, g in o.groupby("game_id"):
        # Sort by book for determinism
        g2 = g.sort_values("book") if "book" in g.columns else g
        items: list[dict[str, Any]] = []
        for _, r in g2.iterrows():
            items.append({"book": r.get("book"), "total": r.get("total"), "price_over": r.get("price_over"), "price_under": r.get("price_under")})
            if len(items) >= 10:  # cap length
                break
        quotes_map[str(gid)] = items
        if g2["_commence"].notna().any():
            try:
                commence_map[str(gid)] = g2["_commence"].min().strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
    med = o.groupby("game_id")["total"].median().rename("market_total")
    out = med.reset_index()
    out["quotes"] = out["game_id"].map(lambda x: quotes_map.get(str(x), []))
    out["commence_time"] = out["game_id"].map(lambda x: commence_map.get(str(x)))
    return out


def _load_accuracy_summary() -> Dict[str, Any] | None:
    """Load accuracy summary JSON from eval directories if present."""
    for p in [OUT / "eval_last2" / "accuracy_summary.json", OUT / "eval" / "accuracy_summary.json"]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def _load_daily_results_for(date_str: str) -> pd.DataFrame:
    """Load per-day results CSV if it exists: outputs/daily_results/results_YYYY-MM-DD.csv"""
    try:
        p = OUT / "daily_results" / f"results_{date_str}.csv"
        if p.exists():
            df = pd.read_csv(p)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _load_schedule_coverage() -> pd.DataFrame:
    """Load per-day schedule coverage counts and anomaly flags if present."""
    p = OUT / "schedule_coverage.csv"
    if not p.exists():
        # Fallback to day counts if present
        q = OUT / "schedule_day_counts.csv"
        if q.exists():
            try:
                df = pd.read_csv(q)
                # Normalize to expected columns
                if {"date", "n_games"}.issubset(df.columns):
                    if "anomaly" not in df.columns:
                        df["anomaly"] = False
                    return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _load_branding_map() -> dict[str, dict[str, Any]]:
    """Load team branding: logo URL and colors.

    CSV columns (flexible headers accepted):
      - team (canonical team name)
      - logo (URL or /static path)
      - primary_color, secondary_color, text_color (CSS color strings)

    Returns dict keyed by normalized team name.
    """
    path = settings.data_dir / "team_branding.csv"
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
        # Resolve likely column names
        cols = {c.lower().strip(): c for c in df.columns}
        team_col = cols.get("team") or cols.get("name") or cols.get("canonical")
        logo_col = cols.get("logo") or cols.get("logo_url")
        pri_col = cols.get("primary_color") or cols.get("primary")
        sec_col = cols.get("secondary_color") or cols.get("secondary")
        txt_col = cols.get("text_color") or cols.get("text")
        if not team_col:
            return out
        for _, r in df.iterrows():
            team = str(r.get(team_col) or "").strip()
            if not team:
                continue
            key = normalize_name(team)
            out[key] = {
                "logo": r.get(logo_col) if logo_col and pd.notna(r.get(logo_col)) else None,
                "primary": r.get(pri_col) if pri_col and pd.notna(r.get(pri_col)) else None,
                "secondary": r.get(sec_col) if sec_col and pd.notna(r.get(sec_col)) else None,
                "text": r.get(txt_col) if txt_col and pd.notna(r.get(txt_col)) else None,
                "team": team,
            }
    except Exception:
        return {}
    return out


def _load_d1_team_set() -> set[str]:
    """Load normalized Division I team names from data/d1_conferences.csv.

    Returns a set of normalized team names using normalize_name(). If the file
    is missing or unreadable, returns an empty set.
    """
    path = settings.data_dir / "d1_conferences.csv"
    d1set: set[str] = set()
    if not path.exists():
        return d1set
    try:
        df = pd.read_csv(path)
        # Prefer a column named 'team'/'school'/'name', else fallback to first column
        cols = {c.lower().strip(): c for c in df.columns}
        team_col = cols.get("team") or cols.get("school") or cols.get("name") or list(df.columns)[0]
        ser = df[team_col].astype(str).map(normalize_name)
        d1set = set(ser.dropna().astype(str))
    except Exception:
        d1set = set()
    return d1set


def _load_all_finals(limit: int | None = 1000) -> pd.DataFrame:
    """Load all per-day results files and return a consolidated DataFrame of finals.

    Columns include: date, game_id, home_team, away_team, home_score, away_score, actual_total,
    plus optional predicted and market totals when present.
    """
    dr_dir = OUT / "daily_results"
    if not dr_dir.exists():
        return pd.DataFrame()
    files = sorted([p for p in dr_dir.glob("results_*.csv")])
    if not files:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
            # Normalize date from filename if column missing or unreliable
            date_iso = p.stem.split("_")[-1]
            df["date"] = pd.to_datetime(df.get("date", date_iso), errors="coerce").dt.strftime("%Y-%m-%d")
            # Keep key columns
            keep = [
                "date", "game_id", "home_team", "away_team", "home_score", "away_score",
                "actual_total", "pred_total", "closing_total",
            ]
            cols = [c for c in keep if c in df.columns]
            sub = df[cols].copy()
            parts.append(sub)
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Only rows with completed games
    if "actual_total" in out.columns:
        try:
            out["actual_total"] = pd.to_numeric(out["actual_total"], errors="coerce")
            out = out[out["actual_total"] > 0]
        except Exception:
            pass
    # Sort by date desc then home team
    if "date" in out.columns:
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.sort_values(["date", "home_team"], ascending=[False, True]).reset_index(drop=True)
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # Apply optional limit
    if limit is not None and len(out) > int(limit):
        out = out.head(int(limit))
    return out


def _build_results_df(date_str: str, force_use_daily: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct a per-date results DataFrame similar to index() but lean for API.

    Preference order:
      1. Use daily_results file if it has scores or predictive columns (or forced).
      2. Else build from predictions merged with games metadata for that date.
      3. Else fallback to games-only slate for that date.

    Returns (df, meta) where meta contains summary counts & flags.
    """
    meta: dict[str, Any] = {"date": date_str, "daily_used": False}
    games = _load_games_current()
    preds = _load_predictions_current()
    odds = _load_odds_joined(date_str)
    # Normalize date columns
    for df_ in (games, preds):
        if not df_.empty and "date" in df_.columns:
            try:
                df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    # Ensure game_id is consistently string-typed across inputs to avoid merge dtype conflicts
    for df_ in (games, preds):
        try:
            if not df_.empty and "game_id" in df_.columns:
                df_["game_id"] = df_["game_id"].astype(str)
        except Exception:
            pass
    # Filter to date
    if date_str:
        if "date" in games.columns:
            games = games[games["date"].astype(str) == date_str]
        if "date" in preds.columns:
            preds = preds[preds["date"].astype(str) == date_str]
    daily_df = _load_daily_results_for(date_str)
    if not daily_df.empty:
        has_scores = False
        has_preds = False
        try:
            if {"home_score","away_score"}.issubset(daily_df.columns):
                sc_sum = pd.to_numeric(daily_df["home_score"], errors="coerce") + pd.to_numeric(daily_df["away_score"], errors="coerce")
                has_scores = bool((sc_sum > 0).any())
            for c in ("pred_total","pred_margin","market_total","closing_total"):
                if c in daily_df.columns and daily_df[c].notna().any():
                    has_preds = True
                    break
        except Exception:
            has_scores = False
        # Past-date retention: keep placeholder daily rows even if no scores/preds
        is_past = False
        try:
            if date_str:
                is_past = dt.date.fromisoformat(date_str) < _today_local()
        except Exception:
            is_past = False
        if force_use_daily or has_scores or has_preds or is_past:
            df = daily_df.copy()
            meta["daily_used"] = True
            meta["results_pending"] = (not has_scores)
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    # Build upcoming slate if daily unused / empty
    if df.empty:
        if not preds.empty:
            df = preds.copy()
            if not games.empty and "game_id" in preds.columns and "game_id" in games.columns:
                try:
                    preds["game_id"] = preds["game_id"].astype(str)
                except Exception:
                    pass
                try:
                    games["game_id"] = games["game_id"].astype(str)
                except Exception:
                    pass
                keep = [c for c in ["game_id","home_team","away_team","home_score","away_score","start_time","status","date"] if c in games.columns]
                df = df.merge(games[keep], on="game_id", how="left", suffixes=("","_g"))
        elif not games.empty:
            base_cols = [c for c in ["game_id","home_team","away_team","home_score","away_score","start_time","status","date"] if c in games.columns]
            df = games[base_cols].copy()
        meta["results_pending"] = True

    # Odds-only fallback: if still empty but odds contain rows for the requested date, build a minimal slate from odds
    try:
        if df.empty and not odds.empty and date_str:
            o = odds.copy()
            # Filter odds to totals, full-game, and target date if possible
            if "market" in o.columns:
                o = o[o["market"].astype(str).str.lower() == "totals"]
            if "period" in o.columns:
                vals = o["period"].astype(str).str.lower()
                o = o[vals.isin(["full_game","fg","full game","game","match"]) | vals.isna()]
            # Date filter via commence_time/date_line if available
            if "commence_time" in o.columns:
                try:
                    o["_commence_date"] = pd.to_datetime(o["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
                    o = o[o["_commence_date"] == str(date_str)]
                except Exception:
                    pass
            elif "date_line" in o.columns:
                o = o[o["date_line"].astype(str) == str(date_str)]
            if not o.empty:
                # Prefer grouping by game_id when present, else by normalized team pair
                grp_key = "game_id" if "game_id" in o.columns else None
                rows: list[dict[str, Any]] = []
                if grp_key:
                    o["game_id"] = o["game_id"].astype(str)
                    for gid, g in o.groupby("game_id"):
                        r = {
                            "game_id": str(gid),
                            "home_team": g.get("home_team").dropna().astype(str).iloc[0] if "home_team" in g.columns and g["home_team"].notna().any() else None,
                            "away_team": g.get("away_team").dropna().astype(str).iloc[0] if "away_team" in g.columns and g["away_team"].notna().any() else None,
                            "start_time": None,
                            "pred_total": None,
                            "pred_margin": None,
                            "market_total": pd.to_numeric(g.get("total"), errors="coerce").median() if "total" in g.columns else None,
                            "date": str(date_str),
                        }
                        # earliest commence_time if available
                        if "commence_time" in g.columns:
                            try:
                                t = pd.to_datetime(g["commence_time"], errors="coerce").min()
                                r["start_time"] = t.strftime("%Y-%m-%d %H:%M") if pd.notna(t) else None
                            except Exception:
                                pass
                        rows.append(r)
                else:
                    # group by normalized team pairs
                    o["_home_norm"] = o.get("home_team", pd.Series(dtype=str)).astype(str).map(normalize_name) if "home_team" in o.columns else None
                    o["_away_norm"] = o.get("away_team", pd.Series(dtype=str)).astype(str).map(normalize_name) if "away_team" in o.columns else None
                    def _pair(row):
                        h = row.get("_home_norm")
                        a = row.get("_away_norm")
                        return "::".join(sorted([str(h), str(a)])) if h and a else None
                    try:
                        o["_pair_key"] = o.apply(_pair, axis=1)
                    except Exception:
                        o["_pair_key"] = None
                    o2 = o.dropna(subset=["_pair_key"]) if "_pair_key" in o.columns else pd.DataFrame()
                    if not o2.empty:
                        for pk, g in o2.groupby("_pair_key"):
                            # reconstruct teams from first row; create surrogate game_id
                            hn = g.get("_home_norm").dropna().astype(str).iloc[0] if "_home_norm" in g.columns and g["_home_norm"].notna().any() else None
                            an = g.get("_away_norm").dropna().astype(str).iloc[0] if "_away_norm" in g.columns and g["_away_norm"].notna().any() else None
                            ht = g.get("home_team").dropna().astype(str).iloc[0] if "home_team" in g.columns and g["home_team"].notna().any() else hn
                            at = g.get("away_team").dropna().astype(str).iloc[0] if "away_team" in g.columns and g["away_team"].notna().any() else an
                            gid = f"{date_str}:{an}:{hn}" if an and hn else f"{date_str}:{pk}"
                            r = {
                                "game_id": gid,
                                "home_team": ht,
                                "away_team": at,
                                "start_time": None,
                                "pred_total": None,
                                "pred_margin": None,
                                "market_total": pd.to_numeric(g.get("total"), errors="coerce").median() if "total" in g.columns else None,
                                "date": str(date_str),
                            }
                            if "commence_time" in g.columns:
                                try:
                                    t = pd.to_datetime(g["commence_time"], errors="coerce").min()
                                    r["start_time"] = t.strftime("%Y-%m-%d %H:%M") if pd.notna(t) else None
                                except Exception:
                                    pass
                            rows.append(r)
                if rows:
                    df = pd.DataFrame(rows)
                    results_note = f"Odds-only slate for {date_str} (no games/predictions available)"
                    meta["results_pending"] = True
    except Exception:
        pass
    # Compute actual_total if scores present
    if {"home_score","away_score"}.issubset(df.columns):
        try:
            hs = pd.to_numeric(df["home_score"], errors="coerce")
            as_ = pd.to_numeric(df["away_score"], errors="coerce")
            df["actual_total"] = hs + as_
            df["actual_margin"] = hs - as_
        except Exception:
            pass
    # Attach market_total median from odds when missing
    if not odds.empty and "game_id" in odds.columns:
        try:
            o = odds.copy()
            o["game_id"] = o["game_id"].astype(str)
            if "market" in o.columns:
                o = o[o["market"].astype(str).str.lower() == "totals"]
            if "period" in o.columns:
                vals = o["period"].astype(str).str.lower()
                o = o[vals.isin(["full_game","fg","full game","game","match"]) | vals.isna()]
            if "total" in o.columns:
                med = o.groupby("game_id")["total"].median().rename("market_total")
                df = df.merge(med, on="game_id", how="left") if "game_id" in df.columns else df
        except Exception:
            pass
    # Edge vs market_total if predictions available
    if {"pred_total","market_total"}.issubset(df.columns):
        try:
            df["edge_total"] = pd.to_numeric(df["pred_total"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
        except Exception:
            df["edge_total"] = None
    # Basic meta stats
    meta["n_rows"] = len(df)
    if "actual_total" in df.columns:
        try:
            at = pd.to_numeric(df["actual_total"], errors="coerce")
            meta["n_finals"] = int((at > 0).sum())
            meta["n_pending"] = int(len(df) - (at > 0).sum())
            meta["all_final"] = bool(len(df) > 0 and meta["n_finals"] == len(df))
        except Exception:
            meta["n_finals"] = 0
            meta["n_pending"] = len(df)
            meta["all_final"] = False
    else:
        meta["n_finals"] = 0
        meta["n_pending"] = len(df)
        meta["all_final"] = False
    meta["columns"] = list(df.columns)
    return df, meta


@app.route("/")
def index():
    # Optional date filter (?date=YYYY-MM-DD)
    date_q = (request.args.get("date") or "").strip()
    force_use_daily = (request.args.get("use_daily") or "").strip() in ("1","true","yes")

    # Define today string once for consistent downstream comparisons
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None

    games = _load_games_current()
    preds = _load_predictions_current()
    odds = _load_odds_joined(date_q)
    logger.info("Index request start date=%s games_cols=%s preds_cols=%s odds_cols=%s", date_q, list(games.columns), list(preds.columns), list(odds.columns))
    # Initialize coverage summary; will refine after merges to reflect coalesced closing lines
    coverage_summary = {"full": 0, "partial": 0, "none": 0}
    edges = _load_edges()

    if not preds.empty:
        preds["game_id"] = preds["game_id"].astype(str)
    if not games.empty and "game_id" in games.columns:
        games["game_id"] = games["game_id"].astype(str)

    # Normalize date strings
    for df_ in (preds, games):
        if not df_.empty and "date" in df_.columns:
            try:
                df_["date"] = pd.to_datetime(df_["date"]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    # If no date provided, prefer today if there are any games for today or any in-progress/live games
    if not date_q:
        today_str = _today_local().strftime("%Y-%m-%d")
        has_today = False
        has_live = False
        try:
            if "date" in games.columns:
                has_today = (games["date"].astype(str) == today_str).any()
            if "status" in games.columns:
                st = games["status"].astype(str).str.lower()
                has_live = st.str.contains("in").any() | st.str.contains("live").any()
        except Exception:
            pass
        # Consider odds presence as a signal that today has a slate even if games/preds are missing
        has_today_odds = False
        try:
            if not odds.empty:
                # commence_time preferred
                if "commence_time" in odds.columns:
                    ct = pd.to_datetime(odds["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
                    has_today_odds = (ct == today_str).any()
                # fallback to date_line or _day_diff markers
                if not has_today_odds and "date_line" in odds.columns:
                    has_today_odds = (odds["date_line"].astype(str) == today_str).any()
                if not has_today_odds and "_day_diff" in odds.columns:
                    try:
                        has_today_odds = (pd.to_numeric(odds["_day_diff"], errors="coerce") == 0).any()
                    except Exception:
                        pass
        except Exception:
            has_today_odds = False
        if has_today or has_live or has_today_odds:
            date_q = today_str
        else:
            # Fallback to latest date seen in predictions or games
            if "date" in preds.columns and not preds.empty and preds["date"].notna().any():
                try:
                    date_q = pd.to_datetime(preds["date"]).max().strftime("%Y-%m-%d")
                except Exception:
                    date_q = (preds["date"].dropna().astype(str).max())
            elif "date" in games.columns and not games.empty and games["date"].notna().any():
                try:
                    date_q = pd.to_datetime(games["date"]).max().strftime("%Y-%m-%d")
                except Exception:
                    date_q = (games["date"].dropna().astype(str).max())
            else:
                # As a last resort (e.g., on Render with no games/preds), use the latest daily_results date if present
                try:
                    dr_dir = OUT / "daily_results"
                    if dr_dir.exists():
                        dates: list[str] = []
                        for p in dr_dir.glob("results_*.csv"):
                            stem = p.stem
                            if stem.startswith("results_"):
                                dates.append(stem.replace("results_", ""))
                        if dates:
                            # Pick max date string safely
                            try:
                                date_q = pd.to_datetime(pd.Series(dates)).max().strftime("%Y-%m-%d")
                            except Exception:
                                date_q = sorted(dates)[-1]
                except Exception:
                    pass

    # Apply date filter (prefer to filter predictions if they carry date; else games)
    # Keep originals for fallback when showing live/today slates
    games_all = games.copy()
    preds_all = preds.copy()
    if date_q:
        if "date" in preds.columns:
            preds = preds[preds["date"] == date_q]
        if "date" in games.columns:
            filtered = games[games["date"] == date_q]
            # If targeting today but no rows due to timezone skew, fallback to any live rows regardless of date
            if filtered.empty:
                try:
                    today_str = _today_local().strftime("%Y-%m-%d")
                    st = games_all.get("status")
                    if st is not None:
                        stl = st.astype(str).str.lower()
                        live_mask = stl.str.contains("in") | stl.str.contains("live")
                        live_df = games_all[live_mask]
                        if not live_df.empty and date_q == today_str:
                            games = live_df
                        else:
                            games = filtered
                    else:
                        games = filtered
                except Exception:
                    games = filtered
            else:
                games = filtered
        # Direct date-specific fallback: if games empty for requested date, attempt to load games_<date>.csv
        if games.empty:
            try:
                date_specific = OUT / f"games_{date_q}.csv"
                if date_specific.exists():
                    gdf = pd.read_csv(date_specific)
                    if not gdf.empty:
                        if "date" in gdf.columns:
                            try:
                                gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                            except Exception:
                                pass
                            gdf = gdf[gdf["date"].astype(str) == str(date_q)]
                        if "game_id" in gdf.columns:
                            gdf["game_id"] = gdf["game_id"].astype(str)
                        games = gdf
            except Exception:
                pass

    # If daily_results exist for the date, prefer it only if games have real scores (>0) or explicit predictions;
    # for future/upcoming slates daily_results may be a placeholder with 0 scores and missing preds.
    daily_df = _load_daily_results_for(date_q) if date_q else pd.DataFrame()
    daily_used = False
    results_note = None
    if not daily_df.empty:
        try:
            has_scores = False
            if {"home_score","away_score"}.issubset(daily_df.columns):
                sc_sum = pd.to_numeric(daily_df["home_score"], errors="coerce") + pd.to_numeric(daily_df["away_score"], errors="coerce")
                has_scores = bool((sc_sum > 0).any())
            has_preds = False
            for c in ("pred_total","pred_margin","market_total","closing_total"):
                if c in daily_df.columns and daily_df[c].notna().any():
                    has_preds = True
                    break
            date_obj: dt.date | None = None
            try:
                date_obj = dt.date.fromisoformat(date_q)
            except Exception:
                date_obj = None
            if not has_scores and not has_preds and not force_use_daily:
                # Relax discard: if the date is in the past retain placeholder rows so user can see slate even without preds/scores.
                if date_obj and date_obj < _today_local():
                    daily_used = True
                    results_note = f"No scores/preds captured for {date_q}; retaining placeholder daily slate."  # past date placeholder
                else:
                    daily_df = pd.DataFrame()
            else:
                daily_used = True
                if not has_scores:
                    results_note = f"Results pending for {date_q}: scores not yet ingested; showing schedule/predictions."
                elif has_scores and force_use_daily:
                    results_note = f"Showing daily results for {date_q}."
        except Exception:
            daily_df = pd.DataFrame()

    # Merge predictions with game metadata
    if not daily_df.empty:
        # Normalize shape to what the template expects
        df = daily_df.copy()
        # Backfill missing/zero scores from master games file for past dates
        try:
            date_obj = dt.date.fromisoformat(date_q) if date_q else None
        except Exception:
            date_obj = None
        if date_obj and date_obj < _today_local():
            try:
                master_path = OUT / "games_all.csv"
                if master_path.exists() and "game_id" in df.columns:
                    master = pd.read_csv(master_path, usecols=[c for c in [
                        "game_id","date","home_score","away_score","status",
                        "home_score_1h","away_score_1h","home_score_2h","away_score_2h"
                    ] if c])
                    master["game_id"] = master["game_id"].astype(str)
                    if "date" in master.columns:
                        master["date"] = pd.to_datetime(master["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                        master = master[master["date"] == str(date_q)]
                    df = df.merge(master.drop(columns=[c for c in ["date"] if c in master.columns]), on="game_id", how="left", suffixes=("", "_m"))
                    # Coalesce scores: prefer existing (>0) else master (>0)
                    for side in ("home","away"):
                        prov = pd.to_numeric(df.get(f"{side}_score"), errors="coerce")
                        mast = pd.to_numeric(df.get(f"{side}_score_m"), errors="coerce")
                        df[f"{side}_score"] = np.where((prov>0) | prov.notna(), df.get(f"{side}_score"), mast)
                    # Coalesce halftime and 2H scores when present in master
                    for half_col in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
                        if half_col in df.columns or f"{half_col}_m" in df.columns:
                            cur = pd.to_numeric(df.get(half_col), errors="coerce") if half_col in df.columns else pd.Series(np.nan, index=df.index)
                            mval = pd.to_numeric(df.get(f"{half_col}_m"), errors="coerce") if f"{half_col}_m" in df.columns else pd.Series(np.nan, index=df.index)
                            df[half_col] = np.where(cur.notna(), cur, mval)
                    if "status_m" in df.columns:
                        df["status"] = df.get("status").where(df.get("status").notna(), df.get("status_m"))
            except Exception:
                pass
        # If fused historical file present, attempt to enrich missing start_time or scores from it
        try:
            fused_path = OUT / "games_hist_fused.csv"
            if fused_path.exists() and "game_id" in df.columns:
                fused_df = pd.read_csv(fused_path, usecols=["game_id","start_time"])
                fused_df["game_id"] = fused_df["game_id"].astype(str)
                df = df.merge(fused_df, on="game_id", how="left", suffixes=("", "_fused"))
                if "start_time" not in df.columns or df["start_time"].isna().all():
                    df["start_time"] = df.get("start_time_fused")
                df = df.drop(columns=[c for c in ["start_time_fused"] if c in df.columns])
        except Exception:
            pass
        # Enrich with games metadata for commence_time and venue/city/state if available
        try:
            if not games.empty and "game_id" in games.columns and "game_id" in df.columns:
                g2 = games.copy()
                g2["game_id"] = g2["game_id"].astype(str)
                keep = [c for c in [
                    "game_id", "start_time", "commence_time",
                    "venue", "venue_full", "arena", "stadium", "location", "site", "site_name", "city", "state"
                ] if c in g2.columns]
                if keep:
                    df = df.merge(g2[keep], on="game_id", how="left", suffixes=("", "_g"))
        except Exception:
            pass
        # Ensure actual_total is present/coalesced for all rows when scores are available
        try:
            if {"home_score","away_score"}.issubset(df.columns):
                hs = pd.to_numeric(df["home_score"], errors="coerce")
                aw = pd.to_numeric(df["away_score"], errors="coerce")
                if "actual_total" not in df.columns:
                    df["actual_total"] = np.where(hs.notna() & aw.notna(), hs + aw, np.nan)
                else:
                    mask = df["actual_total"].isna()
                    df.loc[mask, "actual_total"] = hs[mask] + aw[mask]
        except Exception:
            pass
        # Pick best available start_time among candidates (prefer one that contains a time component)
        try:
            cands = ["start_time", "start_time_g", "commence_time", "commence_time_g"]
            have = [c for c in cands if c in df.columns]
            if have:
                def _pick_start(row):
                    vals = []
                    for c in have:
                        v = row.get(c)
                        if pd.notna(v) and str(v).strip() != "":
                            vals.append(str(v).strip())
                    if not vals:
                        return None
                    # Prefer any with a time component
                    for v in vals:
                        if ":" in v:
                            return v
                    return vals[0]
                df["start_time"] = df.apply(_pick_start, axis=1)
        except Exception:
            pass
        if "game_id" in df.columns:
            try:
                df["game_id"] = df["game_id"].astype(str)
            except Exception:
                pass
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        # Market total: prefer closing_total when present; coalesce NaNs as well
        if "closing_total" in df.columns:
            if "market_total" not in df.columns:
                df["market_total"] = df["closing_total"]
            else:
                try:
                    df["market_total"] = df["market_total"].where(df["market_total"].notna(), df["closing_total"])
                except Exception:
                    pass
    else:
        # Build from preds + games (older behavior) to show upcoming slates
        if preds.empty and not games.empty:
            # No predictions for this date: show games-only slate
            base_cols = [c for c in ["game_id","date","home_team","away_team","home_score","away_score","start_time"] if c in games.columns]
            df = games[base_cols].copy()
            # Ensure required columns exist
            for c in ["pred_total","pred_margin"]:
                if c not in df.columns:
                    df[c] = None
        else:
            df = preds.copy()
            if not games.empty:
                # Try to pick best team columns available
                home_cols = [c for c in ["home_team", "home_team_name", "home"] if c in games.columns]
                away_cols = [c for c in ["away_team", "away_team_name", "away"] if c in games.columns]
                keep = [c for c in [
                    "game_id", "date", "status", "home_score", "away_score", "home_score_1h", "away_score_1h",
                    "start_time", "commence_time", "neutral_site",
                    # venue/location candidates
                    "venue", "venue_full", "arena", "stadium", "location", "site", "site_name", "city", "state"
                ] if c in games.columns]
                keep += (home_cols[:1] or []) + (away_cols[:1] or [])
                # Use suffixes so left (preds) keeps original names, right (games) gets _g on collisions
                df = df.merge(games[keep], on="game_id", how="left", suffixes=("", "_g"))

            def _pick(df_, candidates: list[str]) -> str | None:
                for name in candidates:
                    if name in df_.columns:
                        return name
                return None

            # Standardize team fields: create/patch home_team and away_team
            if "home_team" not in df.columns:
                src = _pick(df, [
                    "home_team_g", "home_team", "home_team_name_g", "home_team_name", "home_g", "home"
                ])
                if src is not None:
                    df["home_team"] = df[src]
            if "away_team" not in df.columns:
                src = _pick(df, [
                    "away_team_g", "away_team", "away_team_name_g", "away_team_name", "away_g", "away"
                ])
                if src is not None:
                    df["away_team"] = df[src]
        # Ensure actual_total is present for upcoming/preds builds
        try:
            if {"home_score","away_score"}.issubset(df.columns):
                hs = pd.to_numeric(df["home_score"], errors="coerce")
                aw = pd.to_numeric(df["away_score"], errors="coerce")
                if "actual_total" not in df.columns:
                    df["actual_total"] = np.where(hs.notna() & aw.notna(), hs + aw, np.nan)
                else:
                    mask = df["actual_total"].isna()
                    df.loc[mask, "actual_total"] = hs[mask] + aw[mask]
        except Exception:
            pass

    # Odds-only fallback (index route): if df still empty but odds present for selected date, synthesize minimal slate
    try:
        if df.empty and not odds.empty and date_q:
            o = odds.copy()
            # Restrict to totals + full game period markers
            if "market" in o.columns:
                o = o[o["market"].astype(str).str.lower() == "totals"]
            if "period" in o.columns:
                vals = o["period"].astype(str).str.lower()
                o = o[vals.isin(["full_game","fg","full game","game","match"]) | vals.isna()]
            # Date filter via commence_time or date_line
            if "commence_time" in o.columns:
                try:
                    o["_commence_date"] = pd.to_datetime(o["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
                    o = o[o["_commence_date"] == str(date_q)]
                except Exception:
                    pass
            elif "date_line" in o.columns:
                o = o[o["date_line"].astype(str) == str(date_q)]
            if not o.empty:
                rows: list[dict[str, Any]] = []
                o["game_id"] = o.get("game_id", pd.Series(range(len(o)))).astype(str)
                for gid, g in o.groupby("game_id"):
                    r = {
                        "game_id": str(gid),
                        "home_team": g.get("home_team").dropna().astype(str).iloc[0] if "home_team" in g.columns and g["home_team"].notna().any() else g.get("home_team_name").dropna().astype(str).iloc[0] if "home_team_name" in g.columns and g["home_team_name"].notna().any() else None,
                        "away_team": g.get("away_team").dropna().astype(str).iloc[0] if "away_team" in g.columns and g["away_team"].notna().any() else g.get("away_team_name").dropna().astype(str).iloc[0] if "away_team_name" in g.columns and g["away_team_name"].notna().any() else None,
                        "start_time": None,
                        "pred_total": None,
                        "pred_margin": None,
                        "market_total": pd.to_numeric(g.get("total"), errors="coerce").median() if "total" in g.columns else None,
                        "date": str(date_q),
                        "home_score": None,
                        "away_score": None,
                    }
                    if "commence_time" in g.columns:
                        try:
                            t = pd.to_datetime(g["commence_time"], errors="coerce").min()
                            r["start_time"] = t.strftime("%Y-%m-%d %H:%M") if pd.notna(t) else None
                        except Exception:
                            pass
                    rows.append(r)
                if rows:
                    df = pd.DataFrame(rows)
                    results_note = f"Odds-only slate for {date_q} (no games/predictions available)"
                    coverage_summary = {"full": 0, "partial": 0, "none": 0}
    except Exception:
        pass

    # Ensure game_id present: if missing attempt deterministic construction (only when teams available)
    if "game_id" not in df.columns:
        home_col = next((c for c in ["home_team","home"] if c in df.columns), None)
        away_col = next((c for c in ["away_team","away"] if c in df.columns), None)
        if home_col and away_col:
            try:
                # Build surrogate id: date + normalized team names
                date_part = (df.get("date") or pd.Series([date_q]*len(df))).astype(str)
                home_norm = df[home_col].astype(str).map(normalize_name)
                away_norm = df[away_col].astype(str).map(normalize_name)
                df["game_id"] = [f"{d}:{a}:{h}" for d,a,h in zip(date_part, away_norm, home_norm)]
                logger.warning("Constructed surrogate game_id for %d rows (date/team based)", len(df))
            except Exception:
                logger.exception("Failed to construct surrogate game_id")
    if "game_id" in df.columns:
        try:
            df["game_id"] = df["game_id"].astype(str)
        except Exception:
            logger.warning("Could not cast game_id to string")

    # Coalesce start_time from games merge (start_time_g) if primary start_time is missing/blank
    try:
        if "start_time_g" in df.columns:
            if "start_time" not in df.columns:
                df["start_time"] = df["start_time_g"]
            else:
                st_primary = df["start_time"].astype(str).str.strip()
                # Fill global if entirely missing/blank
                if (df["start_time"].isna().all()) or st_primary.eq("").all():
                    df["start_time"] = df["start_time_g"]
                else:
                    mask_missing = df["start_time"].isna() | st_primary.eq("")
                    if mask_missing.any():
                        df.loc[mask_missing, "start_time"] = df.loc[mask_missing, "start_time_g"]
    except Exception:
        pass

    # Attach odds (line totals) when available, unless daily_df already provided market_total
    # Distinguish strict last odds vs heuristic closing: if source file name contains 'last', label later.
    source_last = False
    try:
        # crude detection: odds DataFrame path not retained; infer from columns typical to last_odds vs closing_lines
        # last_odds has no 'prio' or synthetic markers; rely on absence of 'prio' + presence of many individual book rows
        source_last = True if "event_id" in odds.columns and "total" in odds.columns and "closing_total" not in odds.columns else False
    except Exception:
        source_last = False
    # Skip odds enrichment entirely if base df is empty (nothing to merge onto)
    if not odds.empty and not df.empty:
        oddf = odds.copy()
        if "market" in oddf.columns:
            oddf = oddf[oddf["market"].astype(str).str.lower() == "totals"]
        if "period" in oddf.columns:
            # Accept a broader set of full-game labels
            vals = oddf["period"].astype(str).str.lower()
            oddf = oddf[vals.isin(["full_game", "full game", "fg", "game", "match"]) | vals.isna()]
        if "game_id" in oddf.columns:
            oddf["game_id"] = oddf["game_id"].astype(str)
            # Direct merge of already-aggregated columns if present on odds frame
            try:
                direct_cols = {}
                if "market_total" in odds.columns:
                    direct_cols["market_total"] = "market_total"
                if "closing_total" in odds.columns:
                    direct_cols["closing_total"] = "closing_total"
                # Spread and ML flexible source names on odds dataframe
                if "spread_home" in odds.columns:
                    direct_cols["spread_home"] = "spread_home"
                elif "home_spread" in odds.columns:
                    direct_cols["home_spread"] = "spread_home"
                if "ml_home" in odds.columns:
                    direct_cols["ml_home"] = "ml_home"
                elif "moneyline_home" in odds.columns:
                    direct_cols["moneyline_home"] = "ml_home"
                if direct_cols:
                    sub = odds.copy()
                    sub["game_id"] = sub["game_id"].astype(str)
                    keep = ["game_id"] + list(direct_cols.keys())
                    sub = sub[[c for c in keep if c in sub.columns]]
                    if len(sub.columns) > 1:
                        # Aggregate by median in case of multiple rows per game
                        agg = sub.groupby("game_id").median(numeric_only=True).reset_index()
                        # Rename to canonical output names
                        agg = agg.rename(columns=direct_cols)
                        for c in ["market_total","closing_total","spread_home","ml_home"]:
                            if c in agg.columns:
                                if c in df.columns:
                                    try:
                                        df[c] = df[c].where(df[c].notna(), df["game_id"].map(agg.set_index("game_id")[c]))
                                    except Exception:
                                        pass
                                else:
                                    df = df.merge(agg[["game_id", c]], on="game_id", how="left")
            except Exception:
                pass
            if "total" in oddf.columns:
                # Ensure groupby result is a DataFrame with an explicit game_id column for safe merge
                line_by_game = (
                    oddf.groupby("game_id", as_index=False)["total"]
                        .median()
                        .rename(columns={"total": "_market_total_from_odds"})
                )
                if not line_by_game.empty:
                    if "game_id" in df.columns:
                        try:
                            df = df.merge(line_by_game, on="game_id", how="left")
                        except Exception:
                            logger.exception("Odds total merge failed; proceeding without _market_total_from_odds")
                    else:
                        # Attempt to synthesize game_id on the fly if teams present
                        home_col = next((c for c in ["home_team","home"] if c in df.columns), None)
                        away_col = next((c for c in ["away_team","away"] if c in df.columns), None)
                        date_col = "date" if "date" in df.columns else None
                        if home_col and away_col:
                            try:
                                date_part = (df.get(date_col) or pd.Series([None]*len(df))).astype(str)
                                home_norm = df[home_col].astype(str).map(normalize_name)
                                away_norm = df[away_col].astype(str).map(normalize_name)
                                df["game_id"] = [f"{d}:{a}:{h}" for d,a,h in zip(date_part, away_norm, home_norm)]
                                logger.warning("Synthesized game_id for odds total merge (%d rows)", len(df))
                                df = df.merge(line_by_game, on="game_id", how="left")
                            except Exception:
                                logger.exception("Failed synthesizing game_id for odds total merge; skipping")
                        else:
                            logger.warning("Skipping odds total merge: df missing game_id and team columns")
                    # Coalesce per-row to fill missing market_total only if helper column now exists
                if "_market_total_from_odds" in df.columns:
                    if "market_total" in df.columns:
                        try:
                            df["market_total"] = df["market_total"].where(df["market_total"].notna(), df["_market_total_from_odds"])
                        except Exception:
                            pass
                    else:
                        # Create market_total only when we actually have odds-derived values
                        try:
                            df["market_total"] = df["_market_total_from_odds"]
                        except Exception:
                            logger.warning("Could not assign market_total from _market_total_from_odds despite column presence")
            # Build per-game odds list and start time from commence_time when present
            odds_map: dict[str, list[dict[str, Any]]] = {}
            start_map: dict[str, str] = {}
            # Normalize commence_time to ISO display string if present
            if "commence_time" in oddf.columns:
                try:
                    oddf["_commence"] = pd.to_datetime(oddf["commence_time"], errors="coerce")
                except Exception:
                    oddf["_commence"] = pd.NaT
            else:
                oddf["_commence"] = pd.NaT
            # Select a small subset of columns for the odds list
            for gid, g in oddf.groupby("game_id"):
                # Sort by book name for determinism
                g2 = g.sort_values(["book"]) if "book" in g.columns else g
                items: list[dict[str, Any]] = []
                for _, r in g2.iterrows():
                    item: dict[str, Any] = {
                        "book": r.get("book"),
                        "total": r.get("total"),
                    }
                    # Optional prices/lines
                    for k in ("price_over", "price_under", "over", "under", "line_over", "line_under"):
                        if k in g2.columns:
                            item[k] = r.get(k)
                    items.append(item)
                    if len(items) >= 6:
                        break
                odds_map[str(gid)] = items
                # Earliest commence time for display
                if "_commence" in g2.columns and g2["_commence"].notna().any():
                    try:
                        t = g2["_commence"].min()
                        start_map[str(gid)] = t.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass
            if odds_map and "game_id" in df.columns:
                df["_odds_list"] = df["game_id"].map(lambda x: odds_map.get(str(x), []))
            if start_map and "game_id" in df.columns:
                # Only fill missing start_time from odds commence_time; don't overwrite existing game times
                mapped = df["game_id"].map(lambda x: start_map.get(str(x)))
                if "start_time" in df.columns:
                    try:
                        df["start_time"] = df["start_time"].where(df["start_time"].notna(), mapped)
                    except Exception:
                        df["start_time"] = mapped
                else:
                    df["start_time"] = mapped

    # Fallback odds merge: if market_total still missing for many rows, derive from raw odds feed by team name matching.
    try:
        if "market_total" in df.columns:
            missing_mt_mask = df["market_total"].isna() | (df["market_total"].astype(str).str.lower().isin(["nan","none","null","nat"]))
        else:
            missing_mt_mask = pd.Series([True]*len(df))
        # Threshold: only run fallback if more than 50% missing or all missing
        if len(df) and missing_mt_mask.sum() > 0 and missing_mt_mask.sum() >= int(0.5*len(df)):
            # Determine raw odds file path per date (supports today and historical) with pattern fallback for prefetch variants.
            primary_raw = OUT / ("odds_today.csv" if (not date_q or (today_str and date_q == today_str)) else f"odds_{date_q}.csv")
            # Also support historical snapshots under outputs/odds_history/odds_<date>.csv
            history_raw = OUT / "odds_history" / (f"odds_{date_q}.csv" if date_q else "")
            raw = pd.DataFrame()
            if primary_raw.exists():
                try:
                    raw = pd.read_csv(primary_raw)
                except Exception:
                    raw = pd.DataFrame()
            # Fallback to odds_history for past dates
            if raw.empty and date_q and history_raw.exists():
                try:
                    raw = pd.read_csv(history_raw)
                except Exception:
                    raw = pd.DataFrame()
            if raw.empty and date_q:
                try:
                    # e.g. odds_YYYY-MM-DD_prefetch.csv or other suffixes
                    for p in sorted(OUT.glob(f"odds_{date_q}_*.csv")):
                        tmp = _safe_read_csv(p)
                        if not tmp.empty:
                            raw = tmp
                            break
                except Exception:
                    pass
            if not raw.empty:
                # Constrain by date if commence_time present and date_q provided (avoid cross-day contamination)
                if date_q and "commence_time" in raw.columns:
                    try:
                        raw["_commence_dt"] = pd.to_datetime(raw["commence_time"], errors="coerce")
                        raw_date_mask = raw["_commence_dt"].dt.strftime("%Y-%m-%d") == str(date_q)
                        raw = raw[raw_date_mask | raw["_commence_dt"].isna()]
                    except Exception:
                        pass
                # Resolve team column names flexibly
                home_col = next((c for c in ["home_team_name","home_team","home"] if c in raw.columns), None)
                away_col = next((c for c in ["away_team_name","away_team","away"] if c in raw.columns), None)
                if home_col and away_col:
                    raw["_home_norm"] = raw[home_col].astype(str).map(normalize_name)
                    raw["_away_norm"] = raw[away_col].astype(str).map(normalize_name)
                    agg_rows: dict[str, dict[str, Any]] = {}
                    for _, r in raw.iterrows():
                        hn = r.get("_home_norm"); an = r.get("_away_norm")
                        if not hn or not an:
                            continue
                        pair_key = "::".join(sorted([str(hn), str(an)]))
                        dct = agg_rows.setdefault(pair_key, {"totals": [], "spreads": [], "ml": []})
                        # Flexible column resolution
                        tot = next((r.get(tc) for tc in ["total","over_under","market_total","line_total"] if tc in raw.columns and pd.notna(r.get(tc))), None)
                        spr = next((r.get(sc) for sc in ["spread","home_spread","spread_home","handicap_home"] if sc in raw.columns and pd.notna(r.get(sc))), None)
                        mlh = next((r.get(mc) for mc in ["moneyline_home","ml_home","price_home","h2h_home"] if mc in raw.columns and pd.notna(r.get(mc))), None)
                        if pd.notna(tot):
                            dct["totals"].append(tot)
                        if pd.notna(spr):
                            dct["spreads"].append(spr)
                        if pd.notna(mlh):
                            dct["ml"].append(mlh)
                    # Apply aggregates to df rows
                    for idx, row in df.iterrows():
                        h = normalize_name(str(row.get("home_team") or ""))
                        a = normalize_name(str(row.get("away_team") or ""))
                        if not h or not a:
                            continue
                        pkey = "::".join(sorted([h, a]))
                        ag = agg_rows.get(pkey)
                        if not ag:
                            continue
                        if ("market_total" not in df.columns) or pd.isna(df.loc[idx, "market_total"]) or str(df.loc[idx, "market_total"]).lower() in ("nan","none","null"):
                            if ag["totals"]:
                                df.loc[idx, "market_total"] = float(pd.to_numeric(pd.Series(ag["totals"]), errors="coerce").median())
                        if ("spread_home" not in df.columns or pd.isna(df.loc[idx, "spread_home"]) or str(df.loc[idx, "spread_home"]).lower() in ("nan","none","null")) and ag["spreads"]:
                            df.loc[idx, "spread_home"] = float(pd.to_numeric(pd.Series(ag["spreads"]), errors="coerce").median())
                        if ("ml_home" not in df.columns or pd.isna(df.loc[idx, "ml_home"]) or str(df.loc[idx, "ml_home"]).lower() in ("nan","none","null")) and ag["ml"]:
                            df.loc[idx, "ml_home"] = float(pd.to_numeric(pd.Series(ag["ml"]), errors="coerce").median())
    except Exception:
        pass

    # Additional start_time fallback from games' commence_time (post-merge) and odds_today style columns
    try:
        for cname in ("commence_time", "commence_time_g"):
            if cname in df.columns:
                mapped = pd.to_datetime(df[cname], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                if "start_time" in df.columns:
                    df["start_time"] = df["start_time"].where(df["start_time"].notna(), mapped)
                else:
                    df["start_time"] = mapped
                break
        # odds_today format may have start_time under different columns like game_date+commence_time or missing market column entirely.
        if ("market_total" not in df.columns or df["market_total"].isna().all()) and "total" in df.columns and "_market_total_from_odds" not in df.columns:
            try:
                # If totals aggregated missing, compute median from raw total column.
                if "game_id" in df.columns:
                    df["_market_total_from_odds"] = df.groupby("game_id")["total"].transform(lambda s: pd.to_numeric(s, errors="coerce").median())
                    df["market_total"] = df["market_total"].where(df["market_total"].notna(), df["_market_total_from_odds"]) if "market_total" in df.columns else df["_market_total_from_odds"]
                else:
                    logger.warning("Skipping transform for market_total: df missing game_id column")
            except Exception:
                pass
    except Exception:
        pass

    # Also attach spreads and moneyline medians (full game) for ATS/ML displays and half lines for 1H/2H totals
    try:
        if not odds.empty and "game_id" in odds.columns:
            o2 = odds.copy()
            o2["game_id"] = o2["game_id"].astype(str)
            def _agg_market(market: str, val_col: str, out_col: str) -> pd.Series | None:
                sub = o2[o2["market"].astype(str).str.lower() == market]
                if "period" in sub.columns:
                    sub = sub[sub["period"].astype(str).str.lower().isin(["full_game","fg","full game"])]
                if val_col not in sub.columns or sub.empty:
                    return None
                return sub.groupby("game_id")[val_col].median().rename(out_col)
            s_fg = _agg_market("spreads", "home_spread", "spread_home")
            m_fg = _agg_market("h2h", "moneyline_home", "ml_home")
            for srs in [s_fg, m_fg]:
                if srs is not None and not srs.empty:
                    # Convert Series to DataFrame with explicit key for merge
                    if isinstance(srs, pd.Series):
                        srs = srs.reset_index()  # columns: ["game_id", <named_col>]
                    if "game_id" in df.columns:
                        df = df.merge(srs, on="game_id", how="left")
                    else:
                        logger.warning("Skipping spread/ML merge: df missing game_id column")

            # 1H totals median
            def _agg_market_period(market: str, val_col: str, period_keys: set[str], out_col: str) -> pd.Series | None:
                sub = o2[o2["market"].astype(str).str.lower() == market]
                if "period" in sub.columns:
                    vals = sub["period"].astype(str).str.lower()
                    sub = sub[vals.isin(period_keys)]
                if val_col not in sub.columns or sub.empty:
                    return None
                return sub.groupby("game_id")[val_col].median().rename(out_col)
            t_1h = _agg_market_period("totals", "total", {"1h","1h_1","first_half","1st_half"}, "market_total_1h")
            t_2h = _agg_market_period("totals", "total", {"2h","second_half","2nd_half"}, "market_total_2h")
            s_1h = _agg_market_period("spreads", "home_spread", {"1h","1h_1","first_half","1st_half"}, "spread_home_1h")
            s_2h = _agg_market_period("spreads", "home_spread", {"2h","second_half","2nd_half"}, "spread_home_2h")
            for srs in [t_1h, t_2h, s_1h, s_2h]:
                if srs is not None and not srs.empty:
                    if isinstance(srs, pd.Series):
                        srs = srs.reset_index()
                    if "game_id" in df.columns:
                        df = df.merge(srs, on="game_id", how="left")
                    else:
                        logger.warning("Skipping half-line merge: df missing game_id column")

            # Fallback: derive half spreads from full-game spread if provider 1H/2H spreads missing
            try:
                if "spread_home" in df.columns:
                    sh_full = pd.to_numeric(df["spread_home"], errors="coerce")
                    # Only derive if 1H spread column absent OR all NaN
                    need_1h = ("spread_home_1h" not in df.columns) or (df.get("spread_home_1h").isna().all())
                    if need_1h:
                        df["spread_home_1h"] = np.where(sh_full.notna(), sh_full * 0.5, np.nan)
                        df["spread_home_1h_basis"] = np.where(sh_full.notna(), "derived", None)
                    else:
                        # Mark basis real for existing values
                        df["spread_home_1h_basis"] = np.where(df["spread_home_1h"].notna(), "provider", None)
                    need_2h = ("spread_home_2h" not in df.columns) or (df.get("spread_home_2h").isna().all())
                    if need_2h:
                        # Assume symmetric halves; second half spread equals first half derived
                        # If we derived 1H above, re-use; else derive independently.
                        base_1h = df.get("spread_home_1h") if "spread_home_1h" in df.columns else (sh_full * 0.5)
                        df["spread_home_2h"] = np.where(sh_full.notna(), sh_full - base_1h, np.nan)
                        df["spread_home_2h_basis"] = np.where(sh_full.notna(), "derived", None)
                    else:
                        df["spread_home_2h_basis"] = np.where(df["spread_home_2h"].notna(), "provider", None)
            except Exception:
                pass

        # Coalesce halftime columns into canonical names even if base exists but is empty
        # Prefer non-NaN across [base, base_x, base_y, base_m, base_bs, base_sec, base_ref, base_fused]
        for base in ["home_score_1h","away_score_1h","home_score_2h","away_score_2h"]:
            cands = [c for c in df.columns if c == base or c.startswith(base + "_")]
            if cands:
                vals = None
                for c in [base] + [c for c in cands if c != base]:
                    if c not in df.columns:
                        continue
                    try:
                        s = pd.to_numeric(df[c], errors="coerce")
                    except Exception:
                        s = pd.Series(np.nan, index=df.index)
                    vals = s if vals is None else vals.where(vals.notna(), s)
                if vals is not None:
                    df[base] = vals
    except Exception:
        pass

    # Hard enrichment: inject start_time and venue from games_curr for the selected date as authoritative source
    try:
        gm = _safe_read_csv(OUT / "games_curr.csv")
        if (gm.empty or (date_q and "date" in gm.columns and not (gm["date"].astype(str) == str(date_q)).any())) and date_q:
            alt = OUT / f"games_{date_q}.csv"
            if alt.exists():
                g2 = _safe_read_csv(alt)
                if not g2.empty:
                    gm = g2
        if not gm.empty and "game_id" in gm.columns:
            gm["game_id"] = gm["game_id"].astype(str)
            if date_q and "date" in gm.columns:
                try:
                    gm["date"] = pd.to_datetime(gm["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    gm = gm[gm["date"] == str(date_q)]
                except Exception:
                    pass
            # Build maps
            m_start = gm.set_index("game_id")["start_time"] if "start_time" in gm.columns else None
            m_venue = gm.set_index("game_id")["venue"] if "venue" in gm.columns else None
            m_neutral = gm.set_index("game_id")["neutral_site"] if "neutral_site" in gm.columns else None
            m_city = gm.set_index("game_id")["city"] if "city" in gm.columns else None
            m_state = gm.set_index("game_id")["state"] if "state" in gm.columns else None
            if "game_id" in df.columns:
                # Ensure string type
                df["game_id"] = df["game_id"].astype(str)
                if m_start is not None:
                    if "start_time" in df.columns:
                        st_str = df["start_time"].astype(str)
                        mask_missing = df["start_time"].isna() | st_str.str.strip().eq("")
                        if mask_missing.any():
                            df.loc[mask_missing, "start_time"] = df.loc[mask_missing, "game_id"].map(m_start)
                    else:
                        df["start_time"] = df["game_id"].map(m_start)
                if m_venue is not None:
                    if "venue" in df.columns:
                        v_str = df["venue"].astype(str)
                        mask_mv = df["venue"].isna() | v_str.str.strip().eq("") | v_str.str.lower().isin(["nan","none","null","nat"]) 
                        if mask_mv.any():
                            df.loc[mask_mv, "venue"] = df.loc[mask_mv, "game_id"].map(m_venue)
                    else:
                        df["venue"] = df["game_id"].map(m_venue)
                if m_neutral is not None and "neutral_site" not in df.columns:
                    df["neutral_site"] = df["game_id"].map(m_neutral)
                # If venue still missing, try City, State
                if "venue" in df.columns and df["venue"].isna().any() and (m_city is not None or m_state is not None):
                    def _city_state(gid):
                        ci = m_city.get(gid) if m_city is not None else None
                        st = m_state.get(gid) if m_state is not None else None
                        if pd.notna(ci) and pd.notna(st):
                            return f"{ci}, {st}"
                        return ci if pd.notna(ci) else (st if pd.notna(st) else None)
                    mask_missing_v = df["venue"].isna()
                    if mask_missing_v.any():
                        df.loc[mask_missing_v, "venue"] = df.loc[mask_missing_v, "game_id"].map(_city_state)
    except Exception:
        pass

    # Fallback: attach closing lines from games_with_closing.csv if closing_total missing (heuristic)
    if ("game_id" in df.columns):
        try:
            closing_path = OUT / "games_with_closing.csv"
            if closing_path.exists():
                cl = pd.read_csv(closing_path)
                if not cl.empty and "game_id" in cl.columns:
                    cl["game_id"] = cl["game_id"].astype(str)
                    # closing_total may not be precomputed; derive from totals market median per game
                    if "closing_total" not in cl.columns and {"market","total"}.issubset(cl.columns):
                        try:
                            totals_fg = cl[(cl["market"].astype(str).str.lower()=="totals") & (cl["period"].astype(str).str.lower().isin(["full_game","full game","fg"]))]
                            if not totals_fg.empty:
                                med_tot = totals_fg.groupby("game_id")["total"].median().rename("closing_total")
                                cl = cl.merge(med_tot, on="game_id", how="left")
                        except Exception:
                            pass
                    # closing spread median from spreads market
                    closing_spread_series = None
                    try:
                        if {"market","home_spread"}.issubset(cl.columns):
                            spreads_fg = cl[(cl["market"].astype(str).str.lower()=="spreads") & (cl["period"].astype(str).str.lower().isin(["full_game","full game","fg"]))]
                            if not spreads_fg.empty:
                                closing_spread_series = spreads_fg.groupby("game_id")["home_spread"].median().rename("closing_spread_home")
                    except Exception:
                        closing_spread_series = None
                    # Merge closing_total if column absent or entirely NaN; also coalesce market_total NaNs from closing_total
                    if "closing_total" in cl.columns:
                        med_ct = cl.groupby("game_id")["closing_total"].median().rename("closing_total")
                        if "closing_total" not in df.columns or df["closing_total"].isna().all():
                            df = df.merge(med_ct, on="game_id", how="left") if "closing_total" not in df.columns else df
                        # Coalesce market_total from closing_total when market_total exists but is NaN (common for past dates)
                        if "market_total" in df.columns:
                            try:
                                df["market_total"] = df["market_total"].where(df["market_total"].notna(), df.get("closing_total"))
                            except Exception:
                                pass
                        else:
                            df["market_total"] = df.get("closing_total")
                    # Merge closing spread
                    if closing_spread_series is not None and not closing_spread_series.empty and "closing_spread_home" not in df.columns:
                        df = df.merge(closing_spread_series, on="game_id", how="left")
                    # Coalesce spread_home from closing_spread_home when missing
                    if "closing_spread_home" in df.columns:
                        if "spread_home" in df.columns:
                            try:
                                df["spread_home"] = df["spread_home"].where(df["spread_home"].notna(), df.get("closing_spread_home"))
                            except Exception:
                                pass
                        else:
                            df["spread_home"] = df.get("closing_spread_home")
        except Exception:
            pass

    # Prediction fallback enrichment: ensure we rarely render a card with odds but no predictions.
    # Removes previous logic that hid odds when predictions were missing; instead we synthesize a baseline prediction.
    try:
        if "pred_total" in df.columns:
            mt_series = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else None
            pm_series = pd.to_numeric(df.get("pred_margin"), errors="coerce") if "pred_margin" in df.columns else None
            missing_pred = df["pred_total"].isna()
            # Fallback 1: if market_total exists but pred_total missing, copy market_total (mark basis) so edge=0 but UI populated.
            if mt_series is not None and missing_pred.any():
                can_copy = missing_pred & mt_series.notna()
                if can_copy.any():
                    df.loc[can_copy, "pred_total"] = mt_series[can_copy]
                    df.loc[can_copy, "pred_total_basis"] = "market_copy"
            # Fallback 2: derive team projections if absent using pred_total & pred_margin
            if {"pred_total","pred_margin"}.issubset(df.columns):
                pt = pd.to_numeric(df["pred_total"], errors="coerce")
                pm2 = pd.to_numeric(df["pred_margin"], errors="coerce")
                need_proj_home = ("proj_home" not in df.columns) or df["proj_home"].isna().all()
                need_proj_away = ("proj_away" not in df.columns) or df["proj_away"].isna().all()
                if need_proj_home:
                    df["proj_home"] = np.where(pt.notna() & pm2.notna(), (pt/2) + (pm2/2), df.get("proj_home"))
                if need_proj_away:
                    # Use proj_home if just created
                    if "proj_home" in df.columns:
                        ph = pd.to_numeric(df["proj_home"], errors="coerce")
                        df["proj_away"] = np.where(pt.notna() & ph.notna(), pt - ph, df.get("proj_away"))
                # Mark adjusted flag when basis is copied from market (for template optional badge)
                if "pred_total_basis" in df.columns:
                    df["pred_total_adjusted"] = np.where(df["pred_total_basis"]=="market_copy", True, df.get("pred_total_adjusted"))
    except Exception:
        pass

    # Targeted per-row fuzzy odds fill for residual missing lines (row-level, regardless of global missing share)
    try:
        from rapidfuzz import process, fuzz  # type: ignore
        if date_q:
            # Resolve a raw odds file for the selected date. Prefer odds_today.csv for today, else odds_<date>.csv; always also consider odds_history.
            candidates = []
            if today_str and date_q == today_str:
                candidates.append(OUT / "odds_today.csv")
            candidates.append(OUT / f"odds_{date_q}.csv")
            candidates.append(OUT / "odds_history" / f"odds_{date_q}.csv")
            # As a last resort, glob any variant odds_<date>_*.csv under outputs and odds_history
            try:
                for p in sorted(OUT.glob(f"odds_{date_q}_*.csv")):
                    candidates.append(p)
                hist_dir = OUT / "odds_history"
                if hist_dir.exists():
                    for p in sorted(hist_dir.glob(f"odds_{date_q}_*.csv")):
                        candidates.append(p)
            except Exception:
                pass
            raw_file = next((p for p in candidates if p.exists()), None)
            if raw_file.exists() and not df.empty and {"home_team","away_team"}.issubset(df.columns):
                # Only attempt for rows still missing critical odds
                miss_mask = df["market_total"].isna() if "market_total" in df.columns else pd.Series([True]*len(df))
                miss_mask |= (df["spread_home"].isna() if "spread_home" in df.columns else False)
                if miss_mask.any():
                    raw = pd.read_csv(raw_file)
                    if not raw.empty:
                        # Resolve team columns flexibly
                        home_col = next((c for c in ["home_team_name","home_team","home"] if c in raw.columns), None)
                        away_col = next((c for c in ["away_team_name","away_team","away"] if c in raw.columns), None)
                        if not (home_col and away_col):
                            raise Exception("raw odds missing team columns")
                        # Constrain by date if commence_time present
                        if "commence_time" in raw.columns:
                            try:
                                _dt = pd.to_datetime(raw["commence_time"], errors="coerce")
                                raw = raw[_dt.dt.strftime("%Y-%m-%d") == str(date_q)]
                            except Exception:
                                pass
                        raw["_home_norm"] = raw[home_col].astype(str).map(normalize_name)
                        raw["_away_norm"] = raw[away_col].astype(str).map(normalize_name)
                        # Pre-build list of pair variants for fuzzy search
                        raw_pairs = []  # (pair_key, index)
                        for i, r in raw.iterrows():
                            hn = r.get("_home_norm"); an = r.get("_away_norm")
                            if not hn or not an:
                                continue
                            pair_key = f"{hn}::{an}"  # keep order
                            raw_pairs.append((pair_key, i))
                        pair_keys = [rk for rk,_ in raw_pairs]
                        # Helper to extract median metrics from subset of matching raw rows
                        def _apply_fill(idx):
                            row = df.loc[idx]
                            hn = normalize_name(str(row.get("home_team") or ""))
                            an = normalize_name(str(row.get("away_team") or ""))
                            if not hn or not an:
                                return
                            target1 = f"{hn}::{an}"; target2 = f"{an}::{hn}"  # order unknown in raw
                            # Fuzzy match both orientations
                            best1 = process.extractOne(target1, pair_keys, scorer=fuzz.token_set_ratio)
                            best2 = process.extractOne(target2, pair_keys, scorer=fuzz.token_set_ratio)
                            best = None
                            for cand in [best1, best2]:
                                if cand and (best is None or cand[1] > best[1]):
                                    best = cand
                            if not best or best[1] < 90:  # threshold
                                return
                            # Gather all raw rows with same unordered team set
                            sel = []
                            for (pk, i_raw) in raw_pairs:
                                parts = pk.split("::")
                                if set(parts) == {hn, an}:
                                    sel.append(i_raw)
                            if not sel:
                                return
                            sub = raw.loc[sel]
                            # Compute medians
                            if ("market_total" not in df.columns) or pd.isna(df.at[idx, "market_total"]):
                                for tc in ["total","over_under","market_total","line_total"]:
                                    if tc in sub.columns and sub[tc].notna().any():
                                        df.at[idx, "market_total"] = float(pd.to_numeric(sub[tc], errors="coerce").median())
                                        break
                            if ("spread_home" not in df.columns) or pd.isna(df.at[idx, "spread_home"]):
                                for sc in ["spread","home_spread","spread_home","handicap_home"]:
                                    if sc in sub.columns and sub[sc].notna().any():
                                        df.at[idx, "spread_home"] = float(pd.to_numeric(sub[sc], errors="coerce").median())
                                        break
                            if ("ml_home" not in df.columns) or pd.isna(df.at[idx, "ml_home"]):
                                for mc in ["moneyline_home","ml_home","price_home","h2h_home"]:
                                    if mc in sub.columns and sub[mc].notna().any():
                                        df.at[idx, "ml_home"] = float(pd.to_numeric(sub[mc], errors="coerce").median())
                                        break
                        for idx in df[miss_mask].index:
                            _apply_fill(idx)
    except Exception:
        pass

    # Compute simple edges when market total present. Label basis if source_last.
    if "market_total" in df.columns and "pred_total" in df.columns:
        df["edge_total"] = df["pred_total"] - df["market_total"]
        if source_last:
            df["market_basis"] = "last"  # strict last pre-tip odds
        else:
            df["market_basis"] = df.get("market_basis", "market")
    # Compute closing edge separately when explicit closing_total present
    if {"closing_total","pred_total"}.issubset(df.columns):
        try:
            df["edge_closing"] = df["pred_total"] - df["closing_total"]
        except Exception:
            df["edge_closing"] = None
    # Merge richer aggregated edge metrics if edges file present (kelly fractions, edge_margin)
    if not edges.empty and "game_id" in df.columns and "game_id" in edges.columns:
        try:
            edges_f = edges.copy()
            edges_f["game_id"] = edges_f["game_id"].astype(str)
            df["game_id"] = df["game_id"].astype(str)
            keep_cols = [c for c in ["game_id","edge_total","edge_margin","kelly_fraction_total","kelly_fraction_ml_home","kelly_fraction_ml_away"] if c in edges_f.columns]
            if keep_cols:
                agg = edges_f[keep_cols].groupby("game_id").median(numeric_only=True).reset_index()
                df = df.merge(agg, on="game_id", how="left", suffixes=("", "_agg"))
        except Exception:
            pass

    # Projected team scores from pred_total and pred_margin
    # Fallback / adjustment for implausibly low or missing totals (e.g., early-season sparse features causing collapsed predictions).
    # We derive a tempo/off/def based estimate from features CSV when available and use it to fill missing
    # predictions or blend if the raw prediction looks implausibly low.
    try:
        if "pred_total" in df.columns:
            # Preserve raw value
            df["pred_total_raw"] = df["pred_total"]
            # Load features to compute derived totals map
            feat_df = pd.DataFrame()
            for name in ("features_curr.csv", "features_all.csv", "features_week.csv", "features_last2.csv"):
                p = OUT / name
                if p.exists():
                    try:
                        feat_df = pd.read_csv(p)
                        break
                    except Exception:
                        feat_df = pd.DataFrame()
            derived_map: dict[str, float] = {}
            derived_margin_map: dict[str, float] = {}
            if not feat_df.empty and "game_id" in feat_df.columns:
                feat_df["game_id"] = feat_df["game_id"].astype(str)
                # Ensure needed columns exist
                h_off = feat_df.get("home_off_rating")
                a_off = feat_df.get("away_off_rating")
                h_def = feat_df.get("home_def_rating")
                a_def = feat_df.get("away_def_rating")
                h_tmp = feat_df.get("home_tempo_rating")
                a_tmp = feat_df.get("away_tempo_rating")
                tmp_sum = feat_df.get("tempo_rating_sum")
                for _, r in feat_df.iterrows():
                    gid = str(r.get("game_id"))
                    try:
                        # Tempo average
                        if pd.notna(r.get("home_tempo_rating")) and pd.notna(r.get("away_tempo_rating")):
                            tempo_avg = (float(r.get("home_tempo_rating")) + float(r.get("away_tempo_rating"))) / 2.0
                        elif pd.notna(r.get("tempo_rating_sum")):
                            tempo_avg = float(r.get("tempo_rating_sum")) / 2.0
                        else:
                            tempo_avg = 70.0  # fallback typical pace
                        off_home = float(r.get("home_off_rating")) if pd.notna(r.get("home_off_rating")) else 100.0
                        off_away = float(r.get("away_off_rating")) if pd.notna(r.get("away_off_rating")) else 100.0
                        def_home = float(r.get("home_def_rating")) if pd.notna(r.get("home_def_rating")) else 100.0
                        def_away = float(r.get("away_def_rating")) if pd.notna(r.get("away_def_rating")) else 100.0
                        # Derived offensive efficiency adjusted total (simple possession model)
                        # Expected points per 100 possessions for each side: off - opp_def, clipped
                        exp_home_pp100 = np.clip(off_home - def_away, 80, 130)
                        exp_away_pp100 = np.clip(off_away - def_home, 80, 130)
                        # Convert to per-game using tempo average (possessions ~ tempo_avg)
                        derived_total = (exp_home_pp100 + exp_away_pp100) / 100.0 * tempo_avg
                        # Simple derived margin: difference in per-100 scaled by tempo
                        derived_margin = (exp_home_pp100 - exp_away_pp100) / 100.0 * tempo_avg
                        # Clamp plausible NCAA range
                        derived_total = float(np.clip(derived_total, 110, 185))
                        derived_margin = float(np.clip(derived_margin, -35, 35))
                        derived_map[gid] = derived_total
                        derived_margin_map[gid] = derived_margin
                    except Exception:
                        continue
            # If pred_total missing, fill from derived when available; also fill pred_margin if missing
            if derived_map:
                if "pred_total" in df.columns:
                    mask_missing_pt = df["pred_total"].isna()
                    if mask_missing_pt.any():
                        df.loc[mask_missing_pt, "pred_total"] = df.loc[mask_missing_pt, "game_id"].map(lambda g: derived_map.get(str(g)))
                        # Mark basis for visibility
                        df.loc[mask_missing_pt, "pred_total_basis"] = df.loc[mask_missing_pt, "pred_total_basis"].where(df.loc[mask_missing_pt, "pred_total_basis"].notna(), "features_derived") if "pred_total_basis" in df.columns else "features_derived"
                if "pred_margin" in df.columns and derived_margin_map:
                    mask_missing_pm = df["pred_margin"].isna()
                    if mask_missing_pm.any():
                        df.loc[mask_missing_pm, "pred_margin"] = df.loc[mask_missing_pm, "game_id"].map(lambda g: derived_margin_map.get(str(g)))
                        df.loc[mask_missing_pm, "pred_margin_basis"] = df.loc[mask_missing_pm, "pred_margin_basis"].where(df.loc[mask_missing_pm, "pred_margin_basis"].notna(), "features_derived") if "pred_margin_basis" in df.columns else "features_derived"
            # Blend predictions when implausibly low vs derived
            pred_total_was_adj = []
            if derived_map:
                adj_vals = []
                for _, r in df.iterrows():
                    gid = str(r.get("game_id"))
                    pred = r.get("pred_total")
                    if pred is None or pd.isna(pred):
                        adj_vals.append(pred)
                        pred_total_was_adj.append(False)
                        continue
                    derived = derived_map.get(gid)
                    if derived is None:
                        adj_vals.append(pred)
                        pred_total_was_adj.append(False)
                        continue
                    # Criteria for adjustment: raw pred < 105 or raw pred < 0.75*derived
                    if pred < 105 or (derived > 0 and pred < 0.75 * derived):
                        blended = 0.5 * float(pred) + 0.5 * float(derived)
                        # final clamp
                        blended = float(np.clip(blended, 110, 185))
                        adj_vals.append(blended)
                        pred_total_was_adj.append(True)
                    else:
                        adj_vals.append(pred)
                        pred_total_was_adj.append(False)
                df["pred_total"] = adj_vals
                df["pred_total_adjusted"] = pred_total_was_adj
            # Compute projected team scores using adjusted pred_total
        if {"pred_total", "pred_margin"}.issubset(df.columns):
            df["proj_home"] = (pd.to_numeric(df["pred_total"], errors="coerce") + pd.to_numeric(df["pred_margin"], errors="coerce")) / 2.0
            df["proj_away"] = pd.to_numeric(df["pred_total"], errors="coerce") - df["proj_home"]
            # Derive half projections for display
            half_ratio = 0.485
            df["pred_total_1h"] = pd.to_numeric(df["pred_total"], errors="coerce") * half_ratio
            df["pred_total_2h"] = pd.to_numeric(df["pred_total"], errors="coerce") * (1.0 - half_ratio)
            df["pred_margin_1h"] = pd.to_numeric(df["pred_margin"], errors="coerce") * 0.5
            df["pred_margin_2h"] = pd.to_numeric(df["pred_margin"], errors="coerce") * 0.5
            # Derived 1H projected team scores and winner/ATS labels
            try:
                pt1 = pd.to_numeric(df["pred_total_1h"], errors="coerce")
                pm1 = pd.to_numeric(df["pred_margin_1h"], errors="coerce")
                df["proj_home_1h"] = (pt1 + pm1) / 2.0
                df["proj_away_1h"] = pt1 - df["proj_home_1h"]
                df["pred_winner_1h"] = np.where(pm1 > 0, "Home", np.where(pm1 < 0, "Away", "Even"))
                # ATS descriptor for predictions (not stored as a single string; computed in template when needed)
            except Exception:
                df["proj_home_1h"] = None
                df["proj_away_1h"] = None
                df["pred_winner_1h"] = None
            # Derived 2H projected team scores and winner label
            try:
                pt2 = pd.to_numeric(df["pred_total_2h"], errors="coerce")
                pm2 = pd.to_numeric(df["pred_margin_2h"], errors="coerce")
                df["proj_home_2h"] = (pt2 + pm2) / 2.0
                df["proj_away_2h"] = pt2 - df["proj_home_2h"]
                df["pred_winner_2h"] = np.where(pm2 > 0, "Home", np.where(pm2 < 0, "Away", "Even"))
            except Exception:
                df["proj_home_2h"] = None
                df["proj_away_2h"] = None
                df["pred_winner_2h"] = None
    except Exception:
        df["proj_home"] = None
        df["proj_away"] = None

    # Compute ATS and ML helpers if odds present
    try:
        if "spread_home" in df.columns and "pred_margin" in df.columns:
            df["edge_ats"] = pd.to_numeric(df["pred_margin"], errors="coerce") - pd.to_numeric(df["spread_home"], errors="coerce")
        if "spread_home_1h" in df.columns and "pred_margin_1h" in df.columns:
            df["edge_ats_1h"] = pd.to_numeric(df["pred_margin_1h"], errors="coerce") - pd.to_numeric(df["spread_home_1h"], errors="coerce")
        if "market_total_1h" in df.columns and "pred_total_1h" in df.columns:
            df["edge_total_1h"] = pd.to_numeric(df["pred_total_1h"], errors="coerce") - pd.to_numeric(df["market_total_1h"], errors="coerce")
        if "market_total_2h" in df.columns and "pred_total_2h" in df.columns:
            df["edge_total_2h"] = pd.to_numeric(df["pred_total_2h"], errors="coerce") - pd.to_numeric(df["market_total_2h"], errors="coerce")
        if "spread_home_2h" in df.columns and "pred_margin_2h" in df.columns:
            df["edge_ats_2h"] = pd.to_numeric(df["pred_margin_2h"], errors="coerce") - pd.to_numeric(df["spread_home_2h"], errors="coerce")
        # Derivative leans for 1H/2H totals and ATS
        if "edge_total_1h" in df.columns:
            et1 = pd.to_numeric(df["edge_total_1h"], errors="coerce")
            df["lean_ou_side_1h"] = np.where(et1 > 0, "Over", np.where(et1 < 0, "Under", None))
            df["lean_ou_edge_abs_1h"] = et1.abs()
        if "edge_total_2h" in df.columns:
            et2 = pd.to_numeric(df["edge_total_2h"], errors="coerce")
            df["lean_ou_side_2h"] = np.where(et2 > 0, "Over", np.where(et2 < 0, "Under", None))
            df["lean_ou_edge_abs_2h"] = et2.abs()
        if "edge_ats_1h" in df.columns:
            ea1 = pd.to_numeric(df["edge_ats_1h"], errors="coerce")
            df["lean_ats_side_1h"] = np.where(ea1 > 0, "Home ATS", np.where(ea1 < 0, "Away ATS", None))
            df["lean_ats_edge_abs_1h"] = ea1.abs()
        if "edge_ats_2h" in df.columns:
            ea2 = pd.to_numeric(df["edge_ats_2h"], errors="coerce")
            df["lean_ats_side_2h"] = np.where(ea2 > 0, "Home ATS", np.where(ea2 < 0, "Away ATS", None))
            df["lean_ats_edge_abs_2h"] = ea2.abs()
        if "ml_home" in df.columns and "pred_margin" in df.columns:
            scale = 7.0
            pm = pd.to_numeric(df["pred_margin"], errors="coerce")
            df["ml_prob_model"] = 1.0 / (1.0 + np.exp(-pm / scale))
            # American odds to implied probability for home side
            def _imp_p(price):
                try:
                    if pd.isna(price):
                        return np.nan
                    price = float(price)
                    return (abs(price) / (abs(price) + 100.0)) if price > 0 else (100.0 / (abs(price) + 100.0))
                except Exception:
                    return np.nan
            df["ml_prob_implied"] = df["ml_home"].map(_imp_p)
            if "ml_prob_implied" in df.columns:
                df["ml_prob_edge"] = df["ml_prob_model"] - df["ml_prob_implied"]
        # Margin direction labels and lean signals
        if "pred_margin" in df.columns:
            pm = pd.to_numeric(df["pred_margin"], errors="coerce")
            df["favored_side"] = np.where(pm > 0, "Home", np.where(pm < 0, "Away", "Even"))
            df["favored_by"] = pm.abs()
        # Over/Under lean based on edge_total (requires market_total)
        if "edge_total" in df.columns:
            et = pd.to_numeric(df["edge_total"], errors="coerce")
            df["lean_ou_side"] = np.where(et > 0, "Over", np.where(et < 0, "Under", None))
            df["lean_ou_edge_abs"] = et.abs()
        # ATS lean based on edge_ats and spread_home
        if "edge_ats" in df.columns and "spread_home" in df.columns:
            ea = pd.to_numeric(df["edge_ats"], errors="coerce")
            df["lean_ats_side"] = np.where(ea > 0, "Home ATS", np.where(ea < 0, "Away ATS", None))
            df["lean_ats_edge_abs"] = ea.abs()
        # Edge vs closing spread if available
        if {"closing_spread_home","pred_margin"}.issubset(df.columns):
            try:
                df["edge_closing_ats"] = pd.to_numeric(df["pred_margin"], errors="coerce") - pd.to_numeric(df["closing_spread_home"], errors="coerce")
            except Exception:
                df["edge_closing_ats"] = None
    except Exception:
        pass

    # Attach picks (per-game) for picks strip display
    top_picks: list[dict[str, Any]] = []
    try:
        # Prefer expanded picks_raw for richer context; fallback to clean picks
        raw_path = OUT / "picks_raw.csv"
        picks_df = pd.read_csv(raw_path) if raw_path.exists() else _load_picks()
        picks_map: dict[str, list[dict[str, Any]]] = {}
        if not picks_df.empty and "game_id" in picks_df.columns:
            picks_df["game_id"] = picks_df["game_id"].astype(str)
            if date_q and "date" in picks_df.columns:
                try:
                    picks_df["date"] = pd.to_datetime(picks_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    picks_df = picks_df[picks_df["date"] == date_q]
                except Exception:
                    pass
            # Normalize columns for market/selection/line/edge
            def pick_col(df_, names):
                return next((c for c in names if c in df_.columns), None)
            col_market = pick_col(picks_df, ["market","bet_type","type"]) or "market"
            col_sel = pick_col(picks_df, ["selection","pick","bet","side","team"]) or "selection"
            col_edge = pick_col(picks_df, ["edge","abs_edge","edge_total","edge_ats"]) or "edge"
            potential_line_cols = [c for c in ["line","line_value","total","home_spread","spread_home","pick_line"] if c in picks_df.columns]
            col_price = pick_col(picks_df, ["price","odds","american_odds"]) or None
            # Coerce edge to numeric and build abs edge
            if col_edge in picks_df.columns:
                picks_df["_edge_val"] = pd.to_numeric(picks_df[col_edge], errors="coerce")
                picks_df["_abs_edge"] = picks_df["_edge_val"].abs()
            else:
                picks_df["_edge_val"] = np.nan
                picks_df["_abs_edge"] = np.nan
            # Build per-game pick lists (dedupe by selection/market/line and cap to top 3)
            for gid, grp in picks_df.groupby("game_id"):
                g = grp.copy()
                if "_abs_edge" in g.columns:
                    g = g.sort_values(["_abs_edge"], ascending=[False])
                # Deduplicate by market+selection+line to avoid many books duplicates
                dedup_cols = [c for c in [col_market, col_sel] if c in g.columns] + (potential_line_cols[:1] or [])
                if dedup_cols:
                    g = g.drop_duplicates(subset=dedup_cols, keep="first")
                items: list[dict[str, Any]] = []
                for _, pr in g.head(3).iterrows():
                    item: dict[str, Any] = {"game_id": str(gid)}
                    item["market"] = pr.get(col_market)
                    item["selection"] = pr.get(col_sel)
                    # Prefer absolute edge for display clarity
                    try:
                        item["edge"] = float(abs(float(pr.get(col_edge)))) if pd.notna(pr.get(col_edge)) else None
                    except Exception:
                        item["edge"] = pr.get(col_edge)
                    # Line detection
                    line_val = None
                    for lc in potential_line_cols:
                        if lc in g.columns and pd.notna(pr.get(lc)):
                            line_val = pr.get(lc)
                            break
                    if line_val is not None:
                        item["line"] = line_val
                    if col_price and col_price in g.columns:
                        item["price"] = pr.get(col_price)
                    items.append(item)
                if items:
                    picks_map[str(gid)] = items
            # Build global top picks: top 12 unique games by abs edge
            try:
                pf = picks_df.copy()
                pf = pf.sort_values(["_abs_edge"], ascending=[False]) if "_abs_edge" in pf.columns else pf
                teams_map: dict[str, tuple[Any, Any]] = {}
                if {"game_id","home_team","away_team"}.issubset(df.columns):
                    for _, rr in df.iterrows():
                        teams_map[str(rr.get("game_id"))] = (rr.get("home_team"), rr.get("away_team"))
                # Keep one best pick per game
                if "_abs_edge" in pf.columns:
                    pf = pf.sort_values(["game_id","_abs_edge"], ascending=[True, False]).drop_duplicates(subset=["game_id"], keep="first")
                top_rows: list[dict[str, Any]] = []
                for _, pr in pf.head(12).iterrows():
                    gid = str(pr.get("game_id"))
                    h, a = teams_map.get(gid, (None, None))
                    item: dict[str, Any] = {"game_id": gid, "home_team": h, "away_team": a}
                    item["market"] = pr.get(col_market)
                    item["selection"] = pr.get(col_sel)
                    try:
                        val = pr.get(col_edge)
                        item["edge"] = float(abs(float(val))) if pd.notna(val) else None
                    except Exception:
                        item["edge"] = pr.get(col_edge)
                    line_val = None
                    for lc in potential_line_cols:
                        v = pr.get(lc)
                        if pd.notna(v):
                            line_val = v
                            break
                    if line_val is not None:
                        item["line"] = line_val
                    if col_price:
                        item["price"] = pr.get(col_price)
                    top_rows.append(item)
                top_picks = top_rows
            except Exception:
                top_picks = []
        df["_picks_list"] = df["game_id"].map(lambda x: picks_map.get(str(x), [])) if picks_map else [[] for _ in range(len(df))]
    except Exception as _ep:
        df["_picks_list"] = [[] for _ in range(len(df))]

    # Order by start_time if available; fallback to date then home_team/game_id; else by abs edge
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    # Accept start_time as either datetime or string; parse robustly with controlled formats to avoid inference warnings
    if "start_time" in df.columns:
        try:
            st_series = df["start_time"].astype(str).str.strip()
            # Normalize common patterns: 'YYYY-MM-DD HH:MM:SS+00:00' or 'YYYY-MM-DD HH:MM'
            st_series = st_series.str.replace("Z", "+00:00", regex=False)
            # If a time is present but no timezone, assume UTC and append +00:00
            has_time = st_series.str.contains(r"\d{1,2}:\d{2}", regex=True)
            needs_tz = has_time & ~st_series.str.contains("[+-][0-9]{2}:[0-9]{2}$", regex=True)
            st_series = st_series.where(~needs_tz, st_series + "+00:00")
            df["_start_dt"] = pd.to_datetime(st_series, errors="coerce", utc=True)
            # If parsing failed for rows with a time component, try stripping seconds or rebuilding ISO
            if "_start_dt" in df.columns:
                mask_fail = df["_start_dt"].isna() & has_time
                if mask_fail.any():
                    raw = st_series[mask_fail]
                    # Remove seconds if present
                    raw2 = raw.str.replace(r":(\d{2})(?::\d{2})?", r":\1", regex=True)
                    reparsed = pd.to_datetime(raw2, errors="coerce", utc=True)
                    df.loc[mask_fail & reparsed.notna(), "_start_dt"] = reparsed[reparsed.notna()]
        except Exception:
            df["_start_dt"] = pd.NaT
    if "_start_dt" in df.columns and df["_start_dt"].notna().any():
        # Use start dt primary, then home_team/game_id for deterministic order
        sort_cols = ["_start_dt", "home_team" if "home_team" in df.columns else "game_id"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    elif "date" in df.columns and df["date"].notna().any():
        sort_cols = ["date", "home_team" if "home_team" in df.columns else "game_id"]
        try:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        except Exception:
            pass
    elif "edge_total" in df.columns:
        df["abs_edge"] = df["edge_total"].abs()
        df = df.sort_values(["abs_edge"], ascending=[False])
    # Convert _start_dt to local time for display and ordering
    try:
        if "_start_dt" in df.columns and df["_start_dt"].notna().any():
            # Convert to system local tz
            local_tz = dt.datetime.now().astimezone().tzinfo
            df["_start_local"] = df["_start_dt"].dt.tz_convert(local_tz)
            df["start_time_local"] = df["_start_local"].dt.strftime("%Y-%m-%d %H:%M")
            # Also provide ISO UTC for client-side rendering in the browser's timezone
            try:
                df["start_time_iso"] = df["_start_dt"].dt.tz_convert(dt.timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                df["start_time_iso"] = df["_start_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Per-row fallback: if any iso/local still missing (NaN), derive from raw start_time string best-effort
            if "start_time" in df.columns:
                st_str = df["start_time"].astype(str)
                # Build simple ISO guess from 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'
                iso_guess = np.where(
                    st_str.str.contains("T"),
                    st_str,
                    st_str.str.replace(" ", "T", regex=False)
                )
                iso_guess = iso_guess.str.replace("\+00:00", "Z", regex=False)
                # Fill missing start_time_iso/local with guesses
                if "start_time_iso" in df.columns:
                    mask_iso_missing = df["start_time_iso"].isna() | (df["start_time_iso"].astype(str).str.strip()=="")
                    df.loc[mask_iso_missing, "start_time_iso"] = iso_guess[mask_iso_missing]
                else:
                    df["start_time_iso"] = iso_guess
                if "start_time_local" in df.columns:
                    mask_loc_missing = df["start_time_local"].isna() | (df["start_time_local"].astype(str).str.strip()=="")
                    disp = st_str.str.replace("T", " ", regex=False).str.replace(r":\d\d(\+\d\d:\d\d|Z)$", "", regex=True)
                    df.loc[mask_loc_missing, "start_time_local"] = disp[mask_loc_missing]
                else:
                    df["start_time_local"] = st_str.str.replace("T", " ", regex=False).str.replace(r":\d\d(\+\d\d:\d\d|Z)$", "", regex=True)
        else:
            # As last resort, if start_time looks like 'YYYY-MM-DD HH:MM:SS+00:00', format local display without tz conversion
            if "start_time" in df.columns:
                st_str = df["start_time"].astype(str)
                # If contains 'T' already, keep as iso; if only date, leave as date (no time available)
                df["start_time_iso"] = np.where(st_str.str.contains("T"), st_str, st_str.str.replace(" ", "T", regex=False).str.replace("\+00:00", "Z", regex=False))
                # For local display, drop seconds and tz if present
                disp = st_str.str.replace("T", " ", regex=False).str.replace(r":\d\d(\+\d\d:\d\d|Z)$", "", regex=True)
                df["start_time_local"] = disp
    except Exception:
        pass

    # Venue and status sanitization/coalescing
    try:
        # Treat common NaN-like strings as missing
        def _clean_str(s):
            if s is None:
                return None
            try:
                s2 = str(s).strip()
            except Exception:
                return None
            if s2 == "" or s2.lower() in {"nan", "none", "nat", "null"}:
                return None
            return s2
        cand_cols = [c for c in ["venue", "venue_full", "arena", "stadium", "site_name", "site", "location"] if c in df.columns]
        # Build venue from first non-empty candidate; if none, fall back to City, State when available
        if cand_cols or ("city" in df.columns or "state" in df.columns):
            venues: list[str | None] = []
            for _, r in df.iterrows():
                # Primary candidates
                val = None
                for c in cand_cols:
                    v = _clean_str(r.get(c))
                    if v:
                        val = v
                        break
                # Fallback to City, State
                if not val:
                    city = _clean_str(r.get("city")) if "city" in df.columns else None
                    state = _clean_str(r.get("state")) if "state" in df.columns else None
                    if city and state:
                        val = f"{city}, {state}"
                    elif city:
                        val = city
                    elif state:
                        val = state
                venues.append(val)
            df["venue"] = venues
        # Clean status string to avoid printing literal 'nan'
        if "status" in df.columns:
            df["status"] = df["status"].map(_clean_str)
        # Clean start_time_local text if any literal 'nan'
        if "start_time_local" in df.columns:
            df["start_time_local"] = df["start_time_local"].map(_clean_str)
    except Exception:
        pass

    # Halves outcomes vs market (postgame)
    try:
        # Compute actual half totals if scores present
        if {"home_score_1h","away_score_1h"}.issubset(df.columns):
            hs1 = pd.to_numeric(df["home_score_1h"], errors="coerce")
            as1 = pd.to_numeric(df["away_score_1h"], errors="coerce")
            df["actual_total_1h"] = np.where(hs1.notna() & as1.notna(), hs1 + as1, df.get("actual_total_1h"))
            df["actual_margin_1h"] = np.where(hs1.notna() & as1.notna(), hs1 - as1, np.nan)
        if {"home_score_2h","away_score_2h"}.issubset(df.columns):
            hs2 = pd.to_numeric(df["home_score_2h"], errors="coerce")
            as2 = pd.to_numeric(df["away_score_2h"], errors="coerce")
            df["actual_total_2h"] = np.where(hs2.notna() & as2.notna(), hs2 + as2, df.get("actual_total_2h"))
            df["actual_margin_2h"] = np.where(hs2.notna() & as2.notna(), hs2 - as2, np.nan)
        # OU results for halves
        if {"actual_total_1h","market_total_1h"}.issubset(df.columns):
            at1 = pd.to_numeric(df["actual_total_1h"], errors="coerce")
            mt1 = pd.to_numeric(df["market_total_1h"], errors="coerce")
            ou1 = np.where(at1 > mt1, "Over", np.where(at1 < mt1, "Under", "Push"))
            df.loc[at1.notna() & mt1.notna(), "ou_result_1h"] = ou1[at1.notna() & mt1.notna()]
            # Eval vs model lean
            if "lean_ou_side_1h" in df.columns:
                l1 = df["lean_ou_side_1h"].astype(str)
                r1 = df["ou_result_1h"].astype(str)
                ok1 = np.where((r1=="Push") | (l1=="None") | (l1=="nan") | (l1==""), np.nan, (l1==r1))
                df.loc[:, "eval_ou_1h_ok"] = ok1
        if {"actual_total_2h","market_total_2h"}.issubset(df.columns):
            at2 = pd.to_numeric(df["actual_total_2h"], errors="coerce")
            mt2 = pd.to_numeric(df["market_total_2h"], errors="coerce")
            ou2 = np.where(at2 > mt2, "Over", np.where(at2 < mt2, "Under", "Push"))
            df.loc[at2.notna() & mt2.notna(), "ou_result_2h"] = ou2[at2.notna() & mt2.notna()]
            if "lean_ou_side_2h" in df.columns:
                l2 = df["lean_ou_side_2h"].astype(str)
                r2 = df["ou_result_2h"].astype(str)
                ok2 = np.where((r2=="Push") | (l2=="None") | (l2=="nan") | (l2==""), np.nan, (l2==r2))
                df.loc[:, "eval_ou_2h_ok"] = ok2
        # ATS results for halves when 1H/2H spreads exist (including derived spreads)
        if {"actual_margin_1h","spread_home_1h"}.issubset(df.columns):
            am1 = pd.to_numeric(df["actual_margin_1h"], errors="coerce")
            sh1 = pd.to_numeric(df["spread_home_1h"], errors="coerce")
            ats1 = np.where(am1 > -sh1, "Home Cover", np.where(am1 < -sh1, "Away Cover", "Push"))
            df.loc[am1.notna() & sh1.notna(), "ats_result_1h"] = ats1[am1.notna() & sh1.notna()]
            if "lean_ats_side_1h" in df.columns:
                l1a = np.where(df["lean_ats_side_1h"].astype(str)=="Home ATS", "Home Cover", np.where(df["lean_ats_side_1h"].astype(str)=="Away ATS", "Away Cover", ""))
                r1a = df["ats_result_1h"].astype(str)
                ok1a = np.where((r1a=="Push") | (l1a==""), np.nan, (l1a==r1a))
                df.loc[:, "eval_ats_1h_ok"] = ok1a
        if {"actual_margin_2h","spread_home_2h"}.issubset(df.columns):
            am2 = pd.to_numeric(df["actual_margin_2h"], errors="coerce")
            sh2 = pd.to_numeric(df["spread_home_2h"], errors="coerce")
            ats2 = np.where(am2 > -sh2, "Home Cover", np.where(am2 < -sh2, "Away Cover", "Push"))
            df.loc[am2.notna() & sh2.notna(), "ats_result_2h"] = ats2[am2.notna() & sh2.notna()]
            if "lean_ats_side_2h" in df.columns:
                l2a = np.where(df["lean_ats_side_2h"].astype(str)=="Home ATS", "Home Cover", np.where(df["lean_ats_side_2h"].astype(str)=="Away ATS", "Away Cover", ""))
                r2a = df["ats_result_2h"].astype(str)
                ok2a = np.where((r2a=="Push") | (l2a==""), np.nan, (l2a==r2a))
                df.loc[:, "eval_ats_2h_ok"] = ok2a
        # Winner correctness for halves (vs predicted winner if available)
        try:
            if {"pred_winner_1h","actual_margin_1h"}.issubset(df.columns):
                pm = df["pred_winner_1h"].astype(str)
                am = pd.to_numeric(df["actual_margin_1h"], errors="coerce")
                act = np.where(am>0, "Home", np.where(am<0, "Away", "Even"))
                ok = np.where((act=="Even") | (pm=="Even") | pm.isna(), np.nan, (pm==act))
                df.loc[:, "eval_winner_1h_ok"] = ok
            if {"pred_winner_2h","actual_margin_2h"}.issubset(df.columns):
                pm2 = df["pred_winner_2h"].astype(str)
                am2n = pd.to_numeric(df["actual_margin_2h"], errors="coerce")
                act2 = np.where(am2n>0, "Home", np.where(am2n<0, "Away", "Even"))
                ok2 = np.where((act2=="Even") | (pm2=="Even") | pm2.isna(), np.nan, (pm2==act2))
                df.loc[:, "eval_winner_2h_ok"] = ok2
        except Exception:
            pass
    except Exception:
        pass

    # Final results & betting outcome reconciliation (ATS & ML) when scores available
    try:
        if {"home_score","away_score"}.issubset(df.columns):
            hs = pd.to_numeric(df["home_score"], errors="coerce")
            as_ = pd.to_numeric(df["away_score"], errors="coerce")
            mask_done = (hs > 0) | (as_ > 0)
            if mask_done.any():
                # Actual totals/margins
                actual_margin = hs - as_
                df.loc[mask_done, "actual_margin"] = actual_margin[mask_done]
                df.loc[mask_done, "actual_total"] = (hs + as_)[mask_done]
                if "spread_home" in df.columns:
                    line = pd.to_numeric(df["spread_home"], errors="coerce")
                    # Home covers if actual margin > -line (because spread_home is home line)
                    ats_res = np.where(actual_margin > -line, "Home Cover", np.where(actual_margin < -line, "Away Cover", "Push"))
                    df.loc[mask_done, "ats_result"] = ats_res[mask_done]
                if "closing_spread_home" in df.columns:
                    c_line = pd.to_numeric(df["closing_spread_home"], errors="coerce")
                    ats_close_res = np.where(actual_margin > -c_line, "Home Cover", np.where(actual_margin < -c_line, "Away Cover", "Push"))
                    df.loc[mask_done, "ats_close_result"] = ats_close_res[mask_done]
                ml_res = np.where(actual_margin > 0, "Home Win", np.where(actual_margin < 0, "Away Win", "Push"))
                df.loc[mask_done, "ml_result"] = ml_res[mask_done]

                # OU reconciliation (full game) vs market and closing
                at = pd.to_numeric(df.get("actual_total"), errors="coerce")
                if {"market_total"}.issubset(df.columns):
                    mt = pd.to_numeric(df.get("market_total"), errors="coerce")
                    ou_full = np.where(at > mt, "Over", np.where(at < mt, "Under", "Push"))
                    df.loc[mask_done & at.notna() & mt.notna(), "ou_result_full"] = ou_full[mask_done & at.notna() & mt.notna()]
                if {"closing_total"}.issubset(df.columns):
                    ct = pd.to_numeric(df.get("closing_total"), errors="coerce")
                    ouc_full = np.where(at > ct, "Over", np.where(at < ct, "Under", "Push"))
                    df.loc[mask_done & at.notna() & ct.notna(), "ou_close_result_full"] = ouc_full[mask_done & at.notna() & ct.notna()]

                # Evaluation flags: model lean vs actual outcome
                # OU vs Market: compare lean_ou_side to ou_result_full
                if {"lean_ou_side","ou_result_full"}.issubset(df.columns):
                    l = df["lean_ou_side"].astype(str)
                    r = df["ou_result_full"].astype(str)
                    ok = np.where((r=="Push") | (l=="None") | (l=="nan") | (l==""), np.nan, (l==r))
                    df.loc[:, "eval_ou_ok"] = ok
                # OU vs Close
                if {"lean_ou_side","ou_close_result_full"}.issubset(df.columns):
                    l = df["lean_ou_side"].astype(str)
                    r = df["ou_close_result_full"].astype(str)
                    ok = np.where((r=="Push") | (l=="None") | (l=="nan") | (l==""), np.nan, (l==r))
                    df.loc[:, "eval_ou_close_ok"] = ok
                # ATS vs actual: map lean_ats_side (Home ATS/Away ATS) to covers
                if {"lean_ats_side","ats_result"}.issubset(df.columns):
                    l = df["lean_ats_side"].astype(str)
                    r = df["ats_result"].astype(str)
                    # Map Home ATS -> Home Cover; Away ATS -> Away Cover
                    l_map = np.where(l=="Home ATS", "Home Cover", np.where(l=="Away ATS", "Away Cover", ""))
                    ok = np.where((r=="Push") | (l_map==""), np.nan, (l_map==r))
                    df.loc[:, "eval_ats_ok"] = ok
                # ATS vs closing (optional) if closing spread present and we computed ats_close_result and edge_closing_ats
                if {"edge_closing_ats","ats_close_result"}.issubset(df.columns):
                    e = pd.to_numeric(df["edge_closing_ats"], errors="coerce")
                    l2 = np.where(e>0, "Home Cover", np.where(e<0, "Away Cover", ""))
                    r2 = df["ats_close_result"].astype(str)
                    ok2 = np.where((r2=="Push") | (l2==""), np.nan, (l2==r2))
                    df.loc[:, "eval_ats_close_ok"] = ok2
                # ML winner: sign of pred_margin vs ml_result
                if {"pred_margin","ml_result"}.issubset(df.columns):
                    pm = pd.to_numeric(df["pred_margin"], errors="coerce")
                    lml = np.where(pm>0, "Home Win", np.where(pm<0, "Away Win", "Push"))
                    rml = df["ml_result"].astype(str)
                    okml = np.where((lml=="Push") | (rml=="Push"), np.nan, (lml==rml))
                    df.loc[:, "eval_ml_ok"] = okml
    except Exception:
        pass

    # Reformat date back to string for template
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Small safeguard: warn if predictions look uniform (little variance)
    uniform_note = None
    if "pred_total" in df.columns and len(df) >= 5:
        try:
            stdv = float(pd.to_numeric(df["pred_total"], errors="coerce").dropna().std())
            if stdv < 1e-3:
                uniform_note = (
                    "Predictions appear uniform. This often happens at season start when rolling features are empty. "
                    "Consider seeding preseason ratings (evaluate-last2) and re-running daily-run."
                )
        except Exception:
            pass

    # Branding enrichment
    branding = _load_branding_map()
    # Precompute pure CSS (no Jinja loops in template) for team colors
    css_lines: list[str] = []
    for key, b in branding.items():
        primary = b.get("primary") or b.get("secondary")
        text = b.get("text") or "#ffffff"
        if primary:
            css_lines.append(f".badge-team.k-{key}{{background:{primary};}}")
        if text:
            css_lines.append(f".badge-team.k-{key} .name{{color:{text};}}")
    dynamic_css = "\n".join(css_lines)
    def _brand_row(row: dict[str, Any]) -> dict[str, Any]:
        for side in ["home", "away"]:
            tkey = normalize_name(str(row.get(f"{side}_team") or ""))
            b = branding.get(tkey) or {}
            row[f"{side}_key"] = tkey
            row[f"{side}_logo"] = b.get("logo")
            row[f"{side}_color"] = b.get("primary") or b.get("secondary") or None
            row[f"{side}_text_color"] = b.get("text") or "#ffffff"
        return row
    # Ensure derivative/odds fields exist to avoid template UndefinedError on missing keys
    try:
        required_cols = [
            # Full game
            "spread_home", "market_total", "pred_total", "pred_margin", "proj_home", "proj_away",
            "ats_result", "actual_total", "ml_result", "closing_total", "closing_spread_home",
            "edge_closing_ats", "edge_total", "lean_ou_side", "edge_ats", "favored_side", "favored_by",
            # 1st half
            "pred_total_1h", "pred_margin_1h", "proj_home_1h", "proj_away_1h", "pred_winner_1h",
            "market_total_1h", "spread_home_1h", "ats_result_1h", "actual_total_1h", "home_score_1h", "away_score_1h",
            "spread_home_1h_basis",
            # 2nd half
            "pred_total_2h", "pred_margin_2h", "proj_home_2h", "proj_away_2h", "pred_winner_2h",
            "market_total_2h", "spread_home_2h", "ats_result_2h", "actual_total_2h", "home_score_2h", "away_score_2h",
            "spread_home_2h_basis",
        ]
        for c in required_cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        pass

    # Ensure half spreads and ATS results are populated (second pass) before template conversion.
    try:
        # Derive half spreads if still missing or all NaN.
        if "spread_home" in df.columns:
            sh_full = pd.to_numeric(df["spread_home"], errors="coerce")
            # 1H
            if ("spread_home_1h" not in df.columns) or df["spread_home_1h"].isna().all():
                df["spread_home_1h"] = np.where(sh_full.notna(), sh_full * 0.5, np.nan)
                df["spread_home_1h_basis"] = np.where(sh_full.notna(), "derived2", df.get("spread_home_1h_basis"))
            # 2H
            if ("spread_home_2h" not in df.columns) or df["spread_home_2h"].isna().all():
                base_1h = pd.to_numeric(df.get("spread_home_1h"), errors="coerce") if "spread_home_1h" in df.columns else (sh_full * 0.5)
                df["spread_home_2h"] = np.where((sh_full.notna()) & (base_1h.notna()), sh_full - base_1h, np.nan)
                df["spread_home_2h_basis"] = np.where(sh_full.notna(), "derived2", df.get("spread_home_2h_basis"))
        # Derive half scores (2H) if missing but 1H + final available
        if {"home_score","away_score","home_score_1h","away_score_1h"}.issubset(df.columns):
            hs = pd.to_numeric(df["home_score"], errors="coerce")
            as_ = pd.to_numeric(df["away_score"], errors="coerce")
            h1 = pd.to_numeric(df["home_score_1h"], errors="coerce")
            a1 = pd.to_numeric(df["away_score_1h"], errors="coerce")
            need_2h_home = ("home_score_2h" not in df.columns) or df["home_score_2h"].isna().all()
            need_2h_away = ("away_score_2h" not in df.columns) or df["away_score_2h"].isna().all()
            if need_2h_home:
                df["home_score_2h"] = np.where(hs.notna() & h1.notna(), hs - h1, np.nan)
            if need_2h_away:
                df["away_score_2h"] = np.where(as_.notna() & a1.notna(), as_ - a1, np.nan)
        # Compute half ATS results if still missing
        if {"home_score_1h","away_score_1h","spread_home_1h"}.issubset(df.columns):
            hs1 = pd.to_numeric(df["home_score_1h"], errors="coerce")
            as1 = pd.to_numeric(df["away_score_1h"], errors="coerce")
            sh1 = pd.to_numeric(df["spread_home_1h"], errors="coerce")
            am1 = hs1 - as1
            mask1 = hs1.notna() & as1.notna() & sh1.notna()
            ats1 = np.where(am1 > -sh1, "Home Cover", np.where(am1 < -sh1, "Away Cover", "Push"))
            if ("ats_result_1h" not in df.columns) or df["ats_result_1h"].isna().all():
                df["ats_result_1h"] = np.nan
            df.loc[mask1 & df["ats_result_1h"].isna(), "ats_result_1h"] = ats1[mask1 & df["ats_result_1h"].isna()]
        if {"home_score_2h","away_score_2h","spread_home_2h"}.issubset(df.columns):
            hs2 = pd.to_numeric(df["home_score_2h"], errors="coerce")
            as2 = pd.to_numeric(df["away_score_2h"], errors="coerce")
            sh2 = pd.to_numeric(df["spread_home_2h"], errors="coerce")
            am2 = hs2 - as2
            mask2 = hs2.notna() & as2.notna() & sh2.notna()
            ats2 = np.where(am2 > -sh2, "Home Cover", np.where(am2 < -sh2, "Away Cover", "Push"))
            if ("ats_result_2h" not in df.columns) or df["ats_result_2h"].isna().all():
                df["ats_result_2h"] = np.nan
            df.loc[mask2 & df["ats_result_2h"].isna(), "ats_result_2h"] = ats2[mask2 & df["ats_result_2h"].isna()]
    except Exception:
        pass
    # Replace NaN with None for template-friendly rendering (avoid 'nan' text everywhere)
    try:
        df_tpl = df.where(pd.notna(df), None)
    except Exception:
        df_tpl = df

    # Post-construction cleanup: ensure no rows reach the template with BOTH missing predictions and odds.
    # Strategy:
    # 1. If pred_total & all market/closing totals are missing, but proj_home/proj_away exist, derive pred_total = proj_home + proj_away.
    # 2. Re-check; if still missing all totals AND no spread or ML quotes, drop the row as it provides no actionable info.
    removed_empty_rows = 0
    try:
        # New behavior: by default KEEP rows even if both predictions and odds are missing.
        # Only drop when explicitly requested via ?drop_empty=1. For backwards compat, ?show_empty=1 also forces keep.
        drop_empty = (request.args.get("drop_empty") or "").strip().lower() in ("1","true","yes")
        allow_empty = (request.args.get("show_empty") or "").strip().lower() in ("1","true","yes")
        needed_cols = {"pred_total", "market_total", "closing_total"}
        if needed_cols.issubset(df_tpl.columns) and drop_empty and not allow_empty:
            # Derive predictions where possible
            have_proj = {"proj_home", "proj_away"}.issubset(df_tpl.columns)
            if have_proj:
                ph = pd.to_numeric(df_tpl.get("proj_home"), errors="coerce")
                pa = pd.to_numeric(df_tpl.get("proj_away"), errors="coerce")
                missing_pred = df_tpl["pred_total"].isna() if "pred_total" in df_tpl.columns else pd.Series(True, index=df_tpl.index)
                can_derive = missing_pred & ph.notna() & pa.notna()
                if can_derive.any():
                    df_tpl.loc[can_derive, "pred_total"] = (ph + pa)[can_derive]
            # Identify rows still empty of both odds & predictions (no totals, no spreads, no ML) after derivation
            still_empty = (
                df_tpl["pred_total"].isna() &
                df_tpl["market_total"].isna() &
                df_tpl["closing_total"].isna()
            )
            if "spread_home" in df_tpl.columns:
                still_empty = still_empty & df_tpl["spread_home"].isna() & df_tpl.get("closing_spread_home", pd.Series([True]*len(df_tpl))).isna()
            if "ml_home" in df_tpl.columns:
                still_empty = still_empty & df_tpl["ml_home"].isna()
            # Drop those rows; capture count for diagnostics
            if still_empty.any():
                removed_empty_rows = int(still_empty.sum())
                df_tpl = df_tpl[~still_empty].reset_index(drop=True)
    except Exception:
        pass

    # Recompute coverage summary post enrichment: treat presence of any of (market_total, closing_total, spread_home, closing_spread_home, ml_home) as odds coverage signals.
    try:
        coverage_summary = {"full": 0, "partial": 0, "none": 0}
        if "game_id" in df_tpl.columns:
            for _, r in df_tpl.iterrows():
                has_total = (r.get("market_total") is not None) or (r.get("closing_total") is not None)
                has_spread = (r.get("spread_home") is not None) or (r.get("closing_spread_home") is not None)
                has_ml = (r.get("ml_home") is not None)
                if has_total and has_spread and has_ml:
                    coverage_summary["full"] += 1
                elif has_total or has_spread or has_ml:
                    coverage_summary["partial"] += 1
                else:
                    coverage_summary["none"] += 1
            # Annotate per-row coverage_status for template use
            cov_status: list[str] = []
            for _, r in df_tpl.iterrows():
                has_total = (r.get("market_total") is not None) or (r.get("closing_total") is not None)
                has_spread = (r.get("spread_home") is not None) or (r.get("closing_spread_home") is not None)
                has_ml = (r.get("ml_home") is not None)
                if has_total and has_spread and has_ml:
                    cov_status.append("full")
                elif has_total or has_spread or has_ml:
                    cov_status.append("partial")
                else:
                    cov_status.append("none")
            df_tpl["coverage_status"] = cov_status
    except Exception:
        pass

    # Final fallback: populate missing half ATS results if spreads and scores now available (including derived spreads)
    try:
        # 1H
        if {"home_score_1h","away_score_1h","spread_home_1h"}.issubset(df_tpl.columns):
            mask_missing_1h = df_tpl["ats_result_1h"].isna() if "ats_result_1h" in df_tpl.columns else pd.Series(True, index=df_tpl.index)
            hs1 = pd.to_numeric(df_tpl["home_score_1h"], errors="coerce")
            as1 = pd.to_numeric(df_tpl["away_score_1h"], errors="coerce")
            sh1 = pd.to_numeric(df_tpl["spread_home_1h"], errors="coerce")
            can_calc_1h = mask_missing_1h & hs1.notna() & as1.notna() & sh1.notna()
            am1 = hs1 - as1
            ats1 = np.where(am1 > -sh1, "Home Cover", np.where(am1 < -sh1, "Away Cover", "Push"))
            if "ats_result_1h" not in df_tpl.columns:
                df_tpl["ats_result_1h"] = None
            df_tpl.loc[can_calc_1h, "ats_result_1h"] = ats1[can_calc_1h]
        # 2H
        if {"home_score_2h","away_score_2h","spread_home_2h"}.issubset(df_tpl.columns):
            mask_missing_2h = df_tpl["ats_result_2h"].isna() if "ats_result_2h" in df_tpl.columns else pd.Series(True, index=df_tpl.index)
            hs2 = pd.to_numeric(df_tpl["home_score_2h"], errors="coerce")
            as2 = pd.to_numeric(df_tpl["away_score_2h"], errors="coerce")
            sh2 = pd.to_numeric(df_tpl["spread_home_2h"], errors="coerce")
            can_calc_2h = mask_missing_2h & hs2.notna() & as2.notna() & sh2.notna()
            am2 = hs2 - as2
            ats2 = np.where(am2 > -sh2, "Home Cover", np.where(am2 < -sh2, "Away Cover", "Push"))
            if "ats_result_2h" not in df_tpl.columns:
                df_tpl["ats_result_2h"] = None
            df_tpl.loc[can_calc_2h, "ats_result_2h"] = ats2[can_calc_2h]
    except Exception:
        pass

    rows = [_brand_row(r) for r in df_tpl.to_dict(orient="records")]

    # Display filter: keep only games with at least one Division I team unless override flag set (?all=1)
    # We apply after all enrichments so market lines/predictions remain intact; this is purely a view-level restriction.
    try:
        show_all = (request.args.get("all") or "").strip().lower() in ("1","true","yes")
        if not show_all and rows:
            d1set = _load_d1_team_set()
            if d1set:
                filtered_rows: list[dict[str, Any]] = []
                for r in rows:
                    h = normalize_name(str(r.get("home_team") or ""))
                    a = normalize_name(str(r.get("away_team") or ""))
                    if (h in d1set) or (a in d1set):
                        filtered_rows.append(r)
                # Only replace if we removed something (avoid accidental emptying on name mismatch)
                if filtered_rows and len(filtered_rows) <= len(rows):
                    rows = filtered_rows
            # If d1set empty, silently skip filter (avoid hiding everything)
    except Exception:
        pass
    total_rows = len(rows)
    accuracy = _load_accuracy_summary()
    coverage_note = None
    try:
        if date_q:
            cov = _load_schedule_coverage()
            if not cov.empty and {"date","anomaly","n_games"}.issubset(cov.columns):
                row = cov[cov["date"].astype(str) == str(date_q)]
                if not row.empty and bool(row.iloc[0]["anomaly"]):
                    coverage_note = f"Schedule anomaly: only {int(row.iloc[0]['n_games'])} games on {date_q}."
    except Exception:
        pass
    # Suppress anomaly note specifically for 2025-11-11 per UI request
    if str(date_q) == "2025-11-11":
        coverage_note = None
    # Archive dates list (daily_results) for navigation
    try:
        archive_dates: list[str] = []
        dr_dir = OUT / "daily_results"
        if dr_dir.exists():
            for p in sorted(dr_dir.glob("results_*.csv")):
                stem = p.stem
                if stem.startswith("results_"):
                    archive_dates.append(stem.replace("results_", ""))
    except Exception:
        archive_dates = []

    # Optional bootstrap/diagnostics hints when today's slate looks underpopulated
    show_bootstrap = False
    bootstrap_url = None
    fused_bootstrap_url = None
    show_diag = False
    diag_url = None
    try:
        tstr = _today_local().strftime("%Y-%m-%d")
        sel_is_today = (str(date_q) == tstr)
        has_pred_col = ("pred_total" in df.columns)
        has_any_preds = bool(has_pred_col and pd.to_numeric(df.get("pred_total"), errors="coerce").notna().any())
        show_bootstrap = bool(sel_is_today and (total_rows == 0 or not has_any_preds))
        if show_bootstrap:
            bootstrap_url = f"/api/bootstrap?date={str(date_q or tstr)}&provider=espn&force=1"
        # If very few rows on today, surface diagnostics and fused bootstrap
        if sel_is_today and (total_rows < 20):
            show_diag = True
            diag_url = f"/api/schedule-diagnostics?date={tstr}&refresh=1"
            fused_bootstrap_url = f"/api/bootstrap?date={tstr}&provider=fused&force=1&refresh=1"
        # Offer odds refresh shortcut when on today's slate
        refresh_odds_url = None
        if sel_is_today:
            refresh_odds_url = f"/api/refresh-odds?date={tstr}"
    except Exception:
        show_bootstrap = False
        refresh_odds_url = None

    return render_template(
        "index.html",
        rows=rows,
        total_rows=total_rows,
        date_val=date_q,
        top_picks=top_picks,
        accuracy=accuracy,
        uniform_note=uniform_note,
        dynamic_css=dynamic_css,
        coverage_note=coverage_note,
        results_note=results_note,
        show_edges=True,
        coverage=coverage_summary,
        archive_dates=archive_dates,
        show_bootstrap=show_bootstrap,
        bootstrap_url=bootstrap_url,
        show_diag=show_diag,
        diag_url=diag_url,
        fused_bootstrap_url=fused_bootstrap_url,
        refresh_odds_url=refresh_odds_url,
        removed_empty_rows=removed_empty_rows,
    )


@app.route("/dashboard")
def dashboard():
    metrics = _load_eval_metrics()
    return render_template("dashboard.html", metrics=metrics)

@app.route("/api/odds")
def api_odds():
    """Return per-game full-game totals odds quotes and aggregated market total.

    Structure: { rows: [ {game_id, market_total, commence_time, quotes: [ {book,total,price_over,price_under} ] } ], n: <int> }
    """
    date_q = (request.args.get("date") or "").strip()
    odds = _load_odds_joined(date_q or None)
    agg = _aggregate_full_game_totals(odds)
    rows = agg.to_dict(orient="records") if not agg.empty else []
    return jsonify({"n": len(rows), "rows": rows})


@app.route("/api/dates")
def api_dates():
    """Return sorted unique dates available from predictions (preferred) or games."""
    for name in ("predictions_week.csv", "predictions.csv", "predictions_all.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty and "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").dropna().unique().tolist()
            except Exception:
                dates = df["date"].dropna().astype(str).unique().tolist()
            dates = sorted(set(dates))
            return jsonify({"dates": dates, "source": name})
    # fallback to daily_results summaries
    dr_dir = OUT / "daily_results"
    if dr_dir.exists():
        try:
            dates = []
            for p in dr_dir.glob("summary_*.json"):
                d = p.stem.split("_")[-1]
                dates.append(d)
            dates = sorted(set(dates))
            if dates:
                return jsonify({"dates": dates, "source": "daily_results"})
        except Exception:
            pass
    # fallback to games
    for name in ("games_curr.csv", "games.csv", "games_all.csv"):
        df = _safe_read_csv(OUT / name)
        if not df.empty and "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").dropna().unique().tolist()
            except Exception:
                dates = df["date"].dropna().astype(str).unique().tolist()
            dates = sorted(set(dates))
            return jsonify({"dates": dates, "source": name})
    return jsonify({"dates": [], "source": None})


@app.route("/api/data-status")
def api_data_status():
    """Quick status of common output files and row counts."""
    files = [
        "games_curr.csv", "games_all.csv", "boxscores.csv", "boxscores_last2.csv",
        "features_curr.csv", "features_all.csv", "features_last2.csv",
        "predictions_week.csv", "predictions_all.csv", "predictions_last2.csv",
        "games_with_odds_today.csv", "games_with_closing.csv", "picks_clean.csv",
    ]
    st: Dict[str, Any] = {}
    for name in files:
        p = OUT / name
        if p.exists():
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = None
            st[name] = {"exists": True, "rows": n}
        else:
            st[name] = {"exists": False, "rows": None}
    acc = _load_accuracy_summary()
    if acc is not None:
        st["accuracy_summary"] = acc
    return jsonify(st)

@app.route("/api/health")
def api_health():
    """Health/status endpoint for deployment diagnostics."""
    try:
        out_dir = OUT
        daily_dir = out_dir / "daily_results"
        games_files = [p.name for p in out_dir.glob("games*.csv")]
        odds_files = [p.name for p in out_dir.glob("odds*.csv")]
        preds_files = [p.name for p in out_dir.glob("predictions*.csv")]
        stake_files = [p.name for p in out_dir.glob("stake_sheet*.csv")]
        daily_results = sorted(daily_dir.glob("results_*.csv")) if daily_dir.exists() else []
        recent_results = [p.stem.split("_")[1] for p in daily_results[-7:]] if daily_results else []
        # Today counts for quick diagnostics
        try:
            today_str = _today_local().strftime("%Y-%m-%d")
        except Exception:
            today_str = None
        games_today_rows = None
        preds_today_rows = None
        try:
            gm = _safe_read_csv(out_dir / "games_curr.csv")
            if not gm.empty and today_str and "date" in gm.columns:
                gm["date"] = gm["date"].astype(str)
                games_today_rows = int((gm["date"] == today_str).sum())
        except Exception:
            games_today_rows = None
        try:
            pr = _load_predictions_current()
            if not pr.empty and today_str and "date" in pr.columns:
                pr["date"] = pd.to_datetime(pr["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                preds_today_rows = int((pr["date"].astype(str) == today_str).sum())
        except Exception:
            preds_today_rows = None
        payload = {
            "status": "ok",
            "games_files": games_files,
            "odds_files": odds_files,
            "predictions_files": preds_files,
            "stake_files": stake_files,
            "daily_results_count": len(daily_results),
            "recent_result_dates": recent_results,
            "today": {
                "date": today_str,
                "games_today_rows": games_today_rows,
                "preds_today_rows": preds_today_rows,
            },
        }
        return jsonify(payload), 200
    except Exception as e:
        logger.exception("/api/health failure")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bootstrap", methods=["POST", "GET"])
def api_bootstrap():
    """On-demand bootstrap to populate today's (or requested) games, features & predictions.

    Query/JSON params:
      - date: YYYY-MM-DD (optional; defaults to today)
      - provider: espn|ncaa|fused (optional; default espn)
      - force: if '1' or true, run even if predictions already exist for date

    Returns JSON with counts and any warnings. Safe to call multiple times; will skip heavy work if data present.
    """
    if cli_daily_run is None:
        return jsonify({"status": "error", "message": "daily_run not importable in this environment"}), 500
    date_param = (request.args.get("date") or (request.json.get("date") if request.is_json else None) or "").strip()
    provider_param = (request.args.get("provider") or (request.json.get("provider") if request.is_json else None) or "espn").strip().lower()
    force_param = (request.args.get("force") or (request.json.get("force") if request.is_json else None) or "").strip().lower() in ("1","true","yes","force")
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None
    # use_cache override (refresh) param
    use_cache_param = (request.args.get("use_cache") or (request.json.get("use_cache") if request.is_json else None) or "").strip().lower()
    refresh_param = (request.args.get("refresh") or (request.json.get("refresh") if request.is_json else None) or "").strip().lower()
    # Interpret: if use_cache in ['0','false','no'] OR refresh=1 => disable cache
    disable_cache = use_cache_param in ("0","false","no") or refresh_param in ("1","true","yes")
    target_date = date_param or today_str
    if not target_date:
        return jsonify({"status": "error", "message": "Unable to resolve target date"}), 400

    # If predictions already exist for this date and not forced, short-circuit
    existing_preds = _load_predictions_current()
    already = False
    pred_rows_for_date = 0
    if not existing_preds.empty and "date" in existing_preds.columns:
        try:
            existing_preds["date"] = pd.to_datetime(existing_preds["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        pred_rows_for_date = int((existing_preds["date"].astype(str) == target_date).sum())
        already = pred_rows_for_date > 0
    if already and not force_param:
        return jsonify({
            "status": "ok",
            "message": f"Predictions already present for {target_date}; skipping bootstrap (use force=1 to override)",
            "date": target_date,
            "pred_rows": pred_rows_for_date,
            "skipped": True,
        }), 200

    # Execute daily_run pipeline minimally (avoid retraining / heavy operations on server)
    logs: list[str] = []
    try:
        # Wrap prints by temporarily redirecting stdout if desired; here we just call and rely on server logs
        cli_daily_run(
            date=target_date,
            season=dt.date.fromisoformat(target_date).year,
            region="us",
            provider=provider_param,
            threshold=2.0,
            default_price=-110.0,
            retrain=False,
            segment="none",
            conf_map=None,
            use_cache=(not disable_cache),
            preseason_weight=0.0,
            preseason_only_sparse=True,
            apply_guardrails=True,
            half_ratio=0.485,
            auto_train_halves=False,
            halves_models_dir=OUT / "models_halves",
            enable_ort=False,
            accumulate_schedule=True,
            accumulate_predictions=True,
        )
    except Exception as e:
        logger.exception("Bootstrap daily_run failed")
        return jsonify({"status": "error", "message": f"daily_run failed: {e}"}), 500

    # Reload artifacts to report counts
    games_after = _safe_read_csv(OUT / "games_curr.csv")
    preds_after = _load_predictions_current()
    if "date" in preds_after.columns:
        try:
            preds_after["date"] = pd.to_datetime(preds_after["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    pred_rows_after = int(preds_after[preds_after.get("date","") == target_date].shape[0]) if not preds_after.empty and "date" in preds_after.columns else 0
    game_rows_after = int(games_after[games_after.get("date","") == target_date].shape[0]) if not games_after.empty and "date" in games_after.columns else len(games_after)

    # Optional auto-fallback: if today's slate looks sparse with provider 'espn' or 'ncaa', try fused
    fallback_info: dict[str, Any] = {"triggered": False}
    try:
        min_thresh = int(os.environ.get("NCAAB_MIN_TODAY_GAMES", "25"))
    except Exception:
        min_thresh = 25
    if (
        provider_param != "fused"
        and target_date == today_str
        and isinstance(game_rows_after, int)
        and game_rows_after < min_thresh
    ):
        try:
            cli_daily_run(
                date=target_date,
                season=dt.date.fromisoformat(target_date).year,
                region="us",
                provider="fused",
                threshold=2.0,
                default_price=-110.0,
                retrain=False,
                segment="none",
                conf_map=None,
                use_cache=(not disable_cache),
                preseason_weight=0.0,
                preseason_only_sparse=True,
                apply_guardrails=True,
                half_ratio=0.485,
                auto_train_halves=False,
                halves_models_dir=OUT / "models_halves",
                enable_ort=False,
                accumulate_schedule=True,
                accumulate_predictions=True,
            )
            # Recompute counts after fallback
            games_after2 = _safe_read_csv(OUT / "games_curr.csv")
            preds_after2 = _load_predictions_current()
            if "date" in preds_after2.columns:
                try:
                    preds_after2["date"] = pd.to_datetime(preds_after2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            pred_rows_after2 = int(preds_after2[preds_after2.get("date","") == target_date].shape[0]) if not preds_after2.empty and "date" in preds_after2.columns else 0
            game_rows_after2 = int(games_after2[games_after2.get("date","") == target_date].shape[0]) if not games_after2.empty and "date" in games_after2.columns else len(games_after2)
            fallback_info = {
                "triggered": True,
                "reason": f"sparse slate ({game_rows_after}) with provider={provider_param}; retried fused",
                "prev": {"game_rows": game_rows_after, "pred_rows": pred_rows_after, "provider": provider_param},
                "after": {"game_rows": game_rows_after2, "pred_rows": pred_rows_after2, "provider": "fused"},
            }
            # Promote fused counts in primary response
            game_rows_after = game_rows_after2
            pred_rows_after = pred_rows_after2
            provider_param = f"{provider_param}->fused"
        except Exception as e:
            logger.exception("Fallback to fused provider failed")
            fallback_info = {
                "triggered": True,
                "error": str(e),
                "prev": {"game_rows": game_rows_after, "pred_rows": pred_rows_after, "provider": provider_param},
            }

    return jsonify({
        "status": "ok",
        "message": f"Bootstrap complete for {target_date}",
        "date": target_date,
        "pred_rows": pred_rows_after,
        "game_rows": game_rows_after,
        "provider": provider_param,
        "forced": force_param,
        "cache_used": not disable_cache,
        "fallback": fallback_info,
    }), 200


@app.route("/api/schedule-diagnostics")
def api_schedule_diagnostics():
    """Fetch today's (or requested date) games from ESPN + NCAA live (cache bypass optional) and report counts.

    Params:
      - date: YYYY-MM-DD (default today)
      - refresh=1 to bypass adapter cache
    Returns JSON with counts and sample team lists to help diagnose under-coverage.
    Does not write any output files.
    """
    date_param = (request.args.get("date") or "").strip()
    refresh = (request.args.get("refresh") or "").strip().lower() in ("1","true","yes")
    try:
        target_date = dt.date.fromisoformat(date_param) if date_param else _today_local()
    except Exception:
        return jsonify({"status": "error", "message": f"Invalid date: {date_param}"}), 400
    # Import adapters locally to avoid global import errors
    try:
        from ncaab_model.data.adapters.espn_scoreboard import _fetch_day as _espn_fetch, _parse_games as _espn_parse  # type: ignore
        from ncaab_model.data.adapters.ncaa_scoreboard import _fetch_scoreboard as _ncaa_fetch, _parse_games as _ncaa_parse  # type: ignore
    except Exception as e:
        return jsonify({"status": "error", "message": f"Adapter import failed: {e}"}), 500
    # Fetch raw payloads (optionally bypass cache)
    espn_payload = _espn_fetch(target_date, use_cache=not refresh)
    ncaa_payload = _ncaa_fetch(target_date, use_cache=not refresh)
    espn_games = _espn_parse(target_date, espn_payload) if espn_payload else []
    ncaa_games = _ncaa_parse(target_date, ncaa_payload) if ncaa_payload else []
    # Build fused keys (normalized simple lower-case names)
    def _norm(t: str | None) -> str | None:
        if not t:
            return None
        return "".join(ch for ch in t.lower() if ch.isalnum() or ch in (" ","-"))
    espn_set = {(_norm(g.home_team), _norm(g.away_team)) for g in espn_games}
    ncaa_set = {(_norm(g.home_team), _norm(g.away_team)) for g in ncaa_games}
    fused = espn_set | ncaa_set
    missing_from_espn = sorted(list(ncaa_set - espn_set))
    missing_from_ncaa = sorted(list(espn_set - ncaa_set))
    return jsonify({
        "status": "ok",
        "date": target_date.isoformat(),
        "refresh": refresh,
        "espn_count": len(espn_games),
        "ncaa_count": len(ncaa_games),
        "fused_unique_matchups": len(fused),
        "missing_from_espn": missing_from_espn[:25],
        "missing_from_ncaa": missing_from_ncaa[:25],
    }), 200


@app.route("/recommendations")
def recommendations():
    picks = _load_picks()
    if not picks.empty and "date" in picks.columns:
        try:
            picks["date"] = pd.to_datetime(picks["date"])
            picks = picks.sort_values(["date", "abs_edge" if "abs_edge" in picks.columns else "edge"], ascending=[True, False])
        except Exception:
            pass
    # Ensure projected scores columns exist to satisfy template even if margin absent
    if not picks.empty and "pred_total" in picks.columns:
        if "pred_margin" not in picks.columns:
            # Assume zero margin if not available (neutral projection)
            picks["pred_margin"] = 0.0
        try:
            # Compute projected home/away from total + margin
            pt = pd.to_numeric(picks["pred_total"], errors="coerce")
            pm = pd.to_numeric(picks["pred_margin"], errors="coerce")
            picks["proj_home"] = (pt + pm) / 2.0
            picks["proj_away"] = pt - picks["proj_home"]
        except Exception:
            picks["proj_home"] = None
            picks["proj_away"] = None
    rows = picks.to_dict(orient="records") if not picks.empty else []
    return render_template("recommendations.html", rows=rows, total_rows=len(rows))


@app.route("/picks-raw")
def picks_raw_page():
    """Render expanded multi-market picks with basic filters (date, market, period)."""
    p = OUT / "picks_raw.csv"
    df = _safe_read_csv(p)
    date_q = (request.args.get("date") or "").strip()
    market_q = (request.args.get("market") or "").strip().lower()
    period_q = (request.args.get("period") or "").strip().lower()
    if df.empty:
        return render_template("recommendations.html", rows=[], total_rows=0)
    # Normalize columns
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    if date_q and "date" in df.columns:
        df = df[df["date"] == date_q]
    if market_q and "market" in df.columns:
        df = df[df["market"].astype(str).str.lower() == market_q]
    if period_q and "period" in df.columns:
        df = df[df["period"].astype(str).str.lower() == period_q]
    # Sort by absolute edge desc
    if "edge" in df.columns:
        try:
            df["_abs_edge"] = pd.to_numeric(df["edge"], errors="coerce").abs()
            df = df.sort_values(["_abs_edge"], ascending=[False])
        except Exception:
            pass
    # Branding enrichment for quick display
    branding = _load_branding_map()
    def _enrich_row(r: dict[str, Any]) -> dict[str, Any]:
        for side in ["home","away"]:
            key = normalize_name(str(r.get(f"{side}_team") or ""))
            b = branding.get(key) or {}
            r[f"{side}_key"] = key
            r[f"{side}_logo"] = b.get("logo")
        return r
    rows = [_enrich_row(r) for r in df.to_dict(orient="records")]
    return render_template("picks_raw.html", rows=rows, total_rows=len(rows), date_val=date_q, market_val=market_q, period_val=period_q)


@app.route("/finals")
def finals():
    """Season-to-date final scores table, aggregated from daily_results.*"""
    df = _load_all_finals(limit=2000)
    rows = df.to_dict(orient="records") if not df.empty else []
    # Basic stats: count, last date range
    date_min = df["date"].min() if "date" in df.columns and not df.empty else None
    date_max = df["date"].max() if "date" in df.columns and not df.empty else None
    return render_template("finals.html", rows=rows, total_rows=len(rows), date_min=date_min, date_max=date_max)


@app.route("/api/recommendations")
def api_recommendations():
    picks = _load_picks()
    rows = picks.to_dict(orient="records") if not picks.empty else []
    return jsonify({"rows": len(rows), "data": rows})


@app.route("/api/picks_raw")
def api_picks_raw():
    """Expose expanded picks (totals/spreads/moneyline, incl. halves) from outputs/picks_raw.csv.

    Optional query: ?date=YYYY-MM-DD filters rows by date column when present.
    Enriches with team branding keys and logos if available.
    """
    p = OUT / "picks_raw.csv"
    df = _safe_read_csv(p)
    if df.empty:
        return jsonify({"rows": 0, "data": []})
    # Filter by date if provided
    date_q = (request.args.get("date") or "").strip()
    if date_q and "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df[df["date"] == date_q]
        except Exception:
            pass
    # Branding enrichment
    branding = _load_branding_map()
    def _enrich(row: dict[str, Any]) -> dict[str, Any]:
        for side in ["home","away"]:
            t = str(row.get(f"{side}_team") or "")
            key = normalize_name(t)
            b = branding.get(key) or {}
            row[f"{side}_key"] = key
            row[f"{side}_logo"] = b.get("logo")
            row[f"{side}_color"] = b.get("primary") or b.get("secondary")
            row[f"{side}_text_color"] = b.get("text") or "#ffffff"
        return row
    rows = [_enrich(r) for r in df.to_dict(orient="records")]
    return jsonify({"rows": len(rows), "data": rows})


@app.route("/api/accuracy")
def api_accuracy():
    acc_json = OUT / "eval_last2" / "accuracy_summary.json"
    if acc_json.exists():
        try:
            data = json.loads(acc_json.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "no accuracy report found"}), 404


@app.route("/api/finalize-day")
def api_finalize_day():
    """Run finalize-day for a given date. Example: /api/finalize-day?date=YYYY-MM-DD

    Returns JSON with status and message. This invokes the same logic as the CLI command.
    """
    date_q = (request.args.get("date") or "").strip()
    if not date_q:
        return jsonify({"ok": False, "error": "missing date"}), 400
    if cli_finalize_day is None:
        return jsonify({"ok": False, "error": "finalize-day not available"}), 500
    try:
        cli_finalize_day(date=date_q)  # type: ignore
        return jsonify({"ok": True, "date": date_q})
    except Exception as e:
        # Typer Exit raises BaseException; catch broadly
        if typer is not None and isinstance(e, typer.Exit):  # type: ignore
            # Treat non-zero code as error
            code = getattr(e, "exit_code", 0)
            if code and int(code) != 0:
                return jsonify({"ok": False, "date": date_q, "error": f"exit {code}"}), 500
            return jsonify({"ok": True, "date": date_q, "note": "completed with exit"})
        return jsonify({"ok": False, "date": date_q, "error": str(e)}), 500

    @app.route("/api/refresh-odds")
    def api_refresh_odds():
        """Refresh today's (or provided date's) odds snapshot and rebuild last odds merge.

        Query params:
          - date=YYYY-MM-DD (optional; defaults to local today)
          - region=us (optional TheOddsAPI region)
          - markets=comma,separated markets (defaults spreads,totals,h2h)
          - tolerance=seconds skew tolerance for last selection (default 60)

        Steps:
          1. Fetch current odds snapshot for the date into odds_history/odds_<date>.csv (multi-region not yet).
          2. Rebuild last_odds.csv across odds_history directory.
          3. Join last odds for that date into games_with_last_<date>.csv (any-D1, allow partial).
          4. Refresh master games_with_last.csv for that single date (append/replace row subset).
          5. Return coverage counts.
        """
        date_q = (request.args.get("date") or "").strip()
        region = (request.args.get("region") or "us").strip()
        markets = (request.args.get("markets") or "h2h,spreads,totals").strip()
        tolerance = int((request.args.get("tolerance") or "60").strip() or 60)
        # Default date = local today (same logic as schedule timezone)
        if not date_q:
            try:
                from zoneinfo import ZoneInfo as _ZoneInfo
                tz_name = os.getenv("NCAAB_SCHEDULE_TZ", "America/New_York")
                date_q = dt.datetime.now(_ZoneInfo(tz_name)).date().isoformat()
            except Exception:
                date_q = dt.date.today().isoformat()
        # Validate games file exists
        games_file = OUT / f"games_{date_q}.csv"
        if not games_file.exists():
            return jsonify({"ok": False, "error": f"games file missing for {date_q}"}), 404
        # 1. Fetch snapshot (single region)
        try:
            if TheOddsAPIAdapter is None:  # type: ignore
                raise RuntimeError("Odds adapter unavailable")
            adapter = TheOddsAPIAdapter(region=region)  # type: ignore
            rows = []
            for row in adapter.iter_current_odds_expanded(markets=markets, date_iso=date_q):  # type: ignore
                rows.append(row.model_dump())
            if rows:
                oh_dir = OUT / "odds_history"; oh_dir.mkdir(parents=True, exist_ok=True)
                snap_path = oh_dir / f"odds_{date_q}.csv"
                pd.DataFrame(rows).to_csv(snap_path, index=False)
            else:
                return jsonify({"ok": False, "error": "no odds rows fetched"}), 500
        except Exception as e:
            return jsonify({"ok": False, "error": f"fetch failed: {e}"}), 500
        # 2. Rebuild last_odds.csv
        try:
            from ncaab_model.data.odds_closing import make_last_odds as _make_last
            last_path = OUT / "last_odds.csv"
            _make_last(OUT / "odds_history", last_path, tolerance_seconds=tolerance)  # type: ignore
        except Exception as e:
            return jsonify({"ok": False, "error": f"make_last_odds failed: {e}"}), 500
        # 3. Join for date
        try:
            from ncaab_model.data.join_closing import join_games_with_closing as _join
            last_df = pd.read_csv(last_path)
            games_df = pd.read_csv(games_file)
            # Any-D1 filter (re-apply minimal subset like CLI) - optional
            try:
                d1 = pd.read_csv(DATA / "d1_conferences.csv")
                from ncaab_model.data.merge_odds import normalize_name as _norm
                d1set = set(d1['team'].astype(str).map(_norm))
                games_df['_home_ok'] = games_df['home_team'].astype(str).map(_norm).isin(d1set)
                games_df['_away_ok'] = games_df['away_team'].astype(str).map(_norm).isin(d1set)
                games_df = games_df[games_df['_home_ok'] | games_df['_away_ok']].copy()
                games_df.drop(columns=['_home_ok','_away_ok'], inplace=True, errors='ignore')
            except Exception:
                pass
            merged_date = _join(games_df, last_df)
            per_date_out = OUT / f"games_with_last_{date_q}.csv"
            merged_date.to_csv(per_date_out, index=False)
        except Exception as e:
            return jsonify({"ok": False, "error": f"join failed: {e}"}), 500
        # 4. Refresh master: replace rows for date
        master_path = OUT / "games_with_last.csv"
        try:
            if master_path.exists():
                master_df = pd.read_csv(master_path)
                if 'date' in master_df.columns:
                    master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
                # Remove existing rows for this date (match by date string if date parsing missing)
                mask_remove = (master_df.get('date').dt.strftime('%Y-%m-%d') == date_q) if 'date' in master_df.columns else (master_df.get('date_game', pd.Series(dtype=str)).astype(str) == date_q)
                master_df = master_df[~mask_remove]
                # Append new
                append_df = merged_date.copy()
                master_df = pd.concat([master_df, append_df], ignore_index=True)
            else:
                master_df = merged_date.copy()
            master_df.to_csv(master_path, index=False)
        except Exception as e:
            return jsonify({"ok": False, "error": f"master refresh failed: {e}"}), 500
        # 5. Coverage counts
        try:
            covered_exact = merged_date['game_id'].nunique() if 'game_id' in merged_date.columns else 0
            games_total = len(games_df)
            return jsonify({
                "ok": True,
                "date": date_q,
                "snapshot_rows": len(rows),
                "merged_rows": len(merged_date),
                "games_total": games_total,
                "covered_exact_games": covered_exact,
            })
        except Exception as e:
            return jsonify({"ok": True, "date": date_q, "note": "completed", "coverage_error": str(e)})


@app.route("/api/results")
def api_results():
    """Structured per-date results (scores + predictions + market lines).

    Query params:
      - date=YYYY-MM-DD (required)
      - use_daily=1 to force using daily_results even if scores/preds missing
      - cols=comma,separated to restrict returned columns

    Response JSON:
      {
        "ok": true,
        "meta": { date, daily_used, results_pending, n_rows, n_finals, n_pending, all_final, columns },
        "rows": [ { game_id, home_team, away_team, home_score, away_score, pred_total, market_total, actual_total, edge_total, ... } ]
      }
    """
    date_q = (request.args.get("date") or "").strip()
    if not date_q:
        return jsonify({"ok": False, "error": "missing date"}), 400
    force_daily = (request.args.get("use_daily") or "").strip() in ("1","true","yes")
    df, meta = _build_results_df(date_q, force_use_daily=force_daily)
    if df.empty:
        return jsonify({"ok": True, "meta": meta, "rows": []})
    # Optional column restriction
    cols_req = (request.args.get("cols") or "").strip()
    if cols_req:
        want = [c.strip() for c in cols_req.split(",") if c.strip()]
        have = [c for c in want if c in df.columns]
        if have:
            df = df[have]
    # Ensure primitive types for JSON (avoid numpy types)
    rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.generic,)):
                try:
                    v = v.item()
                except Exception:
                    v = float(v) if hasattr(v, "__float__") else str(v)
            if isinstance(v, (dt.datetime, dt.date)):
                v = str(v)
            clean[k] = v
        rows.append(clean)
    meta["returned_columns"] = list(df.columns)
    return jsonify({"ok": True, "meta": meta, "rows": rows})


# ---------------- New Pages: Stake Sheet, Coverage, Calibration -----------------

@app.route("/stake-sheet")
def stake_sheet_page():
    """Render stake sheet variants and comparison.

    Query params:
      ?view=orig|cal|compare (default orig)
    """
    view = (request.args.get("view") or "orig").strip().lower()
    date_q = (request.args.get("date") or "").strip()
    if view not in {"orig","cal","compare"}:
        view = "orig"
    df = _load_stake_sheet(view, date_q or None)
    # Ensure expected columns exist to avoid Jinja UndefinedError
    expected_common = [
        "game_id","home_team","away_team","market","selection","line","price",
        "prob","kelly_fraction","ev","stake","book","date",
        # deltas (safe to include for all views)
        "delta_prob","delta_kelly","delta_ev","delta_stake",
    ]
    expected_compare = [
        "prob_orig","prob_cal","kelly_fraction_orig","kelly_fraction_cal",
        "ev_orig","ev_cal","stake_orig","stake_cal",
    ]
    if df.empty:
        rows: list[dict[str, Any]] = []
    else:
        for col in expected_common + (expected_compare if view == "compare" else []):
            if col not in df.columns:
                df[col] = None
        rows = df.to_dict(orient="records")
    summary = _summarize_stake_sheet(df)
    return render_template(
        "stake_sheet.html",
        rows=rows,
        total_rows=len(rows),
        view=view,
        date_val=date_q,
        summary=summary,
    )


@app.route("/download/stake")
def download_stake():
    """Download stake sheet CSV for the requested view.

    Query: ?view=orig|cal|compare
    """
    view = (request.args.get("view") or "orig").strip().lower()
    date_q = (request.args.get("date") or "").strip()
    def _cands(kind: str) -> list[Path]:
        lst: list[Path] = []
        if date_q:
            if kind == "orig":
                lst += [OUT / f"stake_sheet_{date_q}.csv"]
            elif kind == "cal":
                lst += [OUT / f"stake_sheet_{date_q}_cal.csv"]
            elif kind == "compare":
                lst += [OUT / f"stake_sheet_{date_q}_compare.csv"]
        if kind == "orig":
            lst += [OUT / "stake_sheet_today.csv"]
        elif kind == "cal":
            lst += [OUT / "stake_sheet_today_cal.csv"]
        elif kind == "compare":
            lst += [OUT / "stake_sheet_today_compare.csv"]
        return lst
    p = None
    for cand in _cands(view):
        if cand.exists():
            p = cand
            break
    if not p or not p.exists():
        return jsonify({"error": "file not found"}), 404
    try:
        return send_file(str(p), as_attachment=True, download_name=p.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/coverage")
def coverage_page():
    snap = _compute_coverage_snapshot()
    return render_template("coverage.html", snap=snap)


@app.route("/calibration")
def calibration_page():
    artifact = _load_calibration_artifact()
    compare_df = _load_stake_sheet("compare")
    compare_summary = _summarize_stake_sheet(compare_df)
    # Derive aggregate deltas if comparison present
    deltas: dict[str, Any] = {}
    if not compare_df.empty:
        for col in ["delta_prob","delta_kelly","delta_ev","delta_stake"]:
            if col in compare_df.columns:
                try:
                    ser = pd.to_numeric(compare_df[col], errors="coerce")
                    deltas[f"mean_{col}"] = float(ser.dropna().mean())
                    deltas[f"median_{col}"] = float(ser.dropna().median())
                    deltas[f"sum_{col}"] = float(ser.dropna().sum())
                except Exception:
                    continue
    rows_compare = compare_df.head(50).to_dict(orient="records") if not compare_df.empty else []
    return render_template(
        "calibration.html",
        artifact=artifact,
        compare_rows=rows_compare,
        compare_summary=compare_summary,
        deltas=deltas,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=True)
