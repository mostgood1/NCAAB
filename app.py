from __future__ import annotations

import json
from typing import Any, Iterable

# Feature fallback utility (rolling averages) ----------------------------------------------------
def _feature_fallback_enrich(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing core ratings (off/def/tempo) using rolling averages by team slug.

    Assumptions:
      - Columns may include home_off_rating, away_off_rating, etc.
      - A team identifier exists (home_team, away_team) in source frames; for features we attempt
        to derive per-team perspective by exploding rows.
    """
    if feat_df.empty:
        return feat_df
    needed = [
        'home_off_rating','away_off_rating','home_def_rating','away_def_rating','home_tempo_rating','away_tempo_rating'
    ]
    present = [c for c in needed if c in feat_df.columns]
    if not present:
        return feat_df
    # Build per-team history frame
    cols_keep = [c for c in ['game_id','date','home_team','away_team'] if c in feat_df.columns]
    hist_rows: list[dict[str, Any]] = []
    for _, r in feat_df.iterrows():
        for side in ['home','away']:
            team = r.get(f'{side}_team')
            if not team:
                continue
            row = {
                'team': team,
                'date': r.get('date'),
                'game_id': r.get('game_id'),
            }
            for metric in ['off','def','tempo']:
                col = f'{side}_{metric}_rating'
                if col in feat_df.columns:
                    row[metric] = r.get(col)
            hist_rows.append(row)
    hist = pd.DataFrame(hist_rows)
    if hist.empty or 'team' not in hist.columns:
        return feat_df
    try:
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    except Exception:
        pass
    # Compute rolling means per team
    filled_map: dict[str, dict[str, float]] = {}
    for team, g in hist.groupby('team'):
        g2 = g.sort_values('date')
        for metric in ['off','def','tempo']:
            if metric in g2.columns:
                vals = pd.to_numeric(g2[metric], errors='coerce')
                if vals.notna().any():
                    filled_map.setdefault(team, {})[metric] = float(vals.dropna().rolling(window=5, min_periods=1).mean().iloc[-1])
    # Apply fallback where ratings missing
    for side in ['home','away']:
        team_col = f'{side}_team'
        for metric in ['off','def','tempo']:
            col = f'{side}_{metric}_rating'
            if col in feat_df.columns:
                ser = pd.to_numeric(feat_df[col], errors='coerce')
                miss_mask = ser.isna()
                if miss_mask.any():
                    teams = feat_df[team_col].astype(str)
                    feat_df.loc[miss_mask, col] = [filled_map.get(t, {}).get(metric, np.nan) for t in teams[miss_mask]]
            else:
                # create column entirely from fallback map
                teams = feat_df[team_col].astype(str)
                feat_df[col] = [filled_map.get(t, {}).get(metric, np.nan) for t in teams]
    # Recompute tempo sum if components exist
    if {'home_tempo_rating','away_tempo_rating'}.issubset(feat_df.columns):
        feat_df['tempo_rating_sum'] = pd.to_numeric(feat_df['home_tempo_rating'], errors='coerce') + pd.to_numeric(feat_df['away_tempo_rating'], errors='coerce')
    return feat_df
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
import re

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

# Global diagnostic state variables
_PREDICTIONS_SOURCE_PATH: str | None = None
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
    return out


def _resolve_outputs_dir() -> Path:
    """Determine outputs directory using env override, settings, or fallback."""
    env_dir = os.getenv("NCAAB_OUTPUTS_DIR", "").strip()
    if env_dir:
        try:
            p = Path(env_dir).resolve()
            if p.exists() and p.is_dir():
                logger.info("Using outputs dir (env): %s", p)
                return p
        except Exception:
            pass
    # settings.outputs_dir may be a Path-like or string
    try:
        p2 = Path(str(settings.outputs_dir))
        if p2.exists() and p2.is_dir():
            logger.info("Using outputs dir (settings): %s", p2)
            return p2
    except Exception:
        pass
    # common fallbacks
    for cand in [ROOT / "outputs", ROOT]:
        try:
            if cand.exists() and cand.is_dir():
                logger.warning("Using fallback outputs dir: %s", cand)
                return cand
        except Exception:
            continue
    return ROOT

OUT = _resolve_outputs_dir()
                    


# Optional custom team map: allow overriding/augmenting normalize_name via CSV
_CUSTOM_TEAM_SLUG_MAP: dict[str, str] | None = None

def _load_custom_team_map() -> dict[str, str]:
    """Load data/team_map.csv into a slug->slug mapping for canonicalization.

    CSV columns expected: 'raw', 'canonical' (flexible casing). We canonicalize both sides
    through normalize_name to build a stable mapping that can catch provider variants.
    """
    global _CUSTOM_TEAM_SLUG_MAP
    if _CUSTOM_TEAM_SLUG_MAP is not None:
        return _CUSTOM_TEAM_SLUG_MAP
    mapping: dict[str, str] = {}
    try:
            path = settings.data_dir / "team_map.csv"
            if path.exists():
                df = pd.read_csv(path)
                cols = {c.lower().strip(): c for c in df.columns}
                raw_col = cols.get("raw") or list(df.columns)[0]
                can_col = cols.get("canonical") or (list(df.columns)[1] if len(df.columns) > 1 else raw_col)
                for _, r in df.iterrows():
                    raw = str(r.get(raw_col) or "").strip()
                    can = str(r.get(can_col) or "").strip()
                    if not raw or not can:
                        continue
                    # IMPORTANT: avoid recursive _canon_slug -> _load_custom_team_map calls; use base normalize_name here.
                    raw_slug = normalize_name(raw)
                    can_slug = normalize_name(can)
                    if raw_slug and can_slug and raw_slug != can_slug:
                        mapping[raw_slug] = can_slug
    except Exception:
        mapping = {}
    _CUSTOM_TEAM_SLUG_MAP = mapping
    return mapping

def _canon_slug(name: str) -> str:
    """Normalize a team name to a canonical slug, applying custom map if available."""
    slug = normalize_name(name)
    m = _load_custom_team_map()
    return m.get(slug, slug)


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
    """Load predictions file using priority order with environment override.

    Priority:
      1) NCAAB_PREDICTIONS_FILE (absolute or relative to OUT)
      2) OUT/predictions_week.csv
      3) OUT/predictions.csv
      4) OUT/predictions_all.csv
      5) OUT/predictions_last2.csv
      6) Largest non-empty OUT/predictions_*.csv (by size)
    """
    global _PREDICTIONS_SOURCE_PATH
    _PREDICTIONS_SOURCE_PATH = None
    env_path = (os.getenv("NCAAB_PREDICTIONS_FILE") or "").strip()
    today_str = None
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None
    candidates: list[Path] = []
    # 1) Explicit env override
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = OUT / env_path
        candidates.append(p)
    # 2) Date-specific file for today first (if exists) to avoid picking a large historical file
    if today_str:
        # Prefer blended predictions for today if present
        candidates.append(OUT / f"predictions_blend_{today_str}.csv")
        candidates.append(OUT / f"predictions_{today_str}.csv")
    # 3) Conventional aggregate files
    for name in ("predictions_blend.csv", "predictions_week.csv", "predictions.csv", "predictions_all.csv", "predictions_last2.csv"):
        candidates.append(OUT / name)
    # 4) All predictions_*.csv (other dates) ordered by size so richest historical fallback last
    try:
        globbed = list(OUT.glob("predictions_*.csv"))
        # Put today's file (if present) at front; others sorted by size desc
        globbed_other = [p for p in globbed if not today_str or p.name != f"predictions_{today_str}.csv"]
        globbed_other = sorted(globbed_other, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        for p in globbed_other:
            if p not in candidates:
                candidates.append(p)
    except Exception:
        pass
    loaded: list[pd.DataFrame] = []
    chosen_path: Path | None = None
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if not df.empty:
                    # If this is a dated file and matches today, select immediately
                    if today_str and p.name in {f"predictions_{today_str}.csv", f"predictions_blend_{today_str}.csv"}:
                        logger.info("Loaded today's predictions from: %s (rows=%s)", p, len(df))
                        _PREDICTIONS_SOURCE_PATH = str(p)
                        return df
                    if chosen_path is None:
                        chosen_path = p
                        loaded.append(df)
        except Exception:
            continue
    if loaded:
        df = loaded[0]
        logger.info("Loaded predictions fallback from: %s (rows=%s)", chosen_path, len(df))
        _PREDICTIONS_SOURCE_PATH = str(chosen_path)
        return df
    logger.warning("No predictions file found or all empty in %s; proceeding without predictions", OUT)
    return pd.DataFrame()


def _load_model_predictions(date_str: str | None = None) -> pd.DataFrame:
    """Load model-only prediction outputs produced by inference harness.

    File precedence:
      1) Explicit env var NCAAB_MODEL_PREDICTIONS_FILE (absolute or relative to OUT)
      2) OUT/predictions_model_<date>.csv when date provided
      3) OUT/predictions_model_<today>.csv
      4) Most recently modified OUT/predictions_model_*.csv (fallback)
    Returns empty DataFrame if none found or all empty.
    Sets global _MODEL_PREDICTIONS_SOURCE_PATH for diagnostics.
    Expected columns (if present): game_id, pred_total_model, pred_margin_model, date
    """
    global _MODEL_PREDICTIONS_SOURCE_PATH
    _MODEL_PREDICTIONS_SOURCE_PATH = None
    env_path = (os.getenv("NCAAB_MODEL_PREDICTIONS_FILE") or "").strip()
    today_str = None
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None
    candidates: list[Path] = []
    # 1) Env override
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = OUT / env_path
        candidates.append(p)
    # 2) Explicit date (prefer calibrated then raw)
    if date_str:
        candidates.append(OUT / f"predictions_model_calibrated_{date_str}.csv")
        candidates.append(OUT / f"predictions_model_{date_str}.csv")
    # 3) Today (prefer calibrated)
    if today_str and (not date_str or date_str != today_str):
        candidates.append(OUT / f"predictions_model_calibrated_{today_str}.csv")
        candidates.append(OUT / f"predictions_model_{today_str}.csv")
    # 4) Any historical model predictions – choose newest non-empty
    try:
        globbed = list(OUT.glob("predictions_model_calibrated_*.csv")) + list(OUT.glob("predictions_model_*.csv"))
        globbed = sorted(globbed, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for p in globbed:
            if p not in candidates:
                candidates.append(p)
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if not df.empty:
                    # Normalize calibrated column names to generic model columns for downstream merging
                    if 'pred_total_model' not in df.columns and 'pred_total_calibrated' in df.columns:
                        try:
                            df['pred_total_model'] = pd.to_numeric(df['pred_total_calibrated'], errors='coerce')
                        except Exception:
                            df['pred_total_model'] = df['pred_total_calibrated']
                    if 'pred_margin_model' not in df.columns and 'pred_margin_calibrated' in df.columns:
                        try:
                            df['pred_margin_model'] = pd.to_numeric(df['pred_margin_calibrated'], errors='coerce')
                        except Exception:
                            df['pred_margin_model'] = df['pred_margin_calibrated']
                    # Fallback: some raw inference artifacts may store columns as pred_total / pred_margin only
                    if 'pred_total_model' not in df.columns and 'pred_total' in df.columns:
                        df['pred_total_model'] = pd.to_numeric(df['pred_total'], errors='coerce')
                    if 'pred_margin_model' not in df.columns and 'pred_margin' in df.columns:
                        df['pred_margin_model'] = pd.to_numeric(df['pred_margin'], errors='coerce')
                    _MODEL_PREDICTIONS_SOURCE_PATH = str(p)
                    logger.info("Loaded model predictions from: %s (rows=%s, cols=%s)", p, len(df), list(df.columns))
                    return df
        except Exception:
            continue
    logger.info("No model predictions file resolved for date=%s (env=%s)", date_str, env_path)
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
    candidates = [OUT / "calibration" / "artifact_totals.json", OUT / "calibration_totals.json"]
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

            # Declare globals early to avoid Python complaining about prior usage before global statement
            global _PREDICTIONS_SOURCE_PATH, _MODEL_PREDICTIONS_SOURCE_PATH
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
            return pd.read_csv(p)
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
        # Drop comment/header lines beginning with '#'
        df = df[~df[team_col].astype(str).str.strip().str.startswith('#')]
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

    # Declare globals early for prediction source path tracking
    global _PREDICTIONS_SOURCE_PATH, _MODEL_PREDICTIONS_SOURCE_PATH

    # Define today string once for consistent downstream comparisons
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None

    games = _load_games_current()
    preds = _load_predictions_current()
    odds = _load_odds_joined(date_q)
    model_preds = _load_model_predictions(date_q if date_q else None)
    # Initial diagnostics snapshot before filtering
    diag_enabled = (request.args.get("diag") or "").strip().lower() in ("1","true","yes")
    pipeline_stats: dict[str, Any] = {
        "date_param": date_q,
        "games_load_rows": len(games),
        "preds_load_rows": len(preds),
        "odds_load_rows": len(odds),
        "model_preds_load_rows": len(model_preds),
        "outputs_dir": str(OUT),
    }
    team_variance_total: dict[str, float] | None = None
    team_variance_margin: dict[str, float] | None = None
    # Backtest metrics ingestion (totals/spread/moneyline) if precomputed JSON exists
    # Expected file: outputs/backtest_metrics_<date>.json produced by daily_backtest script.
    try:
        bt_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        bt_path = OUT / f'backtest_metrics_{bt_date}.json'
        if bt_path.exists():
            import json as _json
            bt_payload = _json.loads(bt_path.read_text(encoding='utf-8'))
            pipeline_stats['backtest_ingested'] = True
            pipeline_stats['backtest_date'] = bt_payload.get('date')
            pipeline_stats['backtest_generated_at'] = bt_payload.get('generated_at')
            def _lift(prefix: str, obj: Any):
                if isinstance(obj, dict):
                    for k,v in obj.items():
                        # Avoid extremely large nested structures; only primitive scalars retained
                        if isinstance(v, (int,float,str)) or v is None:
                            pipeline_stats[f'{prefix}_{k}'] = v
            for key in ('totals_closing','spread_closing','moneyline_closing'):
                if key in bt_payload:
                    _lift(key, bt_payload[key])
        else:
            pipeline_stats['backtest_ingested'] = False
    except Exception:
        pipeline_stats['backtest_error'] = True
    # Proper scoring rules (CRPS / log-likelihood) ingestion if JSON exists
    try:
        scoring_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        scoring_path = OUT / f'scoring_{scoring_date}.json'
        if scoring_path.exists():
            import json as _json
            sc_payload = _json.loads(scoring_path.read_text(encoding='utf-8'))
            for k in [
                'totals_crps_mean','totals_loglik_mean','margins_crps_mean','margins_loglik_mean',
                'totals_rows','margins_rows','sigma_total_default_used','sigma_margin_default_used'
            ]:
                if k in sc_payload:
                    pipeline_stats[f'scoring_{k}'] = sc_payload[k]
            pipeline_stats['scoring_loaded'] = True
        else:
            pipeline_stats['scoring_loaded'] = False
    except Exception:
        pipeline_stats['scoring_error'] = True
    # Residual distribution ingestion (totals/margins) if JSON exists
    try:
        resid_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        resid_path = OUT / f'residuals_{resid_date}.json'
        if resid_path.exists():
            import json as _json
            r_payload = _json.loads(resid_path.read_text(encoding='utf-8'))
            pipeline_stats['residuals_ingested'] = True
            def _lift_res(prefix: str, obj: Any):
                if isinstance(obj, dict):
                    for k,v in obj.items():
                        if isinstance(v,(int,float,str)) or v is None:
                            pipeline_stats[f'{prefix}_{k}'] = v
            if 'total_stats' in r_payload: _lift_res('resid_total', r_payload['total_stats'])
            if 'margin_stats' in r_payload: _lift_res('resid_margin', r_payload['margin_stats'])
            for key in ('total_corr','margin_corr'):
                if key in r_payload and isinstance(r_payload[key], (int,float)):
                    pipeline_stats[key] = r_payload[key]
            if 'status' in r_payload:
                pipeline_stats['residuals_status'] = r_payload['status']
        else:
            pipeline_stats['residuals_ingested'] = False
    except Exception:
        pipeline_stats['residuals_error'] = True
    # Team variance ingestion for adaptive sigma scaling
    try:
        tv_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        tv_path = OUT / f'team_variance_{tv_date}.json'
        if tv_path.exists():
            import json as _json
            tv_payload = _json.loads(tv_path.read_text(encoding='utf-8'))
            teams_block = tv_payload.get('teams', {}) if isinstance(tv_payload, dict) else {}
            # Build maps for later per-row application
            team_variance_total = {k: v.get('total_std') for k,v in teams_block.items() if isinstance(v, dict) and v.get('total_std') is not None}
            team_variance_margin = {k: v.get('margin_std') for k,v in teams_block.items() if isinstance(v, dict) and v.get('margin_std') is not None}
            pipeline_stats['team_variance_ingested'] = True
            pipeline_stats['team_variance_teams'] = len(teams_block)
            pipeline_stats['team_variance_total_std_median'] = tv_payload.get('global', {}).get('total_std_median')
            pipeline_stats['team_variance_margin_std_median'] = tv_payload.get('global', {}).get('margin_std_median')
        else:
            pipeline_stats['team_variance_ingested'] = False
    except Exception:
        pipeline_stats['team_variance_error'] = True
    # Recalibration trigger ingestion (evaluates drift/correlation/scoring vs thresholds)
    try:
        rc_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        rc_path = OUT / f'recalibration_{rc_date}.json'
        if rc_path.exists():
            import json as _json
            rc_payload = _json.loads(rc_path.read_text(encoding='utf-8'))
            pipeline_stats['recalibration_ingested'] = True
            pipeline_stats['recalibration_needed'] = rc_payload.get('recalibration_needed')
            pipeline_stats['recalibration_reasons'] = rc_payload.get('reasons')
            metrics_block = rc_payload.get('metrics') if isinstance(rc_payload, dict) else {}
            if isinstance(metrics_block, dict):
                for k,v in metrics_block.items():
                    if isinstance(v, (int,float,str)) or v is None:
                        pipeline_stats[f'recalib_{k}'] = v
        else:
            pipeline_stats['recalibration_ingested'] = False
    except Exception:
        pipeline_stats['recalibration_error'] = True
    # Leakage scan ingestion (suspicious feature columns)
    try:
        leak_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        leak_path = OUT / f'leakage_{leak_date}.json'
        if leak_path.exists():
            import json as _json
            leak_payload = _json.loads(leak_path.read_text(encoding='utf-8'))
            pipeline_stats['leakage_ingested'] = True
            pipeline_stats['leakage_suspicious_cols'] = int(len(leak_payload.get('suspicious_columns', [])))
            if 'summary' in leak_payload and isinstance(leak_payload['summary'], dict):
                for k,v in leak_payload['summary'].items():
                    if isinstance(v,(int,float,str)):
                        pipeline_stats[f'leakage_summary_{k}'] = v
        else:
            pipeline_stats['leakage_ingested'] = False
    except Exception:
        pipeline_stats['leakage_error'] = True
    # Conference fairness ingestion
    try:
        fair_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        fair_path = OUT / f'fairness_{fair_date}.json'
        if fair_path.exists():
            import json as _json
            fair_payload = _json.loads(fair_path.read_text(encoding='utf-8'))
            pipeline_stats['fairness_ingested'] = True
            if 'global' in fair_payload and isinstance(fair_payload['global'], dict):
                for k,v in fair_payload['global'].items():
                    if isinstance(v,(int,float,str)):
                        pipeline_stats[f'fairness_global_{k}'] = v
            records = fair_payload.get('records', [])
            pipeline_stats['fairness_conferences_evaluated'] = int(len(records)) if isinstance(records, list) else 0
            if isinstance(records, list) and records:
                # extreme disparities
                try:
                    max_disp = max((abs(r.get('disparity_z_total')) for r in records if isinstance(r, dict) and r.get('disparity_z_total') is not None), default=None)
                    pipeline_stats['fairness_disparity_max_abs'] = max_disp
                except Exception:
                    pass
                try:
                    max_bias = max((abs(r.get('mean_residual_total')) for r in records if isinstance(r, dict) and r.get('mean_residual_total') is not None), default=None)
                    pipeline_stats['fairness_bias_max_abs'] = max_bias
                except Exception:
                    pass
        else:
            pipeline_stats['fairness_ingested'] = False
    except Exception:
        pipeline_stats['fairness_error'] = True
    # Predictability evaluation ingestion
    try:
        pred_eval_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        pred_eval_path = OUT / f'predictability_{pred_eval_date}.json'
        if pred_eval_path.exists():
            import json as _json
            pe_payload = _json.loads(pred_eval_path.read_text(encoding='utf-8'))
            pipeline_stats['predictability_ingested'] = True
            keep_keys = [
                'residual_mean','residual_std','residual_mae','calibration_slope_total','calibration_intercept_total',
                'corr_pred_market_total','coverage_rows','predictability_score','trailing_residual_std','trailing_residual_mae','trailing_calibration_slope_total'
            ]
            for k in keep_keys:
                if k in pe_payload:
                    pipeline_stats[f'predictability_{k}'] = pe_payload.get(k)
        else:
            pipeline_stats['predictability_ingested'] = False
    except Exception:
        pipeline_stats['predictability_error'] = True
    # Feature importance ingestion
    try:
        imp_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        imp_path = OUT / f'importance_{imp_date}.json'
        if imp_path.exists():
            import json as _json
            imp_payload = _json.loads(imp_path.read_text(encoding='utf-8'))
            pipeline_stats['importance_ingested'] = True
            # Capture top 8 features for totals and margin if present
            if 'totals' in imp_payload and isinstance(imp_payload['totals'], dict):
                feats = imp_payload['totals'].get('features', [])
                if isinstance(feats, list) and feats:
                    top_totals = [f.get('feature') for f in feats[:8] if isinstance(f, dict)]
                    pipeline_stats['importance_totals_top'] = top_totals
            if 'margin' in imp_payload and isinstance(imp_payload['margin'], dict):
                feats = imp_payload['margin'].get('features', [])
                if isinstance(feats, list) and feats:
                    top_margin = [f.get('feature') for f in feats[:8] if isinstance(f, dict)]
                    pipeline_stats['importance_margin_top'] = top_margin
        else:
            pipeline_stats['importance_ingested'] = False
    except Exception:
        pipeline_stats['importance_error'] = True
    # Segment performance ingestion
    try:
        seg_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        seg_path = OUT / f'segment_{seg_date}.json'
        if seg_path.exists():
            import json as _json
            seg_payload = _json.loads(seg_path.read_text(encoding='utf-8'))
            pipeline_stats['segment_ingested'] = True
            segs = seg_payload.get('segments', {}) if isinstance(seg_payload, dict) else {}
            if isinstance(segs, dict):
                # Extract key residual means for representative buckets
                for bucket in ['tempo::Q1_slowest','tempo::Q4_fastest','spread::0_2','spread::9_plus']:
                    if bucket in segs and isinstance(segs[bucket], dict):
                        val = segs[bucket].get('mean_residual_total')
                        pipeline_stats[f"segment_{bucket.replace('::','_')}_mean_residual"] = val
        else:
            pipeline_stats['segment_ingested'] = False
    except Exception:
        pipeline_stats['segment_error'] = True
    # Reliability calibration ingestion
    try:
        rel_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        rel_path = OUT / f'reliability_{rel_date}.json'
        if rel_path.exists():
            import json as _json
            rel_payload = _json.loads(rel_path.read_text(encoding='utf-8'))
            pipeline_stats['reliability_ingested'] = True
            for k in ['calibration_slope','calibration_intercept','rows']:
                if k in rel_payload:
                    pipeline_stats[f'reliability_{k}'] = rel_payload.get(k)
        else:
            pipeline_stats['reliability_ingested'] = False
    except Exception:
        pipeline_stats['reliability_error'] = True
    # Daily performance aggregation ingestion (composite health metrics)
    try:
        perf_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        perf_path = OUT / f'performance_{perf_date}.json'
        if perf_path.exists():
            import json as _json
            perf_payload = _json.loads(perf_path.read_text(encoding='utf-8'))
            pipeline_stats['performance_ingested'] = True
            for k in [
                'model_health','predictability_score','recalibration_needed','fairness_bias_flag','fairness_disparity_flag',
                'leakage_suspicious_cols','total_mean','margin_mean','total_corr','margin_corr','total_mean_z','margin_mean_z','total_corr_z','margin_corr_z'
            ]:
                if k in perf_payload:
                    pipeline_stats[f'perf_{k}'] = perf_payload.get(k)
            for k in ['predictability_score','residual_std','crps_total','total_mean','total_corr']:
                for w in ['7','14']:
                    keyname = f'{k}_trailing_{w}'
                    if keyname in perf_payload:
                        pipeline_stats[f'perf_{keyname}'] = perf_payload.get(keyname)
        else:
            pipeline_stats['performance_ingested'] = False
    except Exception:
        pipeline_stats['performance_error'] = True
    # Season metrics ingestion for rolling health context
    try:
        season_path = OUT / 'season_metrics.json'
        if season_path.exists():
            import json as _json
            season_payload = _json.loads(season_path.read_text(encoding='utf-8'))
            pipeline_stats['season_metrics_ingested'] = True
            summary_block = season_payload.get('summary', {}) if isinstance(season_payload, dict) else {}
            for k in [
                'residual_std_mean','predictability_score_mean','crps_mean_mean','totals_residual_std_mean','margin_residual_std_mean'
            ]:
                if k in summary_block:
                    pipeline_stats[f'season_{k}'] = summary_block.get(k)
        else:
            pipeline_stats['season_metrics_ingested'] = False
    except Exception:
        pipeline_stats['season_metrics_error'] = True
    # Drift/Bias diagnostics integration (load precomputed JSON if present for today or selected date)
    try:
        drift_date = date_q if date_q else _today_local().strftime('%Y-%m-%d')
        drift_path = OUT / f'drift_bias_{drift_date}.json'
        if not drift_path.exists():
            # Fallback: if requesting past date before diagnostics existed, ignore silently
            # Attempt today's file when browsing without date param
            if not date_q:
                alt_path = OUT / f'drift_bias_{_today_local().strftime("%Y-%m-%d")}.json'
                if alt_path.exists():
                    drift_path = alt_path
        if drift_path.exists():
            import json as _json
            payload = _json.loads(drift_path.read_text(encoding='utf-8'))
            # Whitelist expected keys to avoid large blob
            for k in [
                'totals_bias','pace_drift','trailing_mean_pred_total','today_mean_pred_total',
                'conference_margin_bias','source_rows'
            ]:
                if k in payload:
                    pipeline_stats[f'drift_bias_{k}'] = payload[k]
            pipeline_stats['drift_bias_loaded'] = True
        else:
            pipeline_stats['drift_bias_loaded'] = False
    except Exception:
        pipeline_stats['drift_bias_error'] = True
    # Instrumentation: capture early model predictions stats after pipeline_stats exists
    try:
        if not model_preds.empty:
            pipeline_stats['model_preds_source'] = _MODEL_PREDICTIONS_SOURCE_PATH
            pipeline_stats['model_preds_cols'] = list(model_preds.columns)
            if 'pred_total_model' in model_preds.columns:
                ptm_early = pd.to_numeric(model_preds['pred_total_model'], errors='coerce')
                pipeline_stats['model_preds_total_stats_early'] = {
                    'count': int(ptm_early.notna().sum()),
                    'min': float(ptm_early.min()) if ptm_early.notna().any() else None,
                    'max': float(ptm_early.max()) if ptm_early.notna().any() else None,
                    'mean': float(ptm_early.mean()) if ptm_early.notna().any() else None,
                    'std': float(ptm_early.std()) if ptm_early.notna().any() else None,
                    'unique': int(ptm_early.nunique()) if ptm_early.notna().any() else 0
                }
                pipeline_stats['model_preds_total_head'] = ptm_early.head(10).tolist()
            if 'pred_total_calibrated' in model_preds.columns:
                ptc_early = pd.to_numeric(model_preds['pred_total_calibrated'], errors='coerce')
                pipeline_stats['model_preds_total_calibrated_head'] = ptc_early.head(10).tolist()
                pipeline_stats['model_preds_total_calibrated_stats_early'] = {
                    'count': int(ptc_early.notna().sum()),
                    'min': float(ptc_early.min()) if ptc_early.notna().any() else None,
                    'max': float(ptc_early.max()) if ptc_early.notna().any() else None,
                    'mean': float(ptc_early.mean()) if ptc_early.notna().any() else None,
                    'std': float(ptc_early.std()) if ptc_early.notna().any() else None,
                    'unique': int(ptc_early.nunique()) if ptc_early.notna().any() else 0
                }
                # Preferred initial source determination (calibrated when any non-NaN values present)
                if ptc_early.notna().any():
                    pipeline_stats['model_preds_preferred_initial_source'] = 'calibrated'
                else:
                    pipeline_stats['model_preds_preferred_initial_source'] = 'raw_model'
            else:
                pipeline_stats['model_preds_preferred_initial_source'] = 'raw_model'
            if 'pred_margin_model' in model_preds.columns:
                pmm_early = pd.to_numeric(model_preds['pred_margin_model'], errors='coerce')
                pipeline_stats['model_preds_margin_head'] = pmm_early.head(10).tolist()
    except Exception:
        pass
    # Attach source paths if available
    try:
        pipeline_stats["predictions_source"] = _PREDICTIONS_SOURCE_PATH
    except Exception:
        pipeline_stats["predictions_source"] = None
    try:
        pipeline_stats["model_predictions_source"] = _MODEL_PREDICTIONS_SOURCE_PATH
    except Exception:
        pipeline_stats["model_predictions_source"] = None
    try:
        # odds loader can expose attribute _ODDS_SOURCE_PATH similarly if implemented; attempt best-effort detection
        odds_source = None
        for cand in [OUT/"odds_today.csv", OUT/"odds_curr.csv", OUT/"odds_" + str(date_q) + ".csv", OUT/"games_with_last.csv"]:
            try:
                if hasattr(cand, "exists") and cand.exists():
                    # heuristic: if shape matches odds df row count within tolerance, pick
                    if not odds.empty:
                        # allow mismatch; we still record first existing for visibility
                        odds_source = str(cand)
                        break
            except Exception:
                continue
        pipeline_stats["odds_source_guess"] = odds_source
    except Exception:
        pipeline_stats["odds_source_guess"] = None
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
    pipeline_stats["games_after_date"] = len(games)
    pipeline_stats["preds_after_date"] = len(preds)
    pipeline_stats["daily_results_rows"] = len(daily_df)
    # Zero-games reason (pre-synthetic) recorded now; may be updated later if synthetic schedule built
    if pipeline_stats["games_after_date"] == 0:
        if pipeline_stats.get("games_load_rows", 0) == 0:
            pipeline_stats["zero_games_reason"] = "no_games_loaded"
        else:
            pipeline_stats["zero_games_reason"] = "date_filter_eliminated_all"
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

    # ------------------------------------------------------------------
    # Synthetic games fallback: if we have ZERO games rows for the target date
    # but we do have predictions or odds, synthesize a minimal games frame so
    # that cards can render instead of showing an empty slate.
    # ------------------------------------------------------------------
    if games.empty:
        fallback_reason: list[str] = []
        synth_df = pd.DataFrame()
        # Prefer predictions for richer team naming
        try:
            if not preds.empty and "game_id" in preds.columns:
                # Candidate columns already present
                if {"home_team","away_team"}.issubset(preds.columns):
                    cols = [c for c in ["game_id","home_team","away_team","date","start_time"] if c in preds.columns]
                    synth_df = preds[cols].drop_duplicates("game_id")
                    fallback_reason.append("preds_home_away_direct")
                else:
                    # Attempt reconstruction from team/opponent + home_away flag
                    if {"team","opponent"}.issubset(preds.columns):
                        ha_col = None
                        for cand in ["home_away","homeaway","is_home"]:
                            if cand in preds.columns:
                                ha_col = cand
                                break
                        if ha_col:
                            ha = preds[ha_col].astype(str).str.lower()
                            home_mask = ha.isin(["home","h","1","true","t"])
                            away_mask = ha.isin(["away","a","0","false","f"])
                            home_side = preds[home_mask][[c for c in ["game_id","team"] if c in preds.columns]].rename(columns={"team":"home_team"})
                            away_side = preds[away_mask][[c for c in ["game_id","team"] if c in preds.columns]].rename(columns={"team":"away_team"})
                            if not home_side.empty and not away_side.empty:
                                merged = home_side.merge(away_side, on="game_id", how="inner")
                                if "date" in preds.columns:
                                    merged = merged.merge(preds[["game_id","date"]].drop_duplicates(), on="game_id", how="left")
                                synth_df = merged.drop_duplicates("game_id")
                                fallback_reason.append("preds_reconstructed_home_away")
                        # If still empty, last resort: pair team/opponent per row (may duplicate, but we dedupe game_id)
                        if synth_df.empty:
                            tmp = preds.copy()
                            if {"team","opponent"}.issubset(tmp.columns):
                                tmp = tmp.rename(columns={"team":"home_team","opponent":"away_team"})
                                cols = [c for c in ["game_id","home_team","away_team","date","start_time"] if c in tmp.columns]
                                synth_df = tmp[cols].drop_duplicates("game_id")
                                fallback_reason.append("preds_team_opponent_pairing")
        except Exception:
            pass
        # If predictions path failed, attempt odds-based synthesis
        if synth_df.empty:
            try:
                if not odds.empty and {"game_id","home_team","away_team"}.issubset(odds.columns):
                    cols = [c for c in ["game_id","home_team","away_team","date_line","commence_time"] if c in odds.columns]
                    o2 = odds[cols].drop_duplicates("game_id")
                    # Normalize date column name to "date"
                    if "date_line" in o2.columns:
                        o2 = o2.rename(columns={"date_line":"date"})
                    if "commence_time" in o2.columns and "start_time" not in o2.columns:
                        o2 = o2.rename(columns={"commence_time":"start_time"})
                    synth_df = o2
                    fallback_reason.append("odds_game_rows")
            except Exception:
                pass
        # Constrain to requested date if a date column exists
        if not synth_df.empty and date_q and "date" in synth_df.columns:
            try:
                synth_df["date"] = pd.to_datetime(synth_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                synth_df = synth_df[synth_df["date"].astype(str) == str(date_q)]
            except Exception:
                pass
        if not synth_df.empty:
            try:
                synth_df["game_id"] = synth_df["game_id"].astype(str)
            except Exception:
                pass
            games = synth_df.copy()
            pipeline_stats["games_synthetic"] = len(games)
            pipeline_stats["games_fallback_reason"] = fallback_reason
            if results_note:
                results_note += " | Synthetic schedule constructed (" + ",".join(fallback_reason) + ")"
            else:
                results_note = "Synthetic schedule constructed (" + ",".join(fallback_reason) + ")"
        else:
            pipeline_stats["games_synthetic"] = 0
            if not fallback_reason:
                pipeline_stats["games_fallback_reason"] = ["none_available"]
            else:
                pipeline_stats["games_fallback_reason"] = fallback_reason or ["attempted_no_rows"]

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

                # ------------------------------------------------------------------
                # Augment: ensure games without predictions are still shown.
                # Previous behavior dropped games lacking prediction rows because
                # we started from preds and performed a left merge. For partial
                # slates (some preds missing) important matchups disappeared.
                # We now append minimal rows for any games on the requested date
                # that are absent from the predictions set so the UI renders them.
                # ------------------------------------------------------------------
                try:
                    if "game_id" in games.columns and "game_id" in df.columns:
                        # Restrict to target date if date column present
                        if date_q and "date" in games.columns:
                            games_for_date = games[games["date"].astype(str) == str(date_q)]
                        else:
                            games_for_date = games
                        missing = games_for_date[~games_for_date["game_id"].astype(str).isin(df["game_id"].astype(str))]
                        if not missing.empty:
                            add_cols = ["game_id","date","home_team","away_team","home_score","away_score","start_time"]
                            add = missing[[c for c in add_cols if c in missing.columns]].copy()
                            # Guarantee prediction placeholder columns
                            for c in ["pred_total","pred_margin"]:
                                if c not in add.columns:
                                    add[c] = None
                            df = pd.concat([df, add], ignore_index=True)
                except Exception:
                    pass

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
        # Disabled period-based filtering to avoid accidental loss of usable odds rows; retain all rows regardless of period.
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
                    # Initial canonical slug pass
                    raw["_home_norm"] = raw[home_col].astype(str).map(_canon_slug)
                    raw["_away_norm"] = raw[away_col].astype(str).map(_canon_slug)
                    # Build canonical universe from current df (games/preds) for improved matching
                    try:
                        canon_universe: set[str] = set()
                        if {"home_team","away_team"}.issubset(df.columns):
                            canon_universe.update(_canon_slug(t) for t in df["home_team"].astype(str))
                            canon_universe.update(_canon_slug(t) for t in df["away_team"].astype(str))
                        # Fallback simple cleanup for odds names that include mascots not present in internal canonical names.
                        mascot_drop = {"falcons","wildcats","tigers","bulldogs","eagles","hawks","knights","raiders","rams","spartans","vikings","aggies","cardinals","broncos","panthers","lions","gators","longhorns","buckeyes","sooners","rebels","cougars","mountaineers","bearcats","bears","wolfpack","cowboys","dolphins","gaels","miners","pilots","dons","jaguars","gamecocks","hurricanes","gophers","badgers","illini","hoosiers","seminoles","foxfes"}
                        try:
                            from rapidfuzz import process, fuzz  # type: ignore
                            use_fuzzy = True
                        except Exception:
                            use_fuzzy = False
                        def _refine(name: str) -> str:
                            base = _canon_slug(name)
                            if base in canon_universe:
                                return base
                            parts = [p for p in re.sub(r"[^A-Za-z0-9 ]+"," ", name).lower().split() if p]
                            # Drop trailing mascot tokens progressively
                            for k in range(len(parts), 0, -1):
                                cand = _canon_slug(" ".join(p for p in parts[:k] if p not in mascot_drop))
                                if cand in canon_universe:
                                    return cand
                            if use_fuzzy and canon_universe:
                                # Fuzzy match against universe
                                best = process.extractOne(base, list(canon_universe), scorer=fuzz.token_set_ratio)
                                if best and best[1] >= 80:
                                    return best[0]
                            return base
                        raw["_home_norm"] = raw[home_col].astype(str).map(_refine)
                        raw["_away_norm"] = raw[away_col].astype(str).map(_refine)
                    except Exception:
                        pass
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
                    logger.info("Raw odds fallback aggregator pairs=%d", len(agg_rows))
                    for idx, row in df.iterrows():
                        h = _canon_slug(str(row.get("home_team") or ""))
                        a = _canon_slug(str(row.get("away_team") or ""))
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
    # Secondary odds coverage fallback: if majority of market_total still missing, derive directly from raw odds totals without any period filtering.
    try:
        if not odds.empty and "game_id" in odds.columns:
            if ("market_total" not in df.columns) or (df["market_total"].isna().sum() > 0.6 * len(df)):
                o3 = odds.copy()
                o3["game_id"] = o3["game_id"].astype(str)
                if "total" in o3.columns:
                    line_by_game2 = (
                        o3.groupby("game_id", as_index=False)["total"]
                          .median()
                          .rename(columns={"total": "_market_total_from_odds_fallback"})
                    )
                    logger.info("Secondary odds fallback: %d games aggregated; pre-missing=%d", len(line_by_game2), int(df["market_total"].isna().sum() if "market_total" in df.columns else len(df)))
                    # Pair-based aggregation for rows lacking game_id alignment (use canonical slugs)
                    try:
                        o3["_home_norm"] = o3.get("home_team_name", o3.get("home_team", "")).astype(str).map(_canon_slug)
                        o3["_away_norm"] = o3.get("away_team_name", o3.get("away_team", "")).astype(str).map(_canon_slug)
                        o3["_pair_key"] = o3.apply(lambda r: "::".join(sorted([str(r.get("_home_norm")), str(r.get("_away_norm"))])), axis=1)
                        pair_med = (
                            o3.groupby("_pair_key", as_index=False)["total"]
                              .median()
                              .rename(columns={"total": "_market_total_pair_med"})
                        )
                    except Exception:
                        pair_med = pd.DataFrame()
                    if not line_by_game2.empty and "game_id" in df.columns:
                        df = df.merge(line_by_game2, on="game_id", how="left")
                        if "market_total" in df.columns:
                            df["market_total"] = df["market_total"].where(df["market_total"].notna(), df["_market_total_from_odds_fallback"])
                        else:
                            df["market_total"] = df["_market_total_from_odds_fallback"]
                    # Apply pair-based fill if still missing
                    if pair_med is not None and not pair_med.empty and ("home_team" in df.columns and "away_team" in df.columns):
                        try:
                            df["_pair_key"] = df.apply(lambda r: "::".join(sorted([_canon_slug(str(r.get("home_team") or "")), _canon_slug(str(r.get("away_team") or ""))])), axis=1)
                            df = df.merge(pair_med, on="_pair_key", how="left")
                            if "market_total" in df.columns:
                                df["market_total"] = df["market_total"].where(df["market_total"].notna(), df["_market_total_pair_med"])
                            else:
                                df["market_total"] = df["_market_total_pair_med"]
                        except Exception:
                            pass
                        logger.info("Secondary odds fallback applied; post-missing=%d", int(df["market_total"].isna().sum()))
    except Exception:
        pass
    try:
        for cname in ("commence_time", "commence_time_g"):
            if cname in df.columns:
                # Normalize commence_time to user's local timezone and standard display format
                try:
                    _local_tz = dt.datetime.now().astimezone().tzinfo
                except Exception:
                    _local_tz = None
                try:
                    # Parse as UTC to correctly handle 'Z' or offset inputs; treat naive as UTC as well
                    raw_series = df[cname]
                    s = pd.to_datetime(raw_series, utc=True, errors="coerce")
                    # If any entries remained NaT, attempt second pass without forcing UTC then localize
                    if s.isna().any():
                        alt = pd.to_datetime(raw_series, errors="coerce")
                        # Localize naive alt timestamps to UTC before convert
                        if alt.notna().any():
                            # Build result combining successes from alt where s failed
                            fill_mask = s.isna() & alt.notna()
                            # Assign tzinfo UTC to naive alt
                            try:
                                alt_localized = alt.map(lambda x: x.replace(tzinfo=dt.timezone.utc) if (pd.notna(x) and x.tzinfo is None) else x)
                                s[fill_mask] = alt_localized[fill_mask]
                            except Exception:
                                s[fill_mask] = alt[fill_mask]
                    if _local_tz is not None:
                        s = s.dt.tz_convert(_local_tz)
                    else:
                        # Fallback: leave as UTC if local tz not determinable
                        pass
                    mapped = s.dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    # Fallback naive parse/format without tz conversion
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

            # 1H/2H medians with robust period normalization (spaces/hyphens/variants)
            def _norm_period(v: Any) -> str:
                try:
                    s = str(v).lower().strip()
                except Exception:
                    return ""
                s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
                return s
            def _agg_market_period(market: str, val_col: str, period_keys: set[str], out_col: str) -> pd.Series | None:
                sub = o2[o2["market"].astype(str).str.lower() == market]
                if "period" in sub.columns:
                    vals = sub["period"].map(_norm_period)
                    sub = sub[vals.isin(period_keys)]
                if val_col not in sub.columns or sub.empty:
                    return None
                return sub.groupby("game_id")[val_col].median().rename(out_col)
            # Accept many variants for half labels
            PERIOD_1H = {"1h","1st_half","first_half","h1","half_1","1_h","1sthalf","firsthalf"}
            PERIOD_2H = {"2h","2nd_half","second_half","h2","half_2","2_h","2ndhalf","secondhalf"}
            t_1h = _agg_market_period("totals", "total", PERIOD_1H, "market_total_1h")
            t_2h = _agg_market_period("totals", "total", PERIOD_2H, "market_total_2h")
            s_1h = _agg_market_period("spreads", "home_spread", PERIOD_1H, "spread_home_1h")
            s_2h = _agg_market_period("spreads", "home_spread", PERIOD_2H, "spread_home_2h")
            # Half moneylines (when available)
            m_1h_h = _agg_market_period("h2h", "moneyline_home", PERIOD_1H, "ml_home_1h")
            m_2h_h = _agg_market_period("h2h", "moneyline_home", PERIOD_2H, "ml_home_2h")
            m_1h_a = _agg_market_period("h2h", "moneyline_away", PERIOD_1H, "ml_away_1h")
            m_2h_a = _agg_market_period("h2h", "moneyline_away", PERIOD_2H, "ml_away_2h")
            for srs in [t_1h, t_2h, s_1h, s_2h, m_1h_h, m_2h_h, m_1h_a, m_2h_a]:
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

    # --------------------------------------------------------------
    # Start time normalization & diagnostics
    # Ensures start_time conforms to 'YYYY-MM-DD HH:MM' local time to
    # avoid UTC rollover (e.g., 00:00 next day +00:00) hiding games on
    # the intended slate. Records instrumentation for Marshall game.
    # --------------------------------------------------------------
    try:
        if 'start_time' in df.columns and len(df):
            st_raw = df['start_time'].astype(str)
            # Detect timezone offset pattern (e.g. +00:00)
            tz_mask = st_raw.str.contains(r'\+\d{2}:\d{2}')
            local_tz = dt.datetime.now().astimezone().tzinfo
            def _norm_start(v: str) -> str:
                # Attempt offset/UTC-aware parse first; convert to local
                try:
                    ts = pd.to_datetime(v, errors='coerce', utc=True)
                    if ts is not None and not pd.isna(ts):
                        return ts.astimezone(local_tz).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
                # Fallback: naive parse interpreted as LOCAL (previously UTC leading to -offset shift)
                try:
                    ts2 = pd.to_datetime(v, errors='coerce', utc=False)
                    if ts2 is not None and not pd.isna(ts2):
                        if ts2.tzinfo is None:
                            try:
                                ts2 = ts2.replace(tzinfo=local_tz)
                            except Exception:
                                pass
                        else:
                            try:
                                ts2 = ts2.astimezone(local_tz)
                            except Exception:
                                pass
                        return ts2.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
                return v
            # Rows needing normalization (timezone pattern OR non-standard format)
            need_fix_mask = tz_mask | (~st_raw.str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$'))
            if need_fix_mask.any():
                df.loc[need_fix_mask, 'start_time_normalized'] = st_raw[need_fix_mask].map(_norm_start)
                norm_vals = df['start_time_normalized']
                good_norm = norm_vals.notna() & norm_vals.astype(str).str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$')
                df['start_time'] = np.where(good_norm, norm_vals, df['start_time'])
            # Instrumentation
            pipeline_stats['start_time_rows'] = int(len(df))
            pipeline_stats['start_time_tz_pattern_count'] = int(tz_mask.sum())
            pipeline_stats['start_time_normalized_count'] = int(need_fix_mask.sum())
            if 'game_id' in df.columns:
                mmask = df['game_id'].astype(str) == '401827130'
                if mmask.any():
                    pipeline_stats['marshall_start_time_raw'] = st_raw[mmask].iloc[0]
                    pipeline_stats['marshall_start_time_final'] = df['start_time'][mmask].iloc[0]
    except Exception:
        pass

    # Final cleanup: drop placeholder TBD games, re-filter by date using localized start_time, and deduplicate
    try:
        if not df.empty:
            # Drop rows where teams are placeholders (TBD/Unknown)
            bads = {"tbd", "unknown", "na", "n/a", ""}
            def _is_bad(x):
                try:
                    return str(x).strip().lower() in bads
                except Exception:
                    return True
            if {"home_team","away_team"}.issubset(df.columns):
                mask_bad = df["home_team"].map(_is_bad) | df["away_team"].map(_is_bad)
                if mask_bad.any():
                    pipeline_stats["dropped_tbd_rows"] = int(mask_bad.sum())
                    df = df[~mask_bad].copy()
            # If date_q provided, ensure rows align by either explicit date or start_time's date (post-localization)
            if date_q:
                try:
                    ok = pd.Series([True]*len(df))
                    if "date" in df.columns:
                        ok = ok & (df["date"].astype(str) == str(date_q))
                    if "start_time" in df.columns:
                        st_date = pd.to_datetime(df["start_time"], errors="coerce").dt.strftime("%Y-%m-%d")
                        ok = ok | (st_date == str(date_q))
                    before = len(df)
                    df = df[ok].copy()
                    pipeline_stats["post_time_date_filter_dropped"] = int(before - len(df))
                except Exception:
                    pass
            # Deduplicate by game_id when present, else by teams+start_time
            try:
                before = len(df)
                if "game_id" in df.columns:
                    df = df.drop_duplicates(subset=["game_id"], keep="last")
                elif {"home_team","away_team","start_time"}.issubset(df.columns):
                    df = df.drop_duplicates(subset=["home_team","away_team","start_time"], keep="last")
                pipeline_stats["post_cleanup_dedup_dropped"] = int(before - len(df))
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: attach closing lines from games_with_closing_<date>.csv (preferred) or games_with_closing.csv if missing
    if ("game_id" in df.columns):
        try:
            # Prefer date-specific artifact when browsing a specific slate; else fallback to undated file
            closing_path = None
            if date_q:
                cand = OUT / f"games_with_closing_{date_q}.csv"
                if cand.exists():
                    closing_path = cand
            if closing_path is None:
                closing_path = OUT / "games_with_closing.csv"
            if closing_path.exists():
                cl = pd.read_csv(closing_path)
                if not cl.empty and "game_id" in cl.columns:
                    cl["game_id"] = cl["game_id"].astype(str)
                    # Build canonical pair keys to enable fallback joins when game_id mismatches
                    try:
                        ch_col = next((c for c in ["home_team","home_team_name","home"] if c in cl.columns), None)
                        ca_col = next((c for c in ["away_team","away_team_name","away"] if c in cl.columns), None)
                        if ch_col and ca_col:
                            cl["_home_norm"] = cl[ch_col].astype(str).map(_canon_slug)
                            cl["_away_norm"] = cl[ca_col].astype(str).map(_canon_slug)
                            cl["_pair_key"] = cl.apply(lambda r: "::".join(sorted([r.get("_home_norm"), r.get("_away_norm")])), axis=1)
                            if {"home_team","away_team"}.issubset(df.columns):
                                df["_home_norm"] = df["home_team"].astype(str).map(_canon_slug)
                                df["_away_norm"] = df["away_team"].astype(str).map(_canon_slug)
                                df["_pair_key"] = df.apply(lambda r: "::".join(sorted([r.get("_home_norm"), r.get("_away_norm")])), axis=1)
                    except Exception:
                        pass
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
                        # Pair-key based coalesce for spreads + totals if still missing
                        try:
                            if "_pair_key" in df.columns and "_pair_key" in cl.columns:
                                # Totals
                                if "market_total" in df.columns and df["market_total"].isna().sum() > 0 and "closing_total" in cl.columns:
                                    pair_tot = cl.groupby("_pair_key")["closing_total"].median().rename("_closing_total_pair")
                                    df = df.merge(pair_tot, on="_pair_key", how="left")
                                    df["market_total"] = df["market_total"].where(df["market_total"].notna(), df.get("_closing_total_pair"))
                                    if "closing_total" in df.columns:
                                        df["closing_total"] = df["closing_total"].where(df["closing_total"].notna(), df.get("_closing_total_pair"))
                                # Spreads
                                if df["spread_home"].isna().sum() > 0 and "home_spread" in cl.columns:
                                    pair_sp = cl.groupby("_pair_key")["home_spread"].median().rename("_closing_spread_pair")
                                    df = df.merge(pair_sp, on="_pair_key", how="left")
                                    df["spread_home"] = df["spread_home"].where(df["spread_home"].notna(), df.get("_closing_spread_pair"))
                                    if "closing_spread_home" in df.columns:
                                        df["closing_spread_home"] = df["closing_spread_home"].where(df["closing_spread_home"].notna(), df.get("_closing_spread_pair"))
                        except Exception:
                            pass
        except Exception:
            pass

    # Ingest spread logistic calibration constant if available
    try:
        cal_path = OUT / 'calibration_spread_logistic.json'
        if cal_path.exists():
            cal_payload = _json.loads(cal_path.read_text(encoding='utf-8'))
            if isinstance(cal_payload, dict):
                pipeline_stats['spread_logistic_K'] = cal_payload.get('best_K')
                pipeline_stats['spread_logistic_rows'] = cal_payload.get('n_rows')
                pipeline_stats['spread_logistic_generated_at'] = cal_payload.get('generated_at')
    except Exception:
        pipeline_stats['spread_logistic_error'] = True

    # Kelly suggestion (spread): compute fraction using calibrated K (if available) and current spread vs model margin
    try:
        if 'pred_margin' in df.columns and ('spread_home' in df.columns or 'closing_spread_home' in df.columns):
            # Prefer closing spread when available
            if 'spread_home' in df.columns:
                base_spread = df['spread_home']
            else:
                base_spread = df['closing_spread_home']
            pm = pd.to_numeric(df['pred_margin'], errors='coerce')
            sh = pd.to_numeric(base_spread, errors='coerce')
            K = pipeline_stats.get('spread_logistic_K', None)
            if K is None or (isinstance(K, float) and (K <= 0 or pd.isna(K))):
                K = 0.115
            def _kelly(p: float, b: float = 0.909) -> float:
                try:
                    return max(0.0, min(1.0, ((p*(b+1) - 1)/b)))
                except Exception:
                    return 0.0
            # Compute cover probability and kelly fraction per row
            diff = pm - sh
            # Guard where values missing
            diff = diff.where(~(pm.isna() | sh.isna()), np.nan)
            p_cover = 1.0/(1.0 + np.exp(-(diff.astype(float) / float(K))))
            df['kelly_frac_spread'] = p_cover.map(lambda p: _kelly(p) if pd.notna(p) else np.nan)
            # Also expose direction suggestion: HOME if model favors home w.r.t line, else AWAY
            df['kelly_side_spread'] = np.where(diff > 0, 'HOME', np.where(diff < 0, 'AWAY', None))
            pipeline_stats['kelly_spread_populated'] = int(df['kelly_frac_spread'].notna().sum())
    except Exception:
        pipeline_stats['kelly_spread_error'] = True

    # Prediction fallback enrichment: ensure we rarely render a card with odds but no predictions.
    # Removes previous logic that hid odds when predictions were missing; instead we synthesize a baseline prediction.
    try:
        # Ensure prediction columns always exist so fallback logic executes even when model merge failed entirely.
        if "pred_total" not in df.columns:
            df["pred_total"] = np.nan
            pipeline_stats["pred_total_column_injected"] = True
        else:
            pipeline_stats["pred_total_column_injected"] = False
        if "pred_margin" not in df.columns:
            df["pred_margin"] = np.nan
            pipeline_stats["pred_margin_column_injected"] = True
        else:
            pipeline_stats["pred_margin_column_injected"] = False
        if "pred_total" in df.columns:
            # Coerce textual placeholders ('nan','None','') to actual NaN before missing mask
            try:
                df["pred_total"] = pd.to_numeric(df["pred_total"], errors="coerce")
            except Exception:
                pass
            if "pred_margin" in df.columns:
                try:
                    df["pred_margin"] = pd.to_numeric(df["pred_margin"], errors="coerce")
                except Exception:
                    pass
            mt_series = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else None
            pm_series = pd.to_numeric(df.get("pred_margin"), errors="coerce") if "pred_margin" in df.columns else None
            missing_pred = df["pred_total"].isna()
            pipeline_stats["missing_pred_total_rows_initial"] = int(missing_pred.sum())
            # Instrumentation: capture column presence and a small sample of missing prediction rows pre-fill
            try:
                cols_needed = [
                    "market_total","home_tempo_rating","away_tempo_rating","home_rtg","away_rtg",
                    "spread_home","closing_spread_home","home_team","away_team","pred_margin","closing_total"
                ]
                pipeline_stats["pred_pipeline_columns_present"] = [c for c in cols_needed if c in df.columns]
                pipeline_stats["pred_total_missing_initial"] = int(missing_pred.sum())
                if missing_pred.any():
                    sample_cols = [
                        "home_team","away_team","market_total","closing_total","pred_total","pred_total_basis",
                        "home_tempo_rating","away_tempo_rating","home_rtg","away_rtg","spread_home","pred_margin"
                    ]
                    sample_view_cols = [c for c in sample_cols if c in df.columns]
                    pipeline_stats["pred_total_missing_sample"] = (
                        df.loc[missing_pred, sample_view_cols].head(5).to_dict(orient="records")
                    )
            except Exception:
                pass
            # If market_total column itself missing, create it from closing_total or leave None so downstream logic can still inspect.
            if mt_series is None:
                if "closing_total" in df.columns:
                    df["market_total"] = pd.to_numeric(df["closing_total"], errors="coerce")
                else:
                    df["market_total"] = np.nan
                mt_series = pd.to_numeric(df.get("market_total"), errors="coerce")
            # If entire market_total is NaN, we'll still attempt a tempo-based synthetic baseline.
            all_market_nan = mt_series.isna().all() if mt_series is not None else True
            # Fallback 1 (revised): synthetic baseline when pred_total missing.
            # Older behavior copied market_total verbatim, collapsing edge variance.
            # New: blend league average, optional tempo, partial market anchor, deterministic noise.
            if missing_pred.any():
                can_fill = missing_pred & mt_series.notna()
                if can_fill.any():
                    import zlib
                    baseline_league_avg = 141.5
                    if {"home_tempo_rating","away_tempo_rating"}.issubset(df.columns):
                        ht = pd.to_numeric(df.get("home_tempo_rating"), errors="coerce")
                        at = pd.to_numeric(df.get("away_tempo_rating"), errors="coerce")
                        tempo_avg_series = np.where(ht.notna() & at.notna(), (ht+at)/2.0, np.nan)
                    else:
                        tempo_avg_series = np.full(len(df), np.nan)
                    def _stable_noise(home, away):
                        try:
                            key = f"{str(home)}::{str(away)}"
                            code = zlib.adler32(key.encode())
                            return ((code % 1000)/1000.0 - 0.5) * 3.2
                        except Exception:
                            return 0.0
                    for idx in df.index[can_fill]:
                        mt_val = mt_series.loc[idx]
                        h = df.at[idx, "home_team"] if "home_team" in df.columns else ""
                        a = df.at[idx, "away_team"] if "away_team" in df.columns else ""
                        noise = _stable_noise(h, a)
                        tempo_avg = tempo_avg_series[idx] if not (isinstance(tempo_avg_series, float) or pd.isna(tempo_avg_series[idx])) else np.nan
                        tempo_component = ((tempo_avg - 70.0) * 0.55) if (not pd.isna(tempo_avg)) else 0.0
                        val = 0.60 * baseline_league_avg + 0.25 * float(mt_val) + 0.15 * (baseline_league_avg + tempo_component) + noise
                        if abs(val - mt_val) < 0.4:
                            val = val + (1.15 if val <= mt_val else -1.15)
                        # Remove hard lower floor (112) which was causing uniform fallback totals when model preds absent.
                        # Apply only an upper plausibility cap; allow natural lower values for early-season slow tempos.
                        val = float(np.clip(val, 60, 188))
                        try:
                            pipeline_stats.setdefault('synthetic_baseline_fills', 0)
                            pipeline_stats['synthetic_baseline_fills'] += 1
                            pipeline_stats.setdefault('synthetic_baseline_vals', []).append(val)
                        except Exception:
                            pass
                        df.at[idx, "pred_total"] = val
                        if "pred_total_basis" in df.columns and pd.isna(df.at[idx, "pred_total_basis"]):
                            df.at[idx, "pred_total_basis"] = "synthetic_baseline"
                        elif "pred_total_basis" not in df.columns:
                            df.loc[idx, "pred_total_basis"] = "synthetic_baseline"
                # Secondary path: fill rows where market_total is NaN (or entire column NaN) using pure league avg + tempo/noise.
                if (all_market_nan or (missing_pred & mt_series.isna()).any()):
                    import zlib
                    baseline_league_avg = 141.5
                    if {"home_tempo_rating","away_tempo_rating"}.issubset(df.columns):
                        ht = pd.to_numeric(df.get("home_tempo_rating"), errors="coerce")
                        at = pd.to_numeric(df.get("away_tempo_rating"), errors="coerce")
                        tempo_avg_series = np.where(ht.notna() & at.notna(), (ht+at)/2.0, np.nan)
                    else:
                        tempo_avg_series = np.full(len(df), np.nan)
                    def _stable_noise2(home, away):
                        try:
                            key = f"{str(home)}::{str(away)}"
                            code = zlib.adler32(key.encode())
                            return ((code % 1000)/1000.0 - 0.5) * 5.0
                        except Exception:
                            return 0.0
                    can_fill2 = missing_pred & df["pred_total"].isna()
                    for idx in df.index[can_fill2]:
                        h = df.at[idx, "home_team"] if "home_team" in df.columns else ""
                        a = df.at[idx, "away_team"] if "away_team" in df.columns else ""
                        noise = _stable_noise2(h, a)
                        tempo_avg = tempo_avg_series[idx] if not (isinstance(tempo_avg_series, float) or pd.isna(tempo_avg_series[idx])) else np.nan
                        tempo_component = ((tempo_avg - 70.0) * 0.65) if (not pd.isna(tempo_avg)) else 0.0
                        val = baseline_league_avg + tempo_component + noise
                        val = float(np.clip(val, 60, 192))
                        df.at[idx, "pred_total"] = val
                        try:
                            pipeline_stats.setdefault("synthetic_baseline_fills_no_market", 0)
                            pipeline_stats["synthetic_baseline_fills_no_market"] += 1
                        except Exception:
                            pass
                        if "pred_total_basis" in df.columns and pd.isna(df.at[idx, "pred_total_basis"]):
                            df.at[idx, "pred_total_basis"] = "synthetic_baseline_nomkt"
                        elif "pred_total_basis" not in df.columns:
                            df.loc[idx, "pred_total_basis"] = "synthetic_baseline_nomkt"
            # Post synthetic fill instrumentation
            try:
                post_missing = df["pred_total"].isna()
                pipeline_stats["pred_total_missing_post_fill"] = int(post_missing.sum())
                if (pipeline_stats.get("synthetic_baseline_fills", 0) + pipeline_stats.get("synthetic_baseline_fills_no_market", 0)) > 0:
                    sample_cols2 = [
                        "home_team","away_team","market_total","pred_total","pred_total_basis",
                        "home_tempo_rating","away_tempo_rating","home_rtg","away_rtg","spread_home","pred_margin"
                    ]
                    view_cols2 = [c for c in sample_cols2 if c in df.columns]
                    pipeline_stats["pred_total_filled_sample"] = (
                        df.loc[~post_missing, view_cols2].head(5).to_dict(orient="records")
                    )
            except Exception:
                pass
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
                    df["pred_total_adjusted"] = np.where(df["pred_total_basis"].isin(["market_copy","blended_low"]), True, df.get("pred_total_adjusted"))
            # Fallback 3: create pred_margin from spread or rating differentials if entirely missing
            if "pred_margin" in df.columns and df["pred_margin"].isna().all():
                spread_src = None
                for cand in ["spread_home","closing_spread_home"]:
                    if cand in df.columns:
                        spread_src = pd.to_numeric(df[cand], errors="coerce")
                        break
                if spread_src is not None and spread_src.notna().any():
                    df["pred_margin"] = spread_src.map(lambda x: -x if pd.notna(x) else x)
                    df["pred_margin_basis"] = df.get("pred_margin_basis")
                    if "pred_margin_basis" in df.columns:
                        df["pred_margin_basis"] = df["pred_margin_basis"].where(df["pred_margin_basis"].notna(), "synthetic_from_spread")
                else:
                    # Rating differential approach (offensive/defensive or power ratings)
                    if {"home_rtg","away_rtg"}.issubset(df.columns):
                        hr = pd.to_numeric(df["home_rtg"], errors="coerce")
                        ar = pd.to_numeric(df["away_rtg"], errors="coerce")
                        diff = hr - ar
                        if diff.notna().any():
                            df["pred_margin"] = diff
                            if "pred_margin_basis" in df.columns:
                                df["pred_margin_basis"] = df["pred_margin_basis"].where(df["pred_margin_basis"].notna(), "synthetic_rating_diff")
            # Fallback 3b: if still missing pred_margin (no spread, no ratings), force even margin to enable projections
            try:
                if "pred_margin" in df.columns:
                    remaining_pm_nan = df["pred_margin"].isna()
                    if remaining_pm_nan.any():
                        df.loc[remaining_pm_nan, "pred_margin"] = 0.0
                        if "pred_margin_basis" not in df.columns:
                            df["pred_margin_basis"] = None
                        df.loc[remaining_pm_nan, "pred_margin_basis"] = df.loc[remaining_pm_nan, "pred_margin_basis"].where(df.loc[remaining_pm_nan, "pred_margin_basis"].notna(), "synthetic_even")
                        pipeline_stats["pred_margin_even_fills"] = int(remaining_pm_nan.sum())
            except Exception:
                pass
            # Projection population (unconditional): build proj_home/proj_away wherever both pred_total & pred_margin exist
            try:
                if {"pred_total","pred_margin"}.issubset(df.columns):
                    pt_num = pd.to_numeric(df["pred_total"], errors="coerce")
                    pm_num = pd.to_numeric(df["pred_margin"], errors="coerce")
                    if "proj_home" not in df.columns:
                        df["proj_home"] = np.nan
                    if "proj_away" not in df.columns:
                        df["proj_away"] = np.nan
                    mask_proj = pt_num.notna() & pm_num.notna()
                    df.loc[mask_proj, "proj_home"] = (pt_num[mask_proj] / 2.0) + (pm_num[mask_proj] / 2.0)
                    df.loc[mask_proj, "proj_away"] = pt_num[mask_proj] - df.loc[mask_proj, "proj_home"]
                    pipeline_stats["proj_home_rows"] = int(df["proj_home"].notna().sum())
                    pipeline_stats["proj_away_rows"] = int(df["proj_away"].notna().sum())
            except Exception:
                pipeline_stats["proj_population_error"] = True
            # Instrument missing prediction counts
            try:
                pipeline_stats["missing_pred_total_rows"] = int(df["pred_total"].isna().sum())
                pipeline_stats["missing_pred_margin_rows"] = int(df["pred_margin"].isna().sum())
            except Exception:
                pass
            # Synthetic line fallback: if market_total still missing, populate from pred_total or derived_total
            try:
                if "market_total" in df.columns:
                    miss_mt = df["market_total"].isna() | df["market_total"].astype(str).str.lower().isin(["nan","none","null",""])
                    if miss_mt.any():
                        if "pred_total" in df.columns:
                            pmiss = miss_mt & df["pred_total"].notna()
                            if pmiss.any():
                                df.loc[pmiss, "market_total"] = df.loc[pmiss, "pred_total"]
                                df.loc[pmiss, "market_total_basis"] = "synthetic_pred"
                        if "derived_total" in df.columns:
                            rem = miss_mt & df["market_total"].isna() & df["derived_total"].notna()
                            if rem.any():
                                df.loc[rem, "market_total"] = df.loc[rem, "derived_total"]
                                df.loc[rem, "market_total_basis"] = "synthetic_derived"
                # Synthetic spread fallback
                if {"spread_home","pred_margin"}.issubset(df.columns):
                    miss_sp = df["spread_home"].isna() | df["spread_home"].astype(str).str.lower().isin(["nan","none","null",""])
                    if miss_sp.any():
                        pmv = pd.to_numeric(df["pred_margin"], errors="coerce")
                        fill_mask = miss_sp & pmv.notna()
                        if fill_mask.any():
                            # Home favored => negative spread
                            df.loc[fill_mask, "spread_home"] = -pmv[fill_mask]
                            df.loc[fill_mask, "spread_home_basis"] = "synthetic_pred_margin"
            except Exception:
                pass
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
                        raw["_home_norm"] = raw[home_col].astype(str).map(_canon_slug)
                        raw["_away_norm"] = raw[away_col].astype(str).map(_canon_slug)
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
                            hn = _canon_slug(str(row.get("home_team") or ""))
                            an = _canon_slug(str(row.get("away_team") or ""))
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
                # Half-lines fallback fill (1H/2H) using raw odds period labels when joined odds lack halves
                need_1h = ("market_total_1h" not in df.columns) or df["market_total_1h"].isna().sum() >= int(0.5*len(df))
                need_2h = ("market_total_2h" not in df.columns) or df["market_total_2h"].isna().sum() >= int(0.5*len(df))
                if (need_1h or need_2h) and raw_file.exists():
                    raw = pd.read_csv(raw_file)
                    if not raw.empty:
                        # Normalize period in raw
                        per = raw.get("period")
                        if per is not None:
                            raw["_period_norm"] = per.map(lambda v: re.sub(r"[^a-z0-9]+","_", str(v).lower()).strip("_"))
                        else:
                            raw["_period_norm"] = ""
                        # Prepare normalized team keys
                        home_col = next((c for c in ["home_team_name","home_team","home"] if c in raw.columns), None)
                        away_col = next((c for c in ["away_team_name","away_team","away"] if c in raw.columns), None)
                        if home_col and away_col:
                            raw["_home_norm"] = raw[home_col].astype(str).map(_canon_slug)
                            raw["_away_norm"] = raw[away_col].astype(str).map(_canon_slug)
                            # Helper to aggregate from a subset filtered by period and unordered team set
                            def _fill_half_for_idx(idx, period_keys: set[str], tgt_prefix: str):
                                row = df.loc[idx]
                                hn = _canon_slug(str(row.get("home_team") or ""))
                                an = _canon_slug(str(row.get("away_team") or ""))
                                if not hn or not an:
                                    return
                                sub = raw[raw["_period_norm"].isin(period_keys)] if "_period_norm" in raw.columns else raw
                                if sub.empty:
                                    return
                                # Filter to unordered pair
                                sub = sub[((sub["_home_norm"]==hn) & (sub["_away_norm"]==an)) | ((sub["_home_norm"]==an) & (sub["_away_norm"]==hn))]
                                if sub.empty:
                                    return
                                # Totals
                                if (tgt_prefix+"total") in ["market_total_1h","market_total_2h"]:
                                    if (df.get(tgt_prefix+"total") is None) or pd.isna(df.at[idx, tgt_prefix+"total"]):
                                        for tc in ["total","over_under","market_total","line_total"]:
                                            if tc in sub.columns and sub[tc].notna().any():
                                                df.at[idx, tgt_prefix+"total"] = float(pd.to_numeric(sub[tc], errors="coerce").median())
                                                break
                                # Spreads
                                if (tgt_prefix+"spread_home") in ["spread_home_1h","spread_home_2h"]:
                                    if (df.get(tgt_prefix+"spread_home") is None) or pd.isna(df.at[idx, tgt_prefix+"spread_home"]):
                                        for sc in ["home_spread","spread","handicap_home","spread_home"]:
                                            if sc in sub.columns and sub[sc].notna().any():
                                                df.at[idx, tgt_prefix+"spread_home"] = float(pd.to_numeric(sub[sc], errors="coerce").median())
                                                break
                                # Moneyline
                                if (tgt_prefix+"ml_home") in ["ml_home_1h","ml_home_2h"]:
                                    if (df.get(tgt_prefix+"ml_home") is None) or pd.isna(df.at[idx, tgt_prefix+"ml_home"]):
                                        for mc in ["moneyline_home","ml_home","price_home","h2h_home"]:
                                            if mc in sub.columns and sub[mc].notna().any():
                                                df.at[idx, tgt_prefix+"ml_home"] = float(pd.to_numeric(sub[mc], errors="coerce").median())
                                                break
                            # Apply per row
                            PERIOD_1H = {"1h","1st_half","first_half","h1","half_1","1_h","1sthalf","firsthalf"}
                            PERIOD_2H = {"2h","2nd_half","second_half","h2","half_2","2_h","2ndhalf","secondhalf"}
                            for idx in df.index:
                                if need_1h:
                                    _fill_half_for_idx(idx, PERIOD_1H, "market_total_1h"[:-6])  # prefix 'market_total_'
                                    _fill_half_for_idx(idx, PERIOD_1H, "spread_home_1h"[:-3])   # prefix 'spread_home_'
                                    _fill_half_for_idx(idx, PERIOD_1H, "ml_home_1h"[:-3])       # prefix 'ml_home_'
                                if need_2h:
                                    _fill_half_for_idx(idx, PERIOD_2H, "market_total_2h"[:-6])
                                    _fill_half_for_idx(idx, PERIOD_2H, "spread_home_2h"[:-3])
                                    _fill_half_for_idx(idx, PERIOD_2H, "ml_home_2h"[:-3])
    except Exception:
        pass

    # Prefer blended predictions when available (overwrite base preds and mark basis)
    try:
        if not df.empty:
            used_blend = False
            if "pred_total_blend" in df.columns:
                # Preserve original once for diagnostics
                if "pred_total_orig" not in df.columns:
                    df["pred_total_orig"] = df.get("pred_total")
                df["pred_total"] = pd.to_numeric(df["pred_total_blend"], errors="coerce")
                # Stamp basis
                if "pred_total_basis" in df.columns:
                    df["pred_total_basis"] = "blend"
                else:
                    df["pred_total_basis"] = "blend"
                used_blend = True
            if "pred_margin_blend" in df.columns:
                if "pred_margin_orig" not in df.columns:
                    df["pred_margin_orig"] = df.get("pred_margin")
                df["pred_margin"] = pd.to_numeric(df["pred_margin_blend"], errors="coerce")
                if "pred_margin_basis" in df.columns:
                    df["pred_margin_basis"] = "blend"
                else:
                    df["pred_margin_basis"] = "blend"
                used_blend = True
            if used_blend:
                pipeline_stats["using_blended_predictions"] = True
                # Blend diagnostics when available
                try:
                    if "blend_weight" in df.columns:
                        bw = pd.to_numeric(df["blend_weight"], errors="coerce")
                        bw_valid = bw.dropna()
                        if not bw_valid.empty:
                            pipeline_stats["blend_weight_min"] = float(bw_valid.min())
                            pipeline_stats["blend_weight_median"] = float(bw_valid.median())
                            pipeline_stats["blend_weight_max"] = float(bw_valid.max())
                            pipeline_stats["blend_weight_mean"] = float(bw_valid.mean())
                    if "seg_n_rows" in df.columns:
                        sn = pd.to_numeric(df["seg_n_rows"], errors="coerce").dropna()
                        if not sn.empty:
                            pipeline_stats["seg_n_rows_median"] = float(sn.median())
                            pipeline_stats["seg_n_rows_min"] = float(sn.min())
                            pipeline_stats["seg_n_rows_max"] = float(sn.max())
                    # Residual performance comparison baseline vs segmented vs blended (totals & margins)
                    try:
                        # Totals residual metrics vs closing_total
                        if {"closing_total","pred_total_base","pred_total_seg","pred_total"}.issubset(df.columns):
                            ct = pd.to_numeric(df["closing_total"], errors="coerce")
                            base = pd.to_numeric(df["pred_total_base"], errors="coerce")
                            seg = pd.to_numeric(df["pred_total_seg"], errors="coerce")
                            blend = pd.to_numeric(df["pred_total"], errors="coerce")
                            def _mae(a,b):
                                try:
                                    m = (a - b).abs().dropna()
                                    return float(m.mean()) if not m.empty else None
                                except Exception:
                                    return None
                            def _rmse(a,b):
                                try:
                                    d = (a - b).dropna()
                                    return float(np.sqrt((d**2).mean())) if not d.empty else None
                                except Exception:
                                    return None
                            mae_base = _mae(base, ct); mae_seg = _mae(seg, ct); mae_blend = _mae(blend, ct)
                            rmse_base = _rmse(base, ct); rmse_seg = _rmse(seg, ct); rmse_blend = _rmse(blend, ct)
                            if mae_base is not None: pipeline_stats["total_mae_base"] = mae_base
                            if mae_seg is not None: pipeline_stats["total_mae_seg"] = mae_seg
                            if mae_blend is not None: pipeline_stats["total_mae_blend"] = mae_blend
                            if rmse_base is not None: pipeline_stats["total_rmse_base"] = rmse_base
                            if rmse_seg is not None: pipeline_stats["total_rmse_seg"] = rmse_seg
                            if rmse_blend is not None: pipeline_stats["total_rmse_blend"] = rmse_blend
                            if mae_base and mae_blend: pipeline_stats["total_mae_blend_improve_vs_base_pct"] = float((mae_base - mae_blend)/mae_base*100.0)
                            if mae_seg and mae_blend: pipeline_stats["total_mae_blend_improve_vs_seg_pct"] = float((mae_seg - mae_blend)/mae_seg*100.0)
                        # Margin residual metrics vs closing_spread_home
                        if {"closing_spread_home","pred_margin_base","pred_margin_seg","pred_margin"}.issubset(df.columns):
                            cs = pd.to_numeric(df["closing_spread_home"], errors="coerce")
                            mb = pd.to_numeric(df["pred_margin_base"], errors="coerce")
                            ms = pd.to_numeric(df["pred_margin_seg"], errors="coerce")
                            mm = pd.to_numeric(df["pred_margin"], errors="coerce")
                            def _mae2(a,b):
                                try:
                                    m = (a - b).abs().dropna()
                                    return float(m.mean()) if not m.empty else None
                                except Exception:
                                    return None
                            def _rmse2(a,b):
                                try:
                                    d = (a - b).dropna()
                                    return float(np.sqrt((d**2).mean())) if not d.empty else None
                                except Exception:
                                    return None
                            mae_mb = _mae2(mb, cs); mae_ms = _mae2(ms, cs); mae_mm = _mae2(mm, cs)
                            rmse_mb = _rmse2(mb, cs); rmse_ms = _rmse2(ms, cs); rmse_mm = _rmse2(mm, cs)
                            if mae_mb is not None: pipeline_stats["margin_mae_base"] = mae_mb
                            if mae_ms is not None: pipeline_stats["margin_mae_seg"] = mae_ms
                            if mae_mm is not None: pipeline_stats["margin_mae_blend"] = mae_mm
                            if rmse_mb is not None: pipeline_stats["margin_rmse_base"] = rmse_mb
                            if rmse_ms is not None: pipeline_stats["margin_rmse_seg"] = rmse_ms
                            if rmse_mm is not None: pipeline_stats["margin_rmse_blend"] = rmse_mm
                            if mae_mb and mae_mm: pipeline_stats["margin_mae_blend_improve_vs_base_pct"] = float((mae_mb - mae_mm)/mae_mb*100.0)
                            if mae_ms and mae_mm: pipeline_stats["margin_mae_blend_improve_vs_seg_pct"] = float((mae_ms - mae_mm)/mae_ms*100.0)
                    except Exception:
                        pipeline_stats["blend_perf_error"] = True
                    # Prediction confidence scoring using team variance + segment effective rows (totals & margins)
                    try:
                        seg_rows_series = None
                        if "seg_eff_n_rows" in df.columns:
                            seg_rows_series = pd.to_numeric(df["seg_eff_n_rows"], errors="coerce")
                        elif "seg_n_rows" in df.columns:
                            seg_rows_series = pd.to_numeric(df["seg_n_rows"], errors="coerce")
                        # Totals confidence
                        if team_variance_total and {"home_team","away_team","pred_total"}.issubset(df.columns):
                            vh = df["home_team"].map(lambda t: team_variance_total.get(str(t), np.nan))
                            va = df["away_team"].map(lambda t: team_variance_total.get(str(t), np.nan))
                            avg_var = (vh + va) / 2.0
                            if seg_rows_series is not None:
                                seg_factor = seg_rows_series / (seg_rows_series + 50.0)
                            else:
                                seg_factor = 0.3
                            conf = (1.0 / (1.0 + avg_var)) * (0.5 + 0.5 * seg_factor)
                            try:
                                conf = conf.clip(lower=0.0, upper=1.0)
                            except Exception:
                                pass
                            df["pred_total_confidence"] = conf
                            c_valid = pd.to_numeric(conf, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
                            if not c_valid.empty:
                                pipeline_stats["pred_total_conf_q25"] = float(c_valid.quantile(0.25))
                                pipeline_stats["pred_total_conf_median"] = float(c_valid.median())
                                pipeline_stats["pred_total_conf_q75"] = float(c_valid.quantile(0.75))
                                pipeline_stats["pred_total_conf_mean"] = float(c_valid.mean())
                        # Margin confidence
                        if team_variance_margin and {"home_team","away_team","pred_margin"}.issubset(df.columns):
                            vh_m = df["home_team"].map(lambda t: team_variance_margin.get(str(t), np.nan))
                            va_m = df["away_team"].map(lambda t: team_variance_margin.get(str(t), np.nan))
                            avg_var_m = (vh_m + va_m) / 2.0
                            if seg_rows_series is not None:
                                seg_factor_m = seg_rows_series / (seg_rows_series + 50.0)
                            else:
                                seg_factor_m = 0.3
                            conf_m = (1.0 / (1.0 + avg_var_m)) * (0.5 + 0.5 * seg_factor_m)
                            try:
                                conf_m = conf_m.clip(lower=0.0, upper=1.0)
                            except Exception:
                                pass
                            df["pred_margin_confidence"] = conf_m
                            m_valid = pd.to_numeric(conf_m, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
                            if not m_valid.empty:
                                pipeline_stats["pred_margin_conf_q25"] = float(m_valid.quantile(0.25))
                                pipeline_stats["pred_margin_conf_median"] = float(m_valid.median())
                                pipeline_stats["pred_margin_conf_q75"] = float(m_valid.quantile(0.75))
                                pipeline_stats["pred_margin_conf_mean"] = float(m_valid.mean())
                    except Exception:
                        pipeline_stats["prediction_confidence_error"] = True
                except Exception:
                    pass
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
    # Merge model predictions (if not already merged) then compute model-based edges
    try:
        if 'model_preds' in locals() and not model_preds.empty and 'game_id' in df.columns and 'game_id' in model_preds.columns:
            # Avoid duplicate merges; include calibrated columns when present
            all_model_cols = ['pred_total_model','pred_margin_model','pred_total_calibrated','pred_margin_calibrated','pred_total_model_basis','pred_margin_model_basis']
            need_cols = [c for c in all_model_cols if c not in df.columns and c in model_preds.columns]
            if need_cols:
                mp = model_preds.copy()
                mp['game_id'] = mp['game_id'].astype(str)
                df['game_id'] = df['game_id'].astype(str)
                keep = ['game_id'] + need_cols
                df = df.merge(mp[keep], on='game_id', how='left', suffixes=('','_m'))
        # ------------------------------------------------------------------
        # Team-level universal prediction fallback: ensure ALL games have
        # point & margin predictions even if model inference skipped rows.
        # For any missing pred_total_model/pred_margin_model we derive
        # lightweight estimates from features (tempo/off/def) and basic
        # historical averages. This guarantees downstream staking and
        # edge metrics exist for every team.
        # ------------------------------------------------------------------
        if 'game_id' in df.columns:
            # Identify rows missing model totals
            missing_model_total = ('pred_total_model' not in df.columns) or df['pred_total_model'].isna().all() or df['pred_total_model'].isna()
            missing_model_margin = ('pred_margin_model' not in df.columns) or df['pred_margin_model'].isna().all() or df['pred_margin_model'].isna()
            # Load features once for derived tempo/off/def based estimates
            feat_sources = ["features_curr.csv","features_all.csv","features_week.csv","features_last2.csv"]
            feat_df = pd.DataFrame()
            for name in feat_sources:
                p = OUT / name
                if p.exists():
                    try:
                        ftmp = pd.read_csv(p)
                        if not ftmp.empty and 'game_id' in ftmp.columns:
                            feat_df = ftmp
                            break
                    except Exception:
                        continue
            # Feature completeness fallback enrichment
            try:
                feat_df = _feature_fallback_enrich(feat_df)
            except Exception:
                pass
            if not feat_df.empty and 'game_id' in feat_df.columns:
                feat_df['game_id'] = feat_df['game_id'].astype(str)
                # Precompute derived base total from offensive/def + tempo when available
                if {'home_off_rating','away_off_rating','home_def_rating','away_def_rating','home_tempo_rating','away_tempo_rating'}.issubset(feat_df.columns):
                    try:
                        ho = pd.to_numeric(feat_df['home_off_rating'], errors='coerce')
                        ao = pd.to_numeric(feat_df['away_off_rating'], errors='coerce')
                        hd = pd.to_numeric(feat_df['home_def_rating'], errors='coerce')
                        ad = pd.to_numeric(feat_df['away_def_rating'], errors='coerce')
                        ht = pd.to_numeric(feat_df['home_tempo_rating'], errors='coerce')
                        at = pd.to_numeric(feat_df['away_tempo_rating'], errors='coerce')
                        # Simple possession scale (tempo ratings assumed ~70 baseline)
                        poss_scale = (ht + at) / 140.0
                        raw_total_est = ((ho + ao) / 2.0 + (200 - (hd + ad) / 2.0)) * poss_scale * 0.5
                        feat_df['derived_total_est'] = raw_total_est
                    except Exception:
                        feat_df['derived_total_est'] = np.nan
                else:
                    feat_df['derived_total_est'] = np.nan
                # Margin estimate from off/def differential if available
                if {'home_off_rating','home_def_rating','away_off_rating','away_def_rating'}.issubset(feat_df.columns):
                    try:
                        ho = pd.to_numeric(feat_df['home_off_rating'], errors='coerce')
                        hd = pd.to_numeric(feat_df['home_def_rating'], errors='coerce')
                        ao = pd.to_numeric(feat_df['away_off_rating'], errors='coerce')
                        ad = pd.to_numeric(feat_df['away_def_rating'], errors='coerce')
                        feat_df['derived_margin_est'] = (ho - hd) - (ao - ad)
                    except Exception:
                        feat_df['derived_margin_est'] = np.nan
                else:
                    feat_df['derived_margin_est'] = np.nan
                # Build maps
                total_map = feat_df.set_index('game_id')['derived_total_est'].to_dict()
                margin_map = feat_df.set_index('game_id')['derived_margin_est'].to_dict()
            else:
                total_map = {}
                margin_map = {}
            if isinstance(missing_model_total, pd.Series) and missing_model_total.any():
                # League average fallback if no derived feature
                league_avg_total = 141.5
                rng = random.Random(42)
                new_totals = []
                for idx, row in df.iterrows():
                    if missing_model_total.iloc[idx]:
                        gid = str(row.get('game_id'))
                        base = total_map.get(gid, league_avg_total)
                        # Add tiny deterministic jitter from team names to avoid identical predictions
                        ht = str(row.get('home_team',''))
                        at = str(row.get('away_team',''))
                        seed_val = hash(ht + '|' + at) & 0xffff
                        rng.seed(seed_val)
                        jitter = rng.uniform(-2.2, 2.2)
                        new_totals.append(base + jitter)
                    else:
                        new_totals.append(row.get('pred_total_model'))
                df['pred_total_model'] = new_totals
                df['pred_total_model_basis'] = df.get('pred_total_model_basis')
                df['pred_total_model_basis'] = df['pred_total_model_basis'].where(df['pred_total_model_basis'].notna(), 'fallback_derived' if total_map else 'fallback_league_avg')
            if isinstance(missing_model_margin, pd.Series) and missing_model_margin.any():
                league_avg_margin = 4.8
                rng = random.Random(99)
                new_margins = []
                for idx, row in df.iterrows():
                    if missing_model_margin.iloc[idx]:
                        gid = str(row.get('game_id'))
                        base = margin_map.get(gid, league_avg_margin)
                        ht = str(row.get('home_team',''))
                        at = str(row.get('away_team',''))
                        seed_val = hash('m:' + ht + '|' + at) & 0xffff
                        rng.seed(seed_val)
                        jitter = rng.uniform(-1.5, 1.5)
                        new_margins.append(base + jitter)
                    else:
                        new_margins.append(row.get('pred_margin_model'))
                df['pred_margin_model'] = new_margins
                df['pred_margin_model_basis'] = df.get('pred_margin_model_basis')
                df['pred_margin_model_basis'] = df['pred_margin_model_basis'].where(df['pred_margin_model_basis'].notna(), 'fallback_derived' if margin_map else 'fallback_league_avg')
        # Margin sigma-based Kelly adjustment if available
        try:
            if {'kelly_fraction_total','pred_total_sigma'}.issubset(df.columns):
                rel_scale_t = pd.to_numeric(df['pred_total_sigma'], errors='coerce') / max(float(pipeline_stats.get('pred_total_sigma_mean', 12.0)) or 12.0, 1e-6)
                df['kelly_fraction_total_adj'] = pd.to_numeric(df['kelly_fraction_total'], errors='coerce') / rel_scale_t.clip(lower=0.5, upper=2.5)
            if {'kelly_fraction_ml_home','pred_margin_model'}.issubset(df.columns) and 'pred_margin_sigma' in df.columns:
                rel_scale_m = pd.to_numeric(df['pred_margin_sigma'], errors='coerce') / max(float(pipeline_stats.get('pred_margin_sigma_mean', 8.0)) or 8.0, 1e-6)
                df['kelly_fraction_margin_adj'] = pd.to_numeric(df.get('kelly_fraction_ml_home'), errors='coerce') / rel_scale_m.clip(lower=0.5, upper=2.5)
        except Exception:
            pass
        # Compute model edges vs market & closing
        if {'pred_total_model','market_total'}.issubset(df.columns):
            df['edge_total_model'] = pd.to_numeric(df['pred_total_model'], errors='coerce') - pd.to_numeric(df['market_total'], errors='coerce')
        if {'pred_total_model','closing_total'}.issubset(df.columns):
            try:
                df['edge_closing_model'] = pd.to_numeric(df['pred_total_model'], errors='coerce') - pd.to_numeric(df['closing_total'], errors='coerce')
            except Exception:
                df['edge_closing_model'] = None
        if {'pred_margin_model','spread_home'}.issubset(df.columns):
            try:
                # Spread convention: negative spread means home favored; edge is model margin - (-spread_home)
                sh = pd.to_numeric(df['spread_home'], errors='coerce')
                pm = pd.to_numeric(df['pred_margin_model'], errors='coerce')
                df['edge_margin_model'] = pm + sh  # since sh is negative when home favored
            except Exception:
                df['edge_margin_model'] = None
        # Unify: prefer model predictions as primary displayed values; override legacy pred_total/pred_margin if present
        try:
            if 'pred_total_model' in df.columns:
                # Optional bias correction using model_tuning.json totals_bias (if loaded earlier in pipeline_stats)
                bias = None
                try:
                    bias = float(pipeline_stats.get('totals_bias')) if 'pipeline_stats' in locals() and isinstance(pipeline_stats.get('totals_bias'), (int,float)) else None
                except Exception:
                    bias = None
                raw_model_total = pd.to_numeric(df['pred_total_model'], errors='coerce')
                if bias is not None and not raw_model_total.isna().all():
                    raw_model_total = raw_model_total + bias  # shift raw model totals by bias if tuning bias provided
                # Prefer calibrated totals if present; fall back to raw model
                calibrated_available = 'pred_total_calibrated' in df.columns
                calibrated_total = pd.to_numeric(df['pred_total_calibrated'], errors='coerce') if calibrated_available else pd.Series([np.nan]*len(df))
                # Use calibrated when it has any non-NaN values; else fallback to raw
                use_calibrated = calibrated_available and calibrated_total.notna().any()
                chosen_total = calibrated_total if use_calibrated else raw_model_total
                # Bootstrap totals uncertainty (global) if historical residuals available
                try:
                    if 'pred_total_sigma_bootstrap' not in df.columns:
                        boot_sigma_t = np.nan
                        hist_dir_t = OUT / 'daily_results'
                        if hist_dir_t.exists():
                            res_files_t = sorted(hist_dir_t.glob('results_*.csv'))[-60:]
                            actual_list_t = []
                            pred_list_t = []
                            for rp_t in reversed(res_files_t):
                                try:
                                    rdf_t = pd.read_csv(rp_t)
                                except Exception:
                                    continue
                                if rdf_t.empty or not {'home_score','away_score'}.issubset(rdf_t.columns):
                                    continue
                                pt_col = 'pred_total_model' if 'pred_total_model' in rdf_t.columns else ('pred_total' if 'pred_total' in rdf_t.columns else None)
                                if pt_col is None:
                                    continue
                                hs_t = pd.to_numeric(rdf_t['home_score'], errors='coerce')
                                as_t = pd.to_numeric(rdf_t['away_score'], errors='coerce')
                                done_t = hs_t.notna() & as_t.notna() & ((hs_t + as_t) > 0)
                                if done_t.sum() == 0:
                                    continue
                                actual_tot = (hs_t + as_t)[done_t]
                                pred_tot_hist = pd.to_numeric(rdf_t[pt_col], errors='coerce')[done_t]
                                good_t = actual_tot.notna() & pred_tot_hist.notna()
                                if good_t.sum() == 0:
                                    continue
                                actual_list_t.append(actual_tot[good_t])
                                pred_list_t.append(pred_tot_hist[good_t])
                                if sum(len(x) for x in actual_list_t) >= 400:
                                    break
                            hist_rows_t = sum(len(x) for x in actual_list_t)
                            if hist_rows_t >= 25:
                                actual_all_t = pd.concat(actual_list_t, ignore_index=True)
                                pred_all_t = pd.concat(pred_list_t, ignore_index=True)
                                # Use simple residuals of raw predictions (ignore calibration for bootstrap variety)
                                try:
                                    residuals_t = actual_all_t - pred_all_t
                                    boot_sigma_t = float(np.std(residuals_t)) if residuals_t.notna().any() else np.nan
                                except Exception:
                                    boot_sigma_t = np.nan
                                pipeline_stats['totals_bootstrap_rows'] = int(hist_rows_t)
                        df['pred_total_sigma_bootstrap'] = boot_sigma_t
                        pipeline_stats['pred_total_sigma_bootstrap_global'] = boot_sigma_t if not pd.isna(boot_sigma_t) else None
                except Exception:
                    pipeline_stats['pred_total_sigma_bootstrap_error'] = True
                # Record stats for both raw and calibrated distributions
                try:
                    pipeline_stats['pred_total_model_raw_stats'] = {
                        'count': int(raw_model_total.notna().sum()),
                        'min': float(raw_model_total.min()) if raw_model_total.notna().any() else None,
                        'max': float(raw_model_total.max()) if raw_model_total.notna().any() else None,
                        'mean': float(raw_model_total.mean()) if raw_model_total.notna().any() else None,
                        'std': float(raw_model_total.std()) if raw_model_total.notna().any() else None,
                        'unique': int(raw_model_total.nunique()) if raw_model_total.notna().any() else 0
                    }
                except Exception:
                    pass
                if use_calibrated:
                    try:
                        pipeline_stats['pred_total_model_calibrated_stats'] = {
                            'count': int(calibrated_total.notna().sum()),
                            'min': float(calibrated_total.min()) if calibrated_total.notna().any() else None,
                            'max': float(calibrated_total.max()) if calibrated_total.notna().any() else None,
                            'mean': float(calibrated_total.mean()) if calibrated_total.notna().any() else None,
                            'std': float(calibrated_total.std()) if calibrated_total.notna().any() else None,
                            'unique': int(calibrated_total.nunique()) if calibrated_total.notna().any() else 0
                        }
                        pipeline_stats['pred_total_model_calibrated_head'] = calibrated_total.head(10).tolist()
                    except Exception:
                        pass
                try:
                    pipeline_stats['pred_total_model_unified_stats'] = {
                        'count': int(chosen_total.notna().sum()),
                        'min': float(chosen_total.min()) if chosen_total.notna().any() else None,
                        'max': float(chosen_total.max()) if chosen_total.notna().any() else None,
                        'mean': float(chosen_total.mean()) if chosen_total.notna().any() else None,
                        'std': float(chosen_total.std()) if chosen_total.notna().any() else None,
                        'unique': int(chosen_total.nunique()) if chosen_total.notna().any() else 0,
                        'source': 'calibrated' if use_calibrated else 'raw_model'
                    }
                    pipeline_stats['pred_total_model_head_unified'] = chosen_total.head(10).tolist()
                except Exception:
                    pass
                # Preserve raw model under separate column for diagnostics if calibrated used
                df['pred_total_model_raw'] = raw_model_total
                df['pred_total'] = chosen_total
                df['pred_total_basis'] = df.get('pred_total_basis')
                df['pred_total_basis'] = df['pred_total_basis'].where(df['pred_total_basis'].notna(), 'model_calibrated' if use_calibrated else 'model_raw')
                # Recompute edge_total using unified pred_total
                if 'market_total' in df.columns:
                    df['edge_total'] = pd.to_numeric(df['pred_total'], errors='coerce') - pd.to_numeric(df['market_total'], errors='coerce')
                # Meta ensemble totals (optional): combine raw/calibrated/market/derived components with learned weights
                try:
                    if 'pred_total_meta' not in df.columns:
                        meta_path = OUT / 'meta_ensemble_totals.txt'
                        if meta_path.exists():
                            try:
                                import lightgbm as lgb  # type: ignore
                            except Exception:
                                meta_path = None
                        if meta_path and meta_path.exists():
                            # Build feature frame
                            feat_cols = []
                            base_map = {
                                'pred_total_model_raw': 'raw_model_total',
                                'pred_total_calibrated': 'calibrated_total',
                                'market_total': 'market_total',
                                'pred_total': 'unified_pred_total'
                            }
                            tmp = pd.DataFrame({'game_id': df.get('game_id')})
                            for out_col, src_col in base_map.items():
                                if src_col in locals() or src_col in df.columns or out_col in df.columns:
                                    # Use df columns (some already set) for consistency
                                    source_series = df[out_col] if out_col in df.columns else (df[src_col] if src_col in df.columns else None)
                                    if source_series is not None:
                                        tmp[out_col] = pd.to_numeric(source_series, errors='coerce')
                                        feat_cols.append(out_col)
                            # Add market_total and spread derived if present
                            for c in ['market_total','closing_total']:
                                if c in df.columns and c not in tmp.columns:
                                    tmp[c] = pd.to_numeric(df[c], errors='coerce')
                                    feat_cols.append(c)
                            # LightGBM expects no NaNs; simple impute with column means
                            if feat_cols:
                                for c in feat_cols:
                                    if c in tmp.columns:
                                        ser = tmp[c]
                                        if ser.isna().any():
                                            mval = ser.mean()
                                            tmp[c] = ser.fillna(mval)
                                try:
                                    booster = lgb.Booster(model_file=str(meta_path))
                                    meta_pred = booster.predict(tmp[feat_cols])
                                    df['pred_total_meta'] = meta_pred
                                    pipeline_stats['meta_ensemble_totals_rows'] = int(len(df))
                                    pipeline_stats['meta_ensemble_totals_features'] = feat_cols
                                except Exception:
                                    pipeline_stats['meta_ensemble_totals_error'] = True
                except Exception:
                    pipeline_stats['meta_ensemble_totals_error'] = True
            if 'pred_margin_model' in df.columns:
                adj_model_margin = pd.to_numeric(df['pred_margin_model'], errors='coerce')
                # ------------------------------
                # Margin calibration expansion
                # Fit linear calibration: actual_margin ~ pred_margin_model
                # Uses recent daily_results history; falls back to raw if insufficient.
                # ------------------------------
                calib_intercept = 0.0
                calib_slope = 1.0
                calib_rows = 0
                actual_all = None
                pred_all = None
                try:
                    hist_dir = OUT / 'daily_results'
                    if hist_dir.exists():
                        result_files = sorted(hist_dir.glob('results_*.csv'))[-60:]  # last ~60 days max
                        actual_list = []
                        pred_list = []
                        for rp in reversed(result_files):  # iterate newest first for recency weighting
                            try:
                                rdf = pd.read_csv(rp)
                            except Exception:
                                continue
                            if rdf.empty or not {'home_score','away_score'}.issubset(rdf.columns):
                                continue
                            pm_col = 'pred_margin_model' if 'pred_margin_model' in rdf.columns else ('pred_margin' if 'pred_margin' in rdf.columns else None)
                            if pm_col is None:
                                continue
                            hs = pd.to_numeric(rdf['home_score'], errors='coerce')
                            as_ = pd.to_numeric(rdf['away_score'], errors='coerce')
                            done_mask = hs.notna() & as_.notna() & ((hs + as_) > 0)
                            if done_mask.sum() == 0:
                                continue
                            actual_margin = (hs - as_)[done_mask]
                            pred_margin_hist = pd.to_numeric(rdf[pm_col], errors='coerce')[done_mask]
                            good_mask = actual_margin.notna() & pred_margin_hist.notna()
                            if good_mask.sum() == 0:
                                continue
                            actual_list.append(actual_margin[good_mask])
                            pred_list.append(pred_margin_hist[good_mask])
                            calib_rows = sum(len(x) for x in actual_list)
                            if calib_rows >= 400:
                                break
                        if calib_rows >= 25:
                            actual_all = pd.concat(actual_list, ignore_index=True)
                            pred_all = pd.concat(pred_list, ignore_index=True)
                            try:
                                var_pred = float(np.var(pred_all))
                                if var_pred > 1e-8:
                                    cov = float(np.cov(pred_all, actual_all)[0,1])
                                    calib_slope = cov / var_pred
                                    calib_intercept = float(np.mean(actual_all)) - calib_slope * float(np.mean(pred_all))
                                # Guardrails
                                calib_slope = float(np.clip(calib_slope, 0.4, 1.6))
                                calib_intercept = float(np.clip(calib_intercept, -8.0, 8.0))
                            except Exception:
                                calib_intercept, calib_slope = 0.0, 1.0
                        pipeline_stats['margin_calibration_rows'] = int(calib_rows)
                        pipeline_stats['margin_calibration_intercept'] = float(calib_intercept)
                        pipeline_stats['margin_calibration_slope'] = float(calib_slope)
                except Exception:
                    pipeline_stats['margin_calibration_error'] = True
                # Apply calibration decision
                if calib_rows >= 25:
                    calibrated_margin = calib_intercept + calib_slope * adj_model_margin
                    df['pred_margin_calibrated'] = calibrated_margin
                    chosen_margin = calibrated_margin
                    margin_basis_code = 'model_calibrated'
                else:
                    df['pred_margin_calibrated'] = np.nan
                    chosen_margin = adj_model_margin
                    margin_basis_code = 'model_raw'
                # Bootstrap global margin uncertainty if not already present
                try:
                    if 'pred_margin_sigma_bootstrap' not in df.columns:
                        if calib_rows >= 25 and actual_all is not None and pred_all is not None:
                            residuals = actual_all - (calib_intercept + calib_slope * pred_all)
                            boot_sigma = float(np.std(residuals)) if residuals.notna().any() else np.nan
                        else:
                            boot_sigma = np.nan
                        df['pred_margin_sigma_bootstrap'] = boot_sigma
                        pipeline_stats['pred_margin_sigma_bootstrap_global'] = boot_sigma if not pd.isna(boot_sigma) else None
                except Exception:
                    pass
                df['pred_margin'] = chosen_margin
                df['pred_margin_basis'] = df.get('pred_margin_basis')
                df['pred_margin_basis'] = df['pred_margin_basis'].where(df['pred_margin_basis'].notna(), margin_basis_code)
                if 'spread_home' in df.columns:
                    sh2 = pd.to_numeric(df['spread_home'], errors='coerce')
                    df['edge_ats'] = pd.to_numeric(df['pred_margin'], errors='coerce') - sh2
                # Recompute favored side / by values
                pm_num = pd.to_numeric(df['pred_margin'], errors='coerce')
                df['favored_side'] = np.where(pm_num > 0, 'Home', np.where(pm_num < 0, 'Away', 'Even'))
                df['favored_by'] = pm_num.abs()
                # Meta ensemble margin (optional)
                try:
                    if 'pred_margin_meta' not in df.columns:
                        meta_path_m = OUT / 'meta_ensemble_margin.txt'
                        if meta_path_m.exists():
                            try:
                                import lightgbm as lgb  # type: ignore
                            except Exception:
                                meta_path_m = None
                        if meta_path_m and meta_path_m.exists():
                            feat_cols_m = []
                            tmpm = pd.DataFrame({'game_id': df.get('game_id')})
                            # Candidate sources
                            cand_map_m = {
                                'pred_margin_model': 'pred_margin_model',
                                'pred_margin_calibrated': 'pred_margin_calibrated',
                                'spread_home': 'spread_home'
                            }
                            for out_col, src_col in cand_map_m.items():
                                if src_col in df.columns:
                                    tmpm[out_col] = pd.to_numeric(df[src_col], errors='coerce')
                                    feat_cols_m.append(out_col)
                            if feat_cols_m:
                                for c in feat_cols_m:
                                    ser = tmpm[c]
                                    if ser.isna().any():
                                        tmpm[c] = ser.fillna(ser.mean())
                                try:
                                    import lightgbm as lgb  # type: ignore
                                    booster_m = lgb.Booster(model_file=str(meta_path_m))
                                    meta_margin_pred = booster_m.predict(tmpm[feat_cols_m])
                                    df['pred_margin_meta'] = meta_margin_pred
                                    pipeline_stats['meta_ensemble_margin_rows'] = int(len(df))
                                    pipeline_stats['meta_ensemble_margin_features'] = feat_cols_m
                                except Exception:
                                    pipeline_stats['meta_ensemble_margin_error'] = True
                except Exception:
                    pipeline_stats['meta_ensemble_margin_error'] = True
        except Exception:
            pass
        # Correlation & aggregate diagnostics (legacy divergence vs pred_total removed post-unification)
        if 'pipeline_stats' in locals():
            try:
                if {'pred_total_model','market_total'}.issubset(df.columns):
                    corr_mt = df[['pred_total_model','market_total']].dropna()
                    if not corr_mt.empty:
                        pipeline_stats['corr_pred_total_model_market'] = float(corr_mt.corr().iloc[0,1])
                if 'market_total_basis' in df.columns:
                    pipeline_stats['synthetic_market_total_count'] = int((df['market_total_basis'].astype(str).str.startswith('synthetic')).sum())
                if 'spread_home_basis' in df.columns:
                    pipeline_stats['synthetic_spread_home_count'] = int((df['spread_home_basis'].astype(str).str.startswith('synthetic')).sum())
                if 'edge_total_model' in df.columns:
                    pipeline_stats['model_edge_rows'] = int(df['edge_total_model'].notna().sum())
                # Record bias applied if any
                if 'totals_bias' in pipeline_stats and 'pred_total_model' in df.columns:
                    pipeline_stats['model_totals_bias_applied'] = pipeline_stats.get('totals_bias')
                # Adaptive team variance application: enrich per-row sigma columns using team-level rolling std
                try:
                    if team_variance_total and 'home_team' in df.columns and 'away_team' in df.columns:
                        def _tv_total(row):
                            ht = str(row.get('home_team'))
                            at = str(row.get('away_team'))
                            v1 = team_variance_total.get(ht)
                            v2 = team_variance_total.get(at)
                            if v1 is None and v2 is None:
                                return None
                            if v1 is None: v1 = v2
                            if v2 is None: v2 = v1
                            return float(np.mean([v for v in [v1, v2] if v is not None])) if v1 or v2 else None
                        df['pred_total_sigma_team'] = df.apply(_tv_total, axis=1)
                    if team_variance_margin and 'home_team' in df.columns and 'away_team' in df.columns:
                        def _tv_margin(row):
                            ht = str(row.get('home_team'))
                            at = str(row.get('away_team'))
                            v1 = team_variance_margin.get(ht)
                            v2 = team_variance_margin.get(at)
                            if v1 is None and v2 is None:
                                return None
                            if v1 is None: v1 = v2
                            if v2 is None: v2 = v1
                            return float(np.mean([v for v in [v1, v2] if v is not None])) if v1 or v2 else None
                        df['pred_margin_sigma_team'] = df.apply(_tv_margin, axis=1)
                    # Combine baseline sigma with team component when available (geometric mean for stability)
                    if 'pred_total_sigma' in df.columns and 'pred_total_sigma_team' in df.columns:
                        base = pd.to_numeric(df['pred_total_sigma'], errors='coerce')
                        teamc = pd.to_numeric(df['pred_total_sigma_team'], errors='coerce')
                        combo = np.sqrt(base * teamc)
                        df['pred_total_sigma_adaptive'] = np.where(teamc.notna(), combo, base)
                    if 'pred_margin_sigma' in df.columns and 'pred_margin_sigma_team' in df.columns:
                        basem = pd.to_numeric(df['pred_margin_sigma'], errors='coerce')
                        teamm = pd.to_numeric(df['pred_margin_sigma_team'], errors='coerce')
                        combo_m = np.sqrt(basem * teamm)
                        df['pred_margin_sigma_adaptive'] = np.where(teamm.notna(), combo_m, basem)
                    # Pipeline stats aggregates for adaptive sigma
                    if 'pred_total_sigma_adaptive' in df.columns:
                        pipeline_stats['pred_total_sigma_adaptive_mean'] = float(pd.to_numeric(df['pred_total_sigma_adaptive'], errors='coerce').mean())
                    if 'pred_margin_sigma_adaptive' in df.columns:
                        pipeline_stats['pred_margin_sigma_adaptive_mean'] = float(pd.to_numeric(df['pred_margin_sigma_adaptive'], errors='coerce').mean())
                    # Context note when recalibration flagged
                    if pipeline_stats.get('recalibration_needed'):
                        pipeline_stats['recalibration_variance_context'] = 'Adaptive sigmas computed; recalibration flag active.'
                except Exception:
                    pipeline_stats['adaptive_team_variance_error'] = True
            except Exception:
                pass
    except Exception:
        pass
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

    # Dynamic weighting: blend model predictions with baseline totals when health degraded.
    try:
        if 'perf_model_health' in pipeline_stats and 'pred_total_model' in df.columns and 'pred_total' in df.columns:
            health = pipeline_stats.get('perf_model_health')
            ps = pipeline_stats.get('perf_predictability_score')
            if isinstance(ps, (int, float)):
                base_w = max(0.3, min(1.0, ps / 0.8))  # scale relative to desired high score ~0.8
            else:
                base_w = 0.6
            if health == 'critical':
                w = base_w * 0.55
            elif health == 'degraded':
                w = base_w * 0.75
            else:
                w = base_w
            w = max(0.25, min(0.95, w))
            pm = pd.to_numeric(df['pred_total_model'], errors='coerce')
            pb = pd.to_numeric(df['pred_total'], errors='coerce')
            blended = (w * pm) + ((1 - w) * pb)
            mask = pm.notna() & pb.notna()
            if mask.any():
                df.loc[mask, 'pred_total_model_blended'] = blended[mask]
                df.loc[mask, 'pred_total_model_weight'] = w
                pipeline_stats['dynamic_weight_applied'] = int(mask.sum())
                pipeline_stats['dynamic_weight_w'] = w
    except Exception:
        pipeline_stats['dynamic_weight_error'] = True

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
            # Pair-based fallback maps (using canonical slugs) to handle game_id mismatches due to normalization
            derived_pair_map: dict[str, float] = {}
            derived_margin_pair_map: dict[str, float] = {}
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
                # Precompute canonical pair key when possible
                try:
                    if {"home_team","away_team"}.issubset(feat_df.columns):
                        feat_df["_home_slug"] = feat_df["home_team"].astype(str).map(_canon_slug)
                        feat_df["_away_slug"] = feat_df["away_team"].astype(str).map(_canon_slug)
                        feat_df["_pair_key"] = feat_df.apply(lambda rr: "::".join(sorted([str(rr.get("_home_slug")), str(rr.get("_away_slug"))])), axis=1)
                except Exception:
                    pass
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
                        # Also record by pair when available
                        pk = r.get("_pair_key")
                        if isinstance(pk, str) and pk:
                            derived_pair_map[pk] = derived_total
                            derived_margin_pair_map[pk] = derived_margin
                    except Exception:
                        continue
            # If pred_total missing, fill from derived when available; also fill pred_margin if missing
            if derived_map:
                if "pred_total" in df.columns:
                    mask_missing_pt = df["pred_total"].isna()
                    if mask_missing_pt.any():
                        # Try by game_id; if missing, fallback via pair-key lookup
                        def _fill_pt(idx, gid):
                            val = derived_map.get(str(gid))
                            if val is not None:
                                return val
                            # Build row pair key
                            try:
                                hn = _canon_slug(str(df.at[idx, "home_team"]))
                                an = _canon_slug(str(df.at[idx, "away_team"]))
                                pk = "::".join(sorted([hn, an]))
                                return derived_pair_map.get(pk)
                            except Exception:
                                return None
                        for idx in df.index[mask_missing_pt]:
                            df.loc[idx, "pred_total"] = _fill_pt(idx, df.at[idx, "game_id"])
                        # Mark basis for visibility
                        df.loc[mask_missing_pt, "pred_total_basis"] = df.loc[mask_missing_pt, "pred_total_basis"].where(df.loc[mask_missing_pt, "pred_total_basis"].notna(), "features_derived") if "pred_total_basis" in df.columns else "features_derived"
                if "pred_margin" in df.columns and derived_margin_map:
                    mask_missing_pm = df["pred_margin"].isna()
                    if mask_missing_pm.any():
                        def _fill_pm(idx, gid):
                            val = derived_margin_map.get(str(gid))
                            if val is not None:
                                return val
                            try:
                                hn = _canon_slug(str(df.at[idx, "home_team"]))
                                an = _canon_slug(str(df.at[idx, "away_team"]))
                                pk = "::".join(sorted([hn, an]))
                                return derived_margin_pair_map.get(pk)
                            except Exception:
                                return None
                        for idx in df.index[mask_missing_pm]:
                            df.loc[idx, "pred_margin"] = _fill_pm(idx, df.at[idx, "game_id"])
                        df.loc[mask_missing_pm, "pred_margin_basis"] = df.loc[mask_missing_pm, "pred_margin_basis"].where(df.loc[mask_missing_pm, "pred_margin_basis"].notna(), "features_derived") if "pred_margin_basis" in df.columns else "features_derived"
            # Blend predictions when implausibly low vs derived; capture instrumentation for diagnostics
            pred_total_was_adj = []
            blend_weights_model: list[float] = []
            blend_weights_derived: list[float] = []
            pred_total_derived_used: list[float | None] = []
            pred_total_raw_list: list[float | None] = []
            blend_severe_flags: list[bool] = []
            if derived_map:
                adj_vals = []
                for _, r in df.iterrows():
                    gid = str(r.get("game_id"))
                    pred = r.get("pred_total")
                    pred_total_raw_list.append(r.get("pred_total_raw"))
                    derived = derived_map.get(gid)
                    if derived is None:
                        try:
                            hn = _canon_slug(str(r.get("home_team")))
                            an = _canon_slug(str(r.get("away_team")))
                            pk = "::".join(sorted([hn, an]))
                            derived = derived_pair_map.get(pk)
                        except Exception:
                            derived = None
                    pred_total_derived_used.append(derived)
                    if pred is None or pd.isna(pred) or derived is None:
                        # No adjustment possible
                        adj_vals.append(pred)
                        pred_total_was_adj.append(False)
                        blend_weights_model.append(np.nan)
                        blend_weights_derived.append(np.nan)
                        blend_severe_flags.append(False)
                        continue
                    # Revised adjustment criteria:
                    # Only blend when derived appears substantially higher AND credible (>=130) and raw pred is far lower.
                    # Skip adjustment if derived is near lower clamp (<=120) to avoid uniform floor inflation.
                    derived_val = float(derived) if derived is not None else None
                    pred_val = float(pred) if pred is not None else None
                    severe_gap = False
                    do_blend = False
                    if derived_val is not None and pred_val is not None:
                        if derived_val >= 130 and pred_val < 0.78 * derived_val and (derived_val - pred_val) >= 18:
                            do_blend = True
                            severe_gap = pred_val < 0.70 * derived_val
                    if do_blend:
                        w_model = 0.45 if severe_gap else 0.55  # lean slightly more to model than before
                        w_derived = 1.0 - w_model
                        blended = w_model * pred_val + w_derived * derived_val
                        # Allow lower values (remove hard 112 floor); retain upper plausibility cap
                        blended = float(np.clip(blended, 100, 188))
                        adj_vals.append(blended)
                        pred_total_was_adj.append(True)
                        blend_weights_model.append(w_model)
                        blend_weights_derived.append(w_derived)
                        blend_severe_flags.append(severe_gap)
                    else:
                        adj_vals.append(pred_val)
                        pred_total_was_adj.append(False)
                        blend_weights_model.append(1.0)
                        blend_weights_derived.append(0.0)
                        blend_severe_flags.append(False)
                df["pred_total"] = adj_vals
                df["pred_total_adjusted"] = pred_total_was_adj
                df["pred_total_blend_w_model"] = blend_weights_model
                df["pred_total_blend_w_derived"] = blend_weights_derived
                df["pred_total_blend_severe"] = blend_severe_flags
                try:
                    pt_post = pd.to_numeric(df['pred_total'], errors='coerce')
                    pipeline_stats['pred_total_post_blend_stats'] = {
                        'min': float(pt_post.min()) if pt_post.notna().any() else None,
                        'max': float(pt_post.max()) if pt_post.notna().any() else None,
                        'mean': float(pt_post.mean()) if pt_post.notna().any() else None,
                        'std': float(pt_post.std()) if pt_post.notna().any() else None,
                        'unique': int(pt_post.nunique()) if pt_post.notna().any() else 0,
                        'adjusted_rows': int(pd.Series(pred_total_was_adj).sum()),
                        'skipped_rows': int(len(pred_total_was_adj) - pd.Series(pred_total_was_adj).sum())
                    }
                except Exception:
                    pass
                df["derived_total"] = pred_total_derived_used if "derived_total" not in df.columns else df["derived_total"].where(df["derived_total"].notna(), pred_total_derived_used)
                # Keep raw value accessible if overwritten
                if "pred_total_raw" not in df.columns:
                    df["pred_total_raw"] = pred_total_raw_list
                # Basis column for template badge
                if "pred_total_basis" not in df.columns:
                    df["pred_total_basis"] = None
                df["pred_total_basis"] = np.where(df["pred_total_adjusted"], df["pred_total_basis"].where(df["pred_total_basis"].notna(), "blended_low"), df["pred_total_basis"])
                try:
                    final_pt = pd.to_numeric(df['pred_total'], errors='coerce')
                    pipeline_stats['pred_total_final_stats'] = {
                        'count': int(final_pt.notna().sum()),
                        'min': float(final_pt.min()) if final_pt.notna().any() else None,
                        'max': float(final_pt.max()) if final_pt.notna().any() else None,
                        'mean': float(final_pt.mean()) if final_pt.notna().any() else None,
                        'std': float(final_pt.std()) if final_pt.notna().any() else None,
                        'unique': int(final_pt.nunique()) if final_pt.notna().any() else 0,
                        'adjusted_rows': int(df.get('pred_total_adjusted', pd.Series(dtype=int)).sum() if 'pred_total_adjusted' in df.columns else 0)
                    }
                    pipeline_stats['pred_total_head_final'] = final_pt.head(10).tolist()
                    try:
                        pipeline_stats['pred_total_missing_final'] = int(final_pt.isna().sum())
                    except Exception:
                        pass
                except Exception:
                    pass
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
            # ----------------------------------------------------------
            # Augment picks with model uncertainty & adjusted Kelly if available
            # Adds columns: total_sigma, margin_sigma, kelly_total_adj, kelly_margin_adj
            # Persists back to picks_raw.csv if new columns introduced.
            # ----------------------------------------------------------
            try:
                if not df.empty and 'game_id' in df.columns:
                    gmap_sigma_t = df.set_index('game_id')['pred_total_sigma'] if 'pred_total_sigma' in df.columns else None
                    gmap_sigma_m = df.set_index('game_id')['pred_margin_sigma'] if 'pred_margin_sigma' in df.columns else None
                    gmap_sigma_t_boot = df.set_index('game_id')['pred_total_sigma_bootstrap'] if 'pred_total_sigma_bootstrap' in df.columns else None
                    gmap_sigma_m_boot = df.set_index('game_id')['pred_margin_sigma_bootstrap'] if 'pred_margin_sigma_bootstrap' in df.columns else None
                    gmap_kelly_t_adj = df.set_index('game_id')['kelly_fraction_total_adj'] if 'kelly_fraction_total_adj' in df.columns else None
                    gmap_kelly_m_adj = df.set_index('game_id')['kelly_fraction_margin_adj'] if 'kelly_fraction_margin_adj' in df.columns else None
                    if gmap_sigma_t is not None and 'total_sigma' not in picks_df.columns:
                        picks_df['total_sigma'] = picks_df['game_id'].map(gmap_sigma_t)
                    if gmap_sigma_m is not None and 'margin_sigma' not in picks_df.columns:
                        picks_df['margin_sigma'] = picks_df['game_id'].map(gmap_sigma_m)
                    if gmap_sigma_t_boot is not None and 'total_sigma_bootstrap' not in picks_df.columns:
                        picks_df['total_sigma_bootstrap'] = picks_df['game_id'].map(gmap_sigma_t_boot)
                    if gmap_sigma_m_boot is not None and 'margin_sigma_bootstrap' not in picks_df.columns:
                        picks_df['margin_sigma_bootstrap'] = picks_df['game_id'].map(gmap_sigma_m_boot)
                    if gmap_kelly_t_adj is not None and 'kelly_total_adj' not in picks_df.columns:
                        picks_df['kelly_total_adj'] = picks_df['game_id'].map(gmap_kelly_t_adj)
                    if gmap_kelly_m_adj is not None and 'kelly_margin_adj' not in picks_df.columns:
                        picks_df['kelly_margin_adj'] = picks_df['game_id'].map(gmap_kelly_m_adj)
                    # Persist enriched picks back to disk (best-effort)
                    try:
                        raw_path_out = OUT / 'picks_raw.csv'
                        picks_df.to_csv(raw_path_out, index=False)
                        pipeline_stats['picks_raw_enriched'] = True
                    except Exception:
                        pipeline_stats['picks_raw_enriched'] = False
            except Exception:
                pipeline_stats['picks_raw_enriched_error'] = True
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
    # ------------------------------------------------------------------
    # Coverage supplementation: append any games or model prediction rows
    # that are missing after daily_df or merge logic. This addresses cases
    # where a game (e.g., Marshall vs Arkansas-Pine Bluff) exists in
    # games_curr.csv and predictions_model_<date>.csv but is absent from
    # the unified display DataFrame due to earlier branch selection.
    # ------------------------------------------------------------------
    try:
        # Ensure df exists and has a game_id column
        if isinstance(df, pd.DataFrame) and 'game_id' in df.columns:
            existing_ids = set(df['game_id'].astype(str))
            games_curr_path = OUT / 'games_curr.csv'
            games_curr_df = _safe_read_csv(games_curr_path) if games_curr_path.exists() else pd.DataFrame()
            if not games_curr_df.empty and 'game_id' in games_curr_df.columns:
                games_curr_df['game_id'] = games_curr_df['game_id'].astype(str)
                if date_q and 'date' in games_curr_df.columns:
                    try:
                        games_curr_df['date'] = pd.to_datetime(games_curr_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        games_curr_df = games_curr_df[games_curr_df['date'] == str(date_q)]
                    except Exception:
                        pass
                # Identify missing schedule rows
                missing_game_ids = [gid for gid in games_curr_df['game_id'] if gid not in existing_ids]
                if missing_game_ids:
                    add_games = games_curr_df[games_curr_df['game_id'].isin(missing_game_ids)].copy()
                    # Guarantee prediction placeholder columns
                    for col in ('pred_total','pred_margin'):
                        if col not in add_games.columns:
                            add_games[col] = np.nan
                    df = pd.concat([df, add_games], ignore_index=True)
                    pipeline_stats['appended_missing_games_rows'] = int(len(add_games))
                    existing_ids.update(missing_game_ids)
            # Add missing model predictions as rows
            if 'model_preds' in locals() and isinstance(model_preds, pd.DataFrame) and not model_preds.empty and 'game_id' in model_preds.columns:
                mp = model_preds.copy()
                mp['game_id'] = mp['game_id'].astype(str)
                missing_model_ids = [gid for gid in mp['game_id'] if gid not in existing_ids]
                if missing_model_ids:
                    add_mp = mp[mp['game_id'].isin(missing_model_ids)].copy()
                    # Enrich with schedule metadata if available
                    if not games_curr_df.empty and 'game_id' in games_curr_df.columns:
                        enrich_cols = [c for c in ['game_id','date','home_team','away_team','start_time','venue'] if c in games_curr_df.columns]
                        if enrich_cols:
                            add_mp = add_mp.merge(games_curr_df[enrich_cols], on='game_id', how='left')
                    # Promote model predictions
                    if 'pred_total_model' in add_mp.columns and 'pred_total' not in add_mp.columns:
                        add_mp['pred_total'] = add_mp['pred_total_model']
                        add_mp['pred_total_basis'] = add_mp.get('pred_total_model_basis','model_raw')
                    if 'pred_margin_model' in add_mp.columns and 'pred_margin' not in add_mp.columns:
                        add_mp['pred_margin'] = add_mp['pred_margin_model']
                        add_mp['pred_margin_basis'] = add_mp.get('pred_margin_model_basis','model')
                    df = pd.concat([df, add_mp], ignore_index=True)
                    pipeline_stats['appended_missing_model_rows'] = int(len(add_mp)) + int(pipeline_stats.get('appended_missing_model_rows', 0))
            # Explicit safeguard: if specific known game_id (Marshall vs Arkansas-Pine Bluff) still missing, force add from sources
            target_ids = []
            try:
                # Try to detect from todays games_curr if no date filter (or matches date_q)
                if not date_q or date_q == today_str:
                    # Hard-coded ID observed earlier
                    target_ids.append('401827130')
            except Exception:
                pass
            for tgt in target_ids:
                if tgt not in set(df['game_id'].astype(str)):
                    # Attempt to build a single-row DataFrame from games_curr + model_preds
                    row_parts = []
                    if not games_curr_df.empty and tgt in set(games_curr_df['game_id']):
                        row_parts.append(games_curr_df[games_curr_df['game_id'] == tgt])
                    if 'model_preds' in locals() and not model_preds.empty:
                        mp_row = model_preds[model_preds['game_id'].astype(str) == tgt]
                        if not mp_row.empty:
                            row_parts.append(mp_row)
                    if row_parts:
                        force_row = row_parts[0].copy()
                        # Add model columns if second part
                        if len(row_parts) > 1:
                            for _, r in row_parts[1].iterrows():
                                for c, v in r.items():
                                    if c not in force_row.columns:
                                        force_row[c] = v
                        # Promote prediction fields
                        if 'pred_total_model' in force_row.columns and 'pred_total' not in force_row.columns:
                            force_row['pred_total'] = force_row['pred_total_model']
                            force_row['pred_total_basis'] = force_row.get('pred_total_model_basis','model_raw')
                        if 'pred_margin_model' in force_row.columns and 'pred_margin' not in force_row.columns:
                            force_row['pred_margin'] = force_row['pred_margin_model']
                            force_row['pred_margin_basis'] = force_row.get('pred_margin_model_basis','model')
                        df = pd.concat([df, force_row], ignore_index=True)
                        pipeline_stats['forced_game_insertion'] = pipeline_stats.get('forced_game_insertion', []) + [tgt]
            pipeline_stats['post_coverage_rows'] = int(len(df))
            pipeline_stats['post_coverage_unique_games'] = int(df['game_id'].nunique())
    except Exception:
        pass

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    # Accept start_time as either datetime or string; parse robustly.
    # Revised: Treat naive timestamps as LOCAL timezone (previously assumed UTC causing double offset subtraction).
    if "start_time" in df.columns:
        try:
            st_series_orig = df["start_time"].astype(str).str.strip()
            st_series = st_series_orig.str.replace("Z", "+00:00", regex=False)
            has_time = st_series.str.contains(r"\d{1,2}:\d{2}", regex=True)
            has_offset = st_series.str.contains(r"[+-]\d{2}:\d{2}$", regex=True) | st_series.str.endswith("Z")
            local_tz = dt.datetime.now().astimezone().tzinfo
            # Parse offset-aware values as UTC then convert to UTC tzinfo; naive values localized to system tz directly.
            parsed = pd.to_datetime(st_series.where(has_offset, None), errors="coerce", utc=True)
            naive_part = pd.to_datetime(st_series.where(~has_offset, None), errors="coerce", utc=False)
            # Localize naive part
            if naive_part.notna().any():
                # Attach local tz (interpret given clock time as local)
                naive_part = naive_part.map(lambda x: x.replace(tzinfo=local_tz) if pd.notna(x) else x)
            # Combine
            combined = parsed.where(parsed.notna(), naive_part)
            df["_start_dt"] = combined
            # Fallback reparsing for failures with time component
            if "_start_dt" in df.columns:
                mask_fail = df["_start_dt"].isna() & has_time
                if mask_fail.any():
                    raw = st_series[mask_fail]
                    raw2 = raw.str.replace(r":(\d{2})(?::\d{2})?", r":\1", regex=True)
                    reparsed_offset = pd.to_datetime(raw2.where(has_offset[mask_fail], None), errors="coerce", utc=True)
                    reparsed_naive = pd.to_datetime(raw2.where(~has_offset[mask_fail], None), errors="coerce", utc=False)
                    if reparsed_naive.notna().any():
                        reparsed_naive = reparsed_naive.map(lambda x: x.replace(tzinfo=local_tz) if pd.notna(x) else x)
                    reparsed_combined = reparsed_offset.where(reparsed_offset.notna(), reparsed_naive)
                    df.loc[mask_fail & reparsed_combined.notna(), "_start_dt"] = reparsed_combined[reparsed_combined.notna()]
            # Parse mode instrumentation
            parse_mode = np.where(df["_start_dt"].isna(), "fail", np.where(has_offset, "offset", np.where(has_time, "naive_local", "date_only")))
            df["start_time_parse_mode"] = parse_mode
            pipeline_stats["start_time_naive_local_count"] = int((parse_mode == "naive_local").sum())
            pipeline_stats["start_time_offset_count"] = int((parse_mode == "offset").sum())
            pipeline_stats["start_time_fail_count"] = int((parse_mode == "fail").sum())
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
            raw_name = str(row.get(f"{side}_team") or "")
            try:
                # Prefer canonical slug that applies custom team_map overrides; fallback to base normalize_name
                tkey = _canon_slug(raw_name)  # type: ignore
            except Exception:
                tkey = normalize_name(raw_name)
            # Attempt branding lookup with canonical key first; fallback to normalized raw if absent
            b = branding.get(tkey) or branding.get(normalize_name(raw_name)) or {}
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
            "spread_home_1h_basis", "market_total_1h_basis",
            # 2nd half
            "pred_total_2h", "pred_margin_2h", "proj_home_2h", "proj_away_2h", "pred_winner_2h",
            "market_total_2h", "spread_home_2h", "ats_result_2h", "actual_total_2h", "home_score_2h", "away_score_2h",
            "spread_home_2h_basis", "market_total_2h_basis",
        ]
        # Half moneylines optional columns for template guards
        extra_opt_cols = [
            "ml_home_1h", "ml_away_1h", "ml_home_2h", "ml_away_2h"
        ]
        for c in required_cols:
            if c not in df.columns:
                df[c] = None
        for c in extra_opt_cols:
            if c not in df.columns:
                df[c] = None
    except Exception:
        pass

    # Ensure half spreads and ATS results are populated (second pass) before template conversion.
    try:
        # Derive half totals if provider halves are missing but full-game total exists
        half_ratio = 0.485
        if "market_total" in df.columns:
            mt_full = pd.to_numeric(df["market_total"], errors="coerce")
            # 1H total
            if ("market_total_1h" not in df.columns) or df.get("market_total_1h").isna().all():
                df["market_total_1h"] = np.where(mt_full.notna(), mt_full * half_ratio, df.get("market_total_1h"))
                # mark that these were derived when no provider 1H line exists
                df["market_total_1h_basis"] = np.where(mt_full.notna(), "derived", df.get("market_total_1h_basis"))
            # 2H total derived as remainder from full game
            if ("market_total_2h" not in df.columns) or df.get("market_total_2h").isna().all():
                # Prefer existing 1H (provider or derived) then subtract from full
                mt1 = pd.to_numeric(df.get("market_total_1h"), errors="coerce") if "market_total_1h" in df.columns else pd.Series(np.nan, index=df.index)
                df["market_total_2h"] = np.where(mt_full.notna() & mt1.notna(), mt_full - mt1, np.where(mt_full.notna(), mt_full * (1.0 - half_ratio), df.get("market_total_2h")))
                df["market_total_2h_basis"] = np.where(mt_full.notna(), "derived", df.get("market_total_2h_basis"))
            # Recompute half OU edges if we just filled totals
            try:
                if {"pred_total_1h","market_total_1h"}.issubset(df.columns):
                    pt1 = pd.to_numeric(df["pred_total_1h"], errors="coerce")
                    mt1 = pd.to_numeric(df["market_total_1h"], errors="coerce")
                    need = ("edge_total_1h" not in df.columns) or df["edge_total_1h"].isna()
                    df["edge_total_1h"] = np.where(need, pt1 - mt1, df.get("edge_total_1h"))
                if {"pred_total_2h","market_total_2h"}.issubset(df.columns):
                    pt2 = pd.to_numeric(df["pred_total_2h"], errors="coerce")
                    mt2 = pd.to_numeric(df["market_total_2h"], errors="coerce")
                    need2 = ("edge_total_2h" not in df.columns) or df["edge_total_2h"].isna()
                    df["edge_total_2h"] = np.where(need2, pt2 - mt2, df.get("edge_total_2h"))
            except Exception:
                pass
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
            if ("ats_result_1h" not in df.columns):
                df["ats_result_1h"] = pd.Series([None]*len(df), dtype="object")
            else:
                try:
                    df["ats_result_1h"] = df["ats_result_1h"].astype("object")
                except Exception:
                    pass
            df.loc[mask1 & df["ats_result_1h"].isna(), "ats_result_1h"] = ats1[mask1 & df["ats_result_1h"].isna()]
        if {"home_score_2h","away_score_2h","spread_home_2h"}.issubset(df.columns):
            hs2 = pd.to_numeric(df["home_score_2h"], errors="coerce")
            as2 = pd.to_numeric(df["away_score_2h"], errors="coerce")
            sh2 = pd.to_numeric(df["spread_home_2h"], errors="coerce")
            am2 = hs2 - as2
            mask2 = hs2.notna() & as2.notna() & sh2.notna()
            ats2 = np.where(am2 > -sh2, "Home Cover", np.where(am2 < -sh2, "Away Cover", "Push"))
            if ("ats_result_2h" not in df.columns):
                df["ats_result_2h"] = pd.Series([None]*len(df), dtype="object")
            else:
                try:
                    df["ats_result_2h"] = df["ats_result_2h"].astype("object")
                except Exception:
                    pass
            df.loc[mask2 & df["ats_result_2h"].isna(), "ats_result_2h"] = ats2[mask2 & df["ats_result_2h"].isna()]
    except Exception:
        pass
    # Replace NaN with None for template-friendly rendering (avoid 'nan' text everywhere)
    try:
        df_tpl = df.where(pd.notna(df), None)
    except Exception:
        df_tpl = df

    # Diagnostics enrichment (after full enrichment & NaN replacement): summarize missing odds & low predictions.
    try:
        # Odds coverage metrics
        if "market_total" in df_tpl.columns:
            pipeline_stats["post_missing_market_total"] = int(pd.to_numeric(df_tpl["market_total"], errors="coerce").isna().sum())
        if "spread_home" in df_tpl.columns:
            pipeline_stats["post_missing_spread_home"] = int(pd.to_numeric(df_tpl["spread_home"], errors="coerce").isna().sum())
        # Low prediction totals (<115) flag
        if "pred_total" in df_tpl.columns:
            pt_vals = pd.to_numeric(df_tpl["pred_total"], errors="coerce")
            low_mask = pt_vals < 115
            pipeline_stats["low_pred_count_lt115"] = int(low_mask.sum())
            if low_mask.any() and "game_id" in df_tpl.columns:
                pipeline_stats["low_pred_game_ids"] = list(df_tpl.loc[low_mask, "game_id"].astype(str).head(12))
            # Predictions identical to market_total (rounded to 0.1) – track frequency
            if "market_total" in df_tpl.columns:
                mt_vals = pd.to_numeric(df_tpl["market_total"], errors="coerce")
                eq_mask = pt_vals.notna() & mt_vals.notna() & (pt_vals.round(1) == mt_vals.round(1))
                pipeline_stats["pred_equal_market_count"] = int(eq_mask.sum())
                if eq_mask.any() and "game_id" in df_tpl.columns:
                    pipeline_stats["pred_equal_market_sample"] = list(df_tpl.loc[eq_mask, "game_id"].astype(str).head(8))
            # Basis distribution counts
            if "pred_total_basis" in df_tpl.columns:
                basis_counts = df_tpl["pred_total_basis"].value_counts(dropna=True).to_dict()
                pipeline_stats["pred_total_basis_counts"] = basis_counts
                pipeline_stats["pred_synthetic_baseline_count"] = int(df_tpl["pred_total_basis"].eq("synthetic_baseline").sum())
                pipeline_stats["pred_market_copy_count"] = int(df_tpl["pred_total_basis"].eq("market_copy").sum())
            if "pred_margin_basis" in df_tpl.columns:
                m_basis_counts = df_tpl["pred_margin_basis"].value_counts(dropna=True).to_dict()
                pipeline_stats["pred_margin_basis_counts"] = m_basis_counts
        # Edge quality diagnostics (correlations)
        try:
            if {"pred_margin_model","spread_home"}.issubset(df_tpl.columns):
                pmv = pd.to_numeric(df_tpl["pred_margin_model"], errors="coerce")
                spv = pd.to_numeric(df_tpl["spread_home"], errors="coerce")
                if pmv.notna().any() and spv.notna().any():
                    pipeline_stats["corr_pred_margin_model_spread_home"] = float(pmv.corr(spv))
            if {"edge_total_model","edge_margin_model"}.issubset(df_tpl.columns):
                etm = pd.to_numeric(df_tpl["edge_total_model"], errors="coerce")
                emm = pd.to_numeric(df_tpl["edge_margin_model"], errors="coerce")
                if etm.notna().any() and emm.notna().any():
                    pipeline_stats["corr_edge_total_vs_margin_model"] = float(etm.corr(emm))
        except Exception:
            pass
        # Explicit list of games missing market_total (first 12)
        if {"game_id","market_total"}.issubset(df_tpl.columns):
            miss_mask = df_tpl["market_total"].isna()
            if miss_mask.any():
                pipeline_stats["missing_odds_game_ids"] = list(df_tpl.loc[miss_mask, "game_id"].astype(str).head(12))
    except Exception:
        pass

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
        strict_d1 = (request.args.get("strict_d1") or "").strip().lower() in ("1","true","yes")
        if not show_all and rows:
            d1set = _load_d1_team_set()
            if d1set:
                filtered_rows: list[dict[str, Any]] = []
                excluded_rows: list[dict[str, Any]] = []
                for r in rows:
                    h = normalize_name(str(r.get("home_team") or ""))
                    a = normalize_name(str(r.get("away_team") or ""))
                    # Detect usable data presence (predictions or odds) to allow graceful inclusion
                    pred_val = r.get("pred_total")
                    market_val = r.get("market_total")
                    odds_list = r.get("_odds_list") or []
                    def _has_val(v: Any) -> bool:
                        if v is None: return False
                        try:
                            # Treat NaN/None/"" as missing
                            if isinstance(v, (float,int)) and (pd.isna(v)): return False
                            s = str(v).strip().lower()
                            return s not in ("", "nan", "none", "null")
                        except Exception:
                            return False
                    has_pred = _has_val(pred_val)
                    has_market = _has_val(market_val)
                    has_any_odds = bool(odds_list)
                    in_d1 = (h in d1set) or (a in d1set)
                    # Inclusion logic:
                    #  - Always include if at least one team is D1
                    #  - If strict_d1 flag set, require D1 team (legacy behavior)
                    #  - Else, include non-D1 pair when we have predictions OR odds lines (so legitimate slate data isn’t hidden)
                    if in_d1 or (not strict_d1 and (has_pred or has_market or has_any_odds)):
                        filtered_rows.append(r)
                    else:
                        excluded_rows.append(r)
                # Replace only if we didn't accidentally drop everything
                if filtered_rows and len(filtered_rows) <= len(rows):
                    rows = filtered_rows
                pipeline_stats["rows_excluded_d1"] = len(excluded_rows)
                pipeline_stats["excluded_game_ids_d1"] = [str(er.get("game_id")) for er in excluded_rows if er.get("game_id")]
                pipeline_stats["excluded_non_d1_pairs_strict_count"] = int(sum(1 for er in excluded_rows if strict_d1))
            # If d1set empty, skip filter entirely
        pipeline_stats["rows_after_d1_filter"] = len(rows)
        pipeline_stats["d1_filter_strict"] = strict_d1
    except Exception as _d1e:
        pipeline_stats["d1_filter_error"] = str(_d1e)
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

    if diag_enabled:
        # Summarize missing prediction / odds counts
        mt_missing = 0
        pt_missing = 0
        try:
            if rows:
                for r in rows:
                    if r.get("market_total") is None and r.get("closing_total") is None:
                        mt_missing += 1
                    if r.get("pred_total") is None:
                        pt_missing += 1
        except Exception:
            pass
        pipeline_stats["missing_market_total_rows"] = mt_missing
        pipeline_stats["missing_pred_total_rows"] = pt_missing
        pipeline_stats["final_rows"] = len(rows)
        logger.info("Render pipeline stats: %s", json.dumps(pipeline_stats))

        # Optional diagnostics JSON direct response: /?diag=1&diag_json=1
        try:
            if (request.args.get('diag_json') or '').strip().lower() in ('1','true','yes'):
                pipeline_stats['final_columns'] = list(df.columns)
                if 'pred_total' in df.columns:
                    pt_final = pd.to_numeric(df['pred_total'], errors='coerce')
                    pipeline_stats['pred_total_final_preview'] = pt_final.head(25).tolist()
                if 'pred_total_raw' in df.columns:
                    pt_raw_prev = pd.to_numeric(df['pred_total_raw'], errors='coerce')
                    pipeline_stats['pred_total_raw_preview'] = pt_raw_prev.head(25).tolist()
                if 'derived_total' in df.columns:
                    dt_prev = pd.to_numeric(df['derived_total'], errors='coerce')
                    pipeline_stats['derived_total_preview'] = dt_prev.head(25).tolist()
                from flask import jsonify
                return jsonify(pipeline_stats)
        except Exception:
            pass
    # ------------------------------------------------------------------
    # Unified predictions export & global capture
    # (moved before return to ensure execution)
    # ------------------------------------------------------------------
    try:
        global _LAST_UNIFIED_FRAME
        _LAST_UNIFIED_FRAME = df.copy()
        # ------------------------------------------------------------------
        # Uncertainty estimation (lightweight): derive sigma for totals & margins
        # using residual std over last N days from daily_results. Scales by tempo.
        # ------------------------------------------------------------------
        try:
            recent_files = sorted((OUT / "daily_results").glob("results_*.csv"))[-14:]
            resid_totals: list[float] = []
            resid_margins: list[float] = []
            for fp in recent_files:
                try:
                    ddf = pd.read_csv(fp)
                except Exception:
                    continue
                needed_t = {"pred_total","home_score","away_score"}
                if needed_t.issubset(ddf.columns):
                    hs = pd.to_numeric(ddf["home_score"], errors="coerce")
                    as_ = pd.to_numeric(ddf["away_score"], errors="coerce")
                    pt = pd.to_numeric(ddf["pred_total"], errors="coerce")
                    actual_total = hs + as_
                    resid = (pt - actual_total).dropna()
                    resid_totals.extend(resid.tolist())
                needed_m = {"pred_margin","home_score","away_score"}
                if needed_m.issubset(ddf.columns):
                    hs = pd.to_numeric(ddf["home_score"], errors="coerce")
                    as_ = pd.to_numeric(ddf["away_score"], errors="coerce")
                    pm = pd.to_numeric(ddf["pred_margin"], errors="coerce")
                    actual_margin = hs - as_
                    residm = (pm - actual_margin).dropna()
                    resid_margins.extend(residm.tolist())
            base_sigma_total = float(np.std(resid_totals)) if len(resid_totals) >= 12 else 12.0
            base_sigma_margin = float(np.std(resid_margins)) if len(resid_margins) >= 12 else 8.0
            # Scale by tempo sum if ratings available
            tempo_scale = None
            if {"home_tempo_rating","away_tempo_rating"}.issubset(df.columns):
                tempo_scale = (pd.to_numeric(df["home_tempo_rating"], errors="coerce") + pd.to_numeric(df["away_tempo_rating"], errors="coerce")) / (2 * 69.0)
            df["pred_total_sigma"] = base_sigma_total * (tempo_scale if tempo_scale is not None else 1.0)
            df["pred_margin_sigma"] = base_sigma_margin * (tempo_scale if tempo_scale is not None else 1.0)
            pipeline_stats["pred_total_sigma_mean"] = float(pd.to_numeric(df["pred_total_sigma"], errors="coerce").mean()) if "pred_total_sigma" in df.columns else None
            pipeline_stats["pred_margin_sigma_mean"] = float(pd.to_numeric(df["pred_margin_sigma"], errors="coerce").mean()) if "pred_margin_sigma" in df.columns else None
            # Confidence-weighted staking adjustment: downscale Kelly by relative sigma vs base
            try:
                if {"kelly_fraction_total","pred_total_sigma"}.issubset(df.columns):
                    rel_scale = pd.to_numeric(df["pred_total_sigma"], errors="coerce") / max(base_sigma_total, 1e-9)
                    df["kelly_fraction_total_adj"] = pd.to_numeric(df["kelly_fraction_total"], errors="coerce") / rel_scale.clip(lower=0.5, upper=2.5)
            except Exception:
                pass
        except Exception:
            pipeline_stats["uncertainty_error"] = "sigma_failed"
        export_flag = (request.args.get("export") or "").strip().lower() in ("1","true","yes")
        export_date: str | None = None
        for cand in [date_q, today_str]:
            if cand:
                export_date = cand
                break
        if not export_date and "date" in df.columns and df["date"].notna().any():
            try:
                export_date = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().iloc[0]
            except Exception:
                try:
                    export_date = str(df["date"].dropna().astype(str).iloc[0])
                except Exception:
                    export_date = None
        if export_date and not df.empty and (export_flag or diag_enabled or bool(date_q)):
            cols_pref = [
                "game_id","date","home_team","away_team","pred_total","pred_margin",
                "pred_total_model","pred_margin_model","pred_total_calibrated","pred_margin_calibrated",
                "pred_total_basis","pred_margin_basis","pred_total_model_basis","pred_margin_model_basis",
                "market_total","closing_total","spread_home","closing_spread_home",
                "edge_total","edge_total_model","edge_closing","edge_closing_model","edge_margin_model",
                "proj_home","proj_away","start_time",
                "pred_total_sigma","pred_margin_sigma","kelly_fraction_total","kelly_fraction_total_adj","kelly_fraction_margin_adj"
            ]
            keep = [c for c in cols_pref if c in df.columns]
            uni = df[keep].copy()
            uni_path = OUT / f"predictions_unified_{export_date}.csv"
            try:
                uni.to_csv(uni_path, index=False)
                pipeline_stats["unified_export_path"] = str(uni_path)
                pipeline_stats["unified_export_rows"] = int(len(uni))
            except Exception:
                pipeline_stats["unified_export_error"] = "write_failed"
    except Exception:
        pass

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
        pipeline_stats=pipeline_stats if diag_enabled else None,
    )



@app.route("/api/render-diagnostics")
def api_render_diagnostics():
    """Return detailed diagnostics for the render pipeline for a given date (or resolved date)."""
    date_q = (request.args.get("date") or "").strip()
    # Re-run lightweight portions to avoid duplicating heavy enrichment.
    games = _load_games_current()
    preds = _load_predictions_current()
    odds = _load_odds_joined(date_q)
    out: dict[str, Any] = {
        "date_param": date_q,
        "outputs_dir": str(OUT),
        "games_load_rows": len(games),
        "preds_load_rows": len(preds),
        "odds_load_rows": len(odds),
    }
    # Date resolution logic (similar to index start) for transparency
    if not date_q:
        try:
            today_str = _today_local().strftime("%Y-%m-%d")
        except Exception:
            today_str = None
        if today_str and "date" in games.columns and (games["date"].astype(str) == today_str).any():
            date_q = today_str
        elif "date" in preds.columns and not preds.empty:
            try:
                date_q = pd.to_datetime(preds["date"]).max().strftime("%Y-%m-%d")
            except Exception:
                date_q = preds["date"].dropna().astype(str).max()
        out["resolved_date"] = date_q
    # Filter by date
    games_date = games
    preds_date = preds
    if date_q:
        if "date" in games.columns:
            games_date = games[games["date"].astype(str) == date_q]
        if "date" in preds.columns:
            preds_date = preds[preds["date"].astype(str) == date_q]
    out["games_after_date"] = len(games_date)
    out["preds_after_date"] = len(preds_date)
    # Sample IDs missing preds or odds
    try:
        g_ids = set(games_date.get("game_id", pd.Series()).astype(str)) if not games_date.empty else set()
        p_ids = set(preds_date.get("game_id", pd.Series()).astype(str)) if not preds_date.empty else set()
        o_ids = set(odds.get("game_id", pd.Series()).astype(str)) if not odds.empty else set()
        out["games_without_preds"] = sorted(list(g_ids - p_ids))[:40]
        out["games_without_odds"] = sorted(list(g_ids - o_ids))[:40]
        out["preds_without_games"] = sorted(list(p_ids - g_ids))[:40]
    except Exception:
        pass
    return jsonify(out)


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
        # Expose predictions source path if previously loaded during this process lifetime
        try:
            global _PREDICTIONS_SOURCE_PATH
            predictions_source = _PREDICTIONS_SOURCE_PATH
        except Exception:
            predictions_source = None
        need_bootstrap = bool(today_str and (preds_today_rows is None or preds_today_rows == 0))
        payload = {
            "status": "ok",
            "outputs_dir": str(OUT),
            "games_files": games_files,
            "odds_files": odds_files,
            "predictions_files": preds_files,
            "stake_files": stake_files,
            "daily_results_count": len(daily_results),
            "recent_result_dates": recent_results,
            "predictions_source": predictions_source,
            "need_bootstrap": need_bootstrap,
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
        # Local import to avoid global import errors at app startup
        from ncaab_model.data.adapters.odds_theoddsapi import TheOddsAPIAdapter  # type: ignore
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
            d1 = pd.read_csv(settings.data_dir / "d1_conferences.csv")
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


@app.route("/api/predictions_unified")
def api_predictions_unified():
    """Return unified predictions frame for a given date (or latest).

    Query params:
      - date=YYYY-MM-DD (optional; defaults to today's resolved slate if present)
      - cols=comma,separated list of columns to include (optional)
      - include_sigma=1 to force uncertainty columns even if not exported

    Behavior:
      1. Attempt to load outputs/predictions_unified_<date>.csv when date provided.
      2. If date omitted, try today's file; fallback to global _LAST_UNIFIED_FRAME.
      3. If file missing and global frame exists, filter by date column when possible.
    """
    date_q = (request.args.get("date") or "").strip()
    cols_req = (request.args.get("cols") or "").strip()
    include_sigma = (request.args.get("include_sigma") or "").strip().lower() in ("1","true","yes")
    today_str = None
    try:
        today_str = _today_local().strftime("%Y-%m-%d")
    except Exception:
        today_str = None
    target_date = date_q or today_str

@app.route("/api/backtest")
def api_backtest():
    """Return daily backtest metrics JSON for a given date.

    Query params:
      - date=YYYY-MM-DD (optional; defaults to yesterday if missing to ensure resolution)

    Response: { ok: bool, date: str, metrics: {...} }
    """
    date_q = (request.args.get("date") or "").strip()
    if not date_q:
        # default to yesterday for resolved outcomes
        try:
            date_q = (_today_local() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        except Exception:
            date_q = None
    if not date_q:
        return jsonify({"ok": False, "error": "no_date"}), 400
    path = OUT / f"backtest_metrics_{date_q}.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "not_found", "date": date_q}), 404
    try:
        import json as _json
        payload = _json.loads(path.read_text(encoding='utf-8'))
        return jsonify({"ok": True, "date": date_q, "metrics": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "date": date_q}), 500

@app.route("/api/residuals")
def api_residuals():
    """Return per-day residual distribution summary (totals/margins).

    Query params:
      - date=YYYY-MM-DD (optional; defaults to yesterday for completed games)
    """
    date_q = (request.args.get("date") or "").strip()
    if not date_q:
        try:
            date_q = (_today_local() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        except Exception:
            date_q = None
    if not date_q:
        return jsonify({"ok": False, "error": "no_date"}), 400
    path = OUT / f"residuals_{date_q}.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "not_found", "date": date_q}), 404
    try:
        import json as _json
        payload = _json.loads(path.read_text(encoding='utf-8'))
        return jsonify({"ok": True, "date": date_q, "residuals": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "date": date_q}), 500
    df = pd.DataFrame()
    loaded_path = None
    if target_date:
        path = OUT / f"predictions_unified_{target_date}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                loaded_path = str(path)
            except Exception:
                df = pd.DataFrame()
    # Fallback to global frame
    if df.empty:
        try:
            global _LAST_UNIFIED_FRAME
            if _LAST_UNIFIED_FRAME is not None and isinstance(_LAST_UNIFIED_FRAME, pd.DataFrame) and not _LAST_UNIFIED_FRAME.empty:
                df = _LAST_UNIFIED_FRAME.copy()
                if target_date and "date" in df.columns:
                    df = df[df["date"].astype(str) == target_date]
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        return jsonify({"ok": True, "date": target_date, "rows": [], "note": "no data"})
    # Optional sigma injection if requested and missing
    if include_sigma and "pred_total_sigma" not in df.columns:
        try:
            sigma_t = float(pd.to_numeric(df.get("pred_total"), errors="coerce").std()) if "pred_total" in df.columns else 12.0
            sigma_m = float(pd.to_numeric(df.get("pred_margin"), errors="coerce").std()) if "pred_margin" in df.columns else 8.0
            df["pred_total_sigma"] = sigma_t
            df["pred_margin_sigma"] = sigma_m
        except Exception:
            pass
    if cols_req:
        want = [c.strip() for c in cols_req.split(",") if c.strip()]
        have = [c for c in want if c in df.columns]
        if have:
            df = df[have]
    # Sanitize types for JSON
    out_rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        clean = {}
        for k,v in r.items():
            if isinstance(v, (np.generic,)):
                try:
                    v = v.item()
                except Exception:
                    v = float(v) if hasattr(v, '__float__') else str(v)
            if isinstance(v, (dt.datetime, dt.date)):
                v = str(v)
            clean[k] = v
        out_rows.append(clean)
    meta = {
        "date": target_date,
        "n_rows": len(out_rows),
        "columns": list(df.columns),
        "loaded_path": loaded_path,
        "from_global": loaded_path is None,
    }
    return jsonify({"ok": True, "meta": meta, "rows": out_rows})


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
