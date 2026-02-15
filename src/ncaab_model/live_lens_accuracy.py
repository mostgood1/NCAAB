from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LiveLensAccuracyConfig:
    date: str
    out_dir: Path = Path("outputs")
    daily_results_dir: Path | None = None
    assume_price: float = -110.0
    full_game_only: bool = True


@dataclass(frozen=True)
class LiveLensAccuracyRetunedConfig:
    """Counterfactual accuracy config.

    Reads logged Live Lens signals, optionally applies the same retune penalties
    used by the UI, reclassifies BET/WATCH from strength, then settles against
    final results.
    """

    date: str
    out_dir: Path = Path("outputs")
    daily_results_dir: Path | None = None
    assume_price: float = -110.0
    full_game_only: bool = True

    apply_retune: bool = True
    late_over_strength_penalty: float = 0.0
    late_over_remaining_lo: float = 5.0
    late_over_remaining_hi: float = 10.0
    late_over_margin_abs_min: float = 0.0
    late_over_period_min: float = 2.0

    early_over_strength_penalty: float = 0.0
    early_over_remaining_min: float = 20.0
    early_over_period_max: float = 1.0


def _root_outputs() -> Path:
    return Path(os.getcwd()) / "outputs"


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    # Validate YYYY-MM-DD
    dt.date.fromisoformat(s2)
    return s2


def signals_path(date: str, out_dir: Path | None = None) -> Path:
    d = _safe_date(date)
    out_root = Path(out_dir) if out_dir is not None else _root_outputs()
    return out_root / f"live_lens_signals_{d}.jsonl"


def results_path(date: str, out_dir: Path | None = None, daily_results_dir: Path | None = None) -> Path:
    d = _safe_date(date)
    if daily_results_dir is not None:
        return Path(daily_results_dir) / f"results_{d}.csv"
    out_root = Path(out_dir) if out_dir is not None else _root_outputs()
    return out_root / "daily_results" / f"results_{d}.csv"


def _read_jsonl(p: Path, max_lines: int = 200_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= int(max_lines):
                break
            s = (line or "").strip()
            if not s:
                continue
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    rows.append(j)
            except Exception:
                continue
    return rows


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _final_total_from_results(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit total if present.
    c_total = _pick_col(df, ["total_points", "final_total", "scored_total", "total", "points_total"])
    if c_total:
        return pd.to_numeric(df[c_total], errors="coerce")

    c_home = _pick_col(df, ["home_score", "home_points", "home_pts", "score_home"])
    c_away = _pick_col(df, ["away_score", "away_points", "away_pts", "score_away"])
    if c_home and c_away:
        hs = pd.to_numeric(df[c_home], errors="coerce")
        aw = pd.to_numeric(df[c_away], errors="coerce")
        return hs + aw

    return pd.Series([math.nan] * len(df))


def _filter_results_to_finals(res_df: pd.DataFrame) -> pd.DataFrame:
    """Return only final/settled games when the results file provides that metadata.

    Backward compatible: if status/completed info isn't present (or is empty), returns res_df unchanged.
    """

    df = res_df
    # Prefer explicit completed flag.
    if "completed" in df.columns:
        try:
            c = df["completed"]
            if c.notna().any():
                # Pandas may treat booleans as objects/strings; normalize.
                c2 = c.astype(str).str.strip().str.lower()
                mask = c2.isin(["true", "1", "yes"])  # conservative
                return df[mask].copy()
        except Exception:
            pass

    # Fallback to status strings when present.
    if "status" in df.columns:
        try:
            s = df["status"]
            if s.notna().any():
                s2 = s.astype(str).str.strip().str.upper()
                mask = s2.eq("STATUS_FINAL") | s2.eq("FINAL") | s2.str.contains("FINAL", na=False)
                return df[mask].copy()
        except Exception:
            pass

    return df


def _remaining_from_row(row: pd.Series) -> float | None:
    r = row.get("remaining")
    if r is not None and pd.notna(r):
        try:
            return float(r)
        except Exception:
            pass
    h = row.get("horizon")
    e = row.get("elapsed")
    try:
        if h is not None and e is not None and pd.notna(h) and pd.notna(e):
            return float(h) - float(e)
    except Exception:
        return None
    return None


def _classify_bet_watch(horizon: float, elapsed: float, strength: float) -> tuple[bool, bool]:
    """Replicate the UI's BET/WATCH classification from strength."""

    h = float(horizon)
    e = float(elapsed)
    s = float(strength)

    if h <= 20.5:
        min_elapsed = 4.0
        thr = 5.0
        thr_watch = max(3.0, thr - 1.0)
    else:
        min_elapsed = 6.0
        thr = 7.0
        thr_watch = max(4.0, thr - 2.0)

    if not math.isfinite(e) or not math.isfinite(s):
        return (False, False)
    if e < min_elapsed:
        return (False, False)

    is_bet = s >= thr
    is_watch = (not is_bet) and (s >= thr_watch)
    return (bool(is_bet), bool(is_watch))


def compute_live_lens_accuracy_retuned(cfg: LiveLensAccuracyRetunedConfig) -> dict[str, Any]:
    date = _safe_date(cfg.date)
    out_root = Path(cfg.out_dir) if cfg.out_dir is not None else _root_outputs()

    sig_p = signals_path(date, out_dir=out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)

    signals = _read_jsonl(sig_p)
    if not signals:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals found at {sig_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
        }

    sig_df = pd.DataFrame(signals)
    if sig_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals rows parsed from {sig_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
        }

    # Normalize ids
    if "game_id" not in sig_df.columns and "event_id" in sig_df.columns:
        sig_df["game_id"] = sig_df["event_id"]
    if "game_id" not in sig_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing game_id/event_id; cannot evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "signals_cols": list(sig_df.columns),
            "n_signals_raw": int(len(sig_df)),
        }
    sig_df["game_id"] = sig_df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Flags
    if "is_bet" in sig_df.columns:
        sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
    else:
        sig_df["is_bet"] = True
    if "is_watch" in sig_df.columns:
        sig_df["is_watch"] = sig_df["is_watch"].astype(bool)
    else:
        sig_df["is_watch"] = False

    # Side + line
    if "side" not in sig_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing side (over/under); cannot settle",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(sig_df)),
            "signals_cols": list(sig_df.columns),
        }
    sig_df["side"] = sig_df["side"].astype(str).str.strip().str.lower()
    sig_df = sig_df[sig_df["side"].isin(["over", "under"])].copy()

    if "live_line" in sig_df.columns:
        sig_df["live_line"] = pd.to_numeric(sig_df["live_line"], errors="coerce")
    else:
        sig_df["live_line"] = math.nan
    sig_df = sig_df[sig_df["live_line"].notna()].copy()

    if cfg.full_game_only and "horizon" in sig_df.columns:
        hz = pd.to_numeric(sig_df["horizon"], errors="coerce")
        sig_df = sig_df[hz >= 39].copy()

    # Include WATCH rows in the candidate pool so counterfactual reclassification is meaningful.
    sig_df = sig_df[sig_df["is_bet"] | sig_df["is_watch"]].copy()

    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No BET/WATCH signals with live_line to evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(pd.DataFrame(signals))),
        }

    # Compute counterfactual strength + reclassify
    sig_df["remaining"] = sig_df.apply(_remaining_from_row, axis=1)
    hz2 = pd.to_numeric(sig_df.get("horizon"), errors="coerce") if "horizon" in sig_df.columns else pd.Series([40.0] * len(sig_df))
    el2 = pd.to_numeric(sig_df.get("elapsed"), errors="coerce") if "elapsed" in sig_df.columns else pd.Series([math.nan] * len(sig_df))
    rem2 = pd.to_numeric(sig_df.get("remaining"), errors="coerce")
    if el2.isna().any():
        try:
            el2 = el2.fillna(hz2 - rem2)
        except Exception:
            pass

    cf_strength = pd.to_numeric(sig_df.get("strength"), errors="coerce") if "strength" in sig_df.columns else pd.Series([math.nan] * len(sig_df))

    if cfg.apply_retune:
        pen = float(cfg.late_over_strength_penalty or 0.0)
        lo_r = float(cfg.late_over_remaining_lo)
        hi_r = float(cfg.late_over_remaining_hi)
        if hi_r < lo_r:
            lo_r, hi_r = hi_r, lo_r
        margin_abs_min = float(cfg.late_over_margin_abs_min or 0.0)
        period_min = float(cfg.late_over_period_min or 1.0)

        early_pen = float(cfg.early_over_strength_penalty or 0.0)
        early_min_rem = float(cfg.early_over_remaining_min or 20.0)
        early_period_max = float(cfg.early_over_period_max if cfg.early_over_period_max is not None else 1.0)

        side = sig_df["side"].astype(str).str.strip().str.lower()
        is_over = side.eq("over")
        in_window = (rem2 > lo_r) & (rem2 <= hi_r)

        period_ok = pd.Series([True] * len(sig_df))
        if period_min > 1:
            if "period" in sig_df.columns:
                per = pd.to_numeric(sig_df.get("period"), errors="coerce")
                period_ok = per.notna() & (per >= period_min)
            else:
                period_ok = (hz2 >= 39) & el2.notna() & (el2 >= 20)

        margin_ok = pd.Series([True] * len(sig_df))
        if margin_abs_min > 0:
            if "margin_home" in sig_df.columns:
                m = pd.to_numeric(sig_df.get("margin_home"), errors="coerce")
                margin_ok = m.abs() >= margin_abs_min
            else:
                margin_ok = pd.Series([False] * len(sig_df))

        adj_mask = (pen > 0) & is_over & in_window & period_ok & margin_ok
        try:
            cf_strength = cf_strength.where(~adj_mask, cf_strength - float(pen))
        except Exception:
            pass

        if early_pen > 0:
            try:
                in_early = rem2 >= early_min_rem
                period_ok_e = pd.Series([True] * len(sig_df))
                if "period" in sig_df.columns:
                    per = pd.to_numeric(sig_df.get("period"), errors="coerce")
                    period_ok_e = per.notna() & (per <= early_period_max)
                else:
                    period_ok_e = (hz2 >= 39) & el2.notna() & (el2 < 20)

                early_mask = is_over & in_early & period_ok_e
                cf_strength = cf_strength.where(~early_mask, cf_strength - float(early_pen))
            except Exception:
                pass

    cf_is_bet: list[bool] = []
    cf_is_watch: list[bool] = []
    for h0, e0, s0 in zip(hz2, el2, cf_strength):
        try:
            if pd.isna(h0) or pd.isna(e0) or pd.isna(s0):
                cf_is_bet.append(False)
                cf_is_watch.append(False)
                continue
            b, w = _classify_bet_watch(float(h0), float(e0), float(s0))
            cf_is_bet.append(b)
            cf_is_watch.append(w)
        except Exception:
            cf_is_bet.append(False)
            cf_is_watch.append(False)

    sig_df["cf_strength"] = cf_strength
    sig_df["cf_is_bet"] = cf_is_bet
    sig_df["cf_is_watch"] = cf_is_watch

    # BET-only accuracy is the main metric.
    sig_df = sig_df[sig_df["cf_is_bet"].astype(bool)].copy()
    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No counterfactual BET signals after retune",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
        }

    # Load results
    if not res_p.exists():
        return {
            "status": "missing",
            "date": date,
            "message": f"Missing results file at {res_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    res_df = pd.read_csv(res_p)
    if res_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"Empty results file at {res_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    gid_col = _pick_col(res_df, ["game_id", "event_id", "id", "gid"])
    if not gid_col:
        return {
            "status": "error",
            "date": date,
            "message": "Results file missing game_id/event_id column",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "results_cols": list(res_df.columns),
        }

    res_df["game_id"] = res_df[gid_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    res_df = _filter_results_to_finals(res_df)
    if res_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "Results file has no final/completed games to settle against",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    res_df["final_total"] = _final_total_from_results(res_df)
    merged = sig_df.merge(res_df[["game_id", "final_total"]], on="game_id", how="left")

    y = pd.to_numeric(merged["final_total"], errors="coerce")
    line = pd.to_numeric(merged["live_line"], errors="coerce")

    def _settle(side: str, yv: float, lv: float) -> float | None:
        if not (isinstance(yv, (int, float)) and isinstance(lv, (int, float))):
            return None
        if not (math.isfinite(float(yv)) and math.isfinite(float(lv))):
            return None
        if float(yv) == float(lv):
            return 0.5
        if side == "over":
            return 1.0 if float(yv) > float(lv) else 0.0
        return 1.0 if float(yv) < float(lv) else 0.0

    merged["result"] = [
        _settle(str(s), float(yv) if pd.notna(yv) else float("nan"), float(lv) if pd.notna(lv) else float("nan"))
        for s, yv, lv in zip(merged["side"], y, line)
    ]

    settled = merged[merged["result"].notna()].copy()
    if settled.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "No settled signals (missing final totals for game_id join)",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(merged)),
            "n_results": int(len(res_df)),
        }

    price = float(cfg.assume_price)
    win_profit = 100.0 / abs(price) if price < 0 else (price / 100.0)

    def _profit(res: float) -> float:
        if res == 1.0:
            return float(win_profit)
        if res == 0.0:
            return -1.0
        return 0.0

    settled["profit_units"] = settled["result"].map(_profit)

    wins = int((settled["result"] == 1.0).sum())
    losses = int((settled["result"] == 0.0).sum())
    pushes = int((settled["result"] == 0.5).sum())
    denom = wins + losses
    win_rate = (wins / denom) if denom > 0 else None
    roi = float(settled["profit_units"].sum() / max(1, len(settled)))

    by_bucket: list[dict[str, Any]] = []
    if "elapsed" in settled.columns:
        el = pd.to_numeric(settled["elapsed"], errors="coerce")
        settled["elapsed_bucket"] = (el // 5 * 5).astype("Int64")
        for b, g in settled.dropna(subset=["elapsed_bucket"]).groupby("elapsed_bucket"):
            w = int((g["result"] == 1.0).sum())
            l = int((g["result"] == 0.0).sum())
            p = int((g["result"] == 0.5).sum())
            d2 = w + l
            by_bucket.append(
                {
                    "elapsed_bucket": int(b),
                    "n": int(len(g)),
                    "wins": w,
                    "losses": l,
                    "pushes": p,
                    "win_rate": (w / d2) if d2 > 0 else None,
                    "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
                }
            )
        by_bucket.sort(key=lambda x: x["elapsed_bucket"])

    summary: dict[str, Any] = {
        "status": "ok",
        "policy": "retuned" if cfg.apply_retune else "logged",
        "date": date,
        "signals_path": str(sig_p),
        "results_path": str(res_p),
        "assume_price": price,
        "n_signals": int(len(merged)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi_units_per_bet": roi,
        "by_elapsed_bucket": by_bucket,
        "retune": {
            "late_over_strength_penalty": float(cfg.late_over_strength_penalty),
            "late_over_remaining_lo": float(cfg.late_over_remaining_lo),
            "late_over_remaining_hi": float(cfg.late_over_remaining_hi),
            "late_over_margin_abs_min": float(cfg.late_over_margin_abs_min),
            "late_over_period_min": float(cfg.late_over_period_min),
            "early_over_strength_penalty": float(cfg.early_over_strength_penalty),
            "early_over_remaining_min": float(cfg.early_over_remaining_min),
            "early_over_period_max": float(cfg.early_over_period_max),
        },
    }

    return {
        "summary": summary,
        "rows": settled,
    }


def compute_live_lens_accuracy(cfg: LiveLensAccuracyConfig) -> dict[str, Any]:
    date = _safe_date(cfg.date)
    out_root = Path(cfg.out_dir) if cfg.out_dir is not None else _root_outputs()

    sig_p = signals_path(date, out_dir=out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)

    signals = _read_jsonl(sig_p)
    if not signals:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals found at {sig_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
        }

    sig_df = pd.DataFrame(signals)
    if sig_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals rows parsed from {sig_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
        }

    # Normalize ids
    if "game_id" not in sig_df.columns and "event_id" in sig_df.columns:
        sig_df["game_id"] = sig_df["event_id"]
    if "game_id" not in sig_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing game_id/event_id; cannot evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "signals_cols": list(sig_df.columns),
            "n_signals_raw": int(len(sig_df)),
        }
    sig_df["game_id"] = sig_df["game_id"].astype(str).str.strip()

    # Filter to bet signals with a line
    if "is_bet" in sig_df.columns:
        sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
    else:
        sig_df["is_bet"] = True

    if "live_line" in sig_df.columns:
        sig_df["live_line"] = pd.to_numeric(sig_df["live_line"], errors="coerce")
    else:
        sig_df["live_line"] = math.nan
    sig_df = sig_df[sig_df["is_bet"] & sig_df["live_line"].notna()].copy()

    if cfg.full_game_only and "horizon" in sig_df.columns:
        hz = pd.to_numeric(sig_df["horizon"], errors="coerce")
        sig_df = sig_df[hz >= 39].copy()

    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No bet signals with live_line to evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(pd.DataFrame(signals))),
        }

    # Load results
    if not res_p.exists():
        return {
            "status": "missing",
            "date": date,
            "message": f"Missing results file at {res_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    res_df = pd.read_csv(res_p)
    if res_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"Empty results file at {res_p}",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    gid_col = _pick_col(res_df, ["game_id", "event_id", "id", "gid"])
    if not gid_col:
        return {
            "status": "error",
            "date": date,
            "message": "Results file missing game_id/event_id column",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "results_cols": list(res_df.columns),
        }
    res_df["game_id"] = res_df[gid_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # If this is a same-day / in-progress results file, avoid settling against partial scores.
    res_df = _filter_results_to_finals(res_df)
    if res_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "Results file has no final/completed games to settle against",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(sig_df)),
        }

    res_df["final_total"] = _final_total_from_results(res_df)

    merged = sig_df.merge(res_df[["game_id", "final_total"]], on="game_id", how="left")

    if "side" not in merged.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing side (over/under); cannot settle",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(merged)),
            "signals_cols": list(sig_df.columns),
        }

    merged["side"] = merged["side"].astype(str).str.strip().str.lower()
    merged = merged[merged["side"].isin(["over", "under"])].copy()

    # Outcome
    y = pd.to_numeric(merged["final_total"], errors="coerce")
    line = pd.to_numeric(merged["live_line"], errors="coerce")

    def _settle(side: str, yv: float, lv: float) -> float | None:
        if not (isinstance(yv, (int, float)) and isinstance(lv, (int, float))):
            return None
        if not (math.isfinite(float(yv)) and math.isfinite(float(lv))):
            return None
        if float(yv) == float(lv):
            return 0.5
        if side == "over":
            return 1.0 if float(yv) > float(lv) else 0.0
        return 1.0 if float(yv) < float(lv) else 0.0

    merged["result"] = [
        _settle(str(s), float(yv) if pd.notna(yv) else float("nan"), float(lv) if pd.notna(lv) else float("nan"))
        for s, yv, lv in zip(merged["side"], y, line)
    ]

    settled = merged[merged["result"].notna()].copy()
    if settled.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "No settled signals (missing final totals for game_id join)",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals": int(len(merged)),
            "n_results": int(len(res_df)),
        }

    # Profit per 1u risk at -110 by default
    price = float(cfg.assume_price)
    win_profit = 100.0 / abs(price) if price < 0 else (price / 100.0)

    def _profit(res: float) -> float:
        if res == 1.0:
            return float(win_profit)
        if res == 0.0:
            return -1.0
        return 0.0  # push

    settled["profit_units"] = settled["result"].map(_profit)

    wins = int((settled["result"] == 1.0).sum())
    losses = int((settled["result"] == 0.0).sum())
    pushes = int((settled["result"] == 0.5).sum())
    denom = wins + losses
    win_rate = (wins / denom) if denom > 0 else None
    roi = float(settled["profit_units"].sum() / max(1, len(settled)))

    # Buckets (by elapsed minutes if present)
    by_bucket: list[dict[str, Any]] = []
    if "elapsed" in settled.columns:
        el = pd.to_numeric(settled["elapsed"], errors="coerce")
        settled["elapsed_bucket"] = (el // 5 * 5).astype("Int64")
        for b, g in settled.dropna(subset=["elapsed_bucket"]).groupby("elapsed_bucket"):
            w = int((g["result"] == 1.0).sum())
            l = int((g["result"] == 0.0).sum())
            p = int((g["result"] == 0.5).sum())
            d2 = w + l
            by_bucket.append(
                {
                    "elapsed_bucket": int(b),
                    "n": int(len(g)),
                    "wins": w,
                    "losses": l,
                    "pushes": p,
                    "win_rate": (w / d2) if d2 > 0 else None,
                    "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
                }
            )
        by_bucket.sort(key=lambda x: x["elapsed_bucket"])

    summary: dict[str, Any] = {
        "status": "ok",
        "date": date,
        "signals_path": str(sig_p),
        "results_path": str(res_p),
        "assume_price": price,
        "n_signals": int(len(merged)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi_units_per_bet": roi,
        "by_elapsed_bucket": by_bucket,
    }

    return {
        "summary": summary,
        "rows": settled,
    }


def write_live_lens_accuracy(out_json: Path, payload: dict[str, Any], out_csv: Path | None = None) -> dict[str, Any]:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # `payload` is either {summary, rows} or an error payload
    if "summary" in payload and isinstance(payload.get("rows"), pd.DataFrame):
        summary = payload["summary"]
        rows_df: pd.DataFrame = payload["rows"]
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if out_csv is not None:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            keep = [
                c
                for c in [
                    "ts",
                    "date",
                    "game_id",
                    "horizon",
                    "elapsed",
                    "remaining",
                    "total_points",
                    "live_line",
                    "side",
                    "final_total",
                    "result",
                    "profit_units",
                    "tuning_source",
                ]
                if c in rows_df.columns
            ]
            rows_df[keep].to_csv(out_csv, index=False)
        return {"status": "ok", "out_json": str(out_json), "out_csv": str(out_csv) if out_csv else None, "summary": summary}

    # Error payload
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "ok", "out_json": str(out_json), "out_csv": None, "payload": payload}


def make_signal_id(d: dict[str, Any]) -> str:
    # Helpful for clients/servers wanting idempotency.
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
