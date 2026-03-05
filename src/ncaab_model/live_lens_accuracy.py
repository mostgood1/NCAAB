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
class LiveLensProjectionAccuracyConfig:
    date: str
    out_dir: Path = Path("outputs")
    daily_results_dir: Path | None = None
    full_game_only: bool = False


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

    # Optional learned penalties based on driver flags/tags.
    # When present, each tag's positive value is subtracted from strength.
    apply_driver_tag_penalties: bool = True
    driver_tag_strength_penalties: dict[str, float] | None = None


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


def _candidate_signals_paths(date: str, out_dir: Path) -> list[tuple[str, Path]]:
    d = _safe_date(date)
    out_root = Path(out_dir) if out_dir is not None else _root_outputs()
    return [
        ("canonical", signals_path(d, out_dir=out_root)),
        ("reconstructed", out_root / f"live_lens_signals_reconstructed_{d}.jsonl"),
        ("recovered", out_root / f"live_lens_signals_recovered_{d}.jsonl"),
    ]


def _parse_tags(v: Any) -> list[str]:
    if v is None:
        return []
    try:
        if isinstance(v, float) and (math.isnan(v) or (not math.isfinite(v))):
            return []
    except Exception:
        pass
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            try:
                if isinstance(x, float) and (math.isnan(x) or (not math.isfinite(x))):
                    continue
            except Exception:
                pass
            s = str(x).strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                continue
            if s not in out:
                out.append(s)
        return out
    if isinstance(v, str):
        s0 = v.strip()
        if not s0 or s0.lower() in {"nan", "none", "null"}:
            return []
        # Accept either comma-separated or pipe-separated formats.
        if "," in s0:
            parts = [p.strip() for p in s0.split(",") if p.strip()]
        elif "|" in s0:
            parts = [p.strip() for p in s0.split("|") if p.strip()]
        else:
            parts = [s0]
        out: list[str] = []
        for p in parts:
            if p and p.lower() not in {"nan", "none", "null"} and p not in out:
                out.append(p)
        return out
    return []


def _tags_from_driver_text(driver_text: Any) -> list[str]:
    try:
        s = str(driver_text or "").strip()
    except Exception:
        return []
    if not s or s.lower() in {"none", "null", "nan"}:
        return []

    # Legacy single-token drivers.
    legacy = {"pace_hi", "pace_lo", "pace_mid", "pace_missing", "eff_hi", "eff_lo", "eff_mid", "eff_missing"}
    sl = s.lower()
    if sl in legacy:
        return [sl]

    # Detailed explainer: "edge +3 | d +12 | ..." -> coarse categorical tags.
    tags: list[str] = []

    def add(t: str) -> None:
        if t and t not in tags:
            tags.append(t)

    try:
        toks = [t.strip() for t in s.split("|") if str(t or "").strip()]
        for t in toks:
            tl = t.lower().strip()
            if tl.startswith("edge "):
                add("edge")
            elif tl.startswith("d "):
                add("sim_gap")
            elif tl.startswith("ppp "):
                add("ppp")
            elif tl.startswith("shooting "):
                add("shooting")
            elif tl.startswith("ft rate "):
                add("ft")
            elif tl.startswith("pbp+"):
                add("pbp")
            elif "late-over" in tl:
                add("late_over")
            elif "early-over" in tl:
                add("early_over")
            elif tl.startswith("flags -"):
                add("flags")
    except Exception:
        return tags

    return tags


def _signals_tag_richness(rows: list[dict[str, Any]], *, sample_n: int = 5000) -> tuple[int, int, int]:
    tag_rows = 0
    tag_total = 0
    uniq: set[str] = set()

    for r in (rows or [])[: int(sample_n)]:
        if not isinstance(r, dict):
            continue
        tags = _parse_tags(r.get("driver_tags"))
        tags2 = _tags_from_driver_text(r.get("driver"))
        merged: list[str] = []
        for t in (tags or []) + (tags2 or []):
            s = str(t or "").strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                continue
            if s not in merged:
                merged.append(s)
        if merged:
            tag_rows += 1
            tag_total += len(merged)
            uniq.update(merged)

    return int(tag_rows), int(tag_total), int(len(uniq))


def _load_best_signals_jsonl(date: str, out_dir: Path) -> tuple[list[dict[str, Any]], Path, str, list[str]]:
    candidates = _candidate_signals_paths(date, out_dir)
    tried = [str(p) for _, p in candidates]

    loaded: list[tuple[str, Path, list[dict[str, Any]], tuple[int, int, int]]] = []
    for kind, p in candidates:
        rows = _read_jsonl(p)
        if rows:
            loaded.append((kind, p, rows, _signals_tag_richness(rows)))

    if not loaded:
        return [], candidates[0][1], candidates[0][0], tried

    # Choose best by richness; tie-break by preference.
    pref_rank = {"canonical": 0, "reconstructed": 1, "recovered": 2}
    best = max(loaded, key=lambda x: (x[3], -pref_rank.get(x[0], 99)))
    kind, p, rows, _richness = best
    return rows, p, kind, tried


def projections_path(date: str, out_dir: Path | None = None) -> Path:
    d = _safe_date(date)
    out_root = Path(out_dir) if out_dir is not None else _root_outputs()
    return out_root / f"live_lens_projections_{d}.jsonl"


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
    c_total = _pick_col(df, ["actual_total", "total_points", "final_total", "scored_total", "total", "points_total"])
    if c_total:
        return pd.to_numeric(df[c_total], errors="coerce")

    c_home = _pick_col(df, ["home_score", "home_points", "home_pts", "score_home"])
    c_away = _pick_col(df, ["away_score", "away_points", "away_pts", "score_away"])
    if c_home and c_away:
        hs = pd.to_numeric(df[c_home], errors="coerce")
        aw = pd.to_numeric(df[c_away], errors="coerce")
        return hs + aw

    return pd.Series([math.nan] * len(df))


def _final_margin_from_results(df: pd.DataFrame) -> pd.Series:
    """Return final margin (home - away) from results.

    Prefers explicit margin columns when present; otherwise derives from scores.
    """

    c_margin = _pick_col(df, ["actual_margin", "final_margin", "margin", "margin_home", "home_margin"])
    if c_margin:
        return pd.to_numeric(df[c_margin], errors="coerce")

    c_home = _pick_col(df, ["home_score", "home_points", "home_pts", "score_home"])
    c_away = _pick_col(df, ["away_score", "away_points", "away_pts", "score_away"])
    if c_home and c_away:
        hs = pd.to_numeric(df[c_home], errors="coerce")
        aw = pd.to_numeric(df[c_away], errors="coerce")
        return hs - aw

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

    signals, sig_p, sig_kind, sig_tried = _load_best_signals_jsonl(date, out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)
    if not signals:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals found at {sig_p}",
            "signals_path": str(sig_p),
            "signals_kind": sig_kind,
            "signals_tried": sig_tried,
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
    hz2 = (
        pd.to_numeric(sig_df.get("horizon"), errors="coerce")
        if "horizon" in sig_df.columns
        else pd.Series([40.0] * len(sig_df), index=sig_df.index)
    )
    el2 = (
        pd.to_numeric(sig_df.get("elapsed"), errors="coerce")
        if "elapsed" in sig_df.columns
        else pd.Series([math.nan] * len(sig_df), index=sig_df.index)
    )
    rem2 = pd.to_numeric(sig_df.get("remaining"), errors="coerce")
    if el2.isna().any():
        try:
            el2 = el2.fillna(hz2 - rem2)
        except Exception:
            pass

    cf_strength = (
        pd.to_numeric(sig_df.get("strength"), errors="coerce")
        if "strength" in sig_df.columns
        else pd.Series([math.nan] * len(sig_df), index=sig_df.index)
    )

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

        period_ok = pd.Series([True] * len(sig_df), index=sig_df.index)
        if period_min > 1:
            if "period" in sig_df.columns:
                per = pd.to_numeric(sig_df.get("period"), errors="coerce")
                period_ok = per.notna() & (per >= period_min)
            else:
                period_ok = (hz2 >= 39) & el2.notna() & (el2 >= 20)

        margin_ok = pd.Series([True] * len(sig_df), index=sig_df.index)
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

    # Optional: learned driver-flag penalties.
    if cfg.apply_driver_tag_penalties:
        pen_map = cfg.driver_tag_strength_penalties or {}

        def _parse_tags(v: Any) -> list[str]:
            if v is None:
                return []
            try:
                if pd.isna(v):
                    return []
            except Exception:
                pass
            if isinstance(v, list):
                out: list[str] = []
                for x in v:
                    if x is None:
                        continue
                    s = str(x).strip()
                    if s:
                        out.append(s)
                return out
            # Sometimes serialized as JSON string.
            try:
                s0 = str(v).strip()
                if not s0:
                    return []
                if s0.lower() in {"nan", "none", "null"}:
                    return []
                if s0.startswith("[") and s0.endswith("]"):
                    j = json.loads(s0)
                    if isinstance(j, list):
                        return [str(x).strip() for x in j if x is not None and str(x).strip()]
            except Exception:
                pass
            # Fallback: comma/pipe separated.
            try:
                s1 = str(v)
                parts = [p.strip() for p in s1.replace("|", ",").split(",")]
                return [p for p in parts if p]
            except Exception:
                return []

        if pen_map and ("driver_tags" in sig_df.columns):
            try:
                tag_pen = sig_df["driver_tags"].apply(
                    lambda vv: float(
                        sum(
                            float(pen_map.get(str(t).strip(), 0.0) or 0.0)
                            for t in _parse_tags(vv)
                            if float(pen_map.get(str(t).strip(), 0.0) or 0.0) > 0
                        )
                    )
                )
                sig_df["cf_tag_penalty"] = tag_pen
                cf_strength = cf_strength - pd.to_numeric(tag_pen, errors="coerce").fillna(0.0)
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
        "driver_tag_penalties": (cfg.driver_tag_strength_penalties or {}) if cfg.apply_driver_tag_penalties else {},
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


def compute_live_lens_total_side_accuracy(cfg: LiveLensAccuracyConfig) -> dict[str, Any]:
    """Compute totals O/U accuracy per signal-side, collapsing line changes.

    The UI logs a row whenever the live line changes; for analysis, we want to
    treat repeated rows as the same bet idea (OVER vs UNDER) for a given lens.

    Output rows are deduped to one row per (game_id, lens, side), selecting the
    earliest timestamped BET signal.
    """

    date = _safe_date(cfg.date)
    out_root = Path(cfg.out_dir) if cfg.out_dir is not None else _root_outputs()

    signals, sig_p, sig_kind, sig_tried = _load_best_signals_jsonl(date, out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)

    if not signals:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals found for {date}",
            "signals_path": str(sig_p),
            "signals_kind": str(sig_kind),
            "signals_tried": list(sig_tried),
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

    # Totals-only
    if "kind" in sig_df.columns:
        sig_df["kind"] = sig_df["kind"].astype(str).str.strip().str.lower()
        sig_df = sig_df[sig_df["kind"].eq("total")].copy()
    else:
        sig_df["kind"] = "total"

    # BET-only
    if "is_bet" in sig_df.columns:
        sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
    else:
        sig_df["is_bet"] = True
    sig_df = sig_df[sig_df["is_bet"].astype(bool)].copy()

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

    # Lens normalization: {fg, 1h, 2h}
    if "lens" in sig_df.columns:
        lens0 = sig_df["lens"].astype(str).str.strip().str.lower()
    else:
        # Preserve index alignment (sig_df is typically filtered and keeps original indices).
        lens0 = pd.Series([""] * len(sig_df), index=sig_df.index)

    lens_norm = lens0
    lens_norm = lens_norm.replace({
        "full_game": "fg",
        "full": "fg",
        "game": "fg",
        "fg": "fg",
        "1h": "1h",
        "2h": "2h",
        "first_half": "1h",
        "second_half": "2h",
    })
    lens_norm = lens_norm.where(lens_norm.isin(["fg", "1h", "2h"]), other="")

    # If lens is missing, fall back to horizon heuristic.
    if lens_norm.eq("").any() and "horizon" in sig_df.columns:
        hz = pd.to_numeric(sig_df.get("horizon"), errors="coerce")
        per = (
            pd.to_numeric(sig_df.get("period"), errors="coerce")
            if "period" in sig_df.columns
            else pd.Series([math.nan] * len(sig_df), index=sig_df.index)
        )

        def _infer(h: float | None, p: float | None) -> str:
            try:
                if h is not None and pd.notna(h) and float(h) >= 39:
                    return "fg"
            except Exception:
                pass
            try:
                if h is not None and pd.notna(h) and float(h) <= 21:
                    # Distinguish 1H vs 2H using period when available.
                    if p is not None and pd.notna(p) and int(float(p)) >= 2:
                        return "2h"
                    if p is not None and pd.notna(p) and int(float(p)) == 1:
                        return "1h"
            except Exception:
                pass
            return ""

        inferred = [
            _infer((float(h) if pd.notna(h) else None), (float(p) if pd.notna(p) else None))
            for h, p in zip(hz.tolist(), per.tolist())
        ]
        # IMPORTANT: preserve index alignment (sig_df likely has a filtered/non-contiguous index).
        lens_norm = lens_norm.where(~lens_norm.eq(""), other=pd.Series(inferred, index=lens_norm.index))

    sig_df["lens"] = lens_norm
    sig_df = sig_df[sig_df["lens"].isin(["fg", "1h", "2h"])].copy()

    # Full-game-only option suppresses 1H/2H.
    if cfg.full_game_only:
        sig_df = sig_df[sig_df["lens"].eq("fg")].copy()

    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No totals BET signals with lens+line to evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(pd.DataFrame(signals))),
        }

    # Deduplicate repeated line-change logs: keep earliest per (game_id, lens, side).
    if "ts" in sig_df.columns:
        sig_df["ts"] = sig_df["ts"].astype(str)
        ts_parsed = pd.to_datetime(sig_df["ts"], errors="coerce", utc=True)
    else:
        ts_parsed = pd.Series([pd.NaT] * len(sig_df), index=sig_df.index)
        sig_df["ts"] = None

    sig_df["_ts_parsed"] = ts_parsed
    sig_df["_row"] = range(len(sig_df))
    sig_df = sig_df.sort_values(by=["_ts_parsed", "_row"], ascending=[True, True])
    sig_df = sig_df.groupby(["game_id", "lens", "side"], as_index=False).first()
    sig_df = sig_df.drop(columns=[c for c in ["_ts_parsed", "_row"] if c in sig_df.columns])

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
    if "actual_total_1h" in res_df.columns:
        res_df["total_1h"] = pd.to_numeric(res_df["actual_total_1h"], errors="coerce")
    else:
        res_df["total_1h"] = math.nan
    if "actual_total_2h" in res_df.columns:
        res_df["total_2h"] = pd.to_numeric(res_df["actual_total_2h"], errors="coerce")
    else:
        res_df["total_2h"] = math.nan

    merged = sig_df.merge(res_df[["game_id", "final_total", "total_1h", "total_2h"]], on="game_id", how="left")

    # Pick target total based on lens
    target = pd.Series([math.nan] * len(merged))
    try:
        target = target.where(~merged["lens"].eq("fg"), other=pd.to_numeric(merged["final_total"], errors="coerce"))
        target = target.where(~merged["lens"].eq("1h"), other=pd.to_numeric(merged["total_1h"], errors="coerce"))
        target = target.where(~merged["lens"].eq("2h"), other=pd.to_numeric(merged["total_2h"], errors="coerce"))
    except Exception:
        pass
    merged["target_total"] = pd.to_numeric(target, errors="coerce")

    y = pd.to_numeric(merged["target_total"], errors="coerce")
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
            "message": "No settled signals (missing totals for game_id join)",
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
        return 0.0

    settled["profit_units"] = settled["result"].map(_profit)

    wins = int((settled["result"] == 1.0).sum())
    losses = int((settled["result"] == 0.0).sum())
    pushes = int((settled["result"] == 0.5).sum())
    denom = wins + losses
    win_rate = (wins / denom) if denom > 0 else None
    roi = float(settled["profit_units"].sum() / max(1, len(settled)))

    by_lens_side: list[dict[str, Any]] = []
    for (lens_k, side_k), g in settled.groupby(["lens", "side"]):
        w = int((g["result"] == 1.0).sum())
        l = int((g["result"] == 0.0).sum())
        p = int((g["result"] == 0.5).sum())
        d2 = w + l
        by_lens_side.append(
            {
                "lens": str(lens_k),
                "side": str(side_k),
                "n": int(len(g)),
                "wins": w,
                "losses": l,
                "pushes": p,
                "win_rate": (w / d2) if d2 > 0 else None,
                "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
            }
        )
    by_lens_side.sort(key=lambda x: (x["lens"], x["side"]))

    summary: dict[str, Any] = {
        "status": "ok",
        "policy": "signal_side",
        "date": date,
        "signals_path": str(sig_p),
        "results_path": str(res_p),
        "assume_price": price,
        "dedupe": "first_bet_per_game_lens_side",
        "n_signals_raw": int(len(pd.DataFrame(signals))),
        "n_signals_used": int(len(sig_df)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi_units_per_bet": roi,
        "by_lens_side": by_lens_side,
    }

    return {
        "summary": summary,
        "rows": settled,
    }


def compute_live_lens_ats_side_accuracy(cfg: LiveLensAccuracyConfig) -> dict[str, Any]:
    """Compute ATS accuracy by signal-side, collapsing repeated line-change logs.

    The UI may log multiple ATS BET rows for the same bet idea as the spread moves;
    for analysis, we treat that as one decision per (game_id, lens, side) and keep
    the earliest timestamped BET row.
    """

    date = _safe_date(cfg.date)
    out_root = Path(cfg.out_dir) if cfg.out_dir is not None else _root_outputs()

    signals, sig_p, sig_kind, sig_tried = _load_best_signals_jsonl(date, out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)

    if not signals:
        return {
            "status": "missing",
            "date": date,
            "message": f"No signals found for {date}",
            "signals_path": str(sig_p),
            "signals_kind": str(sig_kind),
            "signals_tried": list(sig_tried),
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

    # ATS-only
    if "kind" in sig_df.columns:
        sig_df["kind"] = sig_df["kind"].astype(str).str.strip().str.lower()
        sig_df = sig_df[sig_df["kind"].eq("ats")].copy()
    else:
        # Backward-compat: older logs may not include `kind`. Infer ATS rows by side.
        # Totals use side=over/under; ATS uses side=home/away.
        if "side" in sig_df.columns:
            side_norm = sig_df["side"].astype(str).str.strip().str.lower()
            sig_df = sig_df[side_norm.isin(["home", "away"])].copy()
            if not sig_df.empty:
                sig_df["kind"] = "ats"
        if sig_df.empty:
            return {
                "status": "empty",
                "date": date,
                "message": "Signals missing kind; cannot isolate ATS (no home/away rows to infer)",
                "signals_path": str(sig_p),
                "results_path": str(res_p),
                "signals_cols": list(sig_df.columns),
                "n_signals_raw": int(len(pd.DataFrame(signals))),
            }

    # BET-only
    if "is_bet" in sig_df.columns:
        sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
    else:
        sig_df["is_bet"] = True
    sig_df = sig_df[sig_df["is_bet"].astype(bool)].copy()

    # Side + line
    if "side" not in sig_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing side (home/away); cannot settle",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(sig_df)),
            "signals_cols": list(sig_df.columns),
        }
    sig_df["side"] = sig_df["side"].astype(str).str.strip().str.lower()
    sig_df = sig_df[sig_df["side"].isin(["home", "away"])].copy()

    if "live_line" in sig_df.columns:
        sig_df["live_line"] = pd.to_numeric(sig_df["live_line"], errors="coerce")
    else:
        sig_df["live_line"] = math.nan
    sig_df = sig_df[sig_df["live_line"].notna()].copy()

    # Lens normalization: {fg, 1h, 2h}
    if "lens" in sig_df.columns:
        lens0 = sig_df["lens"].astype(str).str.strip().str.lower()
    else:
        lens0 = pd.Series([""] * len(sig_df), index=sig_df.index)

    lens_norm = lens0.replace({
        "full_game": "fg",
        "full": "fg",
        "game": "fg",
        "fg": "fg",
        "1h": "1h",
        "2h": "2h",
        "first_half": "1h",
        "second_half": "2h",
    })
    lens_norm = lens_norm.where(lens_norm.isin(["fg", "1h", "2h"]), other="")

    # If lens is missing, fall back to horizon heuristic.
    if lens_norm.eq("").any() and "horizon" in sig_df.columns:
        hz = pd.to_numeric(sig_df.get("horizon"), errors="coerce")
        per = (
            pd.to_numeric(sig_df.get("period"), errors="coerce")
            if "period" in sig_df.columns
            else pd.Series([math.nan] * len(sig_df), index=sig_df.index)
        )

        def _infer(h: float | None, p: float | None) -> str:
            try:
                if h is not None and pd.notna(h) and float(h) >= 39:
                    return "fg"
            except Exception:
                pass
            try:
                if h is not None and pd.notna(h) and float(h) <= 21:
                    if p is not None and pd.notna(p) and int(float(p)) >= 2:
                        return "2h"
                    if p is not None and pd.notna(p) and int(float(p)) == 1:
                        return "1h"
            except Exception:
                pass
            return ""

        inferred = [
            _infer((float(h) if pd.notna(h) else None), (float(p) if pd.notna(p) else None))
            for h, p in zip(hz.tolist(), per.tolist())
        ]
        lens_norm = lens_norm.where(~lens_norm.eq(""), other=pd.Series(inferred, index=lens_norm.index))

    sig_df["lens"] = lens_norm
    sig_df = sig_df[sig_df["lens"].isin(["fg", "1h", "2h"])].copy()

    if cfg.full_game_only:
        sig_df = sig_df[sig_df["lens"].eq("fg")].copy()

    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No ATS BET signals with lens+line to evaluate",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(pd.DataFrame(signals))),
        }

    # Deduplicate repeated line-change logs: keep earliest per (game_id, lens, side).
    if "ts" in sig_df.columns:
        sig_df["ts"] = sig_df["ts"].astype(str)
        ts_parsed = pd.to_datetime(sig_df["ts"], errors="coerce", utc=True)
    else:
        ts_parsed = pd.Series([pd.NaT] * len(sig_df), index=sig_df.index)
        sig_df["ts"] = None

    sig_df["_ts_parsed"] = ts_parsed
    sig_df["_row"] = range(len(sig_df))
    sig_df = sig_df.sort_values(by=["_ts_parsed", "_row"], ascending=[True, True])
    sig_df = sig_df.groupby(["game_id", "lens", "side"], as_index=False).first()
    sig_df = sig_df.drop(columns=[c for c in ["_ts_parsed", "_row"] if c in sig_df.columns])

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

    res_df["final_margin"] = _final_margin_from_results(res_df)
    merged = sig_df.merge(res_df[["game_id", "final_margin"]], on="game_id", how="left")

    # Outcome
    y = pd.to_numeric(merged["final_margin"], errors="coerce")
    line = pd.to_numeric(merged["live_line"], errors="coerce")

    def _settle_ats(side: str, yv: float, lv: float) -> float | None:
        if not (isinstance(yv, (int, float)) and isinstance(lv, (int, float))):
            return None
        if not (math.isfinite(float(yv)) and math.isfinite(float(lv))):
            return None
        s = str(side or "").strip().lower()
        if s == "home":
            v = float(yv) + float(lv)
            if v == 0:
                return 0.5
            return 1.0 if v > 0 else 0.0
        if s == "away":
            if float(yv) == float(lv):
                return 0.5
            return 1.0 if float(yv) < float(lv) else 0.0
        return None

    merged["result"] = [
        _settle_ats(str(s), float(yv) if pd.notna(yv) else float("nan"), float(lv) if pd.notna(lv) else float("nan"))
        for s, yv, lv in zip(merged["side"], y, line)
    ]

    settled = merged[merged["result"].notna()].copy()
    if settled.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "No settled ATS signals (missing final margins for game_id join)",
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

    by_elapsed_bucket: list[dict[str, Any]] = []
    if "elapsed" in settled.columns:
        try:
            el = pd.to_numeric(settled["elapsed"], errors="coerce")
            settled["elapsed_bucket"] = (el // 5 * 5).astype("Int64")
            for b, g in settled.dropna(subset=["elapsed_bucket"]).groupby("elapsed_bucket"):
                w = int((g["result"] == 1.0).sum())
                l = int((g["result"] == 0.0).sum())
                p = int((g["result"] == 0.5).sum())
                d2 = w + l
                by_elapsed_bucket.append(
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
            by_elapsed_bucket.sort(key=lambda x: x["elapsed_bucket"])
        except Exception:
            by_elapsed_bucket = []

    by_side: list[dict[str, Any]] = []
    try:
        for side_k, g in settled.groupby("side"):
            w = int((g["result"] == 1.0).sum())
            l = int((g["result"] == 0.0).sum())
            p = int((g["result"] == 0.5).sum())
            d2 = w + l
            by_side.append(
                {
                    "side": str(side_k),
                    "n": int(len(g)),
                    "wins": w,
                    "losses": l,
                    "pushes": p,
                    "win_rate": (w / d2) if d2 > 0 else None,
                    "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
                }
            )
        by_side.sort(key=lambda x: x["side"])
    except Exception:
        by_side = []

    by_edge_bucket: list[dict[str, Any]] = []
    if "edge" in settled.columns:
        try:
            e = pd.to_numeric(settled["edge"], errors="coerce")
            # Fixed buckets around 0; keep it simple and stable.
            bins = [-1e9, -0.05, -0.02, 0.0, 0.02, 0.05, 1e9]
            labels = ["<-0.05", "-0.05..-0.02", "-0.02..0", "0..0.02", "0.02..0.05", ">=0.05"]
            settled["edge_bucket"] = pd.cut(e, bins=bins, labels=labels, right=False)
            for b, g in settled.dropna(subset=["edge_bucket"]).groupby("edge_bucket"):
                w = int((g["result"] == 1.0).sum())
                l = int((g["result"] == 0.0).sum())
                p = int((g["result"] == 0.5).sum())
                d2 = w + l
                by_edge_bucket.append(
                    {
                        "edge_bucket": str(b),
                        "n": int(len(g)),
                        "wins": w,
                        "losses": l,
                        "pushes": p,
                        "win_rate": (w / d2) if d2 > 0 else None,
                        "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
                    }
                )
            by_edge_bucket.sort(key=lambda x: labels.index(x["edge_bucket"]) if x["edge_bucket"] in labels else 999)
        except Exception:
            by_edge_bucket = []

    by_driver_tag: list[dict[str, Any]] = []
    if "driver_tags" in settled.columns or "driver" in settled.columns:
        try:
            def _parse_tags(v: Any) -> list[str]:
                if v is None:
                    return []
                try:
                    if pd.isna(v):
                        return []
                except Exception:
                    pass
                if isinstance(v, list):
                    return [str(x).strip() for x in v if x is not None and str(x).strip()]
                # Sometimes serialized as JSON string.
                try:
                    s0 = str(v).strip()
                    if not s0 or s0.lower() in {"nan", "none", "null"}:
                        return []
                    if s0.startswith("[") and s0.endswith("]"):
                        j = json.loads(s0)
                        if isinstance(j, list):
                            return [str(x).strip() for x in j if x is not None and str(x).strip()]
                except Exception:
                    pass
                try:
                    parts = [p.strip() for p in str(v).replace("|", ",").split(",")]
                    return [p for p in parts if p]
                except Exception:
                    return []

            tags: list[list[str]] = []
            if "driver_tags" in settled.columns:
                tags = [_parse_tags(v) for v in settled["driver_tags"].tolist()]
            elif "driver" in settled.columns:
                # Fallback: treat driver explainer text as a single “tag”.
                tags = [[str(v).strip()] if v is not None and str(v).strip() else [] for v in settled["driver"].tolist()]

            flat_rows: list[dict[str, Any]] = []
            for tg_list, res0, prof0 in zip(tags, settled["result"].tolist(), settled["profit_units"].tolist()):
                for tg in tg_list:
                    if not tg:
                        continue
                    flat_rows.append({"tag": str(tg), "result": float(res0), "profit_units": float(prof0)})

            if flat_rows:
                tdf = pd.DataFrame(flat_rows)
                # Only keep tags with at least a few samples to reduce noise.
                min_n = 3
                stats: list[dict[str, Any]] = []
                for tg, g in tdf.groupby("tag"):
                    n = int(len(g))
                    if n < min_n:
                        continue
                    w = int((g["result"] == 1.0).sum())
                    l = int((g["result"] == 0.0).sum())
                    p = int((g["result"] == 0.5).sum())
                    d2 = w + l
                    stats.append(
                        {
                            "tag": str(tg),
                            "n": n,
                            "wins": w,
                            "losses": l,
                            "pushes": p,
                            "win_rate": (w / d2) if d2 > 0 else None,
                            "roi_units_per_bet": float(pd.to_numeric(g["profit_units"], errors="coerce").fillna(0.0).sum() / max(1, n)),
                        }
                    )
                # Keep a compact view: worst 25 + best 10.
                stats.sort(key=lambda r: (r.get("roi_units_per_bet") if r.get("roi_units_per_bet") is not None else float("inf")))
                worst = stats[:25]
                best = list(reversed(stats[-10:])) if len(stats) > 25 else []
                by_driver_tag = worst + best
        except Exception:
            by_driver_tag = []

    by_lens_side: list[dict[str, Any]] = []
    for (lens_k, side_k), g in settled.groupby(["lens", "side"]):
        w = int((g["result"] == 1.0).sum())
        l = int((g["result"] == 0.0).sum())
        p = int((g["result"] == 0.5).sum())
        d2 = w + l
        by_lens_side.append(
            {
                "lens": str(lens_k),
                "side": str(side_k),
                "n": int(len(g)),
                "wins": w,
                "losses": l,
                "pushes": p,
                "win_rate": (w / d2) if d2 > 0 else None,
                "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
            }
        )
    by_lens_side.sort(key=lambda x: (x["lens"], x["side"]))

    summary: dict[str, Any] = {
        "status": "ok",
        "policy": "signal_side",
        "date": date,
        "signals_path": str(sig_p),
        "results_path": str(res_p),
        "assume_price": price,
        "market": "ats",
        "dedupe": "first_bet_per_game_lens_side",
        "n_signals_raw": int(len(pd.DataFrame(signals))),
        "n_signals_used": int(len(sig_df)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi_units_per_bet": roi,
        "by_lens_side": by_lens_side,
        "by_side": by_side,
        "by_elapsed_bucket": by_elapsed_bucket,
        "by_edge_bucket": by_edge_bucket,
        "by_driver_tag": by_driver_tag,
    }

    return {
        "summary": summary,
        "rows": settled,
    }


def compute_live_lens_ats_accuracy(cfg: LiveLensAccuracyConfig) -> dict[str, Any]:
    """Compute Live Lens ATS bet ROI / win-rate from logged signals + finalized results.

    Notes:
      - UI logs ATS `live_line` as the handicap for the selected side:
        * side=home -> line == spread_home
        * side=away -> line == -spread_home
      - Settlement rules (actual_margin = home_score - away_score):
        * home wins if actual_margin + line > 0
        * away wins if actual_margin < line
        * push if equality
    """

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

    # ATS-only
    if "kind" in sig_df.columns:
        sig_df["kind"] = sig_df["kind"].astype(str).str.strip().str.lower()
        sig_df = sig_df[sig_df["kind"].eq("ats")].copy()
    else:
        # Backward-compat: older logs may not include `kind`. Infer ATS rows by side.
        if "side" in sig_df.columns:
            side_norm = sig_df["side"].astype(str).str.strip().str.lower()
            sig_df = sig_df[side_norm.isin(["home", "away"])].copy()
            if not sig_df.empty:
                sig_df["kind"] = "ats"
        if sig_df.empty:
            return {
                "status": "empty",
                "date": date,
                "message": "Signals missing kind; cannot isolate ATS (no home/away rows to infer)",
                "signals_path": str(sig_p),
                "results_path": str(res_p),
                "signals_cols": list(sig_df.columns),
                "n_signals_raw": int(len(pd.DataFrame(signals))),
            }

    # BET-only
    if "is_bet" in sig_df.columns:
        sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
    else:
        sig_df["is_bet"] = True
    sig_df = sig_df[sig_df["is_bet"].astype(bool)].copy()

    # Side + line
    if "side" not in sig_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Signals missing side (home/away); cannot settle",
            "signals_path": str(sig_p),
            "results_path": str(res_p),
            "n_signals_raw": int(len(sig_df)),
            "signals_cols": list(sig_df.columns),
        }
    sig_df["side"] = sig_df["side"].astype(str).str.strip().str.lower()
    sig_df = sig_df[sig_df["side"].isin(["home", "away"])].copy()

    if "live_line" in sig_df.columns:
        sig_df["live_line"] = pd.to_numeric(sig_df["live_line"], errors="coerce")
    else:
        sig_df["live_line"] = math.nan
    sig_df = sig_df[sig_df["live_line"].notna()].copy()

    if cfg.full_game_only and "horizon" in sig_df.columns:
        hz = pd.to_numeric(sig_df["horizon"], errors="coerce")
        sig_df = sig_df[hz >= 39].copy()

    if sig_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No ATS bet signals with live_line to evaluate",
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

    # Avoid settling against partial scores.
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

    res_df["final_margin"] = _final_margin_from_results(res_df)
    merged = sig_df.merge(res_df[["game_id", "final_margin"]], on="game_id", how="left")

    # Outcome
    y = pd.to_numeric(merged["final_margin"], errors="coerce")
    line = pd.to_numeric(merged["live_line"], errors="coerce")

    def _settle_ats(side: str, yv: float, lv: float) -> float | None:
        if not (isinstance(yv, (int, float)) and isinstance(lv, (int, float))):
            return None
        if not (math.isfinite(float(yv)) and math.isfinite(float(lv))):
            return None
        s = str(side or "").strip().lower()
        if s == "home":
            v = float(yv) + float(lv)
            if v == 0:
                return 0.5
            return 1.0 if v > 0 else 0.0
        if s == "away":
            if float(yv) == float(lv):
                return 0.5
            return 1.0 if float(yv) < float(lv) else 0.0
        return None

    merged["result"] = [
        _settle_ats(str(s), float(yv) if pd.notna(yv) else float("nan"), float(lv) if pd.notna(lv) else float("nan"))
        for s, yv, lv in zip(merged["side"], y, line)
    ]

    settled = merged[merged["result"].notna()].copy()
    if settled.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "No settled ATS signals (missing final margins for game_id join)",
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
        "date": date,
        "signals_path": str(sig_p),
        "results_path": str(res_p),
        "assume_price": price,
        "market": "ats",
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
                    "signal_id",
                    "ts",
                    "date",
                    "game_id",
                    "lens",
                    "kind",
                    "horizon",
                    "elapsed",
                    "remaining",
                    "total_points",
                    "live_line",
                    "side",
                    "strength",
                    "edge",
                    "model_prob",
                    "market_prob",
                    "driver",
                    "driver_tags",
                    "final_total",
                    "final_margin",
                    "target_total",
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


def compute_live_lens_projection_accuracy(cfg: LiveLensProjectionAccuracyConfig) -> dict[str, Any]:
    date = _safe_date(cfg.date)
    out_root = Path(cfg.out_dir) if cfg.out_dir is not None else _root_outputs()

    proj_p = projections_path(date, out_dir=out_root)
    res_p = results_path(date, out_dir=out_root, daily_results_dir=cfg.daily_results_dir)

    rows = _read_jsonl(proj_p)
    if not rows:
        return {
            "status": "missing",
            "date": date,
            "message": f"No projections found at {proj_p}",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
        }

    proj_df = pd.DataFrame(rows)
    if proj_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"No projection rows parsed from {proj_p}",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
        }

    if "game_id" not in proj_df.columns and "event_id" in proj_df.columns:
        proj_df["game_id"] = proj_df["event_id"]
    if "game_id" not in proj_df.columns:
        return {
            "status": "empty",
            "date": date,
            "message": "Projections missing game_id/event_id; cannot evaluate",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
            "projection_cols": list(proj_df.columns),
            "n_projections_raw": int(len(proj_df)),
        }

    proj_df["game_id"] = proj_df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    if cfg.full_game_only and "lens" in proj_df.columns:
        try:
            lens = proj_df["lens"].astype(str).str.strip().str.lower()
            proj_df = proj_df[lens.isin(["full_game", "fg", "full", "game"])].copy()
        except Exception:
            pass

    if proj_df.empty:
        return {
            "status": "empty",
            "date": date,
            "message": "No projections remaining after filters",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
        }

    # Load results
    if not res_p.exists():
        return {
            "status": "missing",
            "date": date,
            "message": f"Missing results file at {res_p}",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
            "n_projections": int(len(proj_df)),
        }

    res_df = pd.read_csv(res_p)
    if res_df.empty:
        return {
            "status": "missing",
            "date": date,
            "message": f"Empty results file at {res_p}",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
            "n_projections": int(len(proj_df)),
        }

    gid_col = _pick_col(res_df, ["game_id", "event_id", "id", "gid"])
    if not gid_col:
        return {
            "status": "error",
            "date": date,
            "message": "Results file missing game_id/event_id column",
            "projections_path": str(proj_p),
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
            "projections_path": str(proj_p),
            "results_path": str(res_p),
            "n_projections": int(len(proj_df)),
        }

    res_df["final_total"] = _final_total_from_results(res_df)

    c_1h = _pick_col(res_df, ["actual_total_1h", "total_1h", "first_half_total", "total_points_1h"])
    c_2h = _pick_col(res_df, ["actual_total_2h", "total_2h", "second_half_total", "total_points_2h"])
    if c_1h:
        res_df["total_1h"] = pd.to_numeric(res_df[c_1h], errors="coerce")
    else:
        res_df["total_1h"] = math.nan
    if c_2h:
        res_df["total_2h"] = pd.to_numeric(res_df[c_2h], errors="coerce")
    else:
        res_df["total_2h"] = math.nan

    merged = proj_df.merge(res_df[["game_id", "final_total", "total_1h", "total_2h"]], on="game_id", how="left")

    # Pick target per lens when possible.
    if "lens" in merged.columns:
        lens = merged["lens"].astype(str).str.strip().str.lower()
    else:
        lens = pd.Series([""] * len(merged))

    y_final = pd.to_numeric(merged["final_total"], errors="coerce")
    y_1h = pd.to_numeric(merged["total_1h"], errors="coerce")
    y_2h = pd.to_numeric(merged["total_2h"], errors="coerce")

    y = y_final.copy()
    try:
        mask_1h = lens.eq("1h") & y_1h.notna()
        mask_2h = lens.eq("2h") & y_2h.notna()
        y.loc[mask_1h] = y_1h.loc[mask_1h]
        y.loc[mask_2h] = y_2h.loc[mask_2h]
    except Exception:
        pass

    merged["target_total"] = y

    p_blend = pd.to_numeric(merged["proj_blend"], errors="coerce") if "proj_blend" in merged.columns else pd.Series([math.nan] * len(merged))
    p_final = pd.to_numeric(merged["proj_final"], errors="coerce") if "proj_final" in merged.columns else pd.Series([math.nan] * len(merged))

    merged["abs_err_blend"] = (p_blend - y).abs()
    merged["abs_err_final"] = (p_final - y).abs()
    merged["sq_err_blend"] = (p_blend - y) ** 2
    merged["sq_err_final"] = (p_final - y) ** 2

    settled = merged[y.notna()].copy()
    if settled.empty:
        return {
            "status": "missing",
            "date": date,
            "message": "No settled projections (missing final totals for game_id join)",
            "projections_path": str(proj_p),
            "results_path": str(res_p),
            "n_projections": int(len(merged)),
            "n_results": int(len(res_df)),
        }

    def _mae(s: pd.Series) -> float | None:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float(v.mean()) if not v.empty else None

    def _rmse(s: pd.Series) -> float | None:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float(math.sqrt(float(v.mean()))) if not v.empty else None

    by_remaining: list[dict[str, Any]] = []
    if "remaining_bucket" in settled.columns:
        try:
            rb = pd.to_numeric(settled["remaining_bucket"], errors="coerce").astype("Int64")
            settled["remaining_bucket"] = rb
            for b, g in settled.dropna(subset=["remaining_bucket"]).groupby("remaining_bucket"):
                by_remaining.append(
                    {
                        "remaining_bucket": int(b),
                        "n": int(len(g)),
                        "mae_proj_blend": _mae(g["abs_err_blend"]),
                        "rmse_proj_blend": _rmse(g["sq_err_blend"]),
                        "mae_proj_final": _mae(g["abs_err_final"]),
                        "rmse_proj_final": _rmse(g["sq_err_final"]),
                    }
                )
            by_remaining.sort(key=lambda x: x["remaining_bucket"], reverse=True)
        except Exception:
            by_remaining = []

    by_lens: list[dict[str, Any]] = []
    if "lens" in settled.columns:
        try:
            settled["lens"] = settled["lens"].astype(str).str.strip().str.lower()
            for l, g in settled.groupby("lens"):
                by_lens.append(
                    {
                        "lens": str(l),
                        "n": int(len(g)),
                        "mae_proj_blend": _mae(g["abs_err_blend"]),
                        "rmse_proj_blend": _rmse(g["sq_err_blend"]),
                        "mae_proj_final": _mae(g["abs_err_final"]),
                        "rmse_proj_final": _rmse(g["sq_err_final"]),
                    }
                )
            by_lens.sort(key=lambda x: x["n"], reverse=True)
        except Exception:
            by_lens = []

    summary: dict[str, Any] = {
        "status": "ok",
        "date": date,
        "projections_path": str(proj_p),
        "results_path": str(res_p),
        "full_game_only": bool(cfg.full_game_only),
        "n_projections": int(len(merged)),
        "n_settled": int(len(settled)),
        "n_games": int(settled["game_id"].nunique()),
        "mae_proj_blend": _mae(settled["abs_err_blend"]),
        "rmse_proj_blend": _rmse(settled["sq_err_blend"]),
        "mae_proj_final": _mae(settled["abs_err_final"]),
        "rmse_proj_final": _rmse(settled["sq_err_final"]),
        "by_remaining_bucket": by_remaining,
        "by_lens": by_lens,
    }

    return {
        "summary": summary,
        "rows": settled,
    }


def write_live_lens_projection_accuracy(
    out_json: Path,
    payload: dict[str, Any],
    out_csv: Path | None = None,
) -> dict[str, Any]:
    out_json.parent.mkdir(parents=True, exist_ok=True)

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
                    "lens",
                    "horizon",
                    "elapsed",
                    "remaining",
                    "remaining_bucket",
                    "total_points",
                    "live_line",
                    "proj_final",
                    "proj_blend",
                    "target_total",
                    "abs_err_final",
                    "abs_err_blend",
                    "tuning_source",
                ]
                if c in rows_df.columns
            ]
            rows_df[keep].to_csv(out_csv, index=False)
        return {"status": "ok", "out_json": str(out_json), "out_csv": str(out_csv) if out_csv else None, "summary": summary}

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "ok", "out_json": str(out_json), "out_csv": None, "payload": payload}


def make_signal_id(d: dict[str, Any]) -> str:
    # Helpful for clients/servers wanting idempotency.
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
