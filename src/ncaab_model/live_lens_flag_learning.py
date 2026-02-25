from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .live_lens_accuracy import (
    _filter_results_to_finals,
    _final_total_from_results,
    _pick_col,
    _read_jsonl,
    _remaining_from_row,
    _safe_date,
    results_path,
    signals_path,
)


@dataclass(frozen=True)
class FlagLearningConfig:
    start_date: str
    end_date: str
    out_dir: Path = Path("outputs")
    daily_results_dir: Path | None = None

    assume_price: float = -110.0
    full_game_only: bool = True

    # What signal pool to learn from.
    include_watch: bool = True

    # Learning settings
    min_tag_n: int = 25
    min_overall_n: int = 50
    max_penalty: float = 2.0
    step: float = 0.25
    min_improve_roi: float = 0.002
    max_tags: int = 12


def _iter_date_range(start_date: str, end_date: str) -> list[str]:
    s = dt.date.fromisoformat(_safe_date(start_date))
    e = dt.date.fromisoformat(_safe_date(end_date))
    if s > e:
        s, e = e, s
    days: list[str] = []
    cur = s
    while cur <= e:
        days.append(cur.isoformat())
        cur = cur + dt.timedelta(days=1)
    return days


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
            if s.lower() in {"nan", "none", "null"}:
                continue
            if s:
                out.append(s)
        return out
    try:
        s0 = str(v).strip()
        if not s0:
            return []
        if s0.lower() in {"nan", "none", "null"}:
            return []
        if s0.startswith("[") and s0.endswith("]"):
            j = json.loads(s0)
            if isinstance(j, list):
                return [
                    str(x).strip()
                    for x in j
                    if x is not None and str(x).strip() and str(x).strip().lower() not in {"nan", "none", "null"}
                ]
    except Exception:
        pass
    try:
        s1 = str(v)
        parts = [p.strip() for p in s1.replace("|", ",").split(",")]
        return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]
    except Exception:
        return []


def _tags_from_driver_explainer(raw: Any) -> list[str]:
    """Derive canonical driver tags from the raw driver explainer string.

    Mirrors the frontend tag extraction logic (see templates/index.html).
    """

    try:
        s = str(raw or "").strip()
    except Exception:
        return []

    if not s or s in {"–", "-", "none", "null"}:
        return []

    parts = [p.strip() for p in s.split("|")]
    tags: list[str] = []

    def add(t: str) -> None:
        if not t:
            return
        if t in tags:
            return
        tags.append(t)

    for p in parts:
        pl = p.lower().strip()
        if not pl:
            continue
        if pl.startswith("edge "):
            add("EDGE")
        elif pl.startswith("d "):
            add("SIM-GAP")
        elif pl.startswith("pace "):
            add("PACE")
        elif pl.startswith("ppp "):
            add("PPP")
        elif pl.startswith("shooting "):
            add("SHOOTING")
        elif pl.startswith("ft rate "):
            add("FT")
        elif pl.startswith("pbp+"):
            add("PBP")
        elif "late-over" in pl:
            add("LATE-OVER")
        elif "early-over" in pl:
            add("EARLY-OVER")
    return tags


def _apply_over_penalties(
    df: pd.DataFrame,
    *,
    late_over_strength_penalty: float,
    late_over_remaining_lo: float,
    late_over_remaining_hi: float,
    late_over_margin_abs_min: float,
    late_over_period_min: float,
    early_over_strength_penalty: float,
    early_over_remaining_min: float,
    early_over_period_max: float,
) -> pd.Series:
    """Return per-row penalty to subtract from strength."""

    if df.empty:
        return pd.Series([], dtype=float)

    side = df["side"].astype(str).str.strip().str.lower()
    is_over = side.eq("over")

    hz = pd.to_numeric(df.get("horizon"), errors="coerce")
    el = pd.to_numeric(df.get("elapsed"), errors="coerce")
    rem = pd.to_numeric(df.get("remaining"), errors="coerce")

    pen = pd.Series([0.0] * len(df), index=df.index, dtype=float)

    # Late window
    pen_late = float(late_over_strength_penalty or 0.0)
    if pen_late > 0:
        lo_r = float(late_over_remaining_lo)
        hi_r = float(late_over_remaining_hi)
        if hi_r < lo_r:
            lo_r, hi_r = hi_r, lo_r
        in_window = (rem > lo_r) & (rem <= hi_r)

        period_ok = pd.Series([True] * len(df), index=df.index)
        if float(late_over_period_min or 1.0) > 1:
            if "period" in df.columns:
                per = pd.to_numeric(df.get("period"), errors="coerce")
                period_ok = per.notna() & (per >= float(late_over_period_min))
            else:
                period_ok = (hz >= 39) & el.notna() & (el >= 20)

        margin_ok = pd.Series([True] * len(df), index=df.index)
        if float(late_over_margin_abs_min or 0.0) > 0:
            if "margin_home" in df.columns:
                m = pd.to_numeric(df.get("margin_home"), errors="coerce")
                margin_ok = m.abs() >= float(late_over_margin_abs_min)
            else:
                margin_ok = pd.Series([False] * len(df), index=df.index)

        late_mask = is_over & in_window & period_ok & margin_ok
        pen = pen.where(~late_mask, pen + pen_late)

    # Early window
    pen_early = float(early_over_strength_penalty or 0.0)
    if pen_early > 0:
        in_early = rem >= float(early_over_remaining_min or 20.0)

        period_ok_e = pd.Series([True] * len(df), index=df.index)
        if "period" in df.columns:
            per = pd.to_numeric(df.get("period"), errors="coerce")
            period_ok_e = per.notna() & (per <= float(early_over_period_max))
        else:
            period_ok_e = (hz >= 39) & el.notna() & (el < 20)

        early_mask = is_over & in_early & period_ok_e
        pen = pen.where(~early_mask, pen + pen_early)

    return pen


def _settle_totals(df: pd.DataFrame, res_df: pd.DataFrame, *, assume_price: float) -> pd.DataFrame:
    if df.empty or res_df.empty:
        return pd.DataFrame()

    gid_col = _pick_col(res_df, ["game_id", "event_id", "id", "gid"])
    if not gid_col:
        return pd.DataFrame()

    res = res_df.copy()
    res["game_id"] = res[gid_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    res = _filter_results_to_finals(res)
    if res.empty:
        return pd.DataFrame()

    res["final_total"] = _final_total_from_results(res)

    merged = df.merge(res[["game_id", "final_total"]], on="game_id", how="left")
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
        return settled

    price = float(assume_price)
    win_profit = 100.0 / abs(price) if price < 0 else (price / 100.0)

    def _profit(res0: float) -> float:
        if res0 == 1.0:
            return float(win_profit)
        if res0 == 0.0:
            return -1.0
        return 0.0

    settled["profit_units"] = settled["result"].map(_profit)
    return settled


def _roi(df: pd.DataFrame) -> tuple[int, float | None]:
    if df is None or df.empty:
        return 0, None
    try:
        n = int(len(df))
        roi = float(pd.to_numeric(df["profit_units"], errors="coerce").fillna(0.0).sum() / max(1, n))
        return n, roi
    except Exception:
        return int(len(df)), None


def learn_driver_tag_strength_penalties(
    cfg: FlagLearningConfig,
    *,
    late_over: dict[str, float] | None = None,
    early_over: dict[str, float] | None = None,
) -> dict[str, Any]:
    dates = _iter_date_range(cfg.start_date, cfg.end_date)

    all_rows: list[pd.DataFrame] = []
    missing: list[dict[str, Any]] = []

    for d in dates:
        sig_p = signals_path(d, out_dir=cfg.out_dir)
        res_p = results_path(d, out_dir=cfg.out_dir, daily_results_dir=cfg.daily_results_dir)

        signals = _read_jsonl(sig_p)
        if not signals:
            missing.append({"date": d, "status": "missing_signals", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        sig_df = pd.DataFrame(signals)
        if sig_df.empty:
            missing.append({"date": d, "status": "empty_signals", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        if "game_id" not in sig_df.columns and "event_id" in sig_df.columns:
            sig_df["game_id"] = sig_df["event_id"]
        if "game_id" not in sig_df.columns:
            missing.append({"date": d, "status": "bad_signals", "message": "missing game_id", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        sig_df["game_id"] = sig_df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

        if "side" not in sig_df.columns:
            continue
        sig_df["side"] = sig_df["side"].astype(str).str.strip().str.lower()
        sig_df = sig_df[sig_df["side"].isin(["over", "under"])].copy()

        if "live_line" in sig_df.columns:
            sig_df["live_line"] = pd.to_numeric(sig_df["live_line"], errors="coerce")
        else:
            sig_df["live_line"] = math.nan
        sig_df = sig_df[sig_df["live_line"].notna()].copy()

        if sig_df.empty:
            continue

        # Pool selection: BET-only vs BET+WATCH.
        if "is_bet" in sig_df.columns:
            sig_df["is_bet"] = sig_df["is_bet"].astype(bool)
        else:
            sig_df["is_bet"] = True
        if "is_watch" in sig_df.columns:
            sig_df["is_watch"] = sig_df["is_watch"].astype(bool)
        else:
            sig_df["is_watch"] = False

        if cfg.include_watch:
            sig_df = sig_df[sig_df["is_bet"] | sig_df["is_watch"]].copy()
        else:
            sig_df = sig_df[sig_df["is_bet"]].copy()

        if sig_df.empty:
            continue

        if cfg.full_game_only and "horizon" in sig_df.columns:
            hz = pd.to_numeric(sig_df["horizon"], errors="coerce")
            sig_df = sig_df[hz >= 39].copy()

        if sig_df.empty:
            continue

        # Base strength + remaining
        sig_df["remaining"] = sig_df.apply(_remaining_from_row, axis=1)
        sig_df["elapsed"] = pd.to_numeric(sig_df.get("elapsed"), errors="coerce")
        sig_df["horizon"] = pd.to_numeric(sig_df.get("horizon"), errors="coerce")
        sig_df["strength"] = pd.to_numeric(sig_df.get("strength"), errors="coerce")

        sig_df = sig_df[sig_df["elapsed"].notna() & sig_df["strength"].notna() & sig_df["horizon"].notna()].copy()
        if sig_df.empty:
            continue

        sig_df["cf_strength"] = sig_df["strength"].astype(float)

        # Apply the current over-penalty retune knobs so learning is aligned with policy.
        late = late_over or {}
        early = early_over or {}
        retune_pen = _apply_over_penalties(
            sig_df,
            late_over_strength_penalty=float(late.get("strength_penalty") or 0.0),
            late_over_remaining_lo=float(late.get("remaining_lo") or 5.0),
            late_over_remaining_hi=float(late.get("remaining_hi") or 10.0),
            late_over_margin_abs_min=float(late.get("margin_abs_min") or 0.0),
            late_over_period_min=float(late.get("period_min") or 2.0),
            early_over_strength_penalty=float(early.get("strength_penalty") or 0.0),
            early_over_remaining_min=float(early.get("remaining_min") or 20.0),
            early_over_period_max=float(early.get("period_max") or 1.0),
        )
        sig_df["cf_strength"] = sig_df["cf_strength"] - pd.to_numeric(retune_pen, errors="coerce").fillna(0.0)

        # Results
        if not res_p.exists():
            missing.append({"date": d, "status": "missing_results", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue
        try:
            res_df = pd.read_csv(res_p)
        except Exception:
            missing.append({"date": d, "status": "bad_results", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        settled = _settle_totals(sig_df, res_df, assume_price=float(cfg.assume_price))
        if settled.empty:
            continue

        all_rows.append(settled)

    if not all_rows:
        return {
            "status": "missing",
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "message": "No settled signals in window",
            "missing": missing,
        }

    df = pd.concat(all_rows, ignore_index=True)

    # Tags
    tags_list: list[list[str]]
    if "driver_tags" in df.columns:
        tags_list = [[t.strip() for t in _parse_tags(v) if str(t).strip()] for v in df["driver_tags"].tolist()]
    elif "driver" in df.columns:
        tags_list = [_tags_from_driver_explainer(v) for v in df["driver"].tolist()]
    else:
        return {
            "status": "missing",
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "message": "Signals missing driver_tags/driver (need updated frontend/backend logging)",
            "missing": missing,
        }

    # Thresholds (match UI)
    hz = pd.to_numeric(df["horizon"], errors="coerce")
    el = pd.to_numeric(df["elapsed"], errors="coerce")

    min_elapsed = np.where(hz <= 20.5, 4.0, 6.0)
    thr = np.where(hz <= 20.5, 5.0, 7.0)

    base_strength = pd.to_numeric(df["cf_strength"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)

    # Base bet selection.
    base_bet = (el.to_numpy(dtype=float) >= min_elapsed) & (base_strength >= thr)
    base_df = df[base_bet].copy()
    n0, roi0 = _roi(base_df)

    if n0 < int(cfg.min_overall_n):
        return {
            "status": "insufficient",
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "message": f"Too few baseline BETs to learn from (n={n0} < min_overall_n={cfg.min_overall_n})",
            "baseline": {"n": n0, "roi_units_per_bet": roi0},
            "missing": missing,
        }

    # Candidate tags based on baseline bet pool.
    tag_counts: dict[str, int] = {}
    tag_roi: dict[str, float] = {}

    base_profit = pd.to_numeric(base_df["profit_units"], errors="coerce").fillna(0.0)

    for tg in sorted({t for tags in tags_list for t in tags}):
        mask = np.array([tg in tags for tags in tags_list], dtype=bool)
        mask = mask & base_bet
        n = int(mask.sum())
        if n <= 0:
            continue
        tag_counts[tg] = n
        # ROI restricted to rows where the tag is present.
        try:
            roi_t = float(base_profit[mask[base_bet]].sum() / max(1, n))
        except Exception:
            roi_t = float("nan")
        if math.isfinite(roi_t):
            tag_roi[tg] = roi_t

    # Sort by worst ROI first, but only tags with enough samples.
    # Also restrict to tags that underperform the baseline ROI (penalties are meant
    # to suppress bad regimes, not prune historically-good ones).
    cand_tags = [
        t
        for t, n in tag_counts.items()
        if n >= int(cfg.min_tag_n)
        and (roi0 is None or not math.isfinite(float(roi0)) or (tag_roi.get(t) is not None and float(tag_roi[t]) < float(roi0)))
    ]
    cand_tags.sort(key=lambda t: (tag_roi.get(t, float("inf")), -tag_counts.get(t, 0)))
    cand_tags = cand_tags[: max(0, int(cfg.max_tags))]

    penalties: dict[str, float] = {}

    # Track running penalty sum so each greedy step is O(N) for each grid value.
    pen_sum = np.zeros(len(df), dtype=float)

    def eval_roi(pen_sum_extra: np.ndarray) -> tuple[int, float | None]:
        st = base_strength - (pen_sum + pen_sum_extra)
        bet = (el.to_numpy(dtype=float) >= min_elapsed) & (st >= thr)
        bet_df = df[bet]
        return _roi(bet_df)

    best_n, best_roi = n0, roi0

    grid = [round(x, 6) for x in np.arange(0.0, float(cfg.max_penalty) + 1e-9, float(cfg.step)).tolist()]

    for tg in cand_tags:
        has_tag = np.array([tg in tags for tags in tags_list], dtype=float)

        cur_best_pen = 0.0
        cur_best_n = best_n
        cur_best_roi = best_roi

        for pen in grid:
            extra = has_tag * float(pen)
            n1, roi1 = eval_roi(extra)
            if roi1 is None or not math.isfinite(float(roi1)):
                continue
            if n1 < int(cfg.min_overall_n):
                continue
            if cur_best_roi is None or float(roi1) > float(cur_best_roi):
                cur_best_pen = float(pen)
                cur_best_n = int(n1)
                cur_best_roi = float(roi1)

        # Accept only if it improves enough.
        if (
            cur_best_pen > 0
            and cur_best_roi is not None
            and best_roi is not None
            and (float(cur_best_roi) - float(best_roi)) >= float(cfg.min_improve_roi)
        ):
            penalties[tg] = float(cur_best_pen)
            pen_sum = pen_sum + (has_tag * float(cur_best_pen))
            best_n = int(cur_best_n)
            best_roi = float(cur_best_roi)

    return {
        "status": "ok",
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "dates": dates,
        "assume_price": float(cfg.assume_price),
        "full_game_only": bool(cfg.full_game_only),
        "include_watch": bool(cfg.include_watch),
        "baseline": {"n": int(n0), "roi_units_per_bet": roi0},
        "learned": {"n": int(best_n), "roi_units_per_bet": best_roi},
        "candidates": [
            {"tag": t, "n": int(tag_counts.get(t, 0)), "roi_units_per_bet": tag_roi.get(t)}
            for t in cand_tags
        ],
        "driver_tag_strength_penalties": penalties,
        "missing": missing,
    }


def apply_penalties_to_tuning_json(tuning_json_path: Path, penalties: dict[str, float]) -> dict[str, Any]:
    """Merge penalties into outputs/live_lens_tuning.json.

    Writes under tuning.driver_tag_strength_penalties. Preserves any unknown keys.
    """

    path = Path(tuning_json_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(raw, dict):
        raise ValueError("tuning json root must be object")

    t = raw.get("tuning") if isinstance(raw.get("tuning"), dict) else None
    if t is None:
        # Back-compat: treat raw itself as tuning.
        t = raw
        raw = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "tuning": t, "meta": {}}

    if not isinstance(t, dict):
        raise ValueError("tuning must be object")

    # Normalize + keep only positive values.
    out_map: dict[str, float] = {}
    for k, v in (penalties or {}).items():
        try:
            kk = str(k).strip()
            vv = float(v)
            if not kk or not math.isfinite(vv) or vv <= 0:
                continue
            out_map[kk] = float(vv)
        except Exception:
            continue

    t["driver_tag_strength_penalties"] = out_map
    raw["tuning"] = t

    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw
