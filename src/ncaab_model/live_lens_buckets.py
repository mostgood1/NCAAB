from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .live_lens_accuracy import (
    _final_total_from_results,
    _filter_results_to_finals,
    _pick_col,
    _read_jsonl,
    results_path,
    signals_path,
)


@dataclass(frozen=True)
class LiveLensBucketReportConfig:
    dates: list[str]
    out_dir: Path = Path("outputs")
    daily_results_dir: Path | None = None
    assume_price: float = -110.0
    full_game_only: bool = True
    include_watch: bool = True

    # Optional counterfactual retune: adjust strength and reclassify signals.
    # Matches templates/index.html defaults: apply penalty for OVER in (5,10] minutes remaining.
    apply_retune: bool = False
    late_over_strength_penalty: float = 0.0
    late_over_remaining_lo: float = 5.0
    late_over_remaining_hi: float = 10.0
    late_over_margin_abs_min: float = 0.0
    late_over_period_min: float = 2.0

    early_over_strength_penalty: float = 0.0
    early_over_remaining_min: float = 20.0
    early_over_period_max: float = 1.0


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    dt.date.fromisoformat(s2)
    return s2


def _profit_units(result: float, assume_price: float) -> float:
    price = float(assume_price)
    win_profit = 100.0 / abs(price) if price < 0 else (price / 100.0)
    if result == 1.0:
        return float(win_profit)
    if result == 0.0:
        return -1.0
    return 0.0


def _settle(side: str, final_total: float, live_line: float) -> float | None:
    try:
        yv = float(final_total)
        lv = float(live_line)
    except Exception:
        return None
    if not (math.isfinite(yv) and math.isfinite(lv)):
        return None
    if yv == lv:
        return 0.5
    side2 = str(side or "").strip().lower()
    if side2 == "over":
        return 1.0 if yv > lv else 0.0
    if side2 == "under":
        return 1.0 if yv < lv else 0.0
    return None


def _remaining_from_row(row: pd.Series) -> float | None:
    r = row.get("remaining")
    if r is not None and pd.notna(r):
        try:
            return float(r)
        except Exception:
            pass
    # Fallback: horizon - elapsed
    h = row.get("horizon")
    e = row.get("elapsed")
    try:
        if h is not None and e is not None and pd.notna(h) and pd.notna(e):
            return float(h) - float(e)
    except Exception:
        return None
    return None


def _bucket_remaining(remaining: float | None) -> str | None:
    if remaining is None:
        return None
    try:
        r = float(remaining)
    except Exception:
        return None
    if not math.isfinite(r):
        return None
    if r <= 5:
        return "<=5"
    if r <= 10:
        return "5-10"
    if r <= 20:
        return "10-20"
    return ">20"


def _classify_bet_watch(horizon: float, elapsed: float, strength: float) -> tuple[bool, bool]:
    """Replicate the UI's BET/WATCH classification from strength.

    Notes:
      - UI uses horizon<=20.5 for 1H thresholds, but bucket report is full-game only by default.
      - For full-game: minElapsed=6, thr=7, thrWatch=max(4, thr-2)=5.
    """

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


def compute_live_lens_bucket_report(cfg: LiveLensBucketReportConfig) -> dict[str, Any]:
    dates = [_safe_date(d) for d in (cfg.dates or [])]
    if not dates:
        raise ValueError("No dates provided")

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

        # Normalize ids
        if "game_id" not in sig_df.columns and "event_id" in sig_df.columns:
            sig_df["game_id"] = sig_df["event_id"]
        if "game_id" not in sig_df.columns:
            missing.append({
                "date": d,
                "status": "bad_signals",
                "message": "missing game_id/event_id",
                "signals_path": str(sig_p),
                "results_path": str(res_p),
                "signals_cols": list(sig_df.columns),
            })
            continue
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
            missing.append({
                "date": d,
                "status": "bad_signals",
                "message": "missing side",
                "signals_path": str(sig_p),
                "results_path": str(res_p),
                "signals_cols": list(sig_df.columns),
            })
            continue
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

        if not cfg.include_watch:
            sig_df = sig_df[sig_df["is_bet"]].copy()
        else:
            sig_df = sig_df[sig_df["is_bet"] | sig_df["is_watch"]].copy()

        if sig_df.empty:
            missing.append({"date": d, "status": "empty_filtered", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        if not res_p.exists():
            missing.append({"date": d, "status": "missing_results", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue
        res_df = pd.read_csv(res_p)
        if res_df.empty:
            missing.append({"date": d, "status": "empty_results", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        gid_col = _pick_col(res_df, ["game_id", "event_id", "id", "gid"])
        if not gid_col:
            missing.append({
                "date": d,
                "status": "bad_results",
                "message": "missing game_id/event_id",
                "results_path": str(res_p),
                "results_cols": list(res_df.columns),
            })
            continue

        res_df["game_id"] = res_df[gid_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

        # If this is a same-day / in-progress results file, avoid settling against partial scores.
        res_df = _filter_results_to_finals(res_df)
        if res_df.empty:
            missing.append({"date": d, "status": "no_finals", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        res_df["final_total"] = _final_total_from_results(res_df)

        merged = sig_df.merge(res_df[["game_id", "final_total"]], on="game_id", how="left")
        merged["date"] = d

        merged["remaining"] = merged.apply(_remaining_from_row, axis=1)
        merged["remaining_bucket"] = merged["remaining"].map(_bucket_remaining)

        # Counterfactual retune (optional): apply late-game OVER penalty to strength, then
        # recompute is_bet/is_watch and filter based on those instead of logged flags.
        merged["cf_strength"] = pd.to_numeric(merged.get("strength"), errors="coerce") if "strength" in merged.columns else math.nan
        merged["cf_is_bet"] = merged.get("is_bet", False)
        merged["cf_is_watch"] = merged.get("is_watch", False)

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

            # Compute remaining, elapsed, horizon safely.
            hz = pd.to_numeric(merged.get("horizon"), errors="coerce") if "horizon" in merged.columns else pd.Series([40.0] * len(merged))
            el = pd.to_numeric(merged.get("elapsed"), errors="coerce") if "elapsed" in merged.columns else pd.Series([math.nan] * len(merged))
            rem = pd.to_numeric(merged.get("remaining"), errors="coerce")
            if el.isna().any():
                # Fallback if elapsed missing: elapsed = horizon - remaining
                try:
                    el = el.fillna(hz - rem)
                except Exception:
                    pass

            # Apply penalty to OVER in window (lo, hi]
            side = merged["side"].astype(str).str.strip().str.lower()
            in_window = (rem > lo_r) & (rem <= hi_r)
            is_over = side.eq("over")

            period_ok = pd.Series([True] * len(merged))
            if period_min > 1:
                if "period" in merged.columns:
                    per = pd.to_numeric(merged.get("period"), errors="coerce")
                    period_ok = per.notna() & (per >= period_min)
                else:
                    # If period gate is active but we don't have a period field, infer 2H from elapsed.
                    # For full-game horizon, elapsed>=20 implies period>=2.
                    try:
                        hz0 = hz
                        el0 = el
                        period_ok = (hz0 >= 39) & el0.notna() & (el0 >= 20)
                    except Exception:
                        period_ok = pd.Series([False] * len(merged))
            margin_ok = pd.Series([True] * len(merged))
            if margin_abs_min > 0:
                if "margin_home" in merged.columns:
                    m = pd.to_numeric(merged.get("margin_home"), errors="coerce")
                    margin_ok = m.abs() >= margin_abs_min
                else:
                    margin_ok = pd.Series([False] * len(merged))

            can_adjust = pen > 0
            adj_mask = can_adjust & is_over & in_window & period_ok & margin_ok
            try:
                merged.loc[adj_mask, "cf_strength"] = merged.loc[adj_mask, "cf_strength"] - float(pen)
            except Exception:
                pass

            # Early-game OVER penalty (full-game only). Apply when remaining >= threshold and (period<=max OR elapsed<20).
            if early_pen > 0:
                try:
                    in_early = rem >= early_min_rem
                    period_ok_e = pd.Series([True] * len(merged))
                    if "period" in merged.columns:
                        per = pd.to_numeric(merged.get("period"), errors="coerce")
                        period_ok_e = per.notna() & (per <= early_period_max)
                    else:
                        period_ok_e = (hz >= 39) & el.notna() & (el < 20)

                    early_mask = is_over & in_early & period_ok_e
                    merged.loc[early_mask, "cf_strength"] = merged.loc[early_mask, "cf_strength"] - float(early_pen)
                except Exception:
                    pass

            # Reclassify bet/watch from cf_strength when we have enough fields.
            cf_bet: list[bool] = []
            cf_watch: list[bool] = []
            for h0, e0, s0 in zip(hz, el, merged["cf_strength"]):
                try:
                    if pd.isna(h0) or pd.isna(e0) or pd.isna(s0):
                        cf_bet.append(False)
                        cf_watch.append(False)
                        continue
                    b, w = _classify_bet_watch(float(h0), float(e0), float(s0))
                    cf_bet.append(b)
                    cf_watch.append(w)
                except Exception:
                    cf_bet.append(False)
                    cf_watch.append(False)
            merged["cf_is_bet"] = cf_bet
            merged["cf_is_watch"] = cf_watch

        merged["result"] = [
            _settle(s, y, l)
            for s, y, l in zip(
                merged["side"],
                pd.to_numeric(merged["final_total"], errors="coerce"),
                pd.to_numeric(merged["live_line"], errors="coerce"),
            )
        ]
        merged = merged[merged["result"].notna()].copy()
        if merged.empty:
            missing.append({"date": d, "status": "no_settled", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue

        merged["profit_units"] = merged["result"].map(lambda r: _profit_units(float(r), cfg.assume_price))

        # Select which flags drive inclusion.
        bet_col = "cf_is_bet" if cfg.apply_retune else "is_bet"
        watch_col = "cf_is_watch" if cfg.apply_retune else "is_watch"
        if not cfg.include_watch:
            merged = merged[merged[bet_col].astype(bool)].copy()
        else:
            merged = merged[merged[bet_col].astype(bool) | merged[watch_col].astype(bool)].copy()

        merged["signal_type"] = merged.apply(lambda r: "bet" if bool(r.get(bet_col)) else "watch", axis=1)
        merged["policy"] = "retuned" if cfg.apply_retune else "logged"

        all_rows.append(merged)

    if not all_rows:
        return {
            "status": "missing",
            "dates": dates,
            "message": "No settled signals found for any provided date",
            "missing": missing,
        }

    settled = pd.concat(all_rows, ignore_index=True)

    def _agg(g: pd.DataFrame) -> dict[str, Any]:
        wins = int((g["result"] == 1.0).sum())
        losses = int((g["result"] == 0.0).sum())
        pushes = int((g["result"] == 0.5).sum())
        denom = wins + losses
        return {
            "n": int(len(g)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": (wins / denom) if denom > 0 else None,
            "roi_units_per_bet": float(g["profit_units"].sum() / max(1, len(g))),
        }

    # Overall by bucket x type
    bucket_rows: list[dict[str, Any]] = []
    g0 = settled.dropna(subset=["remaining_bucket"]).copy()

    for (bucket, sig_type), g in g0.groupby(["remaining_bucket", "signal_type"]):
        rec = {"remaining_bucket": str(bucket), "signal_type": str(sig_type)}
        rec.update(_agg(g))
        # Over share in the bucket (for quick diagnosis)
        try:
            rec["over_share"] = float((g["side"] == "over").mean())
        except Exception:
            rec["over_share"] = None
        bucket_rows.append(rec)

    # By bucket x type x side
    side_rows: list[dict[str, Any]] = []
    for (bucket, sig_type, side), g in g0.groupby(["remaining_bucket", "signal_type", "side"]):
        rec = {"remaining_bucket": str(bucket), "signal_type": str(sig_type), "side": str(side)}
        rec.update(_agg(g))
        side_rows.append(rec)

    bucket_df = pd.DataFrame(bucket_rows).sort_values(["signal_type", "remaining_bucket"], kind="stable")
    side_df = pd.DataFrame(side_rows).sort_values(["signal_type", "remaining_bucket", "side"], kind="stable")

    return {
        "status": "ok",
        "dates": dates,
        "n_settled": int(len(settled)),
        "missing": missing,
        "bucket_table": bucket_df,
        "bucket_side_table": side_df,
        "rows": settled,
    }


def iter_date_range(start_date: str, end_date: str) -> list[str]:
    s = dt.date.fromisoformat(str(start_date).strip())
    e = dt.date.fromisoformat(str(end_date).strip())
    if e < s:
        s, e = e, s
    days = (e - s).days
    return [(s + dt.timedelta(days=i)).isoformat() for i in range(days + 1)]
