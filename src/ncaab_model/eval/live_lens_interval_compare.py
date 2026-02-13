from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..live_lens_tuning import DEFAULT_TUNING
from .live_lens_interval_backtest import summarize_interval_backtest


@dataclass(frozen=True)
class IntervalCompareConfig:
    horizon_min: float = 40.0
    edge_eps: float = 0.01

    # Match the UI's full-game defaults in templates/index.html.
    min_elapsed_full: float = 6.0
    thr_full: float = 7.0
    thr_watch_full: float = 5.0

    # 1H settings (not used by default in these backtests, but kept for completeness).
    min_elapsed_1h: float = 4.0
    thr_1h: float = 5.0
    thr_watch_1h: float = 4.0

    pbp_n_scale: float = 70.0
    pace_hi: float = 3.25
    pace_lo: float = 2.75
    pps_hi: float = 1.18
    pps_lo: float = 0.95


def _coerce_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _compute_strength_and_action(df: pd.DataFrame, cfg: IntervalCompareConfig) -> pd.DataFrame:
    out = df.copy()

    out["elapsed_min"] = _coerce_float_series(out.get("elapsed_min"))
    out["elapsed_sec"] = pd.to_numeric(out.get("elapsed_sec"), errors="coerce").astype("Int64")
    out["line_total"] = _coerce_float_series(out.get("line_total"))
    out["proj_blend"] = _coerce_float_series(out.get("proj_blend"))

    edge = out["proj_blend"] - out["line_total"]
    out["edge"] = edge

    side = np.where(edge > float(cfg.edge_eps), "over", np.where(edge < -float(cfg.edge_eps), "under", None))
    out["side"] = side

    # Approximate the UI's PBP adjustment using only what we have in the cached PBP-derived rows:
    # - shot_proxy ~= possessions proxy
    # - poss_rate + ppp thresholds from tuning
    # This keeps comparisons reasonable without needing full fg%/3p%/ft% context.
    poss_rate = _coerce_float_series(out.get("poss_rate"))
    ppp = _coerce_float_series(out.get("ppp"))
    shot_proxy = _coerce_float_series(out.get("shot_proxy"))

    pbp_n = shot_proxy
    w_pbp = (pbp_n / float(cfg.pbp_n_scale)).clip(lower=0.0, upper=1.0)

    pace_adj = np.zeros(len(out), dtype=float)
    eff_adj = np.zeros(len(out), dtype=float)

    over_mask = out["side"] == "over"
    under_mask = out["side"] == "under"

    pr = poss_rate.to_numpy(dtype=float, na_value=np.nan)
    pp = ppp.to_numpy(dtype=float, na_value=np.nan)

    # Pace proxy adjustment
    pace_adj += np.where(over_mask & (pr >= float(cfg.pace_hi)), 1.0, 0.0)
    pace_adj += np.where(over_mask & (pr <= float(cfg.pace_lo)), -1.0, 0.0)
    pace_adj += np.where(under_mask & (pr <= float(cfg.pace_lo)), 1.0, 0.0)
    pace_adj += np.where(under_mask & (pr >= float(cfg.pace_hi)), -1.0, 0.0)

    # Efficiency proxy adjustment
    eff_adj += np.where(over_mask & (pp <= float(cfg.pps_lo)), 1.0, 0.0)
    eff_adj += np.where(over_mask & (pp >= float(cfg.pps_hi)), -1.0, 0.0)
    eff_adj += np.where(under_mask & (pp >= float(cfg.pps_hi)), 1.0, 0.0)
    eff_adj += np.where(under_mask & (pp <= float(cfg.pps_lo)), -1.0, 0.0)

    adj = w_pbp.to_numpy(dtype=float, na_value=0.0) * (pace_adj + eff_adj)

    strength = np.abs(edge.to_numpy(dtype=float, na_value=np.nan)) + adj
    out["strength"] = strength

    # Horizon-based thresholds (full game vs 1H)
    # Our interval backtests are full-game; keep logic horizon-aware anyway.
    horizon = float(cfg.horizon_min)
    is_1h = horizon <= 20.5
    min_elapsed = float(cfg.min_elapsed_1h if is_1h else cfg.min_elapsed_full)
    thr = float(cfg.thr_1h if is_1h else cfg.thr_full)
    thr_watch = float(cfg.thr_watch_1h if is_1h else cfg.thr_watch_full)

    elapsed_ok = out["elapsed_min"].fillna(-1.0) >= min_elapsed
    has_side = out["side"].notna()

    is_bet = elapsed_ok & has_side & (out["strength"] >= thr)
    is_watch = (~is_bet) & elapsed_ok & has_side & (out["strength"] >= thr_watch)

    out["action"] = np.where(is_bet, "bet", np.where(is_watch, "watch", "none"))

    return out


def _first_action_times(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["event_id", "first_watch_sec", "first_bet_sec", "first_side", "first_action"])  # type: ignore[call-arg]

    d = df.copy()
    d["elapsed_sec"] = pd.to_numeric(d.get("elapsed_sec"), errors="coerce")
    d = d.dropna(subset=["event_id", "elapsed_sec"])
    d["elapsed_sec"] = d["elapsed_sec"].astype(int)

    # First BET
    bet = d[d["action"] == "bet"].sort_values(["event_id", "elapsed_sec"])
    first_bet = bet.groupby("event_id", as_index=False).first()[["event_id", "elapsed_sec", "side"]] if not bet.empty else pd.DataFrame(columns=["event_id", "elapsed_sec", "side"])
    first_bet = first_bet.rename(columns={"elapsed_sec": "first_bet_sec", "side": "first_bet_side"})

    # First WATCH-or-BET (aka first signal)
    sig = d[d["action"].isin(["watch", "bet"])].sort_values(["event_id", "elapsed_sec"])
    first_sig = sig.groupby("event_id", as_index=False).first()[["event_id", "elapsed_sec", "side", "action"]] if not sig.empty else pd.DataFrame(columns=["event_id", "elapsed_sec", "side", "action"])
    first_sig = first_sig.rename(columns={"elapsed_sec": "first_watch_sec", "side": "first_side", "action": "first_action"})

    out = first_sig.merge(first_bet, on="event_id", how="outer")
    return out


def compare_interval_backtests(
    *,
    csv_a: Path,
    csv_b: Path,
    out_prefix: Path,
    label_a: str,
    label_b: str,
    config: Optional[IntervalCompareConfig] = None,
) -> dict[str, Any]:
    cfg = config or IntervalCompareConfig(
        horizon_min=40.0,
        pbp_n_scale=float(DEFAULT_TUNING.pbp_n_scale),
        pace_hi=float(DEFAULT_TUNING.pace_hi),
        pace_lo=float(DEFAULT_TUNING.pace_lo),
        pps_hi=float(DEFAULT_TUNING.pps_hi),
        pps_lo=float(DEFAULT_TUNING.pps_lo),
    )

    csv_a = Path(csv_a)
    csv_b = Path(csv_b)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    df_a2 = _compute_strength_and_action(df_a, cfg)
    df_b2 = _compute_strength_and_action(df_b, cfg)

    # Per-file summary blocks
    sum_a = summarize_interval_backtest(df_a)
    sum_b = summarize_interval_backtest(df_b)

    # Action distributions
    def _action_counts(dfx: pd.DataFrame) -> dict[str, int]:
        try:
            vc = dfx["action"].value_counts(dropna=False)
            return {str(k): int(v) for k, v in vc.to_dict().items()}
        except Exception:
            return {}

    def _side_counts_for_action(dfx: pd.DataFrame, action: str) -> dict[str, int]:
        try:
            sub = dfx[dfx["action"] == action]
            vc = sub["side"].value_counts(dropna=False)
            return {str(k): int(v) for k, v in vc.to_dict().items()}
        except Exception:
            return {}

    dist = {
        label_a: {
            "rows": int(len(df_a2)),
            "games": int(df_a2["event_id"].nunique()) if "event_id" in df_a2.columns else None,
            "action_counts": _action_counts(df_a2),
            "bet_side_counts": _side_counts_for_action(df_a2, "bet"),
            "watch_side_counts": _side_counts_for_action(df_a2, "watch"),
        },
        label_b: {
            "rows": int(len(df_b2)),
            "games": int(df_b2["event_id"].nunique()) if "event_id" in df_b2.columns else None,
            "action_counts": _action_counts(df_b2),
            "bet_side_counts": _side_counts_for_action(df_b2, "bet"),
            "watch_side_counts": _side_counts_for_action(df_b2, "watch"),
        },
    }

    # First-signal times per event
    fa = _first_action_times(df_a2)
    fb = _first_action_times(df_b2)

    per_event = fa.merge(fb, on="event_id", how="outer", suffixes=(f"_{label_a}", f"_{label_b}"))

    # Lead-time deltas (positive means A earlier than B)
    def _delta(a: Any, b: Any) -> Any:
        try:
            if pd.isna(a) or pd.isna(b):
                return None
            return int(b) - int(a)
        except Exception:
            return None

    per_event["watch_lead_sec_a_vs_b"] = per_event.apply(
        lambda r: _delta(r.get(f"first_watch_sec_{label_a}"), r.get(f"first_watch_sec_{label_b}")), axis=1
    )
    per_event["bet_lead_sec_a_vs_b"] = per_event.apply(
        lambda r: _delta(r.get(f"first_bet_sec_{label_a}"), r.get(f"first_bet_sec_{label_b}")), axis=1
    )

    # Summary: how often does one step trigger when the other doesn't?
    def _bool_present(x: Any) -> bool:
        return x is not None and (not pd.isna(x))

    wa = per_event.get(f"first_watch_sec_{label_a}")
    wb = per_event.get(f"first_watch_sec_{label_b}")
    ba = per_event.get(f"first_bet_sec_{label_a}")
    bb = per_event.get(f"first_bet_sec_{label_b}")

    watch_a_only = int((wa.apply(_bool_present) & ~wb.apply(_bool_present)).sum()) if wa is not None and wb is not None else 0
    watch_b_only = int((wb.apply(_bool_present) & ~wa.apply(_bool_present)).sum()) if wa is not None and wb is not None else 0
    bet_a_only = int((ba.apply(_bool_present) & ~bb.apply(_bool_present)).sum()) if ba is not None and bb is not None else 0
    bet_b_only = int((bb.apply(_bool_present) & ~ba.apply(_bool_present)).sum()) if ba is not None and bb is not None else 0

    # Lead-time distribution stats
    def _lead_stats(s: pd.Series) -> dict[str, Any]:
        try:
            v = pd.to_numeric(s, errors="coerce").dropna()
            if v.empty:
                return {"n": 0, "mean": None, "p50": None, "p10": None, "p90": None}
            return {
                "n": int(len(v)),
                "mean": float(v.mean()),
                "p10": float(v.quantile(0.10)),
                "p50": float(v.quantile(0.50)),
                "p90": float(v.quantile(0.90)),
            }
        except Exception:
            return {"n": 0, "mean": None, "p50": None, "p10": None, "p90": None}

    lead_summary = {
        "watch": {
            "a_only": watch_a_only,
            "b_only": watch_b_only,
            "lead_sec_a_vs_b": _lead_stats(per_event["watch_lead_sec_a_vs_b"]),
        },
        "bet": {
            "a_only": bet_a_only,
            "b_only": bet_b_only,
            "lead_sec_a_vs_b": _lead_stats(per_event["bet_lead_sec_a_vs_b"]),
        },
    }

    # Write per-event CSV
    out_csv = Path(str(out_prefix) + ".per_event.csv")
    per_event.to_csv(out_csv, index=False)

    payload = {
        "status": "ok",
        "csv_a": str(csv_a),
        "csv_b": str(csv_b),
        "label_a": label_a,
        "label_b": label_b,
        "summary": {
            label_a: sum_a,
            label_b: sum_b,
            "action_dist": dist,
            "lead": lead_summary,
        },
        "out_per_event_csv": str(out_csv),
    }

    out_json = Path(str(out_prefix) + ".summary.json")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
