from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .live_lens_buckets import LiveLensBucketReportConfig, compute_live_lens_bucket_report


@dataclass(frozen=True)
class EarlyOverSearchConfig:
    dates: list[str]
    assume_price: float = -110.0
    remaining_min: float = 20.0
    period_max: float = 1.0
    penalties: list[float] = None  # type: ignore[assignment]


def _extract_bucket_roi(bucket_df: pd.DataFrame, bucket: str, signal_type: str = "bet") -> float | None:
    try:
        m = (bucket_df["remaining_bucket"].astype(str) == str(bucket)) & (bucket_df["signal_type"].astype(str) == str(signal_type))
        r = bucket_df[m]
        if r.empty:
            return None
        v = r.iloc[0].get("roi_units_per_bet")
        return float(v) if v is not None else None
    except Exception:
        return None


def _extract_bucket_n(bucket_df: pd.DataFrame, bucket: str, signal_type: str = "bet") -> int:
    try:
        m = (bucket_df["remaining_bucket"].astype(str) == str(bucket)) & (bucket_df["signal_type"].astype(str) == str(signal_type))
        r = bucket_df[m]
        if r.empty:
            return 0
        return int(r.iloc[0].get("n") or 0)
    except Exception:
        return 0


def search_early_over_retune(base_cfg: LiveLensBucketReportConfig, search_cfg: EarlyOverSearchConfig) -> dict[str, Any]:
    penalties = list(search_cfg.penalties) if search_cfg.penalties is not None else [0.0, 0.5, 1.0, 1.5, 2.0]

    rows: list[dict[str, Any]] = []
    for pen in penalties:
        cfg = LiveLensBucketReportConfig(
            dates=list(search_cfg.dates),
            out_dir=base_cfg.out_dir,
            daily_results_dir=base_cfg.daily_results_dir,
            assume_price=float(search_cfg.assume_price),
            include_watch=False,
            full_game_only=True,
            apply_retune=True,
            late_over_strength_penalty=float(base_cfg.late_over_strength_penalty or 0.0),
            late_over_remaining_lo=float(base_cfg.late_over_remaining_lo),
            late_over_remaining_hi=float(base_cfg.late_over_remaining_hi),
            late_over_margin_abs_min=float(base_cfg.late_over_margin_abs_min),
            late_over_period_min=float(base_cfg.late_over_period_min),
            early_over_strength_penalty=float(pen),
            early_over_remaining_min=float(search_cfg.remaining_min),
            early_over_period_max=float(search_cfg.period_max),
        )
        payload = compute_live_lens_bucket_report(cfg)
        if payload.get("status") != "ok":
            rows.append({"penalty": pen, "status": payload.get("status"), "message": payload.get("message")})
            continue

        bucket_df: pd.DataFrame = payload["bucket_table"]
        roi_gt20 = _extract_bucket_roi(bucket_df, ">20", "bet")
        n_gt20 = _extract_bucket_n(bucket_df, ">20", "bet")

        try:
            settled: pd.DataFrame = payload["rows"]
            overall_roi = float(settled["profit_units"].sum() / max(1, len(settled)))
            overall_n = int(len(settled))
        except Exception:
            overall_roi = None
            overall_n = 0

        rows.append(
            {
                "penalty": float(pen),
                "remaining_min": float(search_cfg.remaining_min),
                "period_max": float(search_cfg.period_max),
                "roi_gt20": roi_gt20,
                "n_gt20": int(n_gt20),
                "overall_roi": overall_roi,
                "overall_n": int(overall_n),
                "status": "ok",
            }
        )

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy() if ("status" in df.columns) else df.copy()
    if not ok.empty:
        ok["roi_gt20_sort"] = pd.to_numeric(ok["roi_gt20"], errors="coerce")
        ok["overall_roi_sort"] = pd.to_numeric(ok["overall_roi"], errors="coerce")
        ok = ok.sort_values(["roi_gt20_sort", "overall_roi_sort", "n_gt20"], ascending=[False, False, False], kind="stable")

    return {
        "status": "ok",
        "dates": list(search_cfg.dates),
        "early": {"remaining_min": float(search_cfg.remaining_min), "period_max": float(search_cfg.period_max)},
        "table": ok,
        "raw": df,
    }
