from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .live_lens_buckets import LiveLensBucketReportConfig, compute_live_lens_bucket_report


@dataclass(frozen=True)
class LateOverSearchConfig:
    dates: list[str]
    assume_price: float = -110.0
    remaining_lo: float = 5.0
    remaining_hi: float = 10.0
    period_min: float = 2.0
    penalties: list[float] = None  # type: ignore[assignment]
    margin_abs_mins: list[float] = None  # type: ignore[assignment]


def _extract_bucket_roi(bucket_df: pd.DataFrame, bucket: str, signal_type: str = "bet") -> float | None:
    try:
        df = bucket_df
        m = (df["remaining_bucket"].astype(str) == str(bucket)) & (df["signal_type"].astype(str) == str(signal_type))
        r = df[m]
        if r.empty:
            return None
        v = r.iloc[0].get("roi_units_per_bet")
        return float(v) if v is not None else None
    except Exception:
        return None


def _extract_bucket_n(bucket_df: pd.DataFrame, bucket: str, signal_type: str = "bet") -> int:
    try:
        df = bucket_df
        m = (df["remaining_bucket"].astype(str) == str(bucket)) & (df["signal_type"].astype(str) == str(signal_type))
        r = df[m]
        if r.empty:
            return 0
        return int(r.iloc[0].get("n") or 0)
    except Exception:
        return 0


def search_late_over_retune(
    base_cfg: LiveLensBucketReportConfig,
    search_cfg: LateOverSearchConfig,
) -> dict[str, Any]:
    penalties = list(search_cfg.penalties) if search_cfg.penalties is not None else [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    margins = list(search_cfg.margin_abs_mins) if search_cfg.margin_abs_mins is not None else [0.0, 5.0, 8.0, 10.0]

    rows: list[dict[str, Any]] = []

    for pen in penalties:
        for mabs in margins:
            cfg = LiveLensBucketReportConfig(
                dates=list(search_cfg.dates),
                out_dir=base_cfg.out_dir,
                daily_results_dir=base_cfg.daily_results_dir,
                assume_price=float(search_cfg.assume_price),
                include_watch=False,  # search on bet-only
                full_game_only=True,
                apply_retune=True,
                late_over_strength_penalty=float(pen),
                late_over_remaining_lo=float(search_cfg.remaining_lo),
                late_over_remaining_hi=float(search_cfg.remaining_hi),
                late_over_margin_abs_min=float(mabs),
                late_over_period_min=float(search_cfg.period_min),
            )
            payload = compute_live_lens_bucket_report(cfg)
            if payload.get("status") != "ok":
                rows.append({"penalty": pen, "margin_abs_min": mabs, "status": payload.get("status"), "message": payload.get("message")})
                continue

            bucket_df: pd.DataFrame = payload["bucket_table"]
            roi_5_10 = _extract_bucket_roi(bucket_df, "5-10", "bet")
            n_5_10 = _extract_bucket_n(bucket_df, "5-10", "bet")

            # Overall ROI across all settled bet rows
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
                    "margin_abs_min": float(mabs),
                    "period_min": float(search_cfg.period_min),
                    "roi_5_10": roi_5_10,
                    "n_5_10": int(n_5_10),
                    "overall_roi": overall_roi,
                    "overall_n": int(overall_n),
                    "status": "ok",
                }
            )

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy() if ("status" in df.columns) else df.copy()

    # Sort primarily by roi_5_10, then by overall_roi, then by n_5_10 (prefer more sample)
    if not ok.empty:
        ok["roi_5_10_sort"] = pd.to_numeric(ok["roi_5_10"], errors="coerce")
        ok["overall_roi_sort"] = pd.to_numeric(ok["overall_roi"], errors="coerce")
        ok = ok.sort_values(["roi_5_10_sort", "overall_roi_sort", "n_5_10"], ascending=[False, False, False], kind="stable")

    return {
        "status": "ok",
        "dates": list(search_cfg.dates),
        "remaining_window": [float(search_cfg.remaining_lo), float(search_cfg.remaining_hi)],
        "period_min": float(search_cfg.period_min),
        "table": ok,
        "raw": df,
    }
