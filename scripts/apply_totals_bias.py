import argparse
import json
import os
from pathlib import Path
import pandas as pd

"""
Apply a simple totals bias correction by comparing model totals vs market totals
on recent history, then adjust today's enriched predictions accordingly.

Outputs:
- Updates today's `outputs/predictions_unified_enriched_<date>.csv` with `totals_bias`
  and `total_adj` (model_total + totals_bias).
- Writes `outputs/bias/totals_bias_summary.json` with window stats.

This is intentionally minimal and robust to missing columns.
"""

DEF_ENRICH_DIR = Path("outputs")
DEF_BIAS_DIR = Path("outputs/bias")
DEF_DRIFT_DAILY = Path("outputs/drift/drift_market_daily.csv")


def compute_bias_from_history(window_days: int) -> float:
    # Try to use drift daily CSV if present; otherwise fall back to predictions history if available.
    if DEF_DRIFT_DAILY.exists():
        try:
            df = pd.read_csv(DEF_DRIFT_DAILY)
            # Expect columns like: date, market_total, model_total, total_delta
            # If not present, attempt to infer.
            col_market = None
            col_model = None
            for c in df.columns:
                lc = c.lower()
                if col_market is None and ("market_total" in lc or "closing_total" in lc or "last_total" in lc or "today_total" in lc):
                    col_market = c
                if col_model is None and ("model_total" in lc or "pred_total" in lc or "our_total" in lc):
                    col_model = c
            if col_market and col_model:
                df = df.dropna(subset=[col_market, col_model])
                df["bias"] = df[col_market] - df[col_model]
                # Use robust center (median) to avoid outlier influence
                return float(df["bias"].median())
        except Exception:
            pass
    # Fallback: zero bias
    return 0.0


def apply_bias_to_today(date_str: str, totals_bias: float) -> str:
    # Find today's enriched predictions file
    enriched_path = DEF_ENRICH_DIR / f"predictions_unified_enriched_{date_str}.csv"
    if not enriched_path.exists():
        raise FileNotFoundError(f"Missing enriched predictions for {date_str}: {enriched_path}")
    df = pd.read_csv(enriched_path)

    # Identify model totals column (robust matching)
    col_model = None
    for c in df.columns:
        lc = c.lower()
        if col_model is None and ("model_total" in lc or "pred_total" in lc or "total_pred" in lc or lc == "total"):
            col_model = c
    if col_model is None:
        # No totals column; write summary and return path unchanged
        return str(enriched_path)

    # Apply bias
    df["totals_bias"] = totals_bias
    df["total_adj"] = df[col_model] + totals_bias

    # Persist back
    df.to_csv(enriched_path, index=False)
    return str(enriched_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD to update")
    parser.add_argument("--window-days", type=int, default=60, help="Window for bias estimation")
    parser.add_argument("--median-threshold", type=float, default=3.0, help="Only apply bias when median |delta_total| exceeds this threshold")
    args = parser.parse_args()

    DEF_BIAS_DIR.mkdir(parents=True, exist_ok=True)

    # Check drift union summary for median gating
    drift_summary_path = Path("outputs/drift/drift_union_summary.json")
    median_total = None
    if drift_summary_path.exists():
        try:
            with open(drift_summary_path, "r") as f:
                ds = json.load(f)
            counts = ds.get("counts", {})
            total = counts.get("total", {})
            median_total = float(total.get("median")) if total.get("median") is not None else None
        except Exception:
            median_total = None

    bias = compute_bias_from_history(args.window_days)

    # Gate application by median threshold if available
    if median_total is not None and abs(median_total) <= args.median_threshold:
        summary = {
            "date": args.date,
            "window_days": args.window_days,
            "totals_bias": bias,
            "median_total": median_total,
            "threshold": args.median_threshold,
            "updated_file": None,
        }
        with open(DEF_BIAS_DIR / "totals_bias_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({"status": "skipped", **summary}))
        return

    out_path = apply_bias_to_today(args.date, bias)

    summary = {
        "date": args.date,
        "window_days": args.window_days,
        "totals_bias": bias,
        "median_total": median_total,
        "threshold": args.median_threshold,
        "updated_file": out_path,
    }
    with open(DEF_BIAS_DIR / "totals_bias_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"status": "updated", **summary}))


if __name__ == "__main__":
    main()
