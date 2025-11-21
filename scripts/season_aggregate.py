#!/usr/bin/env python
"""
Season-wide aggregation of model evaluation artifacts for the current season.

Scans outputs/ for per-date JSON/CSV artifacts and compiles a unified daily metrics
list plus summary statistics. Focused on the in-progress 2025-26 season (starting Nov 2025).

Artifacts consumed (if present per date):
  residuals_<date>.json
  predictability_<date>.json
  fairness_<date>.json
  recalibration_<date>.json
  leakage_<date>.json
  backtest_metrics_<date>.json
  scoring_<date>.json

Outputs:
  outputs/season_metrics.json  (structured summary + daily array)
  outputs/season_metrics.csv   (flat daily row metrics)

Run:
  python scripts/season_aggregate.py --season-start 2025-11-01 --season-end 2025-11-30
  (season-end optional; defaults to today)
"""
from __future__ import annotations
import argparse
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List

import pandas as pd

OUT = Path("outputs")
OUT_DAILY = OUT / "daily_results"

ARTIFACTS = {
    "residuals": "residuals_{d}.json",
    "predictability": "predictability_{d}.json",
    "fairness": "fairness_{d}.json",
    "recalibration": "recalibration_{d}.json",
    "leakage": "leakage_{d}.json",
    "backtest": "backtest_metrics_{d}.json",
    "scoring": "scoring_{d}.json",
}

# Key fields to surface from each artifact (if present)
FIELD_WHITELIST = {
    "residuals": ["totals_residual_mean","totals_residual_std","margin_residual_mean","margin_residual_std","rows","total_corr","margin_corr"],
    "predictability": ["residual_mean","residual_std","residual_mae","predictability_score","calibration_slope_total","calibration_intercept_total","trailing_residual_std"],
    "fairness": ["global_mean_residual_total","global_mean_residual_margin","global_bias_flag","global_disparity_flag","conferences_evaluated"],
    "recalibration": ["recalibration_needed","reasons","residual_mean","corr_pred_market_total","corr_pred_market_margin","crps_degradation"],
    "leakage": ["suspicious_columns_count"],
    "backtest": ["totals_edge_mean","totals_edge_std","spread_edge_mean","spread_edge_std","moneyline_edge_mean","moneyline_edge_std"],
    "scoring": ["crps_total_mean","ll_total_mean","rows"],
}

# Composite daily metrics we derive even if some artifacts missing
DERIVED_DEFAULTS = {
    "date": None,
    "residual_std": None,
    "predictability_score": None,
    "fairness_bias_flag": None,
    "fairness_disparity_flag": None,
    "recalibration_needed": None,
    "leakage_suspicious_cols": None,
    "crps_mean": None,
}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_dates(start: date | None, end: date | None) -> List[str]:
    """Discover available results_<date>.csv dates within [start, end] under outputs/daily_results."""
    base = OUT_DAILY if OUT_DAILY.exists() else OUT
    if not base.exists():
        return []
    dates: List[str] = []
    for p in base.glob("results_*.csv"):
        name = p.name  # results_YYYY-MM-DD.csv
        try:
            ds = name.replace("results_", "").replace(".csv", "")
            dt_obj = date.fromisoformat(ds)
            if start and dt_obj < start:
                continue
            if end and dt_obj > end:
                continue
            dates.append(ds)
        except Exception:
            continue
    dates = sorted(set(dates))
    return dates


def collect_for_date(d: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: v for k, v in DERIVED_DEFAULTS.items()}
    row["date"] = d
    for key, pattern in ARTIFACTS.items():
        fname = pattern.format(d=d)
        # Most artifacts written to outputs/ root; adapt if found under daily_results as a fallback.
        path = OUT / fname
        if not path.exists():
            alt = OUT_DAILY / fname
            if alt.exists():
                path = alt
        if not path.exists():
            continue
        payload = _load_json(path)
        if not payload:
            continue
        # When artifacts store metrics inside nested keys (e.g. fairness global, recalibration metrics)
        surf = payload
        if key == "fairness":
            # Flatten global & summary counts
            g = payload.get("global", {}) if isinstance(payload.get("global"), dict) else {}
            records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
            surf = {
                "global_mean_residual_total": g.get("mean_residual_total"),
                "global_mean_residual_margin": g.get("mean_residual_margin"),
                "global_bias_flag": g.get("bias_flag"),
                "global_disparity_flag": g.get("disparity_flag"),
                "conferences_evaluated": len(records),
            }
        elif key == "recalibration":
            m = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
            surf = {
                "recalibration_needed": payload.get("recalibration_needed"),
                "reasons": payload.get("reasons"),
                "residual_mean": m.get("residual_mean"),
                "corr_pred_market_total": m.get("corr_pred_market_total"),
                "corr_pred_market_margin": m.get("corr_pred_market_margin"),
                "crps_degradation": m.get("crps_degradation"),
            }
        elif key == "leakage":
            surf = {
                "suspicious_columns_count": len(payload.get("suspicious_columns", [])),
            }
        # Extract whitelist fields
        for f in FIELD_WHITELIST.get(key, []):
            if f in surf:
                row[f] = surf[f]
        # Flatten backtest nested sections for ROI/PNL/n_bets
        if key == "backtest":
            for sec, prefix in (
                (payload.get("totals_closing", {}), "backtest_totals_"),
                (payload.get("spread_closing", {}), "backtest_spread_"),
                (payload.get("moneyline_closing", {}), "backtest_moneyline_"),
            ):
                if isinstance(sec, dict):
                    for k in ["roi","pnl_units","n_bets","win_rate","avg_edge"]:
                        if k in sec:
                            row[prefix + k] = sec.get(k)
        # Map a few fields to canonical names
        if key == "predictability":
            if "residual_std" in row:
                pass  # already captured
        if key == "leakage":
            row["leakage_suspicious_cols"] = surf.get("suspicious_columns_count")
    # Derived mapping / canonical consolidation
    row["fairness_bias_flag"] = row.get("global_bias_flag")
    row["fairness_disparity_flag"] = row.get("global_disparity_flag")
    return row


def summarize(daily: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Collect numeric series for aggregation
    def _num(key: str) -> List[float]:
        vals = [v.get(key) for v in daily if isinstance(v.get(key), (int, float))]
        return [float(x) for x in vals]

    summary: Dict[str, Any] = {
        "days": len(daily),
        "start_date": daily[0]["date"] if daily else None,
        "end_date": daily[-1]["date"] if daily else None,
    }
    quant_fields = [
        "residual_std","predictability_score","crps_total_mean","totals_residual_std","margin_residual_std","total_corr","margin_corr",
        "backtest_totals_roi","backtest_spread_roi","backtest_moneyline_roi",
    ]
    for f in quant_fields:
        series = _num(f)
        if series:
            summary[f + "_mean"] = statistics.fmean(series)
            summary[f + "_median"] = statistics.median(series)
            summary[f + "_min"] = min(series)
            summary[f + "_max"] = max(series)
    # Counts / flags
    summary["recalibration_days"] = sum(1 for r in daily if r.get("recalibration_needed") is True)
    summary["fairness_bias_days"] = sum(1 for r in daily if r.get("fairness_bias_flag") is True)
    summary["fairness_disparity_days"] = sum(1 for r in daily if r.get("fairness_disparity_flag") is True)
    summary["leakage_days_nonzero"] = sum(1 for r in daily if isinstance(r.get("leakage_suspicious_cols"), int) and r.get("leakage_suspicious_cols") > 0)
    return summary


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-start", dest="season_start", help="ISO date season start (default auto)" )
    ap.add_argument("--season-end", dest="season_end", help="ISO date season end (default today)")
    args = ap.parse_args(argv)

    today = date.today()
    start = None
    end = today
    if args.season_start:
        try:
            start = date.fromisoformat(args.season_start)
        except Exception:
            print(f"Invalid --season-start {args.season_start}", file=sys.stderr)
            return 2
    if args.season_end:
        try:
            end = date.fromisoformat(args.season_end)
        except Exception:
            print(f"Invalid --season-end {args.season_end}", file=sys.stderr)
            return 2

    # Auto default for start: earliest results_* date in outputs after Aug 1 of current year
    if start is None:
        candidate_dates = discover_dates(None, end)
        season_anchor = date(today.year, 8, 1)
        season_candidates = [d for d in candidate_dates if date.fromisoformat(d) >= season_anchor]
        start = date.fromisoformat(season_candidates[0]) if season_candidates else None

    dates = discover_dates(start, end)
    if not dates:
        print("No dates discovered for season window", file=sys.stderr)
        return 1
    daily_rows: List[Dict[str, Any]] = []
    for d in dates:
        daily_rows.append(collect_for_date(d))

    summary = summarize(daily_rows)
    payload = {
        "summary": summary,
        "daily": daily_rows,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "season_start": start.isoformat() if start else None,
        "season_end": end.isoformat() if end else None,
    }

    OUT.mkdir(exist_ok=True)
    json_path = OUT / "season_metrics.json"
    csv_path = OUT / "season_metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Flatten to CSV
    df = pd.DataFrame(daily_rows)
    df.to_csv(csv_path, index=False)

    print(f"Season metrics written: {json_path} ({len(daily_rows)} days) and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
