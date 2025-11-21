#!/usr/bin/env python
"""Daily performance aggregator.
Combines existing artifacts into a single performance_<date>.json for easier ingestion and health gating.

Artifacts considered:
  residuals_<date>.json
  predictability_<date>.json
  fairness_<date>.json
  recalibration_<date>.json
  leakage_<date>.json
  backtest_metrics_<date>.json
  scoring_<date>.json
  team_variance_<date>.json

Computes composite health signals:
  model_health: one of ['healthy','degraded','critical'] based on rule set
Rules (initial heuristic):
  critical if recalibration_needed and (predictability_score < 0.35 or total_mean_z extreme or totals_corr_z_drop)
  degraded if fairness flags active OR leakage nonzero OR predictability_score < 0.55
  healthy otherwise.

Adds rolling trailing 7-day averages if prior daily performance files exist.

Usage:
  python scripts/daily_performance.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
from typing import Any, Dict
import statistics

OUT = Path("outputs")

ARTS = [
    "residuals_{d}.json",
    "predictability_{d}.json",
    "fairness_{d}.json",
    "recalibration_{d}.json",
    "leakage_{d}.json",
    "backtest_metrics_{d}.json",
    "scoring_{d}.json",
    "team_variance_{d}.json",
]


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _extract(date: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"date": date}
    for pat in ARTS:
        p = OUT / pat.format(d=date)
        payload = _load_json(p)
        if not payload:
            continue
        if pat.startswith("residuals"):
            row["total_mean"] = payload.get("total_stats", {}).get("mean")
            row["margin_mean"] = payload.get("margin_stats", {}).get("mean")
            row["total_corr"] = payload.get("total_corr")
            row["margin_corr"] = payload.get("margin_corr")
        elif pat.startswith("predictability"):
            row["predictability_score"] = payload.get("predictability_score")
            row["residual_std"] = payload.get("residual_std")
        elif pat.startswith("fairness"):
            g = payload.get("global", {})
            row["fairness_bias_flag"] = g.get("bias_flag")
            row["fairness_disparity_flag"] = g.get("disparity_flag")
        elif pat.startswith("recalibration"):
            row["recalibration_needed"] = payload.get("recalibration_needed")
            m = payload.get("metrics", {})
            for k in ["total_mean_z","margin_mean_z","total_corr_z","margin_corr_z"]:
                if k in m:
                    row[k] = m[k]
        elif pat.startswith("leakage"):
            row["leakage_suspicious_cols"] = len(payload.get("suspicious_columns", []))
        elif pat.startswith("scoring"):
            for k in ["crps_total","crps_total_mean","crps_mean","loglik_mean"]:
                if k in payload and (k not in row):
                    row[k] = payload[k]
        elif pat.startswith("backtest_metrics"):
            for k in ["totals_edge_mean","spread_edge_mean","moneyline_edge_mean"]:
                if k in payload and k not in row:
                    row[k] = payload[k]
        elif pat.startswith("team_variance"):
            row["team_variance_ingested"] = True
    return row


def _compute_health(r: Dict[str, Any]) -> str:
    ps = r.get("predictability_score")
    recal = r.get("recalibration_needed") is True
    leakage = r.get("leakage_suspicious_cols", 0) or 0
    bias = r.get("fairness_bias_flag") is True
    disparity = r.get("fairness_disparity_flag") is True
    tmz = r.get("total_mean_z")
    tcz = r.get("total_corr_z")
    # Critical conditions
    if recal and ((isinstance(ps,(int,float)) and ps < 0.35) or (isinstance(tmz,(int,float)) and abs(tmz) > 2.5) or (isinstance(tcz,(int,float)) and tcz < -2.2)):
        return "critical"
    # Degraded conditions
    if (isinstance(ps,(int,float)) and ps < 0.55) or leakage > 0 or bias or disparity:
        return "degraded"
    return "healthy"


def _rolling_trailing(current_date: str, key: str, window: int = 7) -> float | None:
    items = []
    for p in OUT.glob("performance_*.json"):
        ds = p.name.replace("performance_", "").replace(".json", "")
        if ds >= current_date:
            continue
        payload = _load_json(p)
        if isinstance(payload.get(key), (int,float)):
            items.append((ds, float(payload[key])))
    items = sorted(items, key=lambda x: x[0])
    if not items:
        return None
    vals = [v for _, v in items][-window:]
    try:
        return statistics.fmean(vals)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")

    row = _extract(date_str)
    row["model_health"] = _compute_health(row)

    # Compute rolling trailing averages based on historical daily performance artifacts
    hist = []
    for p in OUT.glob("performance_*.json"):
        try:
            ds = p.name.replace("performance_", "").replace(".json", "")
            if ds >= date_str:  # only prior dates
                continue
            payload = _load_json(p)
            if not payload:
                continue
            hist.append((ds, payload))
        except Exception:
            continue
    hist = sorted(hist, key=lambda x: x[0])[-30:]  # last 30
    def _roll(key: str, w: int) -> float | None:
        vals = [float(h[1].get(key)) for h in hist if isinstance(h[1].get(key),(int,float))]
        if not vals:
            return None
        vals = vals[-w:]
        try:
            return statistics.fmean(vals)
        except Exception:
            return None
    for k in ["predictability_score","residual_std","crps_total","total_mean","total_corr"]:
        row[f"{k}_trailing_7"] = _roll(k,7)
        row[f"{k}_trailing_14"] = _roll(k,14)

    payload = row
    payload["generated_at"] = dt.datetime.utcnow().isoformat() + "Z"
    out_path = OUT / f"performance_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Daily performance written -> {out_path}")

if __name__ == "__main__":
    main()
