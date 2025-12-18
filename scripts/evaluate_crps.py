import argparse
import datetime as dt
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ncaab_model.metrics.crps import crps_from_quantiles, gaussian_crps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CRPS for totals/margins using quantiles if available, else Gaussian fallback.")
    p.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD; defaults to today")
    p.add_argument("--pred-csv", type=str, default=None, help="Predictions CSV (unified enriched). Defaults to outputs/predictions_unified_enriched_<date>.csv")
    p.add_argument("--quantiles-csv", type=str, default=None, help="Quantiles sidecar path. Defaults to outputs/quantiles_<date>.csv")
    p.add_argument("--results-csv", type=str, default=None, help="Daily results CSV. Defaults to daily_results/results_<date>.csv")
    p.add_argument("--output-json", type=str, default=None, help="Scoring JSON output. Defaults to outputs/scoring_<date>.json (merged if exists)")
    return p.parse_args()


def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def load_truth(df: pd.DataFrame, results_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(results_path):
        return {"pending": True}
    res = pd.read_csv(results_path)
    if "game_id" not in res.columns:
        return {"pending": True}
    merged = df[["game_id"]].merge(res, on="game_id", how="left")

    total_candidates = [
        "total_final",
        "final_total",
        "total",
        "score_total",
        "total_points",
    ]
    margin_candidates = [
        "margin_final",
        "final_margin",
        "margin",
        "score_diff",
    ]

    truth = {"pending": False}
    for key, cands in [("total", total_candidates), ("margin", margin_candidates)]:
        for c in cands:
            if c in merged.columns:
                truth[key] = pd.to_numeric(merged[c], errors="coerce").to_numpy()
                break
    return truth


def merge_existing_json(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main() -> None:
    args = parse_args()
    date = args.date or today_str()

    pred_csv = args.pred_csv or os.path.join("outputs", f"predictions_unified_enriched_{date}.csv")
    q_csv = args.quantiles_csv or os.path.join("outputs", f"quantiles_{date}.csv")
    results_csv = args.results_csv or os.path.join("daily_results", f"results_{date}.csv")
    out_json = args.output_json or os.path.join("outputs", f"scoring_{date}.json")

    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv}")
    df = pd.read_csv(pred_csv)
    if "game_id" not in df.columns:
        raise ValueError("Predictions CSV must contain 'game_id'")

    truth = load_truth(df, results_csv)
    pending = truth.get("pending", True) or ("total" not in truth and "margin" not in truth)

    qdf = None
    if os.path.exists(q_csv):
        qdf = pd.read_csv(q_csv)
        if "game_id" not in qdf.columns:
            qdf = None

    metrics: Dict[str, object] = {
        "date": date,
        "crps_method": None,
        "pending": pending,
    }

    # Totals
    if not pending and "total" in truth:
        y_total = truth["total"]
        crps_total = None
        method_total = None
        if qdf is not None and set(["pred_total_q10", "pred_total_q50", "pred_total_q90"]).issubset(qdf.columns):
            merged = df[["game_id"]].merge(qdf, on="game_id", how="left")
            q10 = pd.to_numeric(merged["pred_total_q10"], errors="coerce").to_numpy()
            q50 = pd.to_numeric(merged["pred_total_q50"], errors="coerce").to_numpy()
            q90 = pd.to_numeric(merged["pred_total_q90"], errors="coerce").to_numpy()
            crps_total = crps_from_quantiles(y_total, [q10, q50, q90], [0.1, 0.5, 0.9])
            method_total = "quantiles"
        else:
            # Gaussian fallback
            mu = pd.to_numeric(df.get("pred_total"), errors="coerce").to_numpy()
            sigma = None
            for c in ["sigma_total", "pred_total_sigma", "total_sigma"]:
                if c in df.columns:
                    sigma = pd.to_numeric(df[c], errors="coerce").to_numpy()
                    break
            if sigma is None:
                sigma = np.full(len(df), 12.0, dtype=float)
            crps_total = gaussian_crps(y_total, mu, sigma)
            method_total = "gaussian"
        metrics["totals_crps_mean"] = float(np.nanmean(crps_total))
        metrics["totals_crps_count"] = int(np.sum(np.isfinite(crps_total)))
        metrics["totals_crps_method"] = method_total

    # Margins
    if not pending and "margin" in truth:
        y_margin = truth["margin"]
        crps_margin = None
        method_margin = None
        if qdf is not None and set(["pred_margin_q10", "pred_margin_q50", "pred_margin_q90"]).issubset(qdf.columns):
            merged = df[["game_id"]].merge(qdf, on="game_id", how="left")
            q10 = pd.to_numeric(merged["pred_margin_q10"], errors="coerce").to_numpy()
            q50 = pd.to_numeric(merged["pred_margin_q50"], errors="coerce").to_numpy()
            q90 = pd.to_numeric(merged["pred_margin_q90"], errors="coerce").to_numpy()
            crps_margin = crps_from_quantiles(y_margin, [q10, q50, q90], [0.1, 0.5, 0.9])
            method_margin = "quantiles"
        else:
            mu = pd.to_numeric(df.get("pred_margin"), errors="coerce").to_numpy()
            sigma = None
            for c in ["sigma_margin", "pred_margin_sigma", "margin_sigma"]:
                if c in df.columns:
                    sigma = pd.to_numeric(df[c], errors="coerce").to_numpy()
                    break
            if sigma is None:
                sigma = np.full(len(df), 7.0, dtype=float)
            crps_margin = gaussian_crps(y_margin, mu, sigma)
            method_margin = "gaussian"
        metrics["margins_crps_mean"] = float(np.nanmean(crps_margin))
        metrics["margins_crps_count"] = int(np.sum(np.isfinite(crps_margin)))
        metrics["margins_crps_method"] = method_margin

    # Merge with existing scoring JSON if present
    base = merge_existing_json(out_json)
    base.update(metrics)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
"""Compute CRPS and interval coverage using quantiles_history.csv.

Inputs:
  - outputs/quantiles_history.csv
  - outputs/daily_results/results_*.csv

Outputs:
  - outputs/quantile_metrics.csv (per-date CRPS + coverage)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path('outputs')

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob('daily_results/results_*.csv'):
        df = _safe_read(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    return res

def crps_from_quantiles(q10, q50, q90, y):
    # Triangular approximation using piecewise linear CDF from quantiles
    if np.isnan(q10) or np.isnan(q50) or np.isnan(q90) or np.isnan(y):
        return np.nan
    # Simple surrogate: weighted absolute errors
    return 0.25*abs(y - q10) + 0.5*abs(y - q50) + 0.25*abs(y - q90)

def main():
    q = _safe_read(OUTPUTS / 'quantiles_history.csv')
    r = load_results()
    if q.empty or r.empty:
        print('[crps] Missing inputs; aborting.')
        return
    q['game_id'] = q['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    df = r.merge(q, on=['date','game_id'], how='left')
    rows = []
    for _, row in df.iterrows():
        ct = row['actual_total'] if 'actual_total' in row else np.nan
        cm = row['actual_margin'] if 'actual_margin' in row else np.nan
        rows.append({
            'date': row['date'],
            'game_id': row['game_id'],
            'crps_total': crps_from_quantiles(row.get('q10_total', np.nan), row.get('q50_total', np.nan), row.get('q90_total', np.nan), ct),
            'crps_margin': crps_from_quantiles(row.get('q10_margin', np.nan), row.get('q50_margin', np.nan), row.get('q90_margin', np.nan), cm),
            'covered_80_total': float((ct >= row.get('q10_total', np.inf)) and (ct <= row.get('q90_total', -np.inf))) if pd.notna(ct) else np.nan,
            'covered_80_margin': float((cm >= row.get('q10_margin', np.inf)) and (cm <= row.get('q90_margin', -np.inf))) if pd.notna(cm) else np.nan,
        })
    m = pd.DataFrame(rows)
    agg = m.groupby('date', observed=False).agg({
        'crps_total':'mean',
        'crps_margin':'mean',
        'covered_80_total':'mean',
        'covered_80_margin':'mean',
    }).reset_index()
    agg.to_csv(OUTPUTS / 'quantile_metrics.csv', index=False)
    print('[crps] Wrote outputs/quantile_metrics.csv')

if __name__ == '__main__':
    main()
