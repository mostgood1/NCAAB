#!/usr/bin/env python
"""Reliability calibration curve for totals predictions.

Builds binned residual statistics and (optionally) isotonic adjustment guidance.
Writes outputs/reliability_<date>.json.

Approach:
  - Use results_<date>.csv for completed games (actual_total available)
  - Use pred_total_model > pred_total fallback
  - Bin predictions into equal-count bins (up to 12, min 8 rows per bin)
  - For each bin record mean_pred, mean_actual, mean_residual, std_residual
  - Compute slope/intercept for simple linear calibration pred_adj = slope*pred + intercept

Usage:
  python scripts/reliability_calibration.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

OUT = Path("outputs")
RES = OUT / "daily_results"


def load_results(d: str) -> pd.DataFrame:
    p = RES / f"results_{d}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def linear_calibration(x: np.ndarray, y: np.ndarray):
    if len(x) < 5:
        return None, None
    try:
        # Simple least squares slope/intercept
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        if den == 0:
            return None, None
        slope = num / den
        intercept = y_mean - slope * x_mean
        return float(slope), float(intercept)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")

    df = load_results(date_str)
    if df.empty or 'actual_total' not in df.columns:
        payload = {"date": date_str, "status": "no_data"}
        (OUT / f"reliability_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Reliability: no data")
        return

    pred_col = 'pred_total_model' if 'pred_total_model' in df.columns else ('pred_total' if 'pred_total' in df.columns else None)
    if pred_col is None:
        payload = {"date": date_str, "status": "missing_pred"}
        (OUT / f"reliability_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Reliability: missing pred column")
        return

    pred = pd.to_numeric(df[pred_col], errors='coerce')
    actual = pd.to_numeric(df['actual_total'], errors='coerce')
    good = pred.notna() & actual.notna()
    pred = pred[good]
    actual = actual[good]
    if len(pred) < 25:
        payload = {"date": date_str, "status": "insufficient_rows", "rows": int(len(pred))}
        (OUT / f"reliability_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Reliability: insufficient rows")
        return

    # Equal-count binning
    n_bins = min(12, max(5, len(pred) // 8))
    try:
        bins = pd.qcut(pred, q=n_bins, duplicates='drop')
    except Exception:
        bins = pd.qcut(pred.rank(method='first'), q=n_bins, duplicates='drop')
    df_bins = pd.DataFrame({'pred': pred, 'actual': actual, 'bin': bins})
    records = []
    for b, grp in df_bins.groupby('bin'):
        p_vals = grp['pred']
        a_vals = grp['actual']
        resid = p_vals - a_vals
        records.append({
            'bin': str(b),
            'n_games': int(len(grp)),
            'mean_pred': float(p_vals.mean()),
            'mean_actual': float(a_vals.mean()),
            'mean_residual': float(resid.mean()),
            'std_residual': float(resid.std()) if resid.std() > 0 else 0.0,
        })

    slope, intercept = linear_calibration(pred.values, actual.values)

    payload = {
        'date': date_str,
        'generated_at': dt.datetime.utcnow().isoformat() + 'Z',
        'status': 'ok',
        'bins': records,
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'rows': int(len(pred))
    }
    out_path = OUT / f'reliability_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Reliability calibration written -> {out_path}')

if __name__ == '__main__':
    main()
