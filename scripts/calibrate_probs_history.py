"""Calibrate historical probabilities using isotonic regression.

Reads:
  - outputs/prob_method_summary.csv (for completeness; not required)
  - outputs/predictions_history_enriched.csv
  - outputs/daily_results/results_*.csv

Writes:
  - outputs/calibration_params.json (per method isotonic breakpoints)
  - outputs/predictions_history_calibrated.csv (adds calibrated probabilities)
  - outputs/calibration_bins.csv (pre/post calibration reliability bins)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json
import math

OUTPUTS = Path("outputs")
PRED_ENRICHED = OUTPUTS / "predictions_history_enriched.csv"
RESULTS_GLOB = "daily_results/results_*.csv"

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob(RESULTS_GLOB):
        df = _safe_read_csv(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    if {'date','game_id'}.issubset(res.columns):
        res = res.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    return res

def reliability_bins(prob: pd.Series, outcome: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'prob': pd.to_numeric(prob, errors='coerce'), 'y': outcome}).dropna()
    if df.empty:
        return pd.DataFrame(columns=['bin','p_mean','y_rate','count','abs_gap'])
    df['bin'] = pd.qcut(df['prob'].clip(1e-6, 1-1e-6), q=n_bins, duplicates='drop')
    out = df.groupby('bin', observed=False).agg(p_mean=('prob','mean'), y_rate=('y','mean'), count=('y','size')).reset_index()
    out['abs_gap'] = (out['p_mean'] - out['y_rate']).abs()
    out['bin'] = out['bin'].astype(str)
    return out

def fit_isotonic(prob: pd.Series, outcome: pd.Series, n_points: int = 100) -> dict:
    # Pool Adjacent Violators (PAV) algorithm for isotonic calibration
    df = pd.DataFrame({'p': pd.to_numeric(prob, errors='coerce'), 'y': outcome}).dropna().sort_values('p')
    if df.empty:
        return {'x': [], 'y': []}
    # Initialize blocks
    x = df['p'].to_numpy()
    y = df['y'].to_numpy()
    # Start with each point as a block
    blocks = [{'sum_w':1.0, 'sum_y':float(y[i]), 'x_min':float(x[i]), 'x_max':float(x[i])} for i in range(len(x))]
    # PAV merge to enforce non-decreasing average
    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i]['sum_y'] / blocks[i]['sum_w']
        avg_j = blocks[i+1]['sum_y'] / blocks[i+1]['sum_w']
        if avg_i <= avg_j:
            i += 1
        else:
            # merge i and i+1
            merged = {
                'sum_w': blocks[i]['sum_w'] + blocks[i+1]['sum_w'],
                'sum_y': blocks[i]['sum_y'] + blocks[i+1]['sum_y'],
                'x_min': blocks[i]['x_min'],
                'x_max': blocks[i+1]['x_max'],
            }
            blocks[i] = merged
            del blocks[i+1]
            # backtrack if needed
            i = max(i-1, 0)
    # Convert blocks to step function
    xs = []
    ys = []
    for b in blocks:
        xs.append(b['x_min'])
        ys.append(b['sum_y']/b['sum_w'])
        xs.append(b['x_max'])
        ys.append(b['sum_y']/b['sum_w'])
    # Optionally reduce resolution
    if len(xs) > n_points:
        # sample evenly in x
        grid = np.linspace(min(xs), max(xs), n_points)
        # piecewise constant interpolation
        def interp(val):
            # find last xs <= val
            idx = max([i for i, xv in enumerate(xs) if xv <= val], default=0)
            return ys[idx]
        xs = grid.tolist()
        ys = [interp(g) for g in grid]
    return {'x': xs, 'y': ys}

def apply_isotonic(prob: pd.Series, params: dict) -> pd.Series:
    xs = params.get('x', [])
    ys = params.get('y', [])
    if not xs or not ys or len(xs) != len(ys):
        return pd.to_numeric(prob, errors='coerce')
    def map_val(p):
        try:
            p = float(p)
        except Exception:
            return np.nan
        # find rightmost x <= p
        idx = 0
        for i, xv in enumerate(xs):
            if xv <= p:
                idx = i
            else:
                break
        return float(ys[idx])
    return pd.to_numeric(prob, errors='coerce').apply(map_val)

def main():
    preds = _safe_read_csv(PRED_ENRICHED)
    results = load_results()
    if preds.empty or results.empty:
        print('[calibration] Missing inputs; aborting.')
        return
    preds['game_id'] = preds['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    merged = results.merge(preds, on=['date','game_id'], how='left')
    has_market_total = 'market_total' in merged.columns and 'actual_total' in merged.columns
    has_spread_home = 'spread_home' in merged.columns and 'actual_margin' in merged.columns
    if has_market_total:
        merged['ou_outcome'] = np.where(
            merged['market_total'].notna() & merged['actual_total'].notna(),
            (merged['actual_total'] > merged['market_total']).astype(int),
            np.nan,
        )
    else:
        merged['ou_outcome'] = np.nan
    if has_spread_home:
        merged['cover_home_outcome'] = np.where(
            merged['spread_home'].notna() & merged['actual_margin'].notna(),
            (merged['actual_margin'] > merged['spread_home']).astype(int),
            np.nan,
        )
    else:
        merged['cover_home_outcome'] = np.nan
    methods = [
        ('p_over', 'ou_outcome'),
        ('p_home_cover_dist', 'cover_home_outcome'),
    ]
    params_out = {}
    calib_bins_all = []
    for col, outcome_col in methods:
        prob = pd.to_numeric(merged[col], errors='coerce') if col in merged.columns else pd.Series(dtype=float)
        outcome = merged[outcome_col] if outcome_col in merged.columns else pd.Series(dtype=float)
        mask = prob.notna() & outcome.notna()
        if not mask.any():
            continue
        pre_bins = reliability_bins(prob[mask], outcome[mask])
        par = fit_isotonic(prob[mask], outcome[mask])
        params_out[col] = par
        calibrated = apply_isotonic(prob, par)
        merged[f'{col}_cal'] = calibrated
        post_bins = reliability_bins(calibrated[mask], outcome[mask])
        pre_bins['phase'] = 'pre'
        post_bins['phase'] = 'post'
        pre_bins['method'] = col
        post_bins['method'] = col
        calib_bins_all.append(pre_bins)
        calib_bins_all.append(post_bins)
    # Write params
    (OUTPUTS / 'calibration_params.json').write_text(json.dumps(params_out, indent=2))
    # Write calibrated predictions history
    merged.to_csv(OUTPUTS / 'predictions_history_calibrated.csv', index=False)
    # Write bins
    if calib_bins_all:
        pd.concat(calib_bins_all, ignore_index=True).to_csv(OUTPUTS / 'calibration_bins.csv', index=False)
    print('[calibration] Wrote params, calibrated history, and bins.')

if __name__ == '__main__':
    main()
