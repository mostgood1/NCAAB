"""Calibrate historical probabilities using isotonic regression.

Fits an isotonic mapping for a small set of probability columns against realized
outcomes derived from archived results.

Inputs (default under outputs/):
    - predictions_history_enriched.csv
    - daily_results/results_*.csv

Outputs (default under outputs/):
    - calibration_params.json (per-column isotonic breakpoints)
    - predictions_history_calibrated.csv (adds *_cal calibrated probabilities)
    - calibration_bins.csv (pre/post reliability bins)

Usage:
    python scripts/calibrate_probs_history.py
    python scripts/calibrate_probs_history.py --start 2025-12-01 --end 2026-01-13
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _parse_date(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    return s if DATE_RE.match(s) else None

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_results(outputs_dir: Path, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    frames = []
    for p in (outputs_dir / "daily_results").glob("results_*.csv"):
        df = _safe_read_csv(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    if {'date','game_id'}.issubset(res.columns):
        if start:
            res = res[res['date'].astype(str) >= str(start)]
        if end:
            res = res[res['date'].astype(str) <= str(end)]
        res = res.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    return res


def load_predictions(outputs_dir: Path, preds_path: Path, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    preds = _safe_read_csv(preds_path)
    if not preds.empty and {'date', 'game_id'}.issubset(preds.columns):
        preds['game_id'] = preds['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        preds['date'] = pd.to_datetime(preds['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        if start:
            preds = preds[preds['date'].astype(str) >= str(start)]
        if end:
            preds = preds[preds['date'].astype(str) <= str(end)]
        if not preds.empty:
            return preds

    # Fallback: stitch per-day enriched prediction snapshots.
    # These are the daily artifacts the pipeline already produces.
    frames: list[pd.DataFrame] = []
    candidates = sorted(outputs_dir.glob('predictions_unified_enriched_*.csv'))
    if not candidates:
        candidates = sorted(outputs_dir.glob('predictions_enriched_*.csv'))
    for p in candidates:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', p.name)
        if not m:
            continue
        d = m.group(1)
        if start and d < str(start):
            continue
        if end and d > str(end):
            continue
        df = _safe_read_csv(p)
        if df.empty or 'game_id' not in df.columns:
            continue
        if 'date' not in df.columns:
            df['date'] = d
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    preds = pd.concat(frames, ignore_index=True)
    preds = preds.sort_values(['date', 'game_id']).drop_duplicates(subset=['date', 'game_id'], keep='last')
    return preds

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


def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    if out_col in df.columns:
        return
    for c in candidates:
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors='coerce')
            df[out_col] = ser
            break
    if out_col not in df.columns:
        df[out_col] = np.nan


def _coalesce_first_nonnull(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    # Create/overwrite out_col as the first non-null among candidates.
    ser = None
    for c in candidates:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        ser = s if ser is None else ser.combine_first(s)
    df[out_col] = ser if ser is not None else np.nan

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs', help='Outputs directory')
    ap.add_argument('--preds', default=None, help='Predictions history enriched CSV (default: <outputs>/predictions_history_enriched.csv)')
    ap.add_argument('--start', default=None, help='Start date YYYY-MM-DD (inclusive)')
    ap.add_argument('--end', default=None, help='End date YYYY-MM-DD (inclusive)')
    ap.add_argument('--min-rows', type=int, default=500, help='Minimum training rows required per calibrated column')
    ap.add_argument('--bins', type=int, default=10, help='Reliability bin count')
    ap.add_argument('--n-points', type=int, default=200, help='Maximum isotonic breakpoint count (downsampled)')
    ap.add_argument(
        '--prob-source',
        type=str,
        default='history',
        choices=['history', 'quantiles'],
        help=(
            "Which probability source to calibrate. 'history' uses p_over/p_home_cover_dist from predictions history. "
            "'quantiles' derives probabilities from quantiles_history.csv + market lines (matches bankroll-optimize quantile sizing)."
        ),
    )
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    preds_path = Path(args.preds) if args.preds else (outputs_dir / 'predictions_history_enriched.csv')
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    preds = load_predictions(outputs_dir, preds_path, start=start, end=end)
    results = load_results(outputs_dir, start=start, end=end)
    if preds.empty or results.empty:
        print('[calibration] Missing inputs; aborting.')
        return 2

    merged = results.merge(preds, on=['date', 'game_id'], how='left', suffixes=('_res', '_pred'))

    def _cdf_piecewise(q10: float, q50: float, q90: float, x: float) -> float:
        # Piecewise-linear CDF through (q10,0.1),(q50,0.5),(q90,0.9)
        if not (np.isfinite(q10) and np.isfinite(q50) and np.isfinite(q90) and np.isfinite(x)):
            return np.nan
        q10, q50, q90 = sorted([float(q10), float(q50), float(q90)])
        if x <= q10:
            t = 0.1 + (x - q10) * (0.4 / max(q50 - q10, 1e-6))
            return float(np.clip(t, 0.0, 1.0))
        if x <= q50:
            t = 0.1 + (x - q10) * (0.4 / max(q50 - q10, 1e-6))
            return float(np.clip(t, 0.0, 1.0))
        if x <= q90:
            t = 0.5 + (x - q50) * (0.4 / max(q90 - q50, 1e-6))
            return float(np.clip(t, 0.0, 1.0))
        t = 0.9 + (x - q90) * (0.1 / max(q90 - q50, 1e-6))
        return float(np.clip(t, 0.0, 1.0))

    # Some columns exist in both results and predictions history; after merge they'll be
    # suffixed. Coalesce them back into canonical names used below.
    _coalesce_first_nonnull(merged, 'market_total', ['market_total_res', 'market_total_pred', 'market_total'])
    _coalesce_first_nonnull(merged, 'spread_home', ['spread_home_res', 'spread_home_pred', 'spread_home'])

    # Outcomes should always come from the finalized results if present.
    _coalesce_first_nonnull(merged, 'actual_total', ['actual_total_res', 'actual_total_pred', 'actual_total'])
    _coalesce_first_nonnull(merged, 'actual_margin', ['actual_margin_res', 'actual_margin_pred', 'actual_margin'])

    # Probability columns should come from predictions history, but be tolerant.
    _coalesce_cols(merged, 'p_over', ['p_over_pred', 'p_over'])
    _coalesce_cols(merged, 'p_home_cover_dist', ['p_home_cover_dist_pred', 'p_home_cover_dist'])

    if str(args.prob_source).lower() == 'quantiles':
        qh = _safe_read_csv(outputs_dir / 'quantiles_history.csv')
        if not qh.empty and {'date', 'game_id'}.issubset(qh.columns):
            qh['game_id'] = qh['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            qh['date'] = pd.to_datetime(qh['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            merged = merged.merge(qh, on=['date', 'game_id'], how='left', suffixes=('', '_q'))

            # Derive totals/spread probabilities from quantiles (matches bankroll-optimize quantile path).
            if {'q10_total', 'q50_total', 'q90_total', 'market_total'}.issubset(merged.columns):
                q10 = pd.to_numeric(merged['q10_total'], errors='coerce')
                q50 = pd.to_numeric(merged['q50_total'], errors='coerce')
                q90 = pd.to_numeric(merged['q90_total'], errors='coerce')
                line = pd.to_numeric(merged['market_total'], errors='coerce')
                F = [
                    _cdf_piecewise(a, b, c, x)
                    for a, b, c, x in zip(q10.to_numpy(), q50.to_numpy(), q90.to_numpy(), line.to_numpy())
                ]
                p_over_q = (1.0 - pd.Series(F, index=merged.index)).clip(lower=1e-4, upper=1.0 - 1e-4)
                merged['p_over'] = p_over_q

            if {'q10_margin', 'q50_margin', 'q90_margin', 'spread_home'}.issubset(merged.columns):
                q10 = pd.to_numeric(merged['q10_margin'], errors='coerce')
                q50 = pd.to_numeric(merged['q50_margin'], errors='coerce')
                q90 = pd.to_numeric(merged['q90_margin'], errors='coerce')
                spread = pd.to_numeric(merged['spread_home'], errors='coerce')
                thresh = -spread
                F = [
                    _cdf_piecewise(a, b, c, x)
                    for a, b, c, x in zip(q10.to_numpy(), q50.to_numpy(), q90.to_numpy(), thresh.to_numpy())
                ]
                p_home_q = (1.0 - pd.Series(F, index=merged.index)).clip(lower=1e-4, upper=1.0 - 1e-4)
                merged['p_home_cover_dist'] = p_home_q
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
            ((merged['actual_margin'] + merged['spread_home']) > 0).astype(int),
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
        n_train = int(mask.sum())
        if n_train < int(args.min_rows):
            continue
        pre_bins = reliability_bins(prob[mask], outcome[mask], n_bins=int(args.bins))
        par = fit_isotonic(prob[mask], outcome[mask], n_points=int(args.n_points))
        params_out[col] = par
        calibrated = apply_isotonic(prob, par)
        merged[f'{col}_cal'] = calibrated
        post_bins = reliability_bins(calibrated[mask], outcome[mask], n_bins=int(args.bins))
        pre_bins['phase'] = 'pre'
        post_bins['phase'] = 'post'
        pre_bins['method'] = col
        post_bins['method'] = col
        calib_bins_all.append(pre_bins)
        calib_bins_all.append(post_bins)
    # Write params
    (outputs_dir / 'calibration_params.json').write_text(json.dumps(params_out, indent=2))
    # Write calibrated predictions history
    merged.to_csv(outputs_dir / 'predictions_history_calibrated.csv', index=False)
    # Write bins
    if calib_bins_all:
        pd.concat(calib_bins_all, ignore_index=True).to_csv(outputs_dir / 'calibration_bins.csv', index=False)

    print('[calibration] Wrote params, calibrated history, and bins.')
    print(json.dumps({
        'outputs': str(outputs_dir),
        'preds': str(preds_path),
        'start': start,
        'end': end,
        'trained_cols': sorted(list(params_out.keys())),
    }))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
