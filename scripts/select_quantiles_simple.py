"""Generate per-game quantiles (q10/q50/q90) for totals and margins
using residual quantiles from outputs/quantile_model.json and today's
enriched predictions.

Outputs:
- outputs/quantiles_selected.csv (today only)
- outputs/quantiles_history.csv (appended, deduped by date)
- outputs/quantile_model_selection.json (light metadata)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _load_quantile_model() -> dict:
    p = OUT / 'quantile_model.json'
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _latest_enriched_path(date_str: str | None) -> Path | None:
    if date_str:
        p = OUT / f'predictions_unified_enriched_{date_str}.csv'
        return p if p.exists() else None
    # Fallback: pick latest enriched file
    files = sorted([x for x in OUT.glob('predictions_unified_enriched_*.csv')])
    return files[-1] if files else None


def generate_quantiles(date_str: str | None = None, target_cov: float = 0.8) -> dict:
    qm = _load_quantile_model()
    rq = qm.get('residual_quantiles', {}) if isinstance(qm, dict) else {}
    tot = rq.get('total', {}) if isinstance(rq, dict) else {}
    mar = rq.get('margin', {}) if isinstance(rq, dict) else {}
    # Map to low/med/high
    qt = {
        'r_q_low': float(tot.get('q10')) if tot.get('q10') is not None else np.nan,
        'r_q_med': float(tot.get('q50')) if tot.get('q50') is not None else np.nan,
        'r_q_high': float(tot.get('q90')) if tot.get('q90') is not None else np.nan,
    }
    qmarg = {
        'r_q_low': float(mar.get('q10')) if mar.get('q10') is not None else np.nan,
        'r_q_med': float(mar.get('q50')) if mar.get('q50') is not None else np.nan,
        'r_q_high': float(mar.get('q90')) if mar.get('q90') is not None else np.nan,
    }
    # Load enriched predictions for the date
    pred_path = _latest_enriched_path(date_str)
    preds = _safe_read_csv(pred_path) if pred_path else pd.DataFrame()
    if preds.empty:
        raise SystemExit('No predictions_unified_enriched_<date>.csv found')
    # Normalize a few columns
    preds = preds.copy()
    if 'game_id' in preds.columns:
        preds['game_id'] = preds['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    if 'date' not in preds.columns:
        # Derive from filename as backup
        try:
            ds = pred_path.name.replace('predictions_unified_enriched_', '').replace('.csv', '')
        except Exception:
            ds = datetime.utcnow().strftime('%Y-%m-%d')
        preds['date'] = ds
    latest = str(sorted(preds['date'].dropna().astype(str).unique())[-1])
    today = preds[preds['date'].astype(str) == latest].copy()
    # Required prediction columns
    for col in ['pred_total', 'pred_margin']:
        if col not in today.columns:
            raise SystemExit(f'missing column {col} in enriched predictions for {latest}')
    # Apply residual quantiles
    today['q10_total'] = pd.to_numeric(today['pred_total'], errors='coerce') + qt['r_q_low']
    today['q50_total'] = pd.to_numeric(today['pred_total'], errors='coerce') + qt['r_q_med']
    today['q90_total'] = pd.to_numeric(today['pred_total'], errors='coerce') + qt['r_q_high']
    today['q10_margin'] = pd.to_numeric(today['pred_margin'], errors='coerce') + qmarg['r_q_low']
    today['q50_margin'] = pd.to_numeric(today['pred_margin'], errors='coerce') + qmarg['r_q_med']
    today['q90_margin'] = pd.to_numeric(today['pred_margin'], errors='coerce') + qmarg['r_q_high']
    # Enforce monotone increasing per target
    def _mono(row, a, b, c):
        vals = sorted([row[a], row[b], row[c]])
        return pd.Series(vals, index=[a, b, c])
    today[['q10_total','q50_total','q90_total']] = today.apply(lambda r: _mono(r, 'q10_total','q50_total','q90_total'), axis=1)
    today[['q10_margin','q50_margin','q90_margin']] = today.apply(lambda r: _mono(r, 'q10_margin','q50_margin','q90_margin'), axis=1)

    # Persist history and selected
    hist_cols = ['date','game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin']
    qhist_path = OUT / 'quantiles_history.csv'
    if qhist_path.exists():
        try:
            old = pd.read_csv(qhist_path)
            old['game_id'] = old['game_id'].astype(str)
        except Exception:
            old = pd.DataFrame(columns=hist_cols)
        old = old[old['date'].astype(str) != latest]
        new_hist = pd.concat([old, today[hist_cols]], ignore_index=True)
    else:
        new_hist = today[hist_cols]
    new_hist.to_csv(qhist_path, index=False)

    qsel_path = OUT / 'quantiles_selected.csv'
    today[hist_cols].to_csv(qsel_path, index=False)

    # Write lightweight meta
    meta = {
        'method': 'residual_quantiles_simple',
        'source_model': 'outputs/quantile_model.json',
        'predictions_file': str(pred_path.relative_to(ROOT)) if pred_path else None,
        'latest_date': latest,
        'target_coverage': target_cov,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    with open(OUT / 'quantile_model_selection.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Wrote:', qhist_path)
    print('Wrote:', qsel_path)
    print('Wrote:', OUT / 'quantile_model_selection.json')
    return {
        'rows': int(len(today)),
        'date': latest,
        'paths': {
            'history': str(qhist_path),
            'selected': str(qsel_path)
        }
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, default=None, help='Target date YYYY-MM-DD; defaults to latest enriched file')
    ap.add_argument('--target-coverage', type=float, default=0.8)
    args = ap.parse_args()
    generate_quantiles(date_str=args.date, target_cov=args.target_coverage)
