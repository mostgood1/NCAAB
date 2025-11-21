"""Derive moneyline win probability placeholders from spread using logistic transform.

Usage:
  python scripts/moneyline_prob_derive.py --date 2025-11-19

Adds columns:
  ml_prob_model_home, ml_prob_model_away to predictions_model_<date>.csv
If already present, skips.
"""
from __future__ import annotations
import argparse, datetime as dt
from pathlib import Path
import pandas as pd
import math

OUT = Path('outputs')

# Empirical mapping constant: p = 1 / (1 + exp(-k * spread)) where spread is home margin prediction.
# Choose k so that spread=0 => p=0.5; spread=7 => p~0.75 typical moderate favorite; k ~= 0.115
K_CONST = 0.115

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default: today)')
    args = ap.parse_args()
    date_str = args.date or dt.datetime.now().strftime('%Y-%m-%d')
    path = OUT / f'predictions_model_{date_str}.csv'
    df = _safe_csv(path)
    if df.empty:
        print(f'No model predictions for {date_str}')
        return
    if 'ml_prob_model_home' in df.columns and 'ml_prob_model_away' in df.columns:
        print('Moneyline probability columns already present; skipping.')
        return
    if 'pred_margin_model' not in df.columns and 'pred_margin' not in df.columns:
        print('No margin prediction columns available; cannot derive ML probabilities.')
        return
    margin_col = 'pred_margin_model' if 'pred_margin_model' in df.columns else 'pred_margin'
    m = pd.to_numeric(df[margin_col], errors='coerce')
    # Home win prob from predicted margin; assume symmetry
    p_home = 1 / (1 + np.exp(-K_CONST * m))
    # Away probability = 1 - home probability (ignoring tie)
    p_away = 1 - p_home
    df['ml_prob_model_home'] = p_home
    df['ml_prob_model_away'] = p_away
    try:
        df.to_csv(path, index=False)
        print(f'Updated {path} with ML probabilities.')
    except Exception as e:
        print(f'Failed to write updated predictions: {e}')

if __name__ == '__main__':
    import numpy as np  # lazy import inside main scope
    main()
