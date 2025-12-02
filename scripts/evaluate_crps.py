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
