"""Apply split-conformal adjustment to quantile intervals to improve coverage.

Inputs:
  - outputs/quantiles_history.csv (q10/q50/q90 for totals/margins)
  - outputs/daily_results/results_*.csv (actual_total, actual_margin)

Outputs:
  - outputs/quantiles_conformal_today.csv for latest date, with adjusted intervals:
      total_c10, total_c50, total_c90, margin_c10, margin_c50, margin_c90

Method:
  - Compute nonconformity scores on a calibration set (recent dates):
    s_total = max(0, actual_total - q90_total, q10_total - actual_total)
    s_margin = max(0, actual_margin - q90_margin, q10_margin - actual_margin)
  - Buffer B_total = quantile(s_total, 0.80), B_margin = quantile(s_margin, 0.80)
  - Adjust intervals for latest date: [q10 - B, q90 + B], keep q50 as is.
  - This yields marginal 80% coverage under exchangeability assumptions.
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

def main():
    q = _safe_read(OUTPUTS / 'quantiles_history.csv')
    r = load_results()
    if q.empty or r.empty:
        print('[conformal] Missing inputs; aborting.')
        return
    for col in ('game_id','date'):
        if col in q.columns:
            q[col] = q[col].astype(str).str.replace(r'\.0$','', regex=True)
        if col in r.columns:
            r[col] = r[col].astype(str).str.replace(r'\.0$','', regex=True)
    df = r.merge(q, on=['date','game_id'], how='left')
    if df.empty:
        print('[conformal] Empty merge; aborting.')
        return
    # Determine latest date and calibration window (last 30 days if available)
    df['date'] = df['date'].astype(str)
    dates = sorted(df['date'].unique())
    if not dates:
        print('[conformal] No dates found.')
        return
    latest = dates[-1]
    # Calibration set: all except latest; favor recent slice
    cal = df[df['date'] != latest].copy()
    # If too small, use all
    if len(cal) < 50:
        cal = df.copy()
    # Compute scores
    def score_interval(row, actual_key: str, q10_key: str, q90_key: str):
        y = pd.to_numeric(row.get(actual_key), errors='coerce')
        q10 = pd.to_numeric(row.get(q10_key), errors='coerce')
        q90 = pd.to_numeric(row.get(q90_key), errors='coerce')
        if not (np.isfinite(y) and np.isfinite(q10) and np.isfinite(q90)):
            return np.nan
        return max(0.0, y - q90, q10 - y)
    cal['s_total'] = cal.apply(lambda r: score_interval(r, 'actual_total', 'q10_total', 'q90_total'), axis=1)
    cal['s_margin'] = cal.apply(lambda r: score_interval(r, 'actual_margin', 'q10_margin', 'q90_margin'), axis=1)
    B_total = np.nanquantile(cal['s_total'].dropna(), 0.80) if cal['s_total'].dropna().size > 0 else 0.0
    B_margin = np.nanquantile(cal['s_margin'].dropna(), 0.80) if cal['s_margin'].dropna().size > 0 else 0.0
    today = q[q['date'].astype(str) == latest].copy()
    # Apply buffers
    for key in ('q10_total','q90_total','q50_total','q10_margin','q90_margin','q50_margin'):
        if key not in today.columns:
            today[key] = np.nan
    today['total_c10'] = pd.to_numeric(today['q10_total'], errors='coerce') - B_total
    today['total_c50'] = pd.to_numeric(today['q50_total'], errors='coerce')
    today['total_c90'] = pd.to_numeric(today['q90_total'], errors='coerce') + B_total
    today['margin_c10'] = pd.to_numeric(today['q10_margin'], errors='coerce') - B_margin
    today['margin_c50'] = pd.to_numeric(today['q50_margin'], errors='coerce')
    today['margin_c90'] = pd.to_numeric(today['q90_margin'], errors='coerce') + B_margin
    out_cols = ['date','game_id','total_c10','total_c50','total_c90','margin_c10','margin_c50','margin_c90']
    out_df = today[out_cols]
    out_df.to_csv(OUTPUTS / 'quantiles_conformal_today.csv', index=False)
    # Persist buffer parameters for audit
    import json
    params_path = OUTPUTS / f'conformal_params_{latest}.json'
    params_path.write_text(json.dumps({'date': latest, 'B_total': B_total, 'B_margin': B_margin, 'n_cal': int(len(cal))}, indent=2))
    print('[conformal] Wrote outputs/quantiles_conformal_today.csv and', params_path)

if __name__ == '__main__':
    main()
