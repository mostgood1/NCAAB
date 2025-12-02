"""Generate conformal-adjusted quantile intervals for all dates (leave-one-day-out).

For each date D:
 - Use all prior dates as calibration (or all others if scarce)
 - Compute buffers B_total, B_margin from nonconformity scores
 - Write adjusted intervals for D

Inputs:
 - outputs/quantiles_history.csv
 - outputs/daily_results/results_*.csv

Outputs:
 - outputs/quantiles_conformal_history.csv with columns:
   date, game_id, total_c10, total_c50, total_c90, margin_c10, margin_c50, margin_c90
 - outputs/conformal_params_history.csv with per-date buffers and calibration counts
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
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def score_interval(row, actual_key: str, q10_key: str, q90_key: str):
    y = pd.to_numeric(row.get(actual_key), errors='coerce')
    q10 = pd.to_numeric(row.get(q10_key), errors='coerce')
    q90 = pd.to_numeric(row.get(q90_key), errors='coerce')
    if not (np.isfinite(y) and np.isfinite(q10) and np.isfinite(q90)):
        return np.nan
    return max(0.0, y - q90, q10 - y)

def main():
    q = _safe_read(OUTPUTS / 'quantiles_history.csv')
    r = load_results()
    if q.empty or r.empty:
        print('[conformal-all] Missing inputs; aborting.')
        return
    for col in ('game_id','date'):
        if col in q.columns:
            q[col] = q[col].astype(str).str.replace(r'\\.0$','', regex=True)
        if col in r.columns:
            r[col] = r[col].astype(str).str.replace(r'\\.0$','', regex=True)
    df = r.merge(q, on=['date','game_id'], how='left')
    if df.empty:
        print('[conformal-all] Empty merge; aborting.')
        return
    dates = sorted(df['date'].astype(str).unique())
    out_rows = []
    params_rows = []
    for latest in dates:
        cal = df[df['date'] < latest].copy()
        if len(cal) < 30:
            cal = df[df['date'] != latest].copy()
        cal['s_total'] = cal.apply(lambda r: score_interval(r, 'actual_total', 'q10_total', 'q90_total'), axis=1)
        cal['s_margin'] = cal.apply(lambda r: score_interval(r, 'actual_margin', 'q10_margin', 'q90_margin'), axis=1)
        B_total = float(np.nanquantile(cal['s_total'].dropna(), 0.80)) if cal['s_total'].dropna().size > 0 else 0.0
        B_margin = float(np.nanquantile(cal['s_margin'].dropna(), 0.80)) if cal['s_margin'].dropna().size > 0 else 0.0
        today = q[q['date'].astype(str) == latest].copy()
        for key in ('q10_total','q90_total','q50_total','q10_margin','q90_margin','q50_margin'):
            if key not in today.columns:
                today[key] = np.nan
        today['total_c10'] = pd.to_numeric(today['q10_total'], errors='coerce') - B_total
        today['total_c50'] = pd.to_numeric(today['q50_total'], errors='coerce')
        today['total_c90'] = pd.to_numeric(today['q90_total'], errors='coerce') + B_total
        today['margin_c10'] = pd.to_numeric(today['q10_margin'], errors='coerce') - B_margin
        today['margin_c50'] = pd.to_numeric(today['q50_margin'], errors='coerce')
        today['margin_c90'] = pd.to_numeric(today['q90_margin'], errors='coerce') + B_margin
        for row in today[['date','game_id','total_c10','total_c50','total_c90','margin_c10','margin_c50','margin_c90']].itertuples(index=False):
            out_rows.append(row)
        params_rows.append({'date': latest, 'B_total': B_total, 'B_margin': B_margin, 'n_cal': int(len(cal))})
    out_df = pd.DataFrame(out_rows, columns=['date','game_id','total_c10','total_c50','total_c90','margin_c10','margin_c50','margin_c90'])
    out_df.to_csv(OUTPUTS / 'quantiles_conformal_history.csv', index=False)
    pd.DataFrame(params_rows).to_csv(OUTPUTS / 'conformal_params_history.csv', index=False)
    print('[conformal-all] Wrote outputs/quantiles_conformal_history.csv and conformal_params_history.csv')

if __name__ == '__main__':
    main()
