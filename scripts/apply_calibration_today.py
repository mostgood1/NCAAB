"""Apply persisted calibration params to today's enriched predictions.

Reads:
  - outputs/predictions_history_enriched.csv
  - outputs/calibration_params.json

Writes:
  - outputs/predictions_today_calibrated.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json

OUTPUTS = Path('outputs')

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _load_params(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def apply_isotonic_series(prob: pd.Series, params: dict) -> pd.Series:
    xs = params.get('x', [])
    ys = params.get('y', [])
    if not xs or not ys or len(xs) != len(ys):
        return pd.to_numeric(prob, errors='coerce')
    def map_val(p):
        try:
            p = float(p)
        except Exception:
            return np.nan
        idx = 0
        for i, xv in enumerate(xs):
            if xv <= p:
                idx = i
            else:
                break
        return float(ys[idx])
    return pd.to_numeric(prob, errors='coerce').apply(map_val)

def main():
    enriched = _safe_read_csv(OUTPUTS / 'predictions_history_enriched.csv')
    params = _load_params(OUTPUTS / 'calibration_params.json')
    if enriched.empty or not params:
        print('[apply-cal] Missing inputs; aborting.')
        return
    # Pick latest date present
    if 'date' not in enriched.columns:
        print('[apply-cal] No date column; aborting.')
        return
    latest = str(sorted(enriched['date'].dropna().unique())[-1])
    today_df = enriched[enriched['date'].astype(str) == latest].copy()
    # Apply calibration if available
    if 'p_over' in today_df.columns and 'p_over' in params:
        today_df['p_over_cal'] = apply_isotonic_series(today_df['p_over'], params['p_over'])
    if 'p_home_cover_dist' in today_df.columns and 'p_home_cover_dist' in params:
        today_df['p_home_cover_dist_cal'] = apply_isotonic_series(today_df['p_home_cover_dist'], params['p_home_cover_dist'])
    out = OUTPUTS / 'predictions_today_calibrated.csv'
    today_df.to_csv(out, index=False)
    print(f'[apply-cal] Wrote {out} for date {latest}')

if __name__ == '__main__':
    main()
