"""Compute dynamic sigma from historical residuals.

Joins predictions_history_enriched with results to estimate rolling RMSE
for totals and margins. Outputs per-date and global sigma estimates and
augmented calibrated predictions with refined sigmas.

Reads:
  - outputs/predictions_history_calibrated.csv (if exists) else enriched
  - outputs/daily_results/results_*.csv

Writes:
  - outputs/sigma_history.csv (date-level rolling sigmas)
  - outputs/predictions_history_sigma.csv (adds sigma_total,sigma_margin)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path('outputs')

RESULTS_GLOB = 'daily_results/results_*.csv'
PRED_CAL = OUTPUTS / 'predictions_history_calibrated.csv'
PRED_ENRICHED = OUTPUTS / 'predictions_history_enriched.csv'

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob(RESULTS_GLOB):
        df = _safe_read(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    if {'date','game_id'}.issubset(res.columns):
        res = res.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    return res

def load_predictions() -> pd.DataFrame:
    df = _safe_read(PRED_CAL)
    if df.empty:
        df = _safe_read(PRED_ENRICHED)
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    return df

def rolling_rmse(series: pd.Series, window: int = 30) -> pd.Series:
    return series.pow(2).rolling(window=window, min_periods=max(5, window//3)).mean().pow(0.5)

def main():
    preds = load_predictions()
    results = load_results()
    if preds.empty or results.empty:
        print('[sigma] Missing predictions or results; aborting.')
        return
    merged = results.merge(preds, on=['date','game_id'], how='left')
    # Residuals
    if {'actual_total','pred_total'}.issubset(merged.columns):
        merged['res_total'] = merged['actual_total'] - merged['pred_total']
    if {'actual_margin','pred_margin'}.issubset(merged.columns):
        merged['res_margin'] = merged['actual_margin'] - merged['pred_margin']
    # Sort by date for rolling calc
    merged = merged.sort_values('date')
    # Rolling RMSE per date
    sigma_df = merged[['date']].copy()
    sigma_total_roll = rolling_rmse(merged['res_total']) if 'res_total' in merged.columns else pd.Series(index=merged.index, dtype=float)
    sigma_margin_roll = rolling_rmse(merged['res_margin']) if 'res_margin' in merged.columns else pd.Series(index=merged.index, dtype=float)
    sigma_df['sigma_total_roll'] = sigma_total_roll
    sigma_df['sigma_margin_roll'] = sigma_margin_roll
    # Global fallback using overall RMSE
    def overall_rmse(s: pd.Series) -> float:
        s = s.dropna()
        return float(np.sqrt(np.mean(np.square(s)))) if len(s) else np.nan
    sigma_total_global = overall_rmse(merged['res_total']) if 'res_total' in merged.columns else np.nan
    sigma_margin_global = overall_rmse(merged['res_margin']) if 'res_margin' in merged.columns else np.nan
    # Propagate refined sigma into predictions rows by date, with fallback
    by_date = sigma_df.groupby('date', observed=False).agg(
        sigma_total=('sigma_total_roll','last'),
        sigma_margin=('sigma_margin_roll','last'),
    ).reset_index()
    preds2 = preds.merge(by_date, on='date', how='left')
    preds2['sigma_total'] = preds2['sigma_total'].fillna(sigma_total_global)
    preds2['sigma_margin'] = preds2['sigma_margin'].fillna(sigma_margin_global)
    # Write outputs
    sigma_df.to_csv(OUTPUTS / 'sigma_history.csv', index=False)
    preds2.to_csv(OUTPUTS / 'predictions_history_sigma.csv', index=False)
    print('[sigma] Wrote sigma_history and predictions_history_sigma.')

if __name__ == '__main__':
    main()
