"""Compute per-date metrics for drift trends.

Joins enriched predictions with results by date to produce daily
metrics: Brier, log-loss, AUC for p_over and p_home_cover_dist, plus
interval coverage. Writes outputs used by drift monitoring.

Reads:
  - outputs/predictions_history_enriched.csv
  - outputs/daily_results/results_*.csv

Writes:
  - outputs/daily_metrics.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import math

OUTPUTS = Path('outputs')
RESULTS_GLOB = 'daily_results/results_*.csv'

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

def brier(prob: pd.Series, outcome: pd.Series) -> float:
    s = ((prob - outcome)**2).mean()
    return float(s) if pd.notna(s) else math.nan

def log_loss(prob: pd.Series, outcome: pd.Series, eps: float = 1e-12) -> float:
    p = prob.clip(eps, 1 - eps)
    ll = (-(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))).mean()
    return float(ll) if pd.notna(ll) else math.nan

def auc(prob: pd.Series, outcome: pd.Series) -> float:
    df = pd.DataFrame({'p': prob, 'y': outcome}).dropna()
    if df['y'].nunique() != 2:
        return math.nan
    pos = df[df.y == 1].sort_values('p'); neg = df[df.y == 0].sort_values('p')
    if pos.empty or neg.empty:
        return math.nan
    df_sorted = df.sort_values('p'); df_sorted['rank'] = np.arange(1, len(df_sorted)+1)
    rank_sum_pos = df_sorted[df_sorted.y == 1]['rank'].sum()
    n_pos = len(pos); n_neg = len(neg)
    auc_val = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return float(auc_val)

def coverage_rate(actual: pd.Series, low: pd.Series, high: pd.Series) -> float:
    mask = actual.notna() & low.notna() & high.notna()
    if not mask.any():
        return math.nan
    covered = ((actual >= low) & (actual <= high)) & mask
    return float(covered.mean())

def main():
    preds = _safe_read(OUTPUTS / 'predictions_history_enriched.csv')
    results = load_results()
    if preds.empty or results.empty:
        print('[daily-metrics] Missing inputs; aborting.')
        return
    preds['game_id'] = preds['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    merged = results.merge(preds, on=['date','game_id'], how='left')
    has_mt = 'market_total' in merged.columns and 'actual_total' in merged.columns
    has_sp = 'spread_home' in merged.columns and 'actual_margin' in merged.columns
    if has_mt:
        merged['ou_outcome'] = np.where(merged['market_total'].notna() & merged['actual_total'].notna(), (merged['actual_total'] > merged['market_total']).astype(int), np.nan)
    else:
        merged['ou_outcome'] = np.nan
    if has_sp:
        merged['cover_home_outcome'] = np.where(merged['spread_home'].notna() & merged['actual_margin'].notna(), (merged['actual_margin'] > merged['spread_home']).astype(int), np.nan)
    else:
        merged['cover_home_outcome'] = np.nan
    rows = []
    for date, g in merged.groupby('date', observed=False):
        def eval_method(col, outcome_col):
            if col not in g.columns:
                return {}
            prob = pd.to_numeric(g[col], errors='coerce')
            outcome = g[outcome_col]
            mask = prob.notna() & outcome.notna()
            if not mask.any():
                return {}
            return {
                f'{col}_brier': brier(prob[mask], outcome[mask]),
                f'{col}_log_loss': log_loss(prob[mask], outcome[mask]),
                f'{col}_auc': auc(prob[mask], outcome[mask]),
                f'{col}_rows': int(mask.sum()),
            }
        rec = {'date': date}
        rec.update(eval_method('p_over', 'ou_outcome'))
        rec.update(eval_method('p_home_cover_dist', 'cover_home_outcome'))
        # Interval coverage for totals
        rec['coverage_90'] = coverage_rate(g.get('actual_total'), g.get('pred_total_low_90'), g.get('pred_total_high_90'))
        rec['coverage_95'] = coverage_rate(g.get('actual_total'), g.get('pred_total_low_95'), g.get('pred_total_high_95'))
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values('date')
    out.to_csv(OUTPUTS / 'daily_metrics.csv', index=False)
    print('[daily-metrics] Wrote outputs/daily_metrics.csv')

if __name__ == '__main__':
    main()
