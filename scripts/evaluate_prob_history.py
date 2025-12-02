"""Compute historical probability performance metrics.

Reads:
  - outputs/predictions_history_enriched.csv (probabilities, intervals)
  - outputs/daily_results/results_*.csv (actual totals & margins)

Outputs:
  - outputs/prob_metrics_history.json (aggregate metrics)
  - outputs/prob_reliability_bins.csv (calibration bins for key probabilities)
  - outputs/prob_method_summary.csv (per-method AUC/Brier/log_loss)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json
import math

OUTPUTS = Path("outputs")

RESULTS_GLOB = "daily_results/results_*.csv"
PRED_HISTORY = OUTPUTS / "predictions_history_enriched.csv"

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
    # Deduplicate by date/game_id keep last
    if {'date','game_id'}.issubset(res.columns):
        res = res.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    return res

def load_predictions() -> pd.DataFrame:
    df = _safe_read(PRED_HISTORY)
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    return df

def brier(prob: pd.Series, outcome: pd.Series) -> float:
    s = ((prob - outcome)**2).mean()
    return float(s) if pd.notna(s) else math.nan

def log_loss(prob: pd.Series, outcome: pd.Series, eps: float = 1e-12) -> float:
    p = prob.clip(eps, 1 - eps)
    ll = (-(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))).mean()
    return float(ll) if pd.notna(ll) else math.nan

def auc(prob: pd.Series, outcome: pd.Series) -> float:
    # Manual AUC (Wilcoxon) to avoid sklearn dependency
    df = pd.DataFrame({'p': prob, 'y': outcome}).dropna()
    if df['y'].nunique() != 2:
        return math.nan
    pos = df[df.y == 1].sort_values('p')
    neg = df[df.y == 0].sort_values('p')
    if pos.empty or neg.empty:
        return math.nan
    # Rank all probs
    df_sorted = df.sort_values('p')
    df_sorted['rank'] = np.arange(1, len(df_sorted)+1)
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

def reliability_bins(prob: pd.Series, outcome: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'prob': prob, 'y': outcome}).dropna()
    if df.empty:
        return pd.DataFrame(columns=['bin','p_mean','y_rate','count','abs_gap'])
    df['bin'] = pd.qcut(df['prob'].clip(1e-6, 1-1e-6), q=n_bins, duplicates='drop')
    out = df.groupby('bin', observed=False).agg(p_mean=('prob','mean'), y_rate=('y','mean'), count=('y','size')).reset_index()
    out['abs_gap'] = (out['p_mean'] - out['y_rate']).abs()
    out['bin'] = out['bin'].astype(str)
    return out

def compute_roi(edge: pd.Series, prob: pd.Series, price: pd.Series) -> float:
    # Placeholder ROI: expected value over implied price; assumes American odds in price
    # Convert American odds to decimal
    def american_to_decimal(a):
        try:
            a = float(a)
        except Exception:
            return math.nan
        if a > 0:
            return (a / 100.0) + 1
        else:
            return (100.0 / abs(a)) + 1
    dec = price.apply(american_to_decimal)
    mask = prob.notna() & dec.notna()
    if not mask.any():
        return math.nan
    ev = (prob * (dec - 1) - (1 - prob))
    return float(ev[mask].mean())

def main():
    preds = load_predictions()
    results = load_results()
    if preds.empty or results.empty:
        print("[metrics] Missing predictions or results; aborting.")
        return
    merged = results.merge(preds, on=['date','game_id'], how='left', suffixes=('_res',''))
    # Outcomes: over = actual_total > market_total, cover_home = actual_margin > spread_home (home covers)
    merged['ou_outcome'] = np.where(merged.get('market_total').notna() & merged.get('actual_total').notna(), (merged['actual_total'] > merged['market_total']).astype(int), np.nan)
    merged['cover_home_outcome'] = np.where(merged.get('spread_home').notna() & merged.get('actual_margin').notna(), (merged['actual_margin'] > merged['spread_home']).astype(int), np.nan)
    metrics_rows = []
    def eval_method(col: str, outcome_col: str):
        if col not in merged.columns:
            return
        prob = pd.to_numeric(merged[col], errors='coerce')
        outcome = merged[outcome_col]
        mask = prob.notna() & outcome.notna()
        if not mask.any():
            return
        m = {
            'method': col,
            'outcome': outcome_col,
            'rows': int(mask.sum()),
            'brier': brier(prob[mask], outcome[mask]),
            'log_loss': log_loss(prob[mask], outcome[mask]),
            'auc': auc(prob[mask], outcome[mask]),
        }
        metrics_rows.append(m)
    for method in ['p_over','p_home_cover_dist']:
        oc = 'ou_outcome' if method.startswith('p_over') else 'cover_home_outcome'
        eval_method(method, oc)
    # Interval coverage
    interval_metrics = {
        'coverage_90': coverage_rate(merged.get('actual_total'), merged.get('pred_total_low_90'), merged.get('pred_total_high_90')),
        'coverage_95': coverage_rate(merged.get('actual_total'), merged.get('pred_total_low_95'), merged.get('pred_total_high_95')),
    }
    # Reliability bins
    rel_over = reliability_bins(pd.to_numeric(merged.get('p_over'), errors='coerce'), merged.get('ou_outcome'))
    rel_cover = reliability_bins(pd.to_numeric(merged.get('p_home_cover_dist'), errors='coerce'), merged.get('cover_home_outcome'))
    rel_csv = OUTPUTS / 'prob_reliability_bins.csv'
    rb_all = pd.concat([
        rel_over.assign(method='p_over') if not rel_over.empty else pd.DataFrame(columns=['bin','p_mean','y_rate','count','abs_gap','method']),
        rel_cover.assign(method='p_home_cover_dist') if not rel_cover.empty else pd.DataFrame(columns=['bin','p_mean','y_rate','count','abs_gap','method'])
    ], ignore_index=True)
    rb_all.to_csv(rel_csv, index=False)
    # Method summary CSV
    summary_csv = OUTPUTS / 'prob_method_summary.csv'
    pd.DataFrame(metrics_rows).to_csv(summary_csv, index=False)
    # Aggregate JSON
    out_json = OUTPUTS / 'prob_metrics_history.json'
    aggregate = {
        'rows_results': len(results),
        'rows_predictions': len(preds),
        'merged_rows': len(merged),
        'methods': metrics_rows,
        'intervals': interval_metrics,
        'reliability_bins_path': str(rel_csv),
        'method_summary_path': str(summary_csv),
    }
    out_json.write_text(json.dumps(aggregate, indent=2))
    print(f"[metrics] Wrote metrics -> {out_json}")

if __name__ == '__main__':
    main()
