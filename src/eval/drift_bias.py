"""Drift & Bias Diagnostics

Computes:
  - Pace drift: mean predicted total vs trailing 7-day mean
  - Totals bias: mean (pred_total - actual_total)
  - Margin bias per conference (home_margin_pred - actual_margin) aggregated
Outputs JSON to outputs/drift_bias_<date>.json
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pathlib, json, datetime as dt

OUT = pathlib.Path(__file__).resolve().parents[2] / 'outputs'
DATA = pathlib.Path(__file__).resolve().parents[2] / 'data'

def load_unified(date_str: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def load_results(date_str: str) -> pd.DataFrame:
    p = OUT / 'daily_results' / f'results_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def main():
    today = dt.date.today().strftime('%Y-%m-%d')
    df = load_unified(today)
    res = load_results(today)
    if df.empty or res.empty:
        print('No data for diagnostics today; exiting.')
        return
    # Join actual scores
    if {'game_id','home_score','away_score'}.issubset(res.columns):
        res['game_id'] = res['game_id'].astype(str)
        df['game_id'] = df['game_id'].astype(str)
        df = df.merge(res[['game_id','home_score','away_score']], on='game_id', how='left', suffixes=('','_r'))
    actual_total = pd.to_numeric(df.get('home_score'), errors='coerce') + pd.to_numeric(df.get('away_score'), errors='coerce')
    pred_total = pd.to_numeric(df.get('pred_total'), errors='coerce')
    bias_series = pred_total - actual_total
    totals_bias = float(bias_series.dropna().mean()) if bias_series.notna().any() else None
    # Pace drift: compare today's mean pred_total vs trailing mean of past 7 days unified exports
    past_dates = []
    for i in range(1, 8):
        past = (dt.date.today() - dt.timedelta(days=i)).strftime('%Y-%m-%d')
        past_dates.append(past)
    past_vals = []
    for d in past_dates:
        u = load_unified(d)
        if not u.empty and 'pred_total' in u.columns:
            pv = pd.to_numeric(u['pred_total'], errors='coerce')
            if pv.notna().any():
                past_vals.append(pv.mean())
    trailing_mean = float(np.mean(past_vals)) if past_vals else None
    today_mean = float(pred_total.mean()) if pred_total.notna().any() else None
    pace_drift = (today_mean - trailing_mean) if (today_mean is not None and trailing_mean is not None) else None
    # Margin bias per conference
    conf_path = DATA / 'd1_conferences.csv'
    conf_df = pd.read_csv(conf_path) if conf_path.exists() else pd.DataFrame()
    conf_col = next((c for c in ['team','school','name','team_name'] if c in conf_df.columns), None)
    if conf_col and {'home_team','away_team','pred_margin','home_score','away_score'}.issubset(df.columns):
        conf_map = {str(t).lower().strip(): str(c) for t,c in zip(conf_df[conf_col], conf_df.get('conference', conf_df[conf_col]))}
        hm = df['home_team'].astype(str).str.lower().str.strip().map(conf_map)
        am = df['away_team'].astype(str).str.lower().str.strip().map(conf_map)
        actual_margin = pd.to_numeric(df['home_score'], errors='coerce') - pd.to_numeric(df['away_score'], errors='coerce')
        pred_margin = pd.to_numeric(df['pred_margin'], errors='coerce')
        margin_bias = pred_margin - actual_margin
        conf_bias = {}
        for conf in sorted(set(hm.dropna().unique()) | set(am.dropna().unique())):
            mask = (hm == conf) | (am == conf)
            mb = margin_bias[mask]
            if mb.notna().any():
                conf_bias[conf] = float(mb.mean())
    else:
        conf_bias = {}
    out = {
        'date': today,
        'totals_bias': totals_bias,
        'pace_drift': pace_drift,
        'trailing_mean_pred_total': trailing_mean,
        'today_mean_pred_total': today_mean,
        'conference_margin_bias': conf_bias,
        'source_rows': len(df)
    }
    path = OUT / f'drift_bias_{today}.json'
    path.write_text(json.dumps(out, indent=2))
    print(f'Wrote drift/bias diagnostics to {path}')

if __name__ == '__main__':
    main()
