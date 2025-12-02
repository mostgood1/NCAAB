"""Evaluate raw vs conformal interval coverage and CRPS deltas for latest date.

Reads:
  - outputs/quantiles_history.csv
  - outputs/quantiles_conformal_today.csv
  - outputs/daily_results/results_*.csv

Writes:
  - outputs/conformal_metrics_<date>.csv with columns:
      date, raw_covered_80_total, conf_covered_80_total, raw_width_total, conf_width_total,
            raw_covered_80_margin, conf_covered_80_margin, raw_width_margin, conf_width_margin
  - outputs/conformal_metrics_<date>.json (same content JSON)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json

OUTPUTS = Path('outputs')

def _safe_read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob('daily_results/results_*.csv'):
        df = _safe_read(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def main():
    qhist = _safe_read(OUTPUTS / 'quantiles_history.csv')
    qconf = _safe_read(OUTPUTS / 'quantiles_conformal_today.csv')
    res = load_results()
    if qhist.empty or qconf.empty or res.empty:
        print('[conformal-eval] Missing inputs; aborting.')
        return
    for col in ('game_id','date'):
        if col in qhist.columns:
            qhist[col] = qhist[col].astype(str).str.replace(r'\\.0$','', regex=True)
        if col in qconf.columns:
            qconf[col] = qconf[col].astype(str).str.replace(r'\\.0$','', regex=True)
        if col in res.columns:
            res[col] = res[col].astype(str).str.replace(r'\\.0$','', regex=True)
    # Choose latest date that exists in both results and quantile history
    q_dates = set(qhist['date'].astype(str).unique()) - {'' , 'nan', 'None'}
    r_dates = set(res['date'].astype(str).unique()) - {'' , 'nan', 'None'}
    overlap = sorted(q_dates.intersection(r_dates))
    if not overlap:
        print('[conformal-eval] No overlapping dates; aborting.')
        return
    latest = overlap[-1]
    raw_today = qhist[qhist['date'].astype(str) == latest]
    conf_today = qconf[qconf['date'].astype(str) == latest]
    base = res[res['date'].astype(str) == latest]
    if raw_today.empty or base.empty:
        print(f'[conformal-eval] Missing data for latest={latest}; aborting.')
        return
    df_raw = base.merge(raw_today[['game_id','q10_total','q90_total','q10_margin','q90_margin']], on='game_id', how='left')
    df_conf = base.merge(conf_today[['game_id','total_c10','total_c90','margin_c10','margin_c90']], on='game_id', how='left')
    def cov80(actual, low, high):
        a = pd.to_numeric(actual, errors='coerce')
        l = pd.to_numeric(low, errors='coerce')
        h = pd.to_numeric(high, errors='coerce')
        return float(((a >= l) & (a <= h)).mean()) if (np.isfinite(a).sum() > 0 and np.isfinite(l).sum() > 0 and np.isfinite(h).sum() > 0) else np.nan
    raw_cov_total = cov80(df_raw.get('actual_total'), df_raw.get('q10_total'), df_raw.get('q90_total'))
    conf_cov_total = cov80(df_conf.get('actual_total'), df_conf.get('total_c10'), df_conf.get('total_c90'))
    raw_cov_margin = cov80(df_raw.get('actual_margin'), df_raw.get('q10_margin'), df_raw.get('q90_margin'))
    conf_cov_margin = cov80(df_conf.get('actual_margin'), df_conf.get('margin_c10'), df_conf.get('margin_c90'))
    def width(l, h):
        l = pd.to_numeric(l, errors='coerce')
        h = pd.to_numeric(h, errors='coerce')
        return float((h - l).mean()) if (np.isfinite(l).sum() > 0 and np.isfinite(h).sum() > 0) else np.nan
    raw_w_total = width(df_raw.get('q10_total'), df_raw.get('q90_total'))
    conf_w_total = width(df_conf.get('total_c10'), df_conf.get('total_c90'))
    raw_w_margin = width(df_raw.get('q10_margin'), df_raw.get('q90_margin'))
    conf_w_margin = width(df_conf.get('margin_c10'), df_conf.get('margin_c90'))
    out = pd.DataFrame([{ 'date': latest,
        'raw_covered_80_total': raw_cov_total,
        'conf_covered_80_total': conf_cov_total,
        'raw_width_total': raw_w_total,
        'conf_width_total': conf_w_total,
        'raw_covered_80_margin': raw_cov_margin,
        'conf_covered_80_margin': conf_cov_margin,
        'raw_width_margin': raw_w_margin,
        'conf_width_margin': conf_w_margin }])
    csv_path = OUTPUTS / f'conformal_metrics_{latest}.csv'
    out.to_csv(csv_path, index=False)
    (OUTPUTS / f'conformal_metrics_{latest}.json').write_text(json.dumps(out.to_dict(orient='records')[0], indent=2))
    print('[conformal-eval] Wrote', csv_path)

if __name__ == '__main__':
    main()
