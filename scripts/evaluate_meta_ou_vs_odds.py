#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate meta OU classifier (p_over) vs odds across dates')
    p.add_argument('--use-closing', action='store_true', help='Prefer closing_total/spread when available')
    p.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for Over based on p_over')
    p.add_argument('--start-date')
    p.add_argument('--end-date')
    return p.parse_args()


def load_day(date: str):
    pr_path = Path(f'outputs/predictions_unified_enriched_{date}.csv')
    res_path = Path(f'outputs/daily_results/results_{date}.csv')
    if not (pr_path.exists() and res_path.exists()):
        return None, None
    try:
        pr = pd.read_csv(pr_path)
        res = pd.read_csv(res_path)
    except Exception:
        return None, None
    for df in (pr, res):
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    return pr, res


def get_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce')
    return pd.Series([np.nan] * len(df))


def eval_day(date: str, use_closing: bool, thr: float) -> dict:
    pr, res = load_day(date)
    if pr is None:
        return {'date': date, 'status': 'missing'}
    keys = ['game_id'] if 'game_id' in pr.columns and 'game_id' in res.columns else ['date','home_team','away_team']
    merged = pr.merge(res, on=keys, how='inner', suffixes=('_pred','_res'))

    # Meta p_over candidates
    p_over = get_series(merged, ['p_over_meta_cal','p_over_meta','p_over_display','p_over_emp'])
    # Market lines
    if use_closing:
        m_total = get_series(merged, ['closing_total','market_total_res','market_total'])
    else:
        m_total = get_series(merged, ['market_total_res','market_total','closing_total'])
    # Actuals
    a_total = get_series(merged, ['actual_total_res','actual_total_pred','actual_total'])

    mask = p_over.notna() & m_total.notna() & a_total.notna()
    if not mask.any():
        return {'date': date, 'n_ou': 0, 'ou_accuracy': None}
    ou_pred_over = p_over[mask] > float(thr)
    ou_actual_over = (a_total[mask] - m_total[mask]) > 0
    acc = float((ou_pred_over == ou_actual_over).mean())
    return {'date': date, 'n_ou': int(mask.sum()), 'ou_accuracy': acc}


def main():
    args = parse_args()
    dates: list[str] = []
    if args.start_date and args.end_date:
        s = pd.to_datetime(args.start_date)
        e = pd.to_datetime(args.end_date)
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(s, e, freq='D')]
    else:
        root = Path('outputs/daily_results')
        for p in sorted(root.glob('results_*.csv')):
            dates.append(p.stem.split('_', 1)[1])
    rows = [eval_day(d, args.use_closing, args.threshold) for d in dates]
    df = pd.DataFrame(rows)
    overall = {
        'dates': int(len(df)),
        'ou_total': int(df['n_ou'].fillna(0).sum()),
        'ou_accuracy_mean': float(df['ou_accuracy'].dropna().mean()) if df['ou_accuracy'].notna().any() else None,
        'use_closing': bool(args.use_closing),
        'threshold': float(args.threshold),
    }
    print(json.dumps({'overall': overall, 'daily': rows}, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
