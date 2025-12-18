from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model accuracy vs odds (OU and ATS) for a date or range")
    p.add_argument('--date', help='Single date YYYY-MM-DD')
    p.add_argument('--start-date', help='Start date YYYY-MM-DD (inclusive)')
    p.add_argument('--end-date', help='End date YYYY-MM-DD (inclusive)')
    p.add_argument('--use-closing', action='store_true', help='Prefer closing_total/spread columns when available')
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
    # normalize ids
    for df in (pr, res):
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    return pr, res


def eval_day(date: str, use_closing: bool) -> dict:
    pr, res = load_day(date)
    if pr is None:
        return {'date': date, 'status': 'missing'}

    # Join on game_id if available, else on date+teams
    keys = ['game_id'] if 'game_id' in pr.columns and 'game_id' in res.columns else ['date','home_team','away_team']
    merged = pr.merge(res, on=keys, how='inner', suffixes=('_pred','_res'))

    def get_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce')
        return pd.Series([np.nan] * len(df))

    # Choose prediction columns
    p_total = get_series(merged, ['pred_total_market_blend','pred_total_calibrated','pred_total'])
    p_margin = get_series(merged, ['pred_margin_market_blend','pred_margin_calibrated','pred_margin'])

    if use_closing:
        m_total = get_series(merged, ['closing_total','market_total_res','market_total'])
        spread_home = get_series(merged, ['closing_spread_home','spread_home_res','spread_home'])
    else:
        m_total = get_series(merged, ['market_total_res','market_total','closing_total'])
        spread_home = get_series(merged, ['spread_home_res','spread_home','closing_spread_home'])
    # Actuals may exist on both sides; prefer _res (from results)
    a_total = get_series(merged, ['actual_total_res','actual_total_pred','actual_total'])
    a_margin = get_series(merged, ['actual_margin_res','actual_margin_pred','actual_margin'])

    # OU accuracy: compare sign of (pred_total - market_total) vs (actual_total - market_total)
    ou_mask = p_total.notna() & m_total.notna() & a_total.notna()
    ou_pred_over = (p_total[ou_mask] - m_total[ou_mask]) > 0
    ou_actual_over = (a_total[ou_mask] - m_total[ou_mask]) > 0
    ou_acc = float((ou_pred_over == ou_actual_over).mean()) if ou_mask.any() else None

    # ATS accuracy: predicted margin vs spread_home (home favored negative spread). Predict home_cover if p_margin > -spread_home
    ats_mask = p_margin.notna() & spread_home.notna() & a_margin.notna()
    # If spread_home is negative (home favored), market expects home_margin ~ -spread_home
    # Predict home_cover when predicted home_margin >= -spread_home
    pred_home_cover = p_margin[ats_mask] >= (-spread_home[ats_mask])
    actual_home_cover = a_margin[ats_mask] >= (-spread_home[ats_mask])
    ats_acc = float((pred_home_cover == actual_home_cover).mean()) if ats_mask.any() else None

    return {
        'date': date,
        'n_ou': int(ou_mask.sum()),
        'n_ats': int(ats_mask.sum()),
        'ou_accuracy': ou_acc,
        'ats_accuracy': ats_acc,
    }


def main():
    args = parse_args()
    dates = []
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        s = pd.to_datetime(args.start_date)
        e = pd.to_datetime(args.end_date)
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(s, e, freq='D')]
    else:
        # Use all available daily_results files
        root = Path('outputs/daily_results')
        for p in sorted(root.glob('results_*.csv')):
            m = p.stem.split('_', 1)[1]
            dates.append(m)

    results = []
    for d in dates:
        r = eval_day(d, args.use_closing)
        results.append(r)

    df = pd.DataFrame(results)
    overall = {
        'dates': int(len(df)),
        'ou_total': int(df['n_ou'].fillna(0).sum()),
        'ats_total': int(df['n_ats'].fillna(0).sum()),
        'ou_accuracy_mean': float(df['ou_accuracy'].dropna().mean()) if df['ou_accuracy'].notna().any() else None,
        'ats_accuracy_mean': float(df['ats_accuracy'].dropna().mean()) if df['ats_accuracy'].notna().any() else None,
        'use_closing': bool(args.use_closing),
    }
    print(json.dumps({'overall': overall, 'daily': results}, indent=2))


if __name__ == '__main__':
    main()
