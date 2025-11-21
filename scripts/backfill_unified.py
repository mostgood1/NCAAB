"""Historical backfill for unified predictions.

Loads past games_<date>.csv / predictions_model_<date>.csv pairs and produces
predictions_unified_<date>.csv for any date missing a unified export.

Usage (PowerShell):
  python scripts/backfill_unified.py -Start 2025-11-01 -End 2025-11-19
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import argparse
import pathlib
import datetime as dt
import json

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def unify_for_date(date_str: str) -> dict:
    games_path = OUT / f'games_{date_str}.csv'
    preds_path = OUT / f'predictions_model_{date_str}.csv'
    unified_path = OUT / f'predictions_unified_{date_str}.csv'
    if unified_path.exists():
        return {'date': date_str, 'skipped': True, 'reason': 'already_exists'}
    games = pd.read_csv(games_path) if games_path.exists() else pd.DataFrame()
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    if games.empty and preds.empty:
        return {'date': date_str, 'skipped': True, 'reason': 'no_source'}
    df = pd.DataFrame()
    if not preds.empty:
        df = preds.copy()
    if not games.empty:
        if 'game_id' in games.columns:
            games['game_id'] = games['game_id'].astype(str)
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str)
        if df.empty:
            df = games[['game_id','date','home_team','away_team','start_time']].copy()
            for c in ['pred_total','pred_margin']:
                if c not in df.columns:
                    df[c] = np.nan
        else:
            g_keep = [c for c in ['game_id','date','home_team','away_team','start_time'] if c in games.columns]
            if g_keep:
                df = df.merge(games[g_keep], on='game_id', how='left', suffixes=('','_g'))
                if 'date_g' in df.columns and 'date' in df.columns:
                    df['date'] = df['date'].fillna(df['date_g'])
    # Minimal basis assignment if missing
    if 'pred_total' in df.columns and 'pred_total_model' in df.columns:
        df['pred_total'] = df['pred_total'].where(df['pred_total'].notna(), df['pred_total_model'])
    if 'pred_margin' in df.columns and 'pred_margin_model' in df.columns:
        df['pred_margin'] = df['pred_margin'].where(df['pred_margin'].notna(), df['pred_margin_model'])
    if 'pred_total_basis' not in df.columns:
        df['pred_total_basis'] = 'model_raw'
    if 'pred_margin_basis' not in df.columns:
        df['pred_margin_basis'] = 'model'
    keep_cols = [c for c in ['game_id','date','home_team','away_team','pred_total','pred_margin','pred_total_basis','pred_margin_basis','start_time'] if c in df.columns]
    out_df = df[keep_cols].copy()
    try:
        out_df.to_csv(unified_path, index=False)
        return {'date': date_str, 'ok': True, 'rows': int(len(out_df))}
    except Exception as e:
        return {'date': date_str, 'ok': False, 'error': str(e)}

def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-Start', required=True, help='Start date YYYY-MM-DD')
    ap.add_argument('-End', required=True, help='End date YYYY-MM-DD')
    args = ap.parse_args()
    try:
        d_start = dt.date.fromisoformat(args.Start)
        d_end = dt.date.fromisoformat(args.End)
    except Exception:
        raise SystemExit('Invalid date format. Use YYYY-MM-DD for -Start and -End.')
    results = []
    for d in daterange(d_start, d_end):
        res = unify_for_date(d.strftime('%Y-%m-%d'))
        results.append(res)
        print(json.dumps(res))
    summary = {
        'range': f'{args.Start}..{args.End}',
        'attempted': len(results),
        'created': sum(1 for r in results if r.get('ok')),
        'skipped': sum(1 for r in results if r.get('skipped')),
        'errors': [r for r in results if r.get('ok') is False]
    }
    print('Summary:', json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
