from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune OU/ATS thresholds vs odds using historical results")
    p.add_argument('--window-days', type=int, default=60, help='Lookback window of days to include')
    p.add_argument('--start-date', type=str, help='Optional start date YYYY-MM-DD')
    p.add_argument('--end-date', type=str, help='Optional end date YYYY-MM-DD')
    p.add_argument('--outputs-dir', type=str, default='outputs', help='Outputs directory')
    p.add_argument('--results-dir', type=str, default='outputs/daily_results', help='Daily results directory')
    return p.parse_args()


def list_dates(results_dir: Path, start: pd.Timestamp | None, end: pd.Timestamp | None, window_days: int) -> List[str]:
    dates = []
    files = sorted(results_dir.glob('results_*.csv'))
    for f in files:
        d = f.stem.split('_', 1)[1]
        try:
            dt = pd.to_datetime(d)
        except Exception:
            continue
        dates.append((d, dt))
    if not dates:
        return []
    if start is None or end is None:
        end = dates[-1][1] if end is None else end
        start = end - pd.Timedelta(days=window_days - 1) if start is None else start
    sel = [d for d, dt in dates if start <= dt <= end]
    return sel


def load_day(outputs_dir: Path, results_dir: Path, date: str) -> pd.DataFrame | None:
    pr_path = outputs_dir / f'predictions_unified_enriched_{date}.csv'
    res_path = results_dir / f'results_{date}.csv'
    if not (pr_path.exists() and res_path.exists()):
        return None
    try:
        pr = pd.read_csv(pr_path)
        res = pd.read_csv(res_path)
    except Exception:
        return None
    for df in (pr, res):
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    keys = ['game_id'] if 'game_id' in pr.columns and 'game_id' in res.columns else ['date','home_team','away_team']
    try:
        m = pr.merge(res, on=keys, how='inner', suffixes=('_pred','_res'))
    except Exception:
        return None
    return m


def accuracy(mask: np.ndarray, pred_sign: np.ndarray, actual_sign: np.ndarray) -> Tuple[float, float]:
    if not mask.any():
        return (np.nan, 0.0)
    acc = (pred_sign[mask] == actual_sign[mask]).mean()
    return float(acc), float(mask.mean())


def tune_ou(df: pd.DataFrame) -> dict:
    p_total = pd.to_numeric(df.get('pred_total_market_blend', df.get('pred_total_calibrated', df.get('pred_total'))), errors='coerce')
    m_total = pd.to_numeric(df.get('market_total_res', df.get('market_total')), errors='coerce')
    a_total = pd.to_numeric(df.get('actual_total'), errors='coerce')
    delta = p_total - m_total
    actual_delta = a_total - m_total
    pred_over = delta > 0
    actual_over = actual_delta > 0
    # grid search thresholds
    taus = np.linspace(0, 25, 51)  # 0.5 step up to 25 points
    best = {'tau': None, 'acc': 0.0, 'coverage': 0.0}
    for t in taus:
        mask = delta.abs() >= t
        acc, cov = accuracy(mask, pred_over, actual_over)
        # prefer higher accuracy; break ties by coverage
        if not np.isnan(acc):
            if acc > best['acc'] or (acc == best['acc'] and cov > best['coverage']):
                best = {'tau': float(t), 'acc': acc, 'coverage': cov}
    return best


def tune_ats(df: pd.DataFrame) -> dict:
    p_margin = pd.to_numeric(df.get('pred_margin_market_blend', df.get('pred_margin_calibrated', df.get('pred_margin'))), errors='coerce')
    spread_home = pd.to_numeric(df.get('spread_home_res', df.get('spread_home')), errors='coerce')
    a_margin = pd.to_numeric(df.get('actual_margin'), errors='coerce')
    # market implies home margin ~ -spread_home
    market_margin = -spread_home
    delta = p_margin - market_margin
    actual_delta = a_margin - market_margin
    pred_home_cover = delta >= 0
    actual_home_cover = actual_delta >= 0
    taus = np.linspace(0, 12, 25)  # 0.5 step up to 12 points
    best = {'tau': None, 'acc': 0.0, 'coverage': 0.0}
    for t in taus:
        mask = delta.abs() >= t
        acc, cov = accuracy(mask, pred_home_cover, actual_home_cover)
        if not np.isnan(acc):
            if acc > best['acc'] or (acc == best['acc'] and cov > best['coverage']):
                best = {'tau': float(t), 'acc': acc, 'coverage': cov}
    return best


def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)

    start = pd.to_datetime(args.start_date) if args.start_date else None
    end = pd.to_datetime(args.end_date) if args.end_date else None
    dates = list_dates(results_dir, start, end, args.window_days)
    frames = []
    for d in dates:
        m = load_day(outputs_dir, results_dir, d)
        if m is not None and len(m) > 0:
            frames.append(m)
    if not frames:
        print(json.dumps({'status': 'error', 'reason': 'no frames'}))
        return
    df = pd.concat(frames, ignore_index=True)

    ou = tune_ou(df)
    ats = tune_ats(df)
    out = {
        'window_days': args.window_days,
        'n_rows': int(len(df)),
        'ou': ou,
        'ats': ats,
    }
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    with open(Path('outputs/metrics/thresholds_vs_odds.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'status': 'ok', **out}))


if __name__ == '__main__':
    main()
