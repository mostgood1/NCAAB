#!/usr/bin/env python
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
RES = OUT / 'daily_results'


def load_enriched(date: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_enriched_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df['game_id'] = df.get('game_id', '').astype(str)
        df['date'] = df.get('date', '').astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def load_results(date: str) -> pd.DataFrame:
    p = RES / f'results_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df['game_id'] = df.get('game_id','').astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def build_frame(days: int) -> pd.DataFrame:
    today = dt.date.today()
    frames = []
    for i in range(1, days+1):
        d = (today - dt.timedelta(days=i)).strftime('%Y-%m-%d')
        enr = load_enriched(d)
        res = load_results(d)
        if enr.empty or res.empty:
            continue
        if {'home_score','away_score'}.issubset(res.columns):
            res['_actual_margin'] = pd.to_numeric(res['home_score'], errors='coerce') - pd.to_numeric(res['away_score'], errors='coerce')
        enr['game_id'] = enr['game_id'].astype(str)
        res['game_id'] = res['game_id'].astype(str)
        df = enr.merge(res[['game_id','_actual_margin']], on='game_id', how='left')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False) -> dict:
    if use_closing:
        hs = pd.to_numeric(df.get('closing_spread_home'), errors='coerce')
    else:
        hs = pd.to_numeric(df.get('home_spread', df.get('spread_home')), errors='coerce')
    mkt_margin = -hs
    pred_blend = pd.to_numeric(df.get('pred_margin_market_blend', df.get('pred_margin')), errors='coerce')
    sigma = pd.to_numeric(df.get('sigma_margin_emp', df.get('sigma_margin_adj')), errors='coerce')
    p_cover = pd.to_numeric(df.get('p_cover_display', df.get('p_home_cover_emp', df.get('p_home_cover'))), errors='coerce')
    actual_margin = pd.to_numeric(df.get('_actual_margin'), errors='coerce')
    mismatch = df.get('flag_market_margin_mismatch')
    if mismatch is None:
        mismatch = pd.Series(False, index=df.index)
    delta = pred_blend.sub(mkt_margin)
    sel = delta.abs().ge(tau)
    if sigma_max > 0 and sigma.notna().any():
        sel = sel & sigma.le(sigma_max)
    if pmin > 0 and p_cover.notna().any():
        sel = sel & p_cover.ge(pmin)
    sel = sel & (~mismatch)
    n = int(sel.sum())
    if n == 0:
        return {'n': 0, 'accuracy': None, 'home_count': 0, 'away_count': 0, 'pushes': 0}
    home_cover = actual_margin.gt(-hs)
    push_mask = actual_margin.eq(-hs)
    pred_home = delta.gt(0)
    valid_mask = sel & (~push_mask)
    if not valid_mask.any():
        return {'n': n, 'accuracy': None, 'home_count': int(pred_home[sel].sum()), 'away_count': int(n - int(pred_home[sel].sum())), 'pushes': int(push_mask[sel].sum())}
    correct = (home_cover == pred_home)
    acc = float(np.mean(correct[valid_mask])) if valid_mask.any() else None
    return {
        'n': n,
        'accuracy': acc,
        'home_count': int(pred_home[sel].sum()),
        'away_count': int(n - int(pred_home[sel].sum())),
        'pushes': int(push_mask[sel].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--use-closing', action='store_true', help='Prefer closing_spread_home for evaluation when available')
    args = ap.parse_args()

    # Load selected ATS policy
    pol = None
    pol_path = OUT / 'metrics' / 'ats_selection_policy.json'
    if pol_path.exists():
        try:
            pol = json.loads(pol_path.read_text(encoding='utf-8')).get('selected')
        except Exception:
            pol = None
    if not pol:
        pol = {'tau': 8.0, 'sigma_max': 0.0, 'pmin': 0.0}

    df = build_frame(args.window_days)
    if df.empty:
        payload = {
            'window_days': args.window_days,
            'policy': pol,
            'metrics': None,
            'status': 'no-data',
            'generated_at': dt.datetime.utcnow().isoformat() + 'Z'
        }
    else:
        met = compute_metrics(df, float(pol['tau']), float(pol.get('sigma_max', 0.0)), float(pol.get('pmin', 0.0)), use_closing=args.use_closing)
        payload = {
            'window_days': args.window_days,
            'policy': pol,
            'metrics': met,
            'status': 'ok',
            'generated_at': dt.datetime.utcnow().isoformat() + 'Z'
        }
    metrics_dir = OUT / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / 'ats_selection_eval.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
