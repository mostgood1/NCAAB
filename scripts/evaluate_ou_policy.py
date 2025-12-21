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
            res['_actual_total'] = pd.to_numeric(res['home_score'], errors='coerce') + pd.to_numeric(res['away_score'], errors='coerce')
        enr['game_id'] = enr['game_id'].astype(str)
        res['game_id'] = res['game_id'].astype(str)
        df = enr.merge(res[['game_id','_actual_total']], on='game_id', how='left')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False) -> dict:
    mkt = pd.to_numeric(df.get('closing_total'), errors='coerce') if use_closing else pd.to_numeric(df.get('market_total'), errors='coerce')
    pred_blend = pd.to_numeric(df.get('pred_total_market_blend', df.get('pred_total')), errors='coerce')
    sigma = pd.to_numeric(df.get('sigma_total_emp', df.get('sigma_total_adj')), errors='coerce')
    p_over = pd.to_numeric(df.get('p_over_display', df.get('p_over_emp')), errors='coerce')
    actual = pd.to_numeric(df.get('_actual_total'), errors='coerce')
    mismatch = df.get('flag_market_total_mismatch')
    if mismatch is None:
        mismatch = pd.Series(False, index=df.index)
    else:
        try:
            mismatch = mismatch.fillna(False).astype(bool)
        except Exception:
            mismatch = mismatch.fillna(False).map(lambda x: bool(x))
    delta = pred_blend.sub(mkt)
    sel = delta.abs().ge(tau)
    if sigma_max > 0 and sigma.notna().any():
        sel = sel & sigma.le(sigma_max)
    if pmin > 0 and p_over.notna().any():
        sel = sel & p_over.ge(pmin)
    sel = sel & (~mismatch)
    n = int(sel.sum())
    if n == 0:
        return {'n': 0, 'accuracy': None, 'over_count': 0, 'under_count': 0}
    went_over = actual.gt(mkt)
    pred_over = delta.gt(0)
    correct = (went_over == pred_over)
    acc = float(np.mean(correct[sel])) if sel.any() else None
    return {
        'n': n,
        'accuracy': acc,
        'over_count': int(pred_over[sel].sum()),
        'under_count': int(n - int(pred_over[sel].sum())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--use-closing', action='store_true', help='Prefer closing_total for evaluation when available')
    args = ap.parse_args()

    # Load selected policy
    pol = None
    pol_path = OUT / 'metrics' / 'ou_selection_policy.json'
    if pol_path.exists():
        try:
            pol = json.loads(pol_path.read_text(encoding='utf-8')).get('selected')
        except Exception:
            pol = None
    if not pol:
        # Fallback to thresholds tau
        tau_fallback = 16.0
        thr_path = OUT / 'metrics' / 'thresholds_vs_odds.json'
        if thr_path.exists():
            try:
                t = json.loads(thr_path.read_text(encoding='utf-8')).get('ou',{}).get('tau')
                if isinstance(t, (int, float)):
                    tau_fallback = float(t)
            except Exception:
                pass
        pol = {'tau': tau_fallback, 'sigma_max': 0.0, 'pmin': 0.0}

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
    out_path = metrics_dir / 'ou_selection_eval.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
