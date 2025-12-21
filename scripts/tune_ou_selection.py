#!/usr/bin/env python
"""Tune O/U selection policy to target >=75% accuracy.

Scans recent days of enriched predictions and daily results to evaluate grids
over delta threshold (|pred_total_market_blend - market_total|), sigma caps,
and minimum over-probability gates. Selects the combination achieving accuracy
>= target (default 0.75) with maximum coverage. Writes metrics to
outputs/metrics/ou_selection_policy.json.
"""
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
        # Join actual totals
        if {'home_score','away_score'}.issubset(res.columns):
            res['_actual_total'] = pd.to_numeric(res['home_score'], errors='coerce') + pd.to_numeric(res['away_score'], errors='coerce')
        enr['game_id'] = enr['game_id'].astype(str)
        res['game_id'] = res['game_id'].astype(str)
        df = enr.merge(res[['game_id','_actual_total']], on='game_id', how='left')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_selection_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False) -> dict:
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
    # Guardrails
    sel = sel & (~mismatch)
    picks = df[sel].copy()
    n = len(picks)
    if n == 0:
        return {'n': 0, 'accuracy': None, 'over_count': 0, 'under_count': 0}
    went_over = actual.gt(mkt)
    pred_over = delta.gt(0)
    correct = (went_over == pred_over)
    acc = float(np.mean(correct[sel])) if sel.any() else None
    return {
        'n': int(n),
        'accuracy': acc,
        'over_count': int(pred_over[sel].sum()),
        'under_count': int(n - int(pred_over[sel].sum())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--target-accuracy', type=float, default=0.75)
    ap.add_argument('--min-coverage', type=int, default=0, help='Minimum number of picks over window to consider a policy viable')
    ap.add_argument('--use-closing', action='store_true', help='Prefer closing_total for tuning/evaluation when available')
    args = ap.parse_args()

    df = build_frame(args.window_days)
    if df.empty:
        print('[warn] No historical enriched+results data available for tuning.')
        return 0
    grid_tau = [10, 12, 14, 16, 18, 20, 22]
    grid_sigma = [0, 12, 10, 9, 8, 7, 6]  # 0 disables sigma gate
    grid_pmin = [0.0, 0.62, 0.65, 0.68, 0.70, 0.75]
    best = None
    evals = []
    for tau in grid_tau:
        for smax in grid_sigma:
            for pmin in grid_pmin:
                met = compute_selection_metrics(df, tau, smax, pmin, use_closing=args.use_closing)
                evals.append({'tau': tau, 'sigma_max': smax, 'pmin': pmin, **met})
    # Selection logic:
    # 1) Max coverage among those meeting target accuracy and min coverage.
    # 2) Else, highest accuracy among those meeting min coverage (tie-break by coverage).
    # 3) Else, highest accuracy among any with n>0 (tie-break by coverage).
    cand_target = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) >= args.min_coverage and float(e['accuracy']) >= args.target_accuracy]
    if cand_target:
        cand_target.sort(key=lambda e: (int(e['n']), float(e['accuracy'])), reverse=True)
        top = cand_target[0]
        best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                 'n': int(top['n']), 'accuracy': float(top['accuracy']),
                 'over_count': int(top.get('over_count', 0)), 'under_count': int(top.get('under_count', 0)) }
    else:
        cand_cov = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) >= args.min_coverage]
        if cand_cov:
            cand_cov.sort(key=lambda e: (float(e['accuracy']), int(e['n'])), reverse=True)
            top = cand_cov[0]
            best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                     'n': int(top['n']), 'accuracy': float(top['accuracy']),
                     'over_count': int(top.get('over_count', 0)), 'under_count': int(top.get('under_count', 0)) }
        else:
            cand_any = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) > 0]
            if cand_any:
                cand_any.sort(key=lambda e: (float(e['accuracy']), int(e['n'])), reverse=True)
                top = cand_any[0]
                best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                         'n': int(top['n']), 'accuracy': float(top['accuracy']),
                         'over_count': int(top.get('over_count', 0)), 'under_count': int(top.get('under_count', 0)) }
    # If no combination hit the target, select a conservative fallback
    if best is None:
        fallback_candidates = [
            (18, 9, 0.70),
            (16, 9, 0.70),
            (16, 9, 0.68),
            (16, 9, 0.65),
            (14, 9, 0.70),
        ]
        fb_best = None
        fb_best_acc = -1.0
        fb_best_cov = -1
        for tau, smax, pmin in fallback_candidates:
            met = compute_selection_metrics(df, tau, smax, pmin)
            if met['n'] > 0 and met['accuracy'] is not None:
                acc = float(met['accuracy'])
                cov = int(met['n'])
                # Prefer higher accuracy; tie-break by coverage
                if acc > fb_best_acc or (acc == fb_best_acc and cov > fb_best_cov):
                    fb_best = {'tau': tau, 'sigma_max': smax, 'pmin': pmin, **met}
                    fb_best_acc = acc
                    fb_best_cov = cov
        # If all fallbacks failed to select any picks, choose the best from the evaluated grid
        if fb_best is None and evals:
            # Maximize accuracy with at least 1 pick; tie-break by coverage
            viable = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) > 0]
            if viable:
                viable.sort(key=lambda e: (float(e['accuracy']), int(e['n'])), reverse=True)
                top = viable[0]
                fb_best = {
                    'tau': float(top['tau']),
                    'sigma_max': float(top['sigma_max']),
                    'pmin': float(top['pmin']),
                    'n': int(top['n']),
                    'accuracy': float(top['accuracy']),
                    'over_count': int(top.get('over_count', 0)),
                    'under_count': int(top.get('under_count', 0)),
                }
        best = fb_best
        # Final guard: still None? Fall back to tuned OU tau (thresholds_vs_odds) or a sane default
        if best is None:
            tau_fallback = 16.0
            try:
                thr_path = OUT / 'metrics' / 'thresholds_vs_odds.json'
                if thr_path.exists():
                    _thr = json.loads(thr_path.read_text(encoding='utf-8'))
                    t = _thr.get('ou', {}).get('tau')
                    if isinstance(t, (int, float)):
                        tau_fallback = float(t)
            except Exception:
                pass
            met = compute_selection_metrics(df, tau_fallback, 0.0, 0.0, use_closing=args.use_closing)
            best = {
                'tau': float(tau_fallback),
                'sigma_max': 0.0,
                'pmin': 0.0,
                'n': int(met.get('n') or 0),
                'accuracy': (float(met['accuracy']) if met.get('accuracy') is not None else None),
                'over_count': int(met.get('over_count') or 0),
                'under_count': int(met.get('under_count') or 0),
            }

    payload = {
        'window_days': args.window_days,
        'target_accuracy': args.target_accuracy,
        'grid': evals,
        'selected': best,
        'generated_at': dt.datetime.utcnow().isoformat() + 'Z'
    }
    metrics_dir = OUT / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / 'ou_selection_policy.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if best:
        acc_str = f"{best['accuracy']:.3f}" if isinstance(best.get('accuracy'), (int, float)) else str(best.get('accuracy'))
        print(f"[policy] OU selection: tau={best['tau']} sigma_max={best['sigma_max']} pmin={best['pmin']} n={best['n']} acc={acc_str}")
    else:
        print('[policy] No viable selection; metrics written with selected=null.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
