#!/usr/bin/env python
"""Tune ATS selection policy to target >=75% accuracy.

Scans recent days of enriched predictions and daily results to evaluate grids
over delta threshold (|pred_margin_market_blend - implied_market_margin|),
sigma caps, and minimum cover-probability gates. Selects the combination
achieving accuracy >= target with coverage floors. Writes metrics to
outputs/metrics/ats_selection_policy.json.
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
        if {'home_score','away_score'}.issubset(res.columns):
            res['_actual_margin'] = pd.to_numeric(res['home_score'], errors='coerce') - pd.to_numeric(res['away_score'], errors='coerce')
        enr['game_id'] = enr['game_id'].astype(str)
        res['game_id'] = res['game_id'].astype(str)
        df = enr.merge(res[['game_id','_actual_margin']], on='game_id', how='left')
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_selection_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False, prob_side: bool = False, prob_threshold: float = 0.5, prob_side_strict: bool = False) -> dict:
    # Market-implied margin is -home_spread
    if use_closing:
        hs = pd.to_numeric(df.get('closing_spread_home'), errors='coerce')
    else:
        # favor 'home_spread' then 'spread_home'
        hs = pd.to_numeric(df.get('home_spread', df.get('spread_home')), errors='coerce')
    mkt_margin = -hs
    pred_blend = pd.to_numeric(df.get('pred_margin_market_blend', df.get('pred_margin')), errors='coerce')
    sigma = pd.to_numeric(df.get('sigma_margin_emp', df.get('sigma_margin_adj')), errors='coerce')
    p_cover = pd.to_numeric(df.get('p_cover_display', df.get('p_home_cover_emp', df.get('p_home_cover'))), errors='coerce')
    actual_margin = pd.to_numeric(df.get('_actual_margin'), errors='coerce')
    mismatch = df.get('flag_market_margin_mismatch')
    if mismatch is None:
        mismatch = pd.Series(False, index=df.index)
    else:
        try:
            # Silence FutureWarning by inferring objects before astype
            mismatch = mismatch.fillna(False).infer_objects(copy=False).astype(bool)
        except Exception:
            mismatch = mismatch.fillna(False).map(lambda x: bool(x))
    delta = pred_blend.sub(mkt_margin)
    sel = delta.abs().ge(tau)
    if sigma_max > 0 and sigma.notna().any():
        sel = sel & sigma.le(sigma_max)
    if pmin > 0 and p_cover.notna().any():
        sel = sel & p_cover.ge(pmin)
    sel = sel & (~mismatch)

    # Probability-side gating: require either home or away probability >= threshold
    if prob_side:
        p_avail = p_cover.notna()
        prob_home_ok = p_cover.ge(prob_threshold)
        prob_away_ok = (1.0 - p_cover).ge(prob_threshold)
        sel_prob = sel & p_avail & (prob_home_ok | prob_away_ok)
        if prob_side_strict:
            sel_final = sel_prob
        else:
            sel_delta = sel & (~p_avail)
            sel_final = sel_prob | sel_delta
    else:
        sel_final = sel

    picks = df[sel_final].copy()
    n = len(picks)
    if n == 0:
        return {'n': 0, 'accuracy': None, 'home_count': 0, 'away_count': 0, 'pushes': 0}
    # Actual home cover when actual_margin > -home_spread (push at equality)
    home_cover = actual_margin.gt(-hs)
    push_mask = actual_margin.eq(-hs)
    # Side decision: probability-based only when enabled and available; otherwise delta sign
    if prob_side:
        p_avail = p_cover.notna()
        pred_home = pd.Series(False, index=df.index)
        # Use probability-based side when available
        pred_home.loc[p_avail] = p_cover.loc[p_avail].ge(prob_threshold)
        # Fallback to delta sign when probability unavailable (unless strict)
        if not prob_side_strict:
            pred_home.loc[~p_avail] = delta.loc[~p_avail].gt(0)
    else:
        pred_home = delta.gt(0)
    valid_mask = sel_final & (~push_mask)
    if not valid_mask.any():
        return {'n': int(sel_final.sum()), 'accuracy': None, 'home_count': int(pred_home[sel_final].sum()), 'away_count': int(int(sel_final.sum()) - int(pred_home[sel_final].sum())), 'pushes': int(push_mask[sel_final].sum())}
    correct = (home_cover == pred_home)
    acc = float(np.mean(correct[valid_mask])) if valid_mask.any() else None
    return {
        'n': int(sel_final.sum()),
        'accuracy': acc,
        'home_count': int(pred_home[sel_final].sum()),
        'away_count': int(int(sel_final.sum()) - int(pred_home[sel_final].sum())),
        'pushes': int(push_mask[sel_final].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--target-accuracy', type=float, default=0.75)
    ap.add_argument('--min-coverage', type=int, default=0)
    ap.add_argument('--use-closing', action='store_true', help='Prefer closing_spread_home for tuning/evaluation when available')
    ap.add_argument('--use-prob-side', action='store_true', help='Use probability-based side (home if p_home_cover>=threshold) when available')
    ap.add_argument('--prob-side-threshold', type=float, default=0.55, help='Threshold for probability-based side; e.g., 0.55 means home if p_home_cover>=0.55, away otherwise')
    ap.add_argument('--prob-side-strict', action='store_true', help='Strict gating: only select rows with probability available and meeting the threshold; no delta fallback when probability missing')
    args = ap.parse_args()

    df = build_frame(args.window_days)
    if df.empty:
        print('[warn] No historical enriched+results data available for ATS tuning.')
        return 0
    grid_tau = [6, 7, 8, 9, 10, 12]
    grid_sigma = [0, 12, 10, 9, 8, 7, 6]  # 0 disables sigma gate
    grid_pmin = [0.0, 0.60, 0.62, 0.65, 0.68, 0.70]
    evals = []
    for tau in grid_tau:
        for smax in grid_sigma:
            for pmin in grid_pmin:
                met = compute_selection_metrics(df, tau, smax, pmin, use_closing=args.use_closing, prob_side=args.use_prob_side, prob_threshold=args.prob_side_threshold, prob_side_strict=args.prob_side_strict)
                evals.append({'tau': tau, 'sigma_max': smax, 'pmin': pmin, **met})
    best = None
    # Selection logic mirrors OU
    cand_target = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) >= args.min_coverage and float(e['accuracy']) >= args.target_accuracy]
    if cand_target:
        cand_target.sort(key=lambda e: (int(e['n']), float(e['accuracy'])), reverse=True)
        top = cand_target[0]
        best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                 'n': int(top['n']), 'accuracy': float(top['accuracy']),
                 'home_count': int(top.get('home_count', 0)), 'away_count': int(top.get('away_count', 0)), 'pushes': int(top.get('pushes', 0)) }
    else:
        cand_cov = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) >= args.min_coverage]
        if cand_cov:
            cand_cov.sort(key=lambda e: (float(e['accuracy']), int(e['n'])), reverse=True)
            top = cand_cov[0]
            best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                     'n': int(top['n']), 'accuracy': float(top['accuracy']),
                     'home_count': int(top.get('home_count', 0)), 'away_count': int(top.get('away_count', 0)), 'pushes': int(top.get('pushes', 0)) }
        else:
            cand_any = [e for e in evals if e.get('accuracy') is not None and e.get('n', 0) > 0]
            if cand_any:
                cand_any.sort(key=lambda e: (float(e['accuracy']), int(e['n'])), reverse=True)
                top = cand_any[0]
                best = { 'tau': float(top['tau']), 'sigma_max': float(top['sigma_max']), 'pmin': float(top['pmin']),
                         'n': int(top['n']), 'accuracy': float(top['accuracy']),
                         'home_count': int(top.get('home_count', 0)), 'away_count': int(top.get('away_count', 0)), 'pushes': int(top.get('pushes', 0)) }
    if best is None:
        # Conservative fallbacks
        fallback_candidates = [
            (9, 9, 0.68),
            (8, 9, 0.68),
            (8, 9, 0.65),
            (10, 9, 0.65),
        ]
        fb_best = None
        fb_best_acc = -1.0
        fb_best_cov = -1
        for tau, smax, pmin in fallback_candidates:
            met = compute_selection_metrics(df, tau, smax, pmin, use_closing=args.use_closing, prob_side=args.use_prob_side, prob_threshold=args.prob_side_threshold, prob_side_strict=args.prob_side_strict)
            if met['n'] > 0 and met['accuracy'] is not None:
                acc = float(met['accuracy']); cov = int(met['n'])
                if acc > fb_best_acc or (acc == fb_best_acc and cov > fb_best_cov):
                    fb_best = {'tau': tau, 'sigma_max': smax, 'pmin': pmin, **met}
                    fb_best_acc = acc; fb_best_cov = cov
        if fb_best is None and evals:
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
                    'home_count': int(top.get('home_count', 0)),
                    'away_count': int(top.get('away_count', 0)),
                    'pushes': int(top.get('pushes', 0)),
                }
        best = fb_best
        if best is None:
            tau_fallback = 8.0
            met = compute_selection_metrics(df, tau_fallback, 0.0, 0.0, use_closing=args.use_closing, prob_side=args.use_prob_side, prob_threshold=args.prob_side_threshold, prob_side_strict=args.prob_side_strict)
            best = {
                'tau': float(tau_fallback),
                'sigma_max': 0.0,
                'pmin': 0.0,
                'n': int(met.get('n') or 0),
                'accuracy': (float(met['accuracy']) if met.get('accuracy') is not None else None),
                'home_count': int(met.get('home_count') or 0),
                'away_count': int(met.get('away_count') or 0),
                'pushes': int(met.get('pushes') or 0),
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
    out_path = metrics_dir / 'ats_selection_policy.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if best:
        acc_str = f"{best['accuracy']:.3f}" if isinstance(best.get('accuracy'), (int, float)) else str(best.get('accuracy'))
        print(f"[policy] ATS selection: tau={best['tau']} sigma_max={best['sigma_max']} pmin={best['pmin']} n={best['n']} acc={acc_str}")
    else:
        print('[policy] No viable ATS selection; metrics written with selected=null.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
