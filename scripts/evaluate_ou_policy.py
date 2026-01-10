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
    # Also include enriched historical probabilities with results to boost coverage
    hist_path = OUT / 'predictions_history_enriched.csv'
    if hist_path.exists():
        try:
            hist = pd.read_csv(hist_path)
            if 'game_id' in hist.columns:
                hist['game_id'] = hist['game_id'].astype(str)
            for i in range(1, days+1):
                d = (today - dt.timedelta(days=i)).strftime('%Y-%m-%d')
                res = load_results(d)
                if res.empty:
                    continue
                if {'home_score','away_score'}.issubset(res.columns):
                    res['_actual_total'] = pd.to_numeric(res['home_score'], errors='coerce') + pd.to_numeric(res['away_score'], errors='coerce')
                res['game_id'] = res['game_id'].astype(str)
                dfh = hist.merge(res[['game_id','_actual_total']], on='game_id', how='inner')
                if not dfh.empty:
                    frames.append(dfh)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    # Merge quantiles history for window to compute p_over_quantile and sigma
    qhist_path = OUT / 'quantiles_history.csv'
    if qhist_path.exists():
        try:
            qh = pd.read_csv(qhist_path)
            qh['game_id'] = qh['game_id'].astype(str)
            qh['date'] = pd.to_datetime(qh['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            window_dates = {(today - dt.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days+1)}
            qh = qh[qh['date'].isin(window_dates)]
            df_all = df_all.merge(qh, on=['game_id'], how='left')
            q10 = pd.to_numeric(df_all.get('q10_total'), errors='coerce')
            q50 = pd.to_numeric(df_all.get('q50_total'), errors='coerce')
            q90 = pd.to_numeric(df_all.get('q90_total'), errors='coerce')
            line = pd.to_numeric(df_all.get('closing_total'), errors='coerce')
            if line.isna().all():
                line = pd.to_numeric(df_all.get('market_total'), errors='coerce')
            cdf = pd.Series(np.nan, index=df_all.index)
            mid1 = line.notna() & q10.notna() & q50.notna() & (line >= q10) & (line <= q50)
            cdf.loc[mid1] = 0.1 + 0.4 * ((line[mid1] - q10[mid1]) / (q50[mid1] - q10[mid1]).replace(0, np.nan))
            mid2 = line.notna() & q50.notna() & q90.notna() & (line > q50) & (line <= q90)
            cdf.loc[mid2] = 0.5 + 0.4 * ((line[mid2] - q50[mid2]) / (q90[mid2] - q50[mid2]).replace(0, np.nan))
            left = line.notna() & q10.notna() & (line < q10)
            cdf.loc[left] = 0.1 * (line[left] / q10[left]).replace(0, np.nan)
            right = line.notna() & q90.notna() & (line > q90)
            cdf.loc[right] = 0.9 + 0.1 * ((line[right] - q90[right]) / q90[right]).replace(0, np.nan)
            df_all['p_over_quantile'] = 1.0 - cdf
            df_all['sigma_total_quantile'] = (q90 - q10) / 2.563103131089201
        except Exception:
            pass
    return df_all


def compute_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False) -> dict:
    # Helper to guarantee Series output
    def _series(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)
    # Market/closing total with robust fallbacks
    mkt_primary = 'closing_total' if use_closing else 'market_total'
    mkt_alt = 'market_total' if use_closing else 'closing_total'
    mkt = _series(mkt_primary)
    if mkt.isna().all():
        mkt = _series('close_total') if (use_closing and 'close_total' in df.columns) else _series(mkt_alt)
        if mkt.isna().all():
            mkt = _series('total')
    # Prediction mean with fallbacks (prefer calibrated model totals first)
    pred_blend = _series('pred_total_calibrated')
    if pred_blend.isna().all():
        pred_blend = _series('pred_total')
        if pred_blend.isna().all():
            pred_blend = _series('pred_total_market_blend')
        if pred_blend.isna().all():
            pred_blend = _series('pred_total_blend')
    # Sigma fallbacks (prefer quantile-derived sigma first)
    sigma = _series('sigma_total_quantile')
    if sigma.isna().all():
        sigma = _series('sigma_total_emp')
        if sigma.isna().all():
            sigma = _series('sigma_total_adj')
            if sigma.isna().all():
                sigma = _series('pred_total_sigma')
    # Probability fallbacks
    p_over = _series('p_over_quantile')
    if p_over.isna().all():
        p_over = _series('p_over')
        if p_over.isna().all():
            p_over = _series('p_over_display')
            if p_over.isna().all():
                p_over = _series('p_over_emp')
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
