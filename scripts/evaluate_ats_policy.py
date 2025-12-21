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


def compute_metrics(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False, prob_side: bool = False, prob_threshold: float = 0.5, prob_side_strict: bool = False) -> dict:
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
    else:
        try:
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
    # Probability-side gating: require home or away probability >= threshold
    if prob_side:
        p_avail = p_cover.notna()
        prob_home_ok = p_cover.ge(prob_threshold)
        prob_away_ok = (1.0 - p_cover).ge(prob_threshold)
        sel_prob = sel & p_avail & (prob_home_ok | prob_away_ok)
        if prob_side_strict:
            # Strict mode: only keep rows with probabilities meeting threshold
            sel_final = sel_prob
        else:
            # Default: include delta-selected rows without probabilities
            sel_delta = sel & (~p_avail)
            sel_final = sel_prob | sel_delta
    else:
        sel_final = sel
    n = int(sel_final.sum())
    if n == 0:
        return {'n': 0, 'accuracy': None, 'home_count': 0, 'away_count': 0, 'pushes': 0}
    home_cover = actual_margin.gt(-hs)
    push_mask = actual_margin.eq(-hs)
    # Side decision: probability-based only when enabled and available; otherwise delta sign
    if prob_side:
        p_avail = p_cover.notna()
        pred_home = pd.Series(False, index=df.index)
        # Use probability-based side when available
        pred_home.loc[p_avail] = p_cover.loc[p_avail].ge(prob_threshold)
        # Fallback to delta sign when probability unavailable (unless strict gating removed these rows)
        if not prob_side_strict:
            pred_home.loc[~p_avail] = delta.loc[~p_avail].gt(0)
    else:
        pred_home = delta.gt(0)
    valid_mask = sel_final & (~push_mask)
    if not valid_mask.any():
        return {'n': n, 'accuracy': None, 'home_count': int(pred_home[sel_final].sum()), 'away_count': int(n - int(pred_home[sel_final].sum())), 'pushes': int(push_mask[sel_final].sum())}
    correct = (home_cover == pred_home)
    acc = float(np.mean(correct[valid_mask])) if valid_mask.any() else None
    return {
        'n': n,
        'accuracy': acc,
        'home_count': int(pred_home[sel_final].sum()),
        'away_count': int(n - int(pred_home[sel_final].sum())),
        'pushes': int(push_mask[sel_final].sum()),
    }


def compute_debug(df: pd.DataFrame, tau: float, sigma_max: float, pmin: float, use_closing: bool = False) -> dict:
    if use_closing:
        hs = pd.to_numeric(df.get('closing_spread_home'), errors='coerce')
    else:
        hs = pd.to_numeric(df.get('home_spread', df.get('spread_home')), errors='coerce')
    mkt_margin = -hs
    pred_blend = pd.to_numeric(df.get('pred_margin_market_blend', df.get('pred_margin')), errors='coerce')
    sigma = pd.to_numeric(df.get('sigma_margin_emp', df.get('sigma_margin_adj')), errors='coerce')
    p_cover = pd.to_numeric(df.get('p_cover_display', df.get('p_home_cover_emp', df.get('p_home_cover'))), errors='coerce')
    mismatch = df.get('flag_market_margin_mismatch')
    if mismatch is None:
        mismatch = pd.Series(False, index=df.index)
    else:
        try:
            mismatch = mismatch.fillna(False).infer_objects(copy=False).astype(bool)
        except Exception:
            mismatch = mismatch.fillna(False).map(lambda x: bool(x))

    delta = pred_blend.sub(mkt_margin)
    delta_valid = delta.dropna()
    total_pos = int(delta_valid.gt(0).sum())
    total_neg = int(delta_valid.lt(0).sum())
    total_zero = int(delta_valid.eq(0).sum())

    sel = delta.abs().ge(tau)
    if sigma_max > 0 and sigma.notna().any():
        sel = sel & sigma.le(sigma_max)
    if pmin > 0 and p_cover.notna().any():
        sel = sel & p_cover.ge(pmin)
    sel = sel & (~mismatch)

    sel_delta = delta[sel].dropna()
    sel_pos = int(sel_delta.gt(0).sum())
    sel_neg = int(sel_delta.lt(0).sum())
    sel_zero = int(sel_delta.eq(0).sum())

    spread_valid = hs.dropna()
    spread_pos = int(spread_valid.gt(0).sum())
    spread_neg = int(spread_valid.lt(0).sum())
    spread_zero = int(spread_valid.eq(0).sum())

    debug = {
        'delta_sign_counts_total': {'pos': total_pos, 'neg': total_neg, 'zero': total_zero},
        'delta_sign_counts_selected': {'pos': sel_pos, 'neg': sel_neg, 'zero': sel_zero},
        'spread_sign_counts': {'pos': spread_pos, 'neg': spread_neg, 'zero': spread_zero},
        'mkt_margin_stats': {
            'mean': (float(mkt_margin.mean()) if mkt_margin.notna().any() else None),
            'min': (float(mkt_margin.min()) if mkt_margin.notna().any() else None),
            'max': (float(mkt_margin.max()) if mkt_margin.notna().any() else None),
        },
        'pred_margin_blend_stats': {
            'mean': (float(pred_blend.mean()) if pred_blend.notna().any() else None),
            'min': (float(pred_blend.min()) if pred_blend.notna().any() else None),
            'max': (float(pred_blend.max()) if pred_blend.notna().any() else None),
        },
        'p_cover_selected_stats': {
            'mean': (float(p_cover[sel].mean()) if p_cover.notna().any() and sel.any() else None),
            'ge_0_5': (int(p_cover[sel].ge(0.5).sum()) if p_cover.notna().any() and sel.any() else 0),
            'lt_0_5': (int(sel.sum()) - int(p_cover[sel].ge(0.5).sum()) if p_cover.notna().any() and sel.any() else 0),
        },
        'selected_n': int(sel.sum()),
    }
    # Include small sample of selected rows
    cols = [
        'date','game_id','closing_spread_home','home_spread','spread_home',
        'pred_margin_market_blend','pred_margin','p_cover_display','p_home_cover_emp','flag_market_margin_mismatch'
    ]
    sample_cols = [c for c in cols if c in df.columns]
    if sample_cols:
        sm = df.loc[sel, sample_cols].head(5).copy()
        # Compute delta for sample too
        sm['_mkt_margin'] = mkt_margin[sel].head(5).values.tolist() if sel.any() else []
        sm['_delta'] = delta[sel].head(5).values.tolist() if sel.any() else []
        try:
            debug['sample_selected'] = sm.to_dict(orient='records')
        except Exception:
            debug['sample_selected'] = []
    else:
        debug['sample_selected'] = []
    return debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--use-closing', action='store_true', help='Prefer closing_spread_home for evaluation when available')
    ap.add_argument('--use-prob-side', action='store_true', help='Use probability-based side (home if p_home_cover>=threshold) when available')
    ap.add_argument('--prob-side-threshold', type=float, default=0.55, help='Threshold for probability-based side; e.g., 0.55 means home if p_home_cover>=0.55, away otherwise')
    ap.add_argument('--prob-side-strict', action='store_true', help='Strict gating: only select rows with probability available and meeting the threshold; no delta fallback when probability missing')
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
        met = compute_metrics(df, float(pol['tau']), float(pol.get('sigma_max', 0.0)), float(pol.get('pmin', 0.0)), use_closing=args.use_closing, prob_side=args.use_prob_side, prob_threshold=args.prob_side_threshold, prob_side_strict=args.prob_side_strict)
        dbg = compute_debug(df, float(pol['tau']), float(pol.get('sigma_max', 0.0)), float(pol.get('pmin', 0.0)), use_closing=args.use_closing)
        payload = {
            'window_days': args.window_days,
            'policy': pol,
            'metrics': met,
            'debug': dbg,
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
