#!/usr/bin/env python
from __future__ import annotations
import argparse, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


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


def select_ats(df: pd.DataFrame, tau: float, use_closing: bool, prob_side: bool, prob_threshold: float, prob_side_strict: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # Inputs
    hs = pd.to_numeric(df.get('closing_spread_home') if use_closing else df.get('home_spread', df.get('spread_home')), errors='coerce')
    mkt_margin = -hs
    pred_blend = pd.to_numeric(df.get('pred_margin_market_blend', df.get('pred_margin')), errors='coerce')
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
    sel = delta.abs().ge(tau) & (~mismatch)

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

    if not sel_final.any():
        return pd.DataFrame()

    # Side decision
    if prob_side:
        p_avail = p_cover.notna()
        pred_home = pd.Series(False, index=df.index)
        pred_home.loc[p_avail] = p_cover.loc[p_avail].ge(prob_threshold)
        if not prob_side_strict:
            pred_home.loc[~p_avail] = delta.loc[~p_avail].gt(0)
    else:
        pred_home = delta.gt(0)

    picks = df[sel_final].copy()
    picks['_mkt_margin'] = mkt_margin[sel_final]
    picks['_pred_margin_blend'] = pred_blend[sel_final]
    picks['_delta'] = delta[sel_final]
    picks['_p_cover'] = p_cover[sel_final]
    picks['ats_side'] = np.where(pred_home[sel_final], 'home', 'away')
    picks['ats_reason'] = np.where(p_cover[sel_final].notna(), 'prob', 'delta') if prob_side else 'delta'

    cols = ['date','game_id','home_team','away_team','closing_spread_home','home_spread','spread_home',
            'ats_side','ats_reason','_delta','_pred_margin_blend','_mkt_margin','_p_cover']
    present = [c for c in cols if c in picks.columns]
    present += [c for c in ['ats_side','ats_reason','_delta','_pred_margin_blend','_mkt_margin','_p_cover'] if c in picks.columns]
    return picks[present]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, default=None, help='Date in YYYY-MM-DD; default today')
    ap.add_argument('--use-closing', action='store_true')
    ap.add_argument('--prob-side', action='store_true')
    ap.add_argument('--prob-side-strict', action='store_true')
    ap.add_argument('--prob-side-threshold', type=float, default=0.55)
    ap.add_argument('--tau', type=float, default=6.0)
    args = ap.parse_args()

    date = args.date or dt.date.today().strftime('%Y-%m-%d')
    df = load_enriched(date)
    if df.empty:
        print(f'[warn] No enriched predictions for {date}.')
        return 0
    picks = select_ats(df, args.tau, args.use_closing, args.prob_side, args.prob_side_threshold, args.prob_side_strict)
    out_dir = OUT / 'picks'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'ats_picks_{date}.csv'
    if picks.empty:
        print('[info] No ATS picks selected.')
    else:
        picks.to_csv(out_path, index=False)
        print(f'[ok] Wrote {len(picks)} ATS picks to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
