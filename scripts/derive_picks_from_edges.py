import argparse
import os
import pandas as pd
import numpy as np

OUT = os.path.join(os.getcwd(), 'outputs')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    args = ap.parse_args()
    date = args.date
    src = os.path.join(OUT, f'align_period_{date}_edges.csv')
    if not os.path.exists(src):
        raise SystemExit(f'edges file not found: {src}')
    edges = pd.read_csv(src, dtype=str, low_memory=False)
    num_cols = [
        'edge_total','edge_margin','total','over_price','under_price',
        'home_spread','home_spread_price','away_spread','away_spread_price',
        'moneyline_home','moneyline_away','pred_total','pred_margin'
    ]
    for c in num_cols:
        if c in edges.columns:
            edges[c] = pd.to_numeric(edges[c], errors='coerce')
    # Keep full game period
    if 'period' in edges.columns:
        try:
            edges = edges[edges['period'].astype(str).str.lower() == 'full_game']
        except Exception:
            pass
    # Totals
    tots = pd.DataFrame()
    if 'market' in edges.columns:
        m = edges['market'].astype(str).str.lower() == 'totals'
        tots = edges[m].copy()
    if not tots.empty:
        def _tot_side(r: pd.Series) -> str:
            pt = float(r.get('pred_total')) if r.get('pred_total') is not None else np.nan
            ln = float(r.get('total')) if r.get('total') is not None else np.nan
            if np.isfinite(pt) and np.isfinite(ln):
                return 'Over' if pt > ln else 'Under'
            return 'Over'
        tots['bet'] = tots.apply(_tot_side, axis=1)
        tots['line'] = tots['total']
        tots['price'] = tots.apply(lambda r: r['over_price'] if str(r.get('bet')).lower()=='over' else r['under_price'], axis=1)
        tots['edge'] = tots['edge_total'].abs() if 'edge_total' in tots.columns else None
        tots['rec_type'] = 'Totals'; tots['rec_code'] = 'OU'
    # Spreads
    sprs = pd.DataFrame()
    if 'market' in edges.columns:
        m = edges['market'].astype(str).str.lower() == 'spreads'
        sprs = edges[m].copy()
    if not sprs.empty:
        def _spr_side(r: pd.Series) -> str:
            em = float(r.get('edge_margin')) if r.get('edge_margin') is not None else np.nan
            if np.isfinite(em):
                return 'home' if em >= 0 else 'away'
            return 'home'
        sprs['bet'] = sprs.apply(_spr_side, axis=1)
        sprs['line'] = sprs.apply(lambda r: (r['home_spread'] if str(r.get('bet')).lower()=='home' else r['away_spread']), axis=1)
        sprs['price'] = sprs.apply(lambda r: (r['home_spread_price'] if str(r.get('bet')).lower()=='home' else r['away_spread_price']), axis=1)
        sprs['edge'] = sprs['edge_margin'].abs() if 'edge_margin' in sprs.columns else None
        sprs['rec_type'] = 'Spread'; sprs['rec_code'] = 'ATS'
    # Moneyline (optional)
    mls = pd.DataFrame()
    if 'market' in edges.columns and {'home_ml_ev','away_ml_ev'}.issubset(edges.columns):
        mls = edges[edges['market'].astype(str).str.lower() == 'h2h'].copy()
        if not mls.empty:
            for c in ['home_ml_ev','away_ml_ev']:
                mls[c] = pd.to_numeric(mls[c], errors='coerce')
            def _ml_side(r: pd.Series) -> str:
                hv = float(r.get('home_ml_ev') or 0.0)
                av = float(r.get('away_ml_ev') or 0.0)
                return 'home' if hv >= av else 'away'
            mls['bet'] = mls.apply(_ml_side, axis=1)
            mls['line'] = None
            mls['price'] = mls.apply(lambda r: (r['moneyline_home'] if str(r.get('bet')).lower()=='home' else r['moneyline_away']), axis=1)
            mls['edge'] = mls.apply(lambda r: max(float(r.get('home_ml_ev') or 0.0), float(r.get('away_ml_ev') or 0.0)), axis=1)
            mls['rec_type'] = 'Moneyline'; mls['rec_code'] = 'ML'
    # Concatenate
    picks_fb = pd.concat([tots, sprs, mls], ignore_index=True) if (not tots.empty or not sprs.empty or not mls.empty) else pd.DataFrame()
    if picks_fb.empty:
        raise SystemExit('no picks derived from edges')
    # Positive edges only
    try:
        picks_fb = picks_fb[pd.to_numeric(picks_fb['edge'], errors='coerce') > 0]
    except Exception:
        pass
    # Normalize columns for display
    if 'game_id' in picks_fb.columns:
        picks_fb['game_id'] = picks_fb['game_id'].astype(str)
    if 'date_game' in picks_fb.columns and 'date' not in picks_fb.columns:
        picks_fb.rename(columns={'date_game':'date'}, inplace=True)
    if 'home_team_name' in picks_fb.columns and 'home_team' not in picks_fb.columns:
        picks_fb.rename(columns={'home_team_name':'home_team','away_team_name':'away_team'}, inplace=True)
    keep_cols = [
        'game_id','date','home_team','away_team','book','market','period',
        'line','price','edge','pred_total','pred_margin','edge_total','edge_margin',
        'home_spread','home_spread_price','away_spread','away_spread_price',
        'total','over_price','under_price',
        'start_time_iso','start_tz_abbr','start_time','display_date','start_time_local'
    ]
    picks = picks_fb[[c for c in keep_cols if c in picks_fb.columns]].copy()
    # Write picks_raw.csv
    out = os.path.join(OUT, 'picks_raw.csv')
    picks.to_csv(out, index=False)
    print(f'Wrote {out} rows={len(picks)}')

if __name__ == '__main__':
    main()
