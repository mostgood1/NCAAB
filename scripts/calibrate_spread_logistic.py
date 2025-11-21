"""Calibrate logistic scale constant K for spread win probability approximation.

We assume model win prob for home covering spread is:
  p_cover = 1 / (1 + exp(- (pred_margin_model - closing_spread_home) / K))

We search K over a grid to minimize log loss vs observed home cover outcomes.
Outputs JSON: outputs/calibration_spread_logistic.json
Optionally filtered by --min-rows and --max-days.
Usage:
  python scripts/calibrate_spread_logistic.py --max-days 90 --min-rows 300
"""
from __future__ import annotations
import argparse, json, math, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _log_loss(p: float, y: int) -> float:
    p = min(max(p, 1e-9), 1-1e-9)
    return - (y * math.log(p) + (1-y) * math.log(1-p))

def gather_rows(max_days: int | None) -> pd.DataFrame:
    rows = []
    today = dt.date.today()
    # Discover prediction model files
    for pred_path in sorted(OUT.glob('predictions_model_*.csv')):
        try:
            date_part = pred_path.stem.replace('predictions_model_','')
            date_obj = dt.date.fromisoformat(date_part)
        except Exception:
            continue
        if max_days is not None and (today - date_obj).days > max_days:
            continue
        preds = _safe_csv(pred_path)
        if preds.empty:
            continue
        if 'game_id' not in preds.columns:
            continue
        # Standardize margin column
        if 'pred_margin_model' in preds.columns:
            preds['pred_margin'] = preds['pred_margin_model']
        # Load games or daily_results for closing spread + final scores
        g_path = OUT / f'games_{date_part}.csv'
        dr_path = OUT / 'daily_results' / f'results_{date_part}.csv'
        games = _safe_csv(dr_path)
        if games.empty:
            games = _safe_csv(g_path)
        if games.empty:
            continue
        if 'game_id' not in games.columns:
            games['game_id'] = games.get('id', pd.Series(range(len(games)))).astype(str)
        else:
            games['game_id'] = games['game_id'].astype(str)
        preds['game_id'] = preds['game_id'].astype(str)
        merged = preds.merge(games, on='game_id', how='inner', suffixes=('_p','_g'))
        if merged.empty:
            continue
        # Require columns
        needed = ['pred_margin','home_score','away_score']
        if not all(c in merged.columns for c in needed):
            continue
        # Determine closing spread: prefer closing_spread_home else existing closing artifact
        if 'closing_spread_home' not in merged.columns:
            # Attempt merge with games_with_closing
            cpath = OUT / f'games_with_closing_{date_part}.csv'
            if cpath.exists():
                cdf = _safe_csv(cpath)
                if not cdf.empty and 'game_id' in cdf.columns and 'closing_spread_home' in cdf.columns:
                    cdf['game_id'] = cdf['game_id'].astype(str)
                    merged = merged.merge(cdf[['game_id','closing_spread_home']], on='game_id', how='left')
        if 'closing_spread_home' not in merged.columns:
            continue
        # Build calibration rows
        margin_pred = pd.to_numeric(merged['pred_margin'], errors='coerce')
        margin_final = pd.to_numeric(merged['home_score'], errors='coerce') - pd.to_numeric(merged['away_score'], errors='coerce')
        spread = pd.to_numeric(merged['closing_spread_home'], errors='coerce')
        cover = (margin_final + spread) > 0  # home covers
        for mp, mf, sp, cv in zip(margin_pred, margin_final, spread, cover):
            if pd.isna(mp) or pd.isna(mf) or pd.isna(sp):
                continue
            rows.append({'date': date_part, 'margin_pred': mp, 'closing_spread': sp, 'margin_final': mf, 'home_cover': int(cv)})
    # Also consider daily_results as a source when standalone model prediction files are not present
    try:
        for dr in sorted((OUT / 'daily_results').glob('results_*.csv')):
            try:
                date_part = dr.stem.replace('results_','')
                date_obj = dt.date.fromisoformat(date_part)
            except Exception:
                continue
            if max_days is not None and (today - date_obj).days > max_days:
                continue
            ddf = _safe_csv(dr)
            if ddf.empty or 'game_id' not in ddf.columns:
                continue
            # Need pred_margin and scores
            if not {'pred_margin','home_score','away_score'}.issubset(ddf.columns):
                continue
            ddf['game_id'] = ddf['game_id'].astype(str)
            # Attach closing_spread via games_with_closing for the date
            cpath = OUT / f'games_with_closing_{date_part}.csv'
            cdf = _safe_csv(cpath)
            if not cdf.empty and 'game_id' in cdf.columns:
                cdf['game_id'] = cdf['game_id'].astype(str)
                if 'closing_spread_home' not in cdf.columns and {'market','home_spread','period'}.issubset(cdf.columns):
                    try:
                        sub = cdf[(cdf['market'].astype(str).str.lower()=='spreads') & (cdf['period'].astype(str).str.lower().isin(['full_game','full game','fg']))]
                        csp = sub.groupby('game_id')['home_spread'].median().rename('closing_spread_home')
                        cdf = cdf.merge(csp, on='game_id', how='left')
                    except Exception:
                        pass
                ddf = ddf.merge(cdf[['game_id','closing_spread_home']], on='game_id', how='left')
            if 'closing_spread_home' not in ddf.columns:
                continue
            mp = pd.to_numeric(ddf['pred_margin'], errors='coerce')
            mf = pd.to_numeric(ddf['home_score'], errors='coerce') - pd.to_numeric(ddf['away_score'], errors='coerce')
            sp = pd.to_numeric(ddf['closing_spread_home'], errors='coerce')
            cover = (mf + sp) > 0
            for a,b,c_,cv in zip(mp, mf, sp, cover):
                if pd.isna(a) or pd.isna(b) or pd.isna(c_):
                    continue
                rows.append({'date': date_part, 'margin_pred': float(a), 'closing_spread': float(c_), 'margin_final': float(b), 'home_cover': int(bool(cv))})
    except Exception:
        pass
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-days', type=int, default=120, help='Limit calibration window to last N days')
    ap.add_argument('--min-rows', type=int, default=200, help='Minimum rows required to output calibration')
    ap.add_argument('--provisional-min-rows', type=int, default=50, help='If rows >= this but < min-rows, still emit a provisional calibration payload')
    ap.add_argument('--force-provisional', action='store_true', help='Always write a calibration payload using available rows (marks provisional when < min-rows)')
    ap.add_argument('--grid-start', type=float, default=0.05)
    ap.add_argument('--grid-end', type=float, default=0.35)
    ap.add_argument('--grid-step', type=float, default=0.01)
    args = ap.parse_args()

    df = gather_rows(args.max_days)
    n = len(df)
    # Handle insufficient rows with graceful fallbacks
    if df.empty:
        # Try to carry forward prior calibration, else default
        prior_path = OUT / 'calibration_spread_logistic.json'
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text(encoding='utf-8'))
                prior['carried_forward'] = True
                prior['source_rows'] = 0
                prior['generated_at'] = dt.datetime.now().isoformat()
                prior_path.write_text(json.dumps(prior, indent=2), encoding='utf-8')
                print(f"No rows; carried forward prior calibration K={prior.get('best_K')}")
                return
            except Exception:
                pass
        # Default payload
        payload = {
            'generated_at': dt.datetime.now().isoformat(),
            'n_rows': 0,
            'best_K': 0.115,
            'best_avg_log_loss': None,
            'grid': {'start': args.grid_start, 'end': args.grid_end, 'step': args.grid_step},
            'fallback_default': True
        }
        (OUT / 'calibration_spread_logistic.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('No rows; wrote default K=0.115')
        return

    if (n < args.min_rows) and not args.force_provisional and (n < args.provisional_min_rows):
        print(f'Insufficient rows for calibration: {n} (< {args.provisional_min_rows}). Skipping write.')
        return

    # Grid search for K minimizing log loss
    best = None
    for K in [round(x,4) for x in list(np.arange(args.grid_start, args.grid_end + 1e-9, args.grid_step))]:
        total_loss = 0.0
        for _, r in df.iterrows():
            diff = (r['margin_pred'] - r['closing_spread']) / K
            p = 1.0/(1.0 + math.exp(-diff))
            total_loss += _log_loss(p, int(r['home_cover']))
        avg_loss = total_loss / len(df)
        if best is None or avg_loss < best['avg_log_loss']:
            best = {'K': K, 'avg_log_loss': avg_loss}

    payload = {
        'generated_at': dt.datetime.now().isoformat(),
        'n_rows': int(n),
        'best_K': best['K'],
        'best_avg_log_loss': best['avg_log_loss'],
        'grid': {'start': args.grid_start, 'end': args.grid_end, 'step': args.grid_step}
    }
    if (n < args.min_rows):
        payload['provisional'] = True
    out_path = OUT / 'calibration_spread_logistic.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Wrote calibration payload to {out_path} (K={best['K']}, rows={n}, provisional={payload.get('provisional', False)})")

if __name__ == '__main__':
    main()
