"""Generate residual statistics for a resolved date.

Usage:
  python scripts/residuals_generate.py --date 2025-11-19

Produces outputs/residuals_<date>.json with keys:
  totals_residual_mean, totals_residual_std, margin_residual_mean, margin_residual_std,
  total_corr, margin_corr, rows, date, generated_at
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd

OUT = Path('outputs')

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default: yesterday)')
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime('%Y-%m-%d')

    results = _safe_csv(OUT / 'daily_results' / f'results_{date_str}.csv')
    preds = _safe_csv(OUT / f'predictions_unified_{date_str}.csv')
    if preds.empty:
        preds = _safe_csv(OUT / f'predictions_model_{date_str}.csv')

    payload = {'date': date_str, 'generated_at': dt.datetime.now().isoformat(), 'rows': 0}
    try:
        if not results.empty and 'game_id' in results.columns and 'home_score' in results.columns and 'away_score' in results.columns:
            results['game_id'] = results['game_id'].astype(str)
            results['actual_total'] = pd.to_numeric(results['home_score'], errors='coerce') + pd.to_numeric(results['away_score'], errors='coerce')
            results['actual_margin'] = pd.to_numeric(results['home_score'], errors='coerce') - pd.to_numeric(results['away_score'], errors='coerce')
            if not preds.empty and 'game_id' in preds.columns:
                preds['game_id'] = preds['game_id'].astype(str)
                merged = results.merge(preds, on='game_id', how='left')
            else:
                merged = results.copy()
            pt = pd.to_numeric(merged.get('pred_total'), errors='coerce')
            pm = pd.to_numeric(merged.get('pred_margin'), errors='coerce')
            at = pd.to_numeric(merged.get('actual_total'), errors='coerce')
            am = pd.to_numeric(merged.get('actual_margin'), errors='coerce')
            merged['residual_total'] = pt - at
            merged['residual_margin'] = pm - am
            payload['rows'] = int(len(merged))
            payload['totals_residual_mean'] = float(merged['residual_total'].mean()) if merged['residual_total'].notna().any() else None
            payload['totals_residual_std'] = float(merged['residual_total'].std(ddof=0)) if merged['residual_total'].notna().any() else None
            payload['margin_residual_mean'] = float(merged['residual_margin'].mean()) if merged['residual_margin'].notna().any() else None
            payload['margin_residual_std'] = float(merged['residual_margin'].std(ddof=0)) if merged['residual_margin'].notna().any() else None
            # Correlations
            try:
                payload['total_corr'] = float(pd.concat([pt, at], axis=1).corr().iloc[0,1]) if pt.notna().any() and at.notna().any() else None
            except Exception:
                payload['total_corr'] = None
            try:
                payload['margin_corr'] = float(pd.concat([pm, am], axis=1).corr().iloc[0,1]) if pm.notna().any() and am.notna().any() else None
            except Exception:
                payload['margin_corr'] = None
        else:
            payload['status'] = 'no_results'
    except Exception as e:
        payload['error'] = str(e)

    out_path = OUT / f'residuals_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote residuals to {out_path}')

if __name__ == '__main__':
    main()
