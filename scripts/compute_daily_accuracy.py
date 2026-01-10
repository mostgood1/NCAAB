import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def compute_for_date(date_str: str, outputs_dir: str = 'outputs'):
    dr_path = Path(outputs_dir) / 'daily_results' / f'results_{date_str}.csv'
    disp_path = Path(outputs_dir) / f'predictions_display_{date_str}.csv'
    if not dr_path.exists() or not disp_path.exists():
        return {
            'date': date_str,
            'error': f'missing files for date: results={dr_path.exists()} display={disp_path.exists()}'
        }
    dr = pd.read_csv(dr_path, dtype={'game_id': str})
    disp = pd.read_csv(disp_path, dtype={'game_id': str})
    cols = ['game_id','pred_total','pred_margin','closing_total','closing_spread_home']
    disp_cols = [c for c in cols if c in disp.columns]
    df = dr.merge(disp[disp_cols], on='game_id', how='left', suffixes=('', '_disp'))

    # Coalesce predictions from display when daily_results lacks them
    def coalesce(col: str) -> pd.Series:
        a = pd.to_numeric(df.get(col), errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)
        b = pd.to_numeric(df.get(f"{col}_disp"), errors='coerce') if f"{col}_disp" in df.columns else pd.Series(np.nan, index=df.index)
        out = a.copy()
        m = out.isna() & b.notna()
        out[m] = b[m]
        return out

    pm = coalesce('pred_margin')
    am = pd.to_numeric(df.get('actual_margin'), errors='coerce')
    mw = pm.notna() & am.notna()
    winners = {
        'n': int(mw.sum()),
        'acc': float(((pm[mw] > 0).astype(int) == (am[mw] > 0).astype(int)).mean()) if mw.sum() > 0 else None,
    }

    pt = coalesce('pred_total')
    line = pd.to_numeric(df.get('market_total'), errors='coerce')
    if line.isna().all() and 'closing_total' in df:
        line = pd.to_numeric(df.get('closing_total'), errors='coerce')
    ou = df.get('ou_result_full')
    mask_t = pt.notna() & line.notna() & ou.notna() & ou.isin(['Over','Under'])
    totals = {
        'n': int(mask_t.sum()),
        'acc': float((((pt[mask_t] > line[mask_t]).astype(int)) == (ou[mask_t] == 'Over').astype(int)).mean()) if mask_t.sum() > 0 else None,
    }

    sp = pd.to_numeric(df.get('spread_home'), errors='coerce')
    ats_res = df.get('ats_result')
    mask_ats = pm.notna() & sp.notna() & ats_res.notna() & ats_res.isin(['Home Cover','Away Cover'])
    ats = {
        'n': int(mask_ats.sum()),
        'acc': float((((pm[mask_ats] > -sp[mask_ats]).astype(int)) == (ats_res[mask_ats] == 'Home Cover').astype(int)).mean()) if mask_ats.sum() > 0 else None,
    }

    return {'date': date_str, 'winners': winners, 'totals': totals, 'ats': ats}


if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    if not date_str:
        print(json.dumps({'error': 'usage: python scripts/compute_daily_accuracy.py YYYY-MM-DD'}))
        sys.exit(1)
    res = compute_for_date(date_str)
    print(json.dumps(res, indent=2))
