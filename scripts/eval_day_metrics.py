import sys, os
import pandas as pd

def eval_day(date_str: str):
    root = os.path.dirname(os.path.dirname(__file__))
    out = os.path.join(root, 'outputs')
    pred_path = os.path.join(out, f'predictions_unified_enriched_{date_str}.csv')
    res_path = os.path.join(out, 'daily_results', f'results_{date_str}.csv')
    exists = {'pred_path_exists': os.path.exists(pred_path), 'res_path_exists': os.path.exists(res_path)}
    print(exists)
    if not (os.path.exists(pred_path) and os.path.exists(res_path)):
        return
    p = pd.read_csv(pred_path)
    r = pd.read_csv(res_path)
    if {'home_score','away_score'}.issubset(r.columns):
        r['actual_total'] = pd.to_numeric(r['home_score'], errors='coerce') + pd.to_numeric(r['away_score'], errors='coerce')
    else:
        r['actual_total'] = None
    for col in ('game_id','date'):
        if col in p.columns: p[col] = p[col].astype(str)
        if col in r.columns: r[col] = r[col].astype(str)
    cols = ['game_id','date','actual_total']
    if set(cols).issubset(r.columns):
        rr = r[cols].copy()
        print({'actual_total_notna': int(pd.to_numeric(rr['actual_total'], errors='coerce').notna().sum())})
        rr = rr.dropna(subset=['actual_total'])
    else:
        rr = pd.DataFrame()
    # Merge on game_id only to avoid date format mismatches
    df = p.merge(rr[['game_id','actual_total']], on=['game_id'], how='left', suffixes=('_p','_r'))
    if 'actual_total_r' in df.columns:
        df['actual_total'] = df['actual_total_r']
    # Debug match counts
    try:
        inter = set(p['game_id'].astype(str)) & set(r['game_id'].astype(str))
        print({'match_count': len(inter), 'pred_rows': len(p), 'result_rows': len(r)})
    except Exception:
        pass
    cands = [
        'pred_total_calibrated',
        'pred_total_model',
        'pred_total_q50',
        'pred_total_view',
        'pred_total'
    ]
    mt_col = 'market_total'
    mt = pd.to_numeric(df.get(mt_col), errors='coerce')
    act = pd.to_numeric(df['actual_total'], errors='coerce') if 'actual_total' in df.columns else pd.Series(dtype='float64')
    try:
        print({'act_notna': int(act.notna().sum())})
    except Exception:
        pass
    metrics = {}
    for c in cands:
        if c in df.columns:
            pred = pd.to_numeric(df[c], errors='coerce')
            try:
                print({c+'_notna': int(pred.notna().sum())})
            except Exception:
                pass
            mask = pred.notna() & act.notna()
            if mask.any():
                mae = float((pred[mask] - act[mask]).abs().mean())
                bias = float((pred[mask] - act[mask]).mean())
                metrics[c] = {'mae':round(mae,3),'bias':round(bias,3),'n':int(mask.sum())}
    mt_mask = mt.notna() & act.notna()
    if mt_mask.any():
        mae = float((mt[mt_mask] - act[mt_mask]).abs().mean())
        bias = float((mt[mt_mask] - act[mt_mask]).mean())
        metrics['market_total'] = {'mae':round(mae,3),'bias':round(bias,3),'n':int(mt_mask.sum())}
    print(metrics)

if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else '2026-01-05'
    eval_day(date_str)
