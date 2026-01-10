import sys, os
import pandas as pd

def load_day(root, date_str):
    out = os.path.join(root, 'outputs')
    pred_path = os.path.join(out, f'predictions_unified_enriched_{date_str}.csv')
    res_path = os.path.join(out, 'daily_results', f'results_{date_str}.csv')
    if not (os.path.exists(pred_path) and os.path.exists(res_path)):
        return pd.DataFrame()
    p = pd.read_csv(pred_path)
    r = pd.read_csv(res_path)
    if {'home_score','away_score'}.issubset(r.columns):
        r['actual_total'] = pd.to_numeric(r['home_score'], errors='coerce') + pd.to_numeric(r['away_score'], errors='coerce')
    else:
        r['actual_total'] = None
    for col in ('game_id','date'):
        if col in p.columns: p[col] = p[col].astype(str)
        if col in r.columns: r[col] = r[col].astype(str)
    rr = r[['game_id','actual_total']]
    df = p.merge(rr, on=['game_id'], how='left', suffixes=('_p','_r'))
    if 'actual_total_r' in df.columns:
        df['actual_total'] = df['actual_total_r']
    return df

def eval_range(dates):
    root = os.path.dirname(os.path.dirname(__file__))
    frames = [load_day(root, d) for d in dates]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print({'error':'no_data'})
        return
    df = pd.concat(frames, ignore_index=True)
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
    metrics = {}
    for c in cands:
        if c in df.columns:
            pred = pd.to_numeric(df[c], errors='coerce')
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
    print({'dates':dates,'metrics':metrics})

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dates = sys.argv[1:]
    else:
        dates = ['2026-01-01','2026-01-02','2026-01-03','2026-01-04']
    eval_range(dates)
