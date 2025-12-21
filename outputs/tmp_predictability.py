import pandas as pd, pathlib, numpy as np
root = pathlib.Path('c:/Users/mostg/OneDrive/Coding/NCAAB')
out = root/'outputs'
date_str = '2025-11-19'
print(f'--- Predictability diagnostics for {date_str} ---')
pred_path = out/f'predictions_model_{date_str}.csv'
align_path = out/f'align_period_{date_str}.csv'
if not pred_path.exists():
    print('Predictions file missing:', pred_path)
else:
    preds = pd.read_csv(pred_path)
    def stats(col):
        x = pd.to_numeric(preds[col], errors='coerce')
        x = x[np.isfinite(x)]
        return {
            'count': int(x.size),
            'mean': round(float(x.mean()),3),
            'std': round(float(x.std(ddof=0)),3),
            'min': round(float(x.min()),3),
            'max': round(float(x.max()),3),
            'cv': round(float(x.std(ddof=0)/x.mean()),3) if x.mean() else None,
            'unique': int(x.nunique())
        }
    tot_stats = stats('pred_total_model') if 'pred_total_model' in preds.columns else {}
    marg_stats = stats('pred_margin_model') if 'pred_margin_model' in preds.columns else {}
    print('Totals stats:', tot_stats)
    print('Margin stats:', marg_stats)
    for col in ['pred_total_model','pred_margin_model']:
        if col in preds.columns:
            vals = pd.to_numeric(preds[col], errors='coerce')
            vals = vals[np.isfinite(vals)]
            if len(vals)>0:
                z = (vals - vals.mean())/vals.std(ddof=0) if vals.std(ddof=0)>0 else np.zeros_like(vals)
                buckets = { '(-inf,-1]': int((z<=-1).sum()), '(-1,0]': int(((z>-1)&(z<=0)).sum()), '(0,1]': int(((z>0)&(z<=1)).sum()), '(1,2]': int(((z>1)&(z<=2)).sum()), '(2,inf)': int((z>2).sum()) }
                print(f'{col} z-buckets:', buckets)
    if align_path.exists():
        align = pd.read_csv(align_path)
        xcols = [c for c in align.columns if 'market_total' in c]
        if 'game_id' in align.columns and 'pred_total_model' in preds.columns:
            merged = preds.merge(align[['game_id']+xcols], on='game_id', how='left')
            corr = {}
            for c in xcols:
                s = pd.to_numeric(merged[c], errors='coerce')
                p = pd.to_numeric(merged['pred_total_model'], errors='coerce')
                mask = np.isfinite(s) & np.isfinite(p)
                if mask.sum()>5:
                    corr[c] = round(float(np.corrcoef(s[mask], p[mask])[0,1]),3)
            print('Correlation total_model vs market lines:', corr)
    else:
        print('Align file missing; skipping market correlation.')
    issues = []
    if tot_stats and tot_stats['std'] < 5: issues.append('Low total std (<5)')
    if marg_stats and marg_stats['std'] < 5: issues.append('Low margin std (<5)')
    if tot_stats and tot_stats['unique'] < max(5, tot_stats['count']//5): issues.append('Few unique total values')
    if issues:
        print('Potential issues:', issues)
    else:
        print('No red-flag uniformity issues detected.')
print('--- Done ---')