import pandas as pd, pathlib, datetime, numpy as np
OUT=pathlib.Path('outputs')
d=datetime.datetime.now().strftime('%Y-%m-%d')
fn=OUT/('predictions_unified_enriched_'+d+'.csv')
print('File exists:', fn.exists(), fn)
if fn.exists():
    df=pd.read_csv(fn)
    miss=df[df['pred_total'].isna()].copy()
    cols=['game_id','home_team','away_team','pred_total','pred_total_model','pred_total_calibrated','pred_total_model_basis']
    print(miss[cols].head(10))
    print('Counts pred_total_model notna among missing pred_total:', miss['pred_total_model'].notna().sum())
    print('Unique bases among missing pred_total:', miss['pred_total_model_basis'].unique())
