import pandas as pd, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'outputs'/f'predictions_unified_enriched_{pd.Timestamp.today().strftime("%Y-%m-%d")}.csv'
if not p.exists():
    print('Missing enriched artifact', p)
    sys.exit(0)
df=pd.read_csv(p)
print('Rows:', len(df))
print('Columns:', list(df.columns))
for c in ['pred_total','pred_margin','market_total','spread_home']:
    if c in df.columns:
        miss=int(pd.to_numeric(df[c], errors='coerce').isna().sum())
        print(f'{c} missing: {miss}')
if 'pred_total' in df.columns:
    miss_df=df[pd.to_numeric(df['pred_total'], errors='coerce').isna()]
    print('First 5 missing pred_total rows:')
    print(miss_df[['game_id','home_team','away_team','pred_total']].head())
