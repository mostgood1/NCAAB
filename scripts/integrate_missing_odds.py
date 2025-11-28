import pandas as pd
from pathlib import Path
from datetime import datetime

OUT = Path('outputs')

def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    fetched_path = OUT / f'missing_odds_fetched_{date_str}.csv'
    enriched_path = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not fetched_path.exists():
        print('Fetched odds file missing:', fetched_path)
        return
    if not enriched_path.exists():
        print('Base enriched artifact missing:', enriched_path)
        return
    f_df = pd.read_csv(fetched_path)
    e_df = pd.read_csv(enriched_path)
    if e_df.empty or f_df.empty:
        print('One of the frames empty; abort.')
        return
    for col in ['game_id','market_total','spread_home']:
        if col not in e_df.columns:
            if col != 'game_id':
                e_df[col] = None
    e_df['game_id'] = e_df['game_id'].astype(str)
    f_df['game_id'] = f_df['game_id'].astype(str)
    merged = e_df.merge(f_df[['game_id','market_total_fetched','spread_home_fetched','fetched_basis','match_score']], on='game_id', how='left')
    # Promote only where original real odds missing
    mt_mask = merged['market_total'].isna() & merged['market_total_fetched'].notna()
    sh_mask = merged['spread_home'].isna() & merged['spread_home_fetched'].notna()
    if mt_mask.any():
        merged.loc[mt_mask, 'market_total'] = merged.loc[mt_mask, 'market_total_fetched']
        merged.loc[mt_mask, 'market_total_basis'] = 'fetched_consensus'
    if sh_mask.any():
        merged.loc[sh_mask, 'spread_home'] = merged.loc[sh_mask, 'spread_home_fetched']
        merged.loc[sh_mask, 'spread_home_basis'] = 'fetched_consensus'
    out_path = OUT / f'predictions_unified_enriched_{date_str}_with_missing_odds.csv'
    merged.to_csv(out_path, index=False)
    print('Integrated with fetched odds written:', out_path)
    print('Filled totals:', int(mt_mask.sum()), 'Filled spreads:', int(sh_mask.sum()))

if __name__ == '__main__':
    main()
