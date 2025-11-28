import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

OUT = Path('outputs')

def main():
    # Allow override date via CLI arg YYYY-MM-DD
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    enriched = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not enriched.exists():
        # Fallback chain: with_missing_inference -> force_fill
        alt1 = OUT / f'predictions_unified_enriched_{date_str}_with_missing_inference.csv'
        alt2 = OUT / f'predictions_unified_enriched_{date_str}_force_fill.csv'
        if alt1.exists():
            enriched = alt1
        elif alt2.exists():
            enriched = alt2
        else:
            print(f'Enriched artifact not found: {enriched}')
            return
    df = pd.read_csv(enriched)
    if df.empty or 'game_id' not in df.columns:
        print('Artifact empty or missing game_id.')
        return
    miss_mask = (
        (df.get('pred_total').isna() if 'pred_total' in df.columns else False) |
        (df.get('pred_margin').isna() if 'pred_margin' in df.columns else False) |
        (df.get('market_total').isna() if 'market_total' in df.columns else False) |
        (df.get('spread_home').isna() if 'spread_home' in df.columns else False)
    )
    if not isinstance(miss_mask, pd.Series) or not miss_mask.any():
        print('No real-data coverage gaps detected.')
        return
    need_cols = [c for c in ['game_id','home_team','away_team','pred_total','pred_margin','market_total','spread_home','pred_total_model','pred_margin_model'] if c in df.columns]
    out_df = df.loc[miss_mask, need_cols].copy()
    out_file = OUT / f'missing_real_coverage_{date_str}.csv'
    out_df.to_csv(out_file, index=False)
    print(f'Missing coverage rows: {len(out_df)} written to {out_file}')
    # Simple summary counts
    print('Counts:')
    print('  pred_total missing:', out_df['pred_total'].isna().sum() if 'pred_total' in out_df.columns else 'n/a')
    print('  pred_margin missing:', out_df['pred_margin'].isna().sum() if 'pred_margin' in out_df.columns else 'n/a')
    print('  market_total missing:', out_df['market_total'].isna().sum() if 'market_total' in out_df.columns else 'n/a')
    print('  spread_home missing:', out_df['spread_home'].isna().sum() if 'spread_home' in out_df.columns else 'n/a')

if __name__ == '__main__':
    main()
