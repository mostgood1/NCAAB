import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    preds_path = Path('outputs') / f'missing_inference_preds_{date_str}.csv'
    enriched_path = Path('outputs') / f'predictions_unified_enriched_{date_str}.csv'
    if not preds_path.exists():
        print('Missing inference predictions not found:', preds_path)
        return
    if not enriched_path.exists():
        print('Enriched artifact not found:', enriched_path)
        return
    preds_df = pd.read_csv(preds_path)
    enriched_df = pd.read_csv(enriched_path)
    if 'game_id' not in preds_df.columns:
        print('Preds file lacks game_id.')
        return
    preds_df['game_id'] = preds_df['game_id'].astype(str)
    enriched_df['game_id'] = enriched_df['game_id'].astype(str)
    merged = enriched_df.merge(preds_df[['game_id','pred_total_model','pred_margin_model']], on='game_id', how='left', suffixes=('','_new'))
    # Promote only where original pred_total / pred_margin are missing
    for col in ['pred_total','pred_margin']:
        model_col = f'{col}_model'
        if model_col not in merged.columns and f'{model_col}_new' in merged.columns:
            merged[model_col] = merged[f'{model_col}_new']
        mask = merged[col].isna() & merged[model_col].notna()
        merged.loc[mask, col] = merged.loc[mask, model_col]
    out_path = Path('outputs') / f'predictions_unified_enriched_{date_str}_with_missing_inference.csv'
    merged.to_csv(out_path, index=False)
    print('Integrated artifact written:', out_path)

if __name__ == '__main__':
    main()
