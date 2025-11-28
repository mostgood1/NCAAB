import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

OUT = Path('outputs')
FEATURE_CANDIDATES = [
    'features_missing_filled_{date}.csv',  # newly synthesized mandatory feature fills
    'features_curr.csv','features_all.csv','features_week.csv','features_last2.csv'
]

def _coerce_numeric_str(val):
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except Exception:
            return False
    return False

def load_features(date_str: str):
    # Prioritize synthesized file for the date if present
    for name in FEATURE_CANDIDATES:
        p = OUT / name.format(date=date_str)
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and 'game_id' in df.columns:
                    return df
            except Exception:
                continue
    # Fallback: first existing candidate irrespective of date formatting
    for name in FEATURE_CANDIDATES[1:]:  # skip the formatted first pattern again
        p = OUT / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and 'game_id' in df.columns:
                    return df
            except Exception:
                continue
    return pd.DataFrame()

def derive_missing_rows_from_enriched(date_str: str):
    enriched = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not enriched.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(enriched)
    except Exception:
        return pd.DataFrame()
    if df.empty or 'game_id' not in df.columns:
        return pd.DataFrame()
    miss_mask = (
        (df.get('pred_total').isna() if 'pred_total' in df.columns else False) |
        (df.get('pred_margin').isna() if 'pred_margin' in df.columns else False) |
        (df.get('market_total').isna() if 'market_total' in df.columns else False) |
        (df.get('spread_home').isna() if 'spread_home' in df.columns else False)
    )
    if not isinstance(miss_mask, pd.Series) or not miss_mask.any():
        return pd.DataFrame()
    need_cols = [c for c in ['game_id','home_team','away_team','pred_total','pred_margin','market_total','spread_home'] if c in df.columns]
    return df.loc[miss_mask, need_cols].copy()

def main():
    # Optional date override via first CLI arg YYYY-MM-DD
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    miss_file = OUT / f'missing_real_coverage_{date_str}.csv'
    if miss_file.exists():
        try:
            miss_df = pd.read_csv(miss_file)
        except Exception:
            miss_df = pd.DataFrame()
    else:
        miss_df = derive_missing_rows_from_enriched(date_str)
        if not miss_df.empty:
            # Persist for downstream scripts to keep consistent workflow
            try:
                miss_df.to_csv(miss_file, index=False)
                print('Persisted derived missing coverage CSV:', miss_file)
            except Exception:
                pass
    if miss_df.empty:
        print('No missing rows present for date', date_str)
        return
    feat_df = load_features(date_str)
    if feat_df.empty:
        print('Feature frame empty; cannot build manifest.')
        return
    feat_df['game_id'] = feat_df['game_id'].astype(str)
    miss_df['game_id'] = miss_df['game_id'].astype(str)
    merged = miss_df.merge(feat_df, on='game_id', how='left')
    manifest_records = []
    core_cols = ['game_id','home_team','away_team']
    for _, row in merged.iterrows():
        rec = {k: row.get(k) for k in core_cols if k in merged.columns}
        for c in merged.columns:
            if c in core_cols:
                continue
            lc = c.lower()
            if lc.startswith('pred_') or lc.endswith('_model'):
                continue  # exclude existing predictions/model outputs
            val = row.get(c)
            if _coerce_numeric_str(val):
                rec[c] = val
        manifest_records.append(rec)
    manifest_path = OUT / f'missing_inference_manifest_{date_str}.json'
    with open(manifest_path,'w',encoding='utf-8') as fh:
        json.dump({'date': date_str,'count': len(manifest_records),'records': manifest_records}, fh, indent=2)
    print('Manifest written:', manifest_path, 'records:', len(manifest_records))

if __name__ == '__main__':
    main()
