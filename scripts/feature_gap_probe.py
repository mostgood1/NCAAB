import pandas as pd
from pathlib import Path
from datetime import datetime
import json

OUT = Path('outputs')
DATE_STR = datetime.now().strftime('%Y-%m-%d')
MISSING = OUT / f'missing_real_coverage_{DATE_STR}.csv'
FEATURE_SOURCES = [
    OUT / 'features_curr.csv',
    OUT / 'features_all.csv',
    OUT / 'features_week.csv',
    OUT / 'features_last2.csv'
]
REPORT = OUT / f'feature_gap_report_{DATE_STR}.json'

MANDATORY_FEATURES = [
    'home_off_rating','away_off_rating','home_def_rating','away_def_rating',
    'home_tempo_rating','away_tempo_rating'
]

OPTIONAL_FEATURES = [
    'home_recent_form','away_recent_form','home_pace_adj','away_pace_adj',
    'home_off_trend','away_off_trend','home_def_trend','away_def_trend'
]


def safe_read(p):
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_union():
    frames = []
    for src in FEATURE_SOURCES:
        df = safe_read(src)
        if not df.empty and 'game_id' in df.columns:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Prioritize first non-empty; fallback merge
    base = frames[0]
    # Merge subsequent frames for extra columns
    for add in frames[1:]:
        try:
            base = base.merge(add, on='game_id', how='left', suffixes=('','_x'))
        except Exception:
            continue
    return base

def main():
    miss = safe_read(MISSING)
    feat_union = load_union()
    report = {'date': DATE_STR, 'missing_rows': len(miss), 'features_rows': len(feat_union)}
    if miss.empty:
        report['status'] = 'no_missing'
    else:
        if 'game_id' in miss.columns:
            miss['game_id'] = miss['game_id'].astype(str)
        if 'game_id' in feat_union.columns:
            feat_union['game_id'] = feat_union['game_id'].astype(str)
        present_mask = feat_union['game_id'].isin(miss['game_id']) if not feat_union.empty else pd.Series(False, index=miss.index)
        present_ids = set(feat_union['game_id'])
        gaps = []
        for r in miss.itertuples():
            gid = str(getattr(r,'game_id'))
            row = feat_union[feat_union['game_id'] == gid]
            missing_cols = []
            for col in MANDATORY_FEATURES:
                if col not in feat_union.columns or row.empty or pd.isna(row[col]).all():
                    missing_cols.append(col)
            optional_missing = []
            for col in OPTIONAL_FEATURES:
                if col not in feat_union.columns or row.empty or pd.isna(row[col]).all():
                    optional_missing.append(col)
            gaps.append({
                'game_id': gid,
                'has_feature_row': gid in present_ids and not row.empty,
                'mandatory_missing': missing_cols,
                'optional_missing': optional_missing,
                'home_team': getattr(r,'home_team',''),
                'away_team': getattr(r,'away_team','')
            })
        report['gaps'] = gaps
        # Aggregate stats
        agg_mandatory = {}
        for g in gaps:
            for c in g['mandatory_missing']:
                agg_mandatory[c] = agg_mandatory.get(c,0)+1
        report['mandatory_feature_missing_counts'] = agg_mandatory
    with open(REPORT,'w',encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print('Feature gap report written:', REPORT)
    if 'mandatory_feature_missing_counts' in report:
        print('Mandatory missing counts:', report['mandatory_feature_missing_counts'])

if __name__ == '__main__':
    main()
