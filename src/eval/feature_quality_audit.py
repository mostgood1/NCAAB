"""Feature Quality Audit Script

Scans feature CSVs in outputs/ for:
  - Missingness rates per column
  - Basic descriptive stats (mean, std) for numeric columns
  - Top absolute correlations (|r|) among key predictive metrics (off_rating, def_rating, tempo)
Outputs JSON report to outputs/feature_quality_audit.json.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / 'outputs'

FEATURE_FILES = [
    'features_curr.csv','features_all.csv','features_week.csv','features_last2.csv'
]

NUMERIC_LIKE = {
    'home_off_rating','away_off_rating','home_def_rating','away_def_rating','home_tempo_rating','away_tempo_rating',
    'tempo_rating_sum','derived_total_est','derived_margin_est'
}

def _load_features() -> pd.DataFrame:
    frames = []
    for name in FEATURE_FILES:
        p = OUT / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    df['__source'] = name
                    frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_report() -> dict:
    df = _load_features()
    if df.empty:
        return {'ok': False, 'error': 'no feature files found'}
    report: dict[str, any] = {'ok': True, 'rows': len(df), 'sources': sorted(df['__source'].unique())}
    miss = {}
    for c in df.columns:
        miss[c] = int(df[c].isna().sum())
    report['missing_counts'] = miss
    # Numeric stats
    num_stats = {}
    for c in df.columns:
        if c.startswith('__'): continue
        vals = pd.to_numeric(df[c], errors='coerce')
        if vals.notna().sum() >= 10:
            num_stats[c] = {
                'count': int(vals.notna().sum()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max())
            }
    report['numeric_stats'] = num_stats
    # Correlations for chosen set
    corr_targets = [c for c in NUMERIC_LIKE if c in df.columns]
    corr_pairs = []
    for i,a in enumerate(corr_targets):
        va = pd.to_numeric(df[a], errors='coerce')
        for b in corr_targets[i+1:]:
            vb = pd.to_numeric(df[b], errors='coerce')
            if va.notna().any() and vb.notna().any():
                r = va.corr(vb)
                if not np.isnan(r):
                    corr_pairs.append({'a': a, 'b': b, 'r': float(r), 'abs_r': float(abs(r))})
    corr_pairs.sort(key=lambda x: x['abs_r'], reverse=True)
    report['top_correlations'] = corr_pairs[:25]
    return report

def main():
    rep = build_report()
    out_path = OUT / 'feature_quality_audit.json'
    try:
        out_path.write_text(json.dumps(rep, indent=2))
        print(f'Wrote feature quality report to {out_path} (rows={rep.get("rows")})')
    except Exception as e:
        print(f'Failed to write report: {e}')

if __name__ == '__main__':
    main()
