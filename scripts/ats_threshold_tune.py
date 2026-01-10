import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), 'outputs')
DR = os.path.join(OUT, 'daily_results')

# Load last N days of results
N_DAYS = 7
files = sorted(glob.glob(os.path.join(DR, 'results_*.csv')))
# Keep only last N days by date in filename
def _date_from_name(p: str) -> str | None:
    base = os.path.basename(p)
    try:
        return base.split('_', 1)[1].split('.csv')[0]
    except Exception:
        return None
by_date = {}
for f in files:
    d = _date_from_name(f)
    if not d:
        continue
    by_date.setdefault(d, f)
if not by_date:
    print(json.dumps({"status": "no-files"}))
    raise SystemExit(0)
all_dates = sorted(by_date.keys())
keep_dates = all_dates[-N_DAYS:]
keep_files = [by_date[d] for d in keep_dates]

parts = []
for p in keep_files:
    try:
        df = pd.read_csv(p, dtype=str, low_memory=False)
        df['source_date'] = _date_from_name(p)
        parts.append(df)
    except Exception:
        continue
if not parts:
    print(json.dumps({"status": "no-data"}))
    raise SystemExit(0)

df = pd.concat(parts, ignore_index=True)
# Coerce needed fields
for c in ['pred_margin','spread_home','actual_margin']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
# Consider ATS rows with all fields present
mask = df[['pred_margin','spread_home','actual_margin']].notna().all(axis=1)
adf = df.loc[mask].copy()
if adf.empty:
    print(json.dumps({"status": "no-ats-rows"}))
    raise SystemExit(0)

# Compute raw signed edges (predicted margin vs line)
adf['edge_margin_signed'] = adf['pred_margin'] - adf['spread_home']
# Prediction correctness from home perspective
adf['pred_home_cover'] = adf['pred_margin'] > (-adf['spread_home'])
adf['actual_home_cover'] = adf['actual_margin'] > (-adf['spread_home'])

# Sweep absolute edge thresholds to find best accuracy
thresholds = np.arange(0.0, 6.0 + 1e-9, 0.5)
records = []
for th in thresholds:
    keep = adf[np.abs(adf['edge_margin_signed']) >= th]
    if keep.empty:
        acc = None
        n = 0
    else:
        acc = float((keep['pred_home_cover'] == keep['actual_home_cover']).mean())
        n = int(len(keep))
    records.append({"threshold": float(th), "accuracy": acc, "n": n})

# Choose best by accuracy, then by n desc
best = None
for r in sorted(records, key=lambda x: ((x['accuracy'] is None), -(x['accuracy'] or 0.0), -x['n'])):
    best = r
    break

out = {
    "dates": keep_dates,
    "overall": {
        "n": int(len(adf)),
        "accuracy": float((adf['pred_home_cover'] == adf['actual_home_cover']).mean()) if len(adf) else None,
    },
    "sweep": records,
    "best": best,
}
print(json.dumps(out, indent=2))
