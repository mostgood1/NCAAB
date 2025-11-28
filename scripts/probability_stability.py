import json, math, datetime
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')
TODAY = datetime.datetime.utcnow().date().isoformat()

# Probability column prefixes to monitor (totals & cover)
PREFIXES = ['p_over','p_home_cover']
BINS = 50
JS_THRESH = 0.05  # soft flag threshold

ARTIFACT = {
    'date': TODAY,
    'js_threshold': JS_THRESH,
    'methods': {},
}

def load_enriched(date: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_enriched_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# Resolve previous date with existing artifact
prev_date = None
for days_back in range(1, 8):
    d = (datetime.datetime.utcnow().date() - datetime.timedelta(days=days_back)).isoformat()
    if (OUT / f'predictions_unified_enriched_{d}.csv').exists():
        prev_date = d
        break

cur = load_enriched(TODAY)
prev = load_enriched(prev_date) if prev_date else pd.DataFrame()

if cur.empty or prev.empty:
    ARTIFACT['warning'] = 'Missing current or previous enriched predictions for stability analysis.'
else:
    # Identify all probability columns matching prefixes
    prob_cols = [c for c in cur.columns if any(c.startswith(pfx) for pfx in PREFIXES)]
    for col in prob_cols:
        if col not in prev.columns:
            continue
        a = pd.to_numeric(cur[col], errors='coerce').clip(0,1).dropna()
        b = pd.to_numeric(prev[col], errors='coerce').clip(0,1).dropna()
        if len(a) < 25 or len(b) < 25:
            continue
        hist_a, edges = np.histogram(a.values, bins=BINS, range=(0,1), density=True)
        hist_b, _ = np.histogram(b.values, bins=BINS, range=(0,1), density=True)
        # Convert to probability mass (normalize)
        pa = hist_a / hist_a.sum()
        pb = hist_b / hist_b.sum()
        m = 0.5 * (pa + pb)
        def kl(p, q):
            mask = (p > 0) & (q > 0)
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
        ARTIFACT['methods'][col] = {
            'rows_today': int(len(a)),
            'rows_prev': int(len(b)),
            'js_divergence': js,
            'flag_instability': bool(js > JS_THRESH),
            'mean_today': float(a.mean()),
            'mean_prev': float(b.mean()),
        }

out_path = OUT / f'prob_stability_{TODAY}.json'
out_path.write_text(json.dumps(ARTIFACT, indent=2))
print('Probability stability analysis complete:', json.dumps(ARTIFACT, indent=2))
