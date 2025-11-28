import json, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

OUT = Path('outputs')
CAL_DIR = OUT / 'calibrators'
CAL_DIR.mkdir(exist_ok=True)

# Probability columns to calibrate (if present with sufficient samples)
PROB_METHODS = [
    'p_over','p_over_dist','p_over_cdf','p_over_skew','p_over_mix','p_over_piecewise_ext','p_over_kde','p_over_ensemble','p_over_final',
    'p_home_cover','p_home_cover_dist','p_home_cover_cdf','p_home_cover_skew','p_home_cover_mix','p_home_cover_piecewise_ext','p_home_cover_kde','p_home_cover_ensemble','p_home_cover_final'
]

MIN_SAMPLES = int(os.getenv('CALIBRATE_MIN_SAMPLES', '150'))


def collect_rows():
    rows = []
    for p in sorted(OUT.glob('predictions_unified_enriched_*.csv')):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Require actual scores & odds lines to derive outcomes
        if not {'home_score','away_score','market_total','spread_home'}.issubset(df.columns):
            continue
        # Derive actual margin/total outcome flags
        hs = pd.to_numeric(df['home_score'], errors='coerce')
        as_ = pd.to_numeric(df['away_score'], errors='coerce')
        mt = pd.to_numeric(df['market_total'], errors='coerce')
        sp = pd.to_numeric(df['spread_home'], errors='coerce')
        actual_total_flag = ((hs + as_) > mt).astype(int)
        actual_cover_flag = ((hs - as_) > sp).astype(int)
        base = pd.DataFrame({
            'actual_over': actual_total_flag,
            'actual_cover': actual_cover_flag
        })
        for c in PROB_METHODS:
            if c in df.columns:
                base[c] = pd.to_numeric(df[c], errors='coerce')
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fit_isotonic(probs: pd.Series, outcome: pd.Series):
    mask = probs.notna() & outcome.notna()
    if mask.sum() < MIN_SAMPLES:
        return None
    x = probs[mask]
    y = outcome[mask]
    # Sort by prob to ensure monotonic fit stability
    order = np.argsort(x.values)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(x.values[order], y.values[order])
    return iso


def main():
    data = collect_rows()
    if data.empty:
        print('No data collected for calibration.')
        return 0
    saved = 0
    for c in PROB_METHODS:
        if c not in data.columns:
            continue
        target = 'actual_over' if c.startswith('p_over') else 'actual_cover'
        iso = fit_isotonic(data[c], data[target])
        if iso is not None:
            import joblib
            out_path = CAL_DIR / f'{c}_iso.joblib'
            joblib.dump(iso, out_path)
            saved += 1
            print('Saved calibrator', out_path)
    print(f'Calibration complete. Saved={saved}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
