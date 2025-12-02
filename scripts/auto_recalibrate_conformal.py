"""Auto-recalibrate conformal interval buffer scales based on recent coverage.

Inputs (preferred):
  - outputs/quantiles_conformal_history.csv with columns: date, game_id, c10_total, c90_total, c10_margin, c90_margin, actual_total, actual_margin
Fallback:
  - outputs/conformal_metrics_all.csv with per-date coverage columns if available

Outputs:
  - outputs/conformal_autotune.json with scale_total and scale_margin to nudge buffers
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import datetime as dt

OUTPUTS = Path('outputs')

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def compute_from_history(df: pd.DataFrame, days: int = 14) -> tuple[float|None, float|None]:
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        cutoff = df['date'].max() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
        def cov(pair_low, pair_high, actual):
            m = (~pd.isna(df[pair_low])) & (~pd.isna(df[pair_high])) & (~pd.isna(df[actual]))
            if m.sum() == 0:
                return None
            return float(((df.loc[m, actual] >= df.loc[m, pair_low]) & (df.loc[m, actual] <= df.loc[m, pair_high])).mean())
        cov_t = cov('c10_total','c90_total','actual_total') if {'c10_total','c90_total','actual_total'}.issubset(df.columns) else None
        cov_m = cov('c10_margin','c90_margin','actual_margin') if {'c10_margin','c90_margin','actual_margin'}.issubset(df.columns) else None
        return cov_t, cov_m
    except Exception:
        return None, None

def compute_from_metrics(df: pd.DataFrame, days: int = 14) -> tuple[float|None, float|None]:
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        cutoff = df['date'].max() - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
        # Try common column names
        cand_total = [c for c in df.columns if 'cov' in c.lower() and 'total' in c.lower() and 'conf' in c.lower() or 'conformal' in c.lower()]
        cand_margin = [c for c in df.columns if 'cov' in c.lower() and 'margin' in c.lower() and ('conf' in c.lower() or 'conformal' in c.lower())]
        cov_t = float(df[cand_total[0]].mean()) if cand_total else None
        cov_m = float(df[cand_margin[0]].mean()) if cand_margin else None
        return cov_t, cov_m
    except Exception:
        return None, None

def main():
    hist = _safe_read(OUTPUTS / 'quantiles_conformal_history.csv')
    cov_t, cov_m = (None, None)
    if not hist.empty:
        cov_t, cov_m = compute_from_history(hist, days=14)
    if cov_t is None and cov_m is None:
        met = _safe_read(OUTPUTS / 'conformal_metrics_all.csv')
        if not met.empty:
            cov_t, cov_m = compute_from_metrics(met, days=28)
    target = 0.80
    res = {'timestamp': dt.datetime.utcnow().isoformat() + 'Z'}
    def scale_from_cov(c):
        if c is None or np.isnan(c):
            return None
        # Limit scale to a reasonable band to avoid thrash
        s = float(target / max(1e-6, c))
        return float(max(0.8, min(1.2, s)))
    res['scale_total'] = scale_from_cov(cov_t)
    res['scale_margin'] = scale_from_cov(cov_m)
    with open(OUTPUTS / 'conformal_autotune.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('[conformal-autotune] Wrote outputs/conformal_autotune.json', res)

if __name__ == '__main__':
    main()
