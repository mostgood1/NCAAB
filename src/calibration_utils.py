from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_calibration_params(path: str | Path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def apply_isotonic_series(prob: pd.Series, params: dict) -> pd.Series:
    xs = params.get('x', [])
    ys = params.get('y', [])
    if not xs or not ys or len(xs) != len(ys):
        return pd.to_numeric(prob, errors='coerce')
    def map_val(p):
        try:
            p = float(p)
        except Exception:
            return np.nan
        idx = 0
        for i, xv in enumerate(xs):
            if xv <= p:
                idx = i
            else:
                break
        return float(ys[idx])
    return pd.to_numeric(prob, errors='coerce').apply(map_val)

def apply_calibration_to_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    if 'p_over' in out.columns and 'p_over' in params:
        out['p_over_cal'] = apply_isotonic_series(out['p_over'], params['p_over'])
    if 'p_home_cover_dist' in out.columns and 'p_home_cover_dist' in params:
        out['p_home_cover_dist_cal'] = apply_isotonic_series(out['p_home_cover_dist'], params['p_home_cover_dist'])
    return out

def apply_sigma_intervals(df: pd.DataFrame, sigma_total_col: str = 'sigma_total') -> pd.DataFrame:
    out = df.copy()
    if {'pred_total', sigma_total_col}.issubset(out.columns):
        # Z-scores for 90/95% two-sided intervals under normal
        z90 = 1.6448536269514722
        z95 = 1.959963984540054
        sigma = pd.to_numeric(out[sigma_total_col], errors='coerce')
        pred = pd.to_numeric(out['pred_total'], errors='coerce')
        out['pred_total_low_90'] = pred - z90 * sigma
        out['pred_total_high_90'] = pred + z90 * sigma
        out['pred_total_low_95'] = pred - z95 * sigma
        out['pred_total_high_95'] = pred + z95 * sigma
    return out
