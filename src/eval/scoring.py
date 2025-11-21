"""Daily Proper Scoring Rules

Computes CRPS and log-likelihood for totals and margins assuming Normal
prediction distributions using per-row mean (pred_total, pred_margin) and
sigma columns when available (pred_total_sigma / pred_margin_sigma) with
fallback constants.

Outputs JSON: outputs/scoring_<date>.json
Fields:
  date
  totals_crps_mean
  totals_loglik_mean
  margins_crps_mean
  margins_loglik_mean
  totals_rows
  margins_rows
  sigma_total_default_used
  sigma_margin_default_used

Usage:
  python -m src.eval.scoring --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt, pathlib
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[2] / 'outputs'

DEFAULT_TOTAL_SIGMA = 12.0
DEFAULT_MARGIN_SIGMA = 8.0

# Normal PDF and CDF
sqrt2 = np.sqrt(2.0)
inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

def phi(z: np.ndarray) -> np.ndarray:
    return (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*z*z)

def Phi(z: np.ndarray) -> np.ndarray:
    return 0.5*(1.0 + erf(z/np.sqrt(2)))

# Use numpy.erf via scipy fallback; implement simple approximation if missing
try:
    from math import erf
except ImportError:  # pragma: no cover
    def erf(x):
        # Numerical approximation (Abramowitz-Stegun 7.1.26)
        # Acceptable for evaluation context
        sign = np.sign(x)
        x = np.abs(x)
        a1,a2,a3,a4,a5,p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
        t = 1.0/(1.0 + p*x)
        y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
        return sign*y

def crps_normal(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CRPS for a Normal distribution (non-negative sigma)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (x - mu) / sigma
    # Formula: CRPS = sigma * [1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1)]
    return sigma * (inv_sqrt_pi - 2.0*phi(z) - z*(2.0*(0.5*(1.0 + erf(z/np.sqrt(2)))) - 1.0))

def loglik_normal(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    return -0.5*np.log(2*np.pi*sigma*sigma) - 0.5*((x-mu)/sigma)**2

def load_unified(date_str: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def load_results(date_str: str) -> pd.DataFrame:
    p = OUT / 'daily_results' / f'results_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def compute(date: str):
    uni = load_unified(date)
    res = load_results(date)
    if uni.empty or res.empty:
        return {
            'date': date,
            'totals_crps_mean': None,
            'totals_loglik_mean': None,
            'margins_crps_mean': None,
            'margins_loglik_mean': None,
            'totals_rows': 0,
            'margins_rows': 0,
            'sigma_total_default_used': True,
            'sigma_margin_default_used': True,
        }
    # Attach actual scores
    if {'game_id','home_score','away_score'}.issubset(res.columns) and 'game_id' in uni.columns:
        res['game_id'] = res['game_id'].astype(str)
        uni['game_id'] = uni['game_id'].astype(str)
        uni = uni.merge(res[['game_id','home_score','away_score']], on='game_id', how='left')
    actual_total = pd.to_numeric(uni.get('home_score'), errors='coerce') + pd.to_numeric(uni.get('away_score'), errors='coerce')
    actual_margin = pd.to_numeric(uni.get('home_score'), errors='coerce') - pd.to_numeric(uni.get('away_score'), errors='coerce')

    pred_total = pd.to_numeric(uni.get('pred_total'), errors='coerce')
    pred_margin = pd.to_numeric(uni.get('pred_margin'), errors='coerce')

    sigma_total = pd.to_numeric(uni.get('pred_total_sigma'), errors='coerce') if 'pred_total_sigma' in uni.columns else pd.Series([np.nan]*len(uni))
    sigma_margin = pd.to_numeric(uni.get('pred_margin_sigma'), errors='coerce') if 'pred_margin_sigma' in uni.columns else pd.Series([np.nan]*len(uni))

    if sigma_total.isna().all():
        sigma_total = pd.Series([DEFAULT_TOTAL_SIGMA]*len(uni))
        sigma_total_default = True
    else:
        # fill NaNs with mean or default
        m = sigma_total.dropna().mean() if sigma_total.dropna().any() else DEFAULT_TOTAL_SIGMA
        sigma_total = sigma_total.fillna(m)
        sigma_total_default = False
    if sigma_margin.isna().all():
        sigma_margin = pd.Series([DEFAULT_MARGIN_SIGMA]*len(uni))
        sigma_margin_default = True
    else:
        m2 = sigma_margin.dropna().mean() if sigma_margin.dropna().any() else DEFAULT_MARGIN_SIGMA
        sigma_margin = sigma_margin.fillna(m2)
        sigma_margin_default = False

    # Totals scoring
    mask_t = actual_total.notna() & pred_total.notna()
    crps_t = crps_normal(pred_total[mask_t].values, sigma_total[mask_t].values, actual_total[mask_t].values) if mask_t.any() else np.array([])
    ll_t = loglik_normal(pred_total[mask_t].values, sigma_total[mask_t].values, actual_total[mask_t].values) if mask_t.any() else np.array([])

    # Margins scoring
    mask_m = actual_margin.notna() & pred_margin.notna()
    crps_m = crps_normal(pred_margin[mask_m].values, sigma_margin[mask_m].values, actual_margin[mask_m].values) if mask_m.any() else np.array([])
    ll_m = loglik_normal(pred_margin[mask_m].values, sigma_margin[mask_m].values, actual_margin[mask_m].values) if mask_m.any() else np.array([])

    out = {
        'date': date,
        'totals_crps_mean': float(np.mean(crps_t)) if crps_t.size else None,
        'totals_loglik_mean': float(np.mean(ll_t)) if ll_t.size else None,
        'margins_crps_mean': float(np.mean(crps_m)) if crps_m.size else None,
        'margins_loglik_mean': float(np.mean(ll_m)) if ll_m.size else None,
        'totals_rows': int(mask_t.sum()),
        'margins_rows': int(mask_m.sum()),
        'sigma_total_default_used': sigma_total_default,
        'sigma_margin_default_used': sigma_margin_default,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, default=dt.date.today().strftime('%Y-%m-%d'))
    args = ap.parse_args()
    result = compute(args.date)
    path = OUT / f'scoring_{args.date}.json'
    path.write_text(json.dumps(result, indent=2))
    print(f"Wrote scoring metrics to {path}")

if __name__ == '__main__':
    main()
