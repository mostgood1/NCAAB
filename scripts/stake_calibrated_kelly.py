"""Generate stake sizing using calibrated probabilities and variance-aware Kelly.

Reads:
  - outputs/predictions_history_calibrated.csv (or today's predictions file if available)
  - outputs/predictions_history_sigma.csv (for sigma_total/sigma_margin)

Writes:
  - outputs/stake_sheet_calibrated.csv (stake %, EV, Kelly fraction, caps applied)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt

OUTPUTS = Path('outputs')

PRED_CAL = OUTPUTS / 'predictions_history_calibrated.csv'
PRED_SIG = OUTPUTS / 'predictions_history_sigma.csv'

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def american_to_decimal(a):
    try:
        a = float(a)
    except Exception:
        return np.nan
    if a > 0:
        return (a / 100.0) + 1
    else:
        return (100.0 / abs(a)) + 1

def kelly_fraction(p: float, dec_odds: float) -> float:
    if not np.isfinite(p) or not np.isfinite(dec_odds):
        return 0.0
    b = dec_odds - 1.0
    return max(0.0, (p * (b + 1) - 1) / b) if b > 0 else 0.0

def variance_scaler(sigma: float, base: float = 1.0) -> float:
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    # Scale down Kelly as uncertainty rises; inverse relation
    return min(1.0, base / sigma)

def main(use_quantiles: bool = True):
    cal = _safe_read(PRED_CAL)
    sig = _safe_read(PRED_SIG)
    if cal.empty:
        print('[stake] Missing calibrated predictions; aborting.')
        return
    # Merge sigma
    if not sig.empty and 'date' in sig.columns:
        cal = cal.merge(sig[['date','game_id','sigma_total','sigma_margin']], on=['date','game_id'], how='left')
    # Decide markets: totals (p_over_cal) and spreads (p_home_cover_dist_cal)
    # Odds price columns may vary; attempt common ones
    # Use closing or market odds if present for american lines
    over_price = cal.get('over_odds') if 'over_odds' in cal.columns else cal.get('closing_over_odds')
    home_price = cal.get('home_odds') if 'home_odds' in cal.columns else cal.get('closing_home_odds')
    cal['dec_over'] = pd.to_numeric(over_price, errors='coerce').apply(american_to_decimal) if over_price is not None else np.nan
    cal['dec_home'] = pd.to_numeric(home_price, errors='coerce').apply(american_to_decimal) if home_price is not None else np.nan
    # Probabilities
    cal['p_over_cal'] = pd.to_numeric(cal.get('p_over_cal'), errors='coerce')
    cal['p_home_cover_dist_cal'] = pd.to_numeric(cal.get('p_home_cover_dist_cal'), errors='coerce')
    # Kelly base
    cal['kelly_over'] = cal.apply(lambda r: kelly_fraction(r['p_over_cal'], r['dec_over']), axis=1)
    cal['kelly_home'] = cal.apply(lambda r: kelly_fraction(r['p_home_cover_dist_cal'], r['dec_home']), axis=1)
    # Optional quantile-aware exposure scaling using interval width
    if use_quantiles and {'total_p10','total_p90'}.issubset(cal.columns):
        try:
            width_total = (pd.to_numeric(cal['total_p90'], errors='coerce') - pd.to_numeric(cal['total_p10'], errors='coerce')).abs()
            width_total = width_total.replace(0, np.nan)
            cal['kelly_over'] = cal['kelly_over'] * (1.0 / width_total.clip(lower=1.0))
        except Exception:
            pass
    if use_quantiles and {'margin_p10','margin_p90'}.issubset(cal.columns):
        try:
            width_margin = (pd.to_numeric(cal['margin_p90'], errors='coerce') - pd.to_numeric(cal['margin_p10'], errors='coerce')).abs()
            width_margin = width_margin.replace(0, np.nan)
            cal['kelly_home'] = cal['kelly_home'] * (1.0 / width_margin.clip(lower=1.0))
        except Exception:
            pass
    # Variance scaling using sigma_total for totals, sigma_margin for spreads
    cal['kelly_over_var'] = cal.apply(lambda r: r['kelly_over'] * variance_scaler(r.get('sigma_total', np.nan)), axis=1)
    cal['kelly_home_var'] = cal.apply(lambda r: r['kelly_home'] * variance_scaler(r.get('sigma_margin', np.nan)), axis=1)
    # Caps
    cap = 0.05
    cal['stake_over'] = cal['kelly_over_var'].clip(lower=0.0, upper=cap)
    cal['stake_home'] = cal['kelly_home_var'].clip(lower=0.0, upper=cap)
    # Expected value
    cal['ev_over'] = cal['p_over_cal'] * (cal['dec_over'] - 1) - (1 - cal['p_over_cal'])
    cal['ev_home'] = cal['p_home_cover_dist_cal'] * (cal['dec_home'] - 1) - (1 - cal['p_home_cover_dist_cal'])
    # Output minimal stake sheet
    cols = [
        'date','game_id','home_team','away_team','market_total','spread_home',
        'p_over_cal','dec_over','kelly_over','stake_over','ev_over',
        'p_home_cover_dist_cal','dec_home','kelly_home','stake_home','ev_home',
    ]
    present = [c for c in cols if c in cal.columns]
    out = cal[present].copy()
    stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    out['generated_at'] = stamp
    out.to_csv(OUTPUTS / 'stake_sheet_calibrated.csv', index=False)
    print('[stake] Wrote outputs/stake_sheet_calibrated.csv')

if __name__ == '__main__':
    import sys
    use_q = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-quantiles':
        use_q = False
    main(use_quantiles=use_q)
