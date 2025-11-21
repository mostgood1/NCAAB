"""Generate simple scoring metrics (CRPS/log-likelihood approximations).

Usage:
  python scripts/scoring_generate.py --date 2025-11-19

If distributional predictions not available, approximates CRPS using normal assumption
with mean=pred_total and std from interval width if present.

Outputs: outputs/scoring_<date>.json
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd
import math

OUT = Path('outputs')

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def normal_crps(mu: float, sigma: float, x: float) -> float:
    """Closed form CRPS for normal distribution."""
    if sigma is None or sigma <= 0 or any(math.isnan(v) for v in [mu, x]):
        return math.nan
    # Reference formula: CRPS(N(mu,sigma); x) = sigma * [1/sqrt(pi) - 2 phi(z) - z(2 Phi(z)-1)] where z=(x-mu)/sigma
    z = (x - mu) / sigma
    from math import sqrt, pi
    # phi and Phi
    phi = math.exp(-0.5*z*z)/math.sqrt(2*math.pi)
    # Approximate Phi via 0.5*(1+erf(z/sqrt(2)))
    Phi = 0.5*(1+math.erf(z/math.sqrt(2)))
    return sigma*(1/math.sqrt(math.pi) - 2*phi - z*(2*Phi - 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default: yesterday)')
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime('%Y-%m-%d')

    results = _safe_csv(OUT / 'daily_results' / f'results_{date_str}.csv')
    preds = _safe_csv(OUT / f'predictions_unified_{date_str}.csv')
    if preds.empty:
        preds = _safe_csv(OUT / f'predictions_model_{date_str}.csv')

    payload = {'date': date_str, 'generated_at': dt.datetime.now().isoformat(), 'rows': 0}
    try:
        if results.empty or 'home_score' not in results.columns or 'away_score' not in results.columns:
            payload['status'] = 'no_results'
        else:
            results['game_id'] = results.get('game_id', results.get('id')).astype(str)
            results['actual_total'] = pd.to_numeric(results['home_score'], errors='coerce') + pd.to_numeric(results['away_score'], errors='coerce')
            merged = results.merge(preds[['game_id','pred_total']].copy() if 'pred_total' in preds.columns else results[['game_id']], on='game_id', how='left')
            mu = pd.to_numeric(merged.get('pred_total'), errors='coerce')
            at = pd.to_numeric(merged.get('actual_total'), errors='coerce')
            # Attempt std from interval predictions or fallback constant
            sigma = None
            for cand_low, cand_high in [('pred_total_p10','pred_total_p90'),('pred_total_lo','pred_total_hi')]:
                if cand_low in preds.columns and cand_high in preds.columns:
                    low = pd.to_numeric(preds[cand_low], errors='coerce')
                    high = pd.to_numeric(preds[cand_high], errors='coerce')
                    # approx: middle 80% width ~ 2.563*sigma for normal -> sigma ~ width/2.563
                    sigma_series = (high - low) / 2.563
                    sigma = sigma_series.reindex(mu.index) if len(sigma_series)==len(mu) else None
                    break
            if sigma is None:
                sigma = pd.Series([12.0]*len(mu))  # conservative early-season variance constant
            crps_vals = []
            for m,a,s in zip(mu, at, sigma):
                if math.isnan(m) or math.isnan(a) or math.isnan(s):
                    crps_vals.append(math.nan)
                else:
                    crps_vals.append(normal_crps(m, s, a))
            merged['crps_total'] = crps_vals
            payload['rows'] = int(len(merged))
            payload['crps_total_mean'] = float(pd.Series(crps_vals).mean(skipna=True)) if len(merged) else None
            # Log-likelihood approximate (normal)
            ll_vals = []
            for m,a,s in zip(mu, at, sigma):
                if math.isnan(m) or math.isnan(a) or math.isnan(s) or s <= 0:
                    ll_vals.append(math.nan)
                else:
                    ll_vals.append(-0.5*math.log(2*math.pi*s*s) - 0.5*((a-m)**2)/(s*s))
            merged['ll_total'] = ll_vals
            payload['ll_total_mean'] = float(pd.Series(ll_vals).mean(skipna=True)) if len(merged) else None
            # Moneyline Brier score and ECE if model probabilities available
            try:
                # Build outcomes: 1 if home won
                results['home_win'] = (pd.to_numeric(results['home_score'], errors='coerce') > pd.to_numeric(results['away_score'], errors='coerce')).astype(float)
                # Join model probs if present
                if 'ml_prob_model' in preds.columns:
                    ml = preds[['game_id','ml_prob_model']].copy()
                    ml['game_id'] = ml['game_id'].astype(str)
                    mlm = results[['game_id','home_win']].merge(ml, on='game_id', how='left')
                    if not mlm.empty and mlm['ml_prob_model'].notna().any():
                        p = pd.to_numeric(mlm['ml_prob_model'], errors='coerce')
                        y = pd.to_numeric(mlm['home_win'], errors='coerce')
                        brier = ((p - y) ** 2)
                        payload['brier_ml_mean'] = float(brier.mean(skipna=True))
                        # ECE (Expected Calibration Error) with 10 bins
                        bins = pd.cut(p, bins=[i/10 for i in range(11)], include_lowest=True)
                        ece = 0.0
                        total_n = 0
                        for b, grp in mlm.groupby(bins):
                            grp = grp.dropna(subset=['ml_prob_model','home_win'])
                            n = len(grp)
                            if n == 0:
                                continue
                            conf = float(pd.to_numeric(grp['ml_prob_model']).mean())
                            acc = float(pd.to_numeric(grp['home_win']).mean())
                            ece += (n * abs(acc - conf))
                            total_n += n
                        payload['ece_ml'] = float(ece / total_n) if total_n > 0 else None
            except Exception:
                pass
    except Exception as e:
        payload['error'] = str(e)

    out_path = OUT / f'scoring_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote scoring metrics to {out_path}')

if __name__ == '__main__':
    main()
