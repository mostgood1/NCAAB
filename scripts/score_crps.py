import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')

# CRPS for a normal distribution N(mu, sigma^2) evaluated at observation x
# Formula: CRPS = sigma * [1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1)] where z=(x-mu)/sigma
# Implement via error function for numerical stability
from math import erf, sqrt, exp, pi

def _phi(z):
    return (1.0/np.sqrt(2*pi)) * np.exp(-0.5*z*z)

def _Phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))

def crps_normal(mu, sigma, x):
    if sigma <= 0 or np.isnan(sigma):
        return np.nan
    z = (x - mu) / sigma
    return sigma * (1.0/np.sqrt(pi) - 2*_phi(z) - z*(2*_Phi(z)-1))

def safe_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, help='Target date YYYY-MM-DD (defaults to today local)')
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--enriched-file', type=str, help='Explicit path to enriched predictions file')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = args.date
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')

    # Prefer explicit file, else conventional naming
    if args.enriched_file:
        enriched_path = Path(args.enriched_file)
    else:
        enriched_path = out_dir / f'predictions_unified_enriched_{date_str}.csv'
    if not enriched_path.exists():
        print('Enriched file missing:', enriched_path)
        return
    df = pd.read_csv(enriched_path)
    if df.empty:
        print('Enriched file empty')
        return

    # Require actual scores to compute realized margin/total
    if {'home_score','away_score'}.issubset(df.columns):
        hs = safe_numeric(df['home_score'])
        as_ = safe_numeric(df['away_score'])
        actual_total = hs + as_
        actual_margin = hs - as_
    else:
        print('Missing scores for CRPS; abort.')
        return

    # Predicted means
    mu_total = safe_numeric(df.get('pred_total_model', df.get('pred_total')))
    mu_margin = safe_numeric(df.get('pred_margin_model', df.get('pred_margin')))

    # Sigma resolution: prefer per-row columns else fallback constant heuristics
    sigma_total = safe_numeric(df.get('pred_total_sigma_bootstrap', df.get('pred_total_sigma', None)))
    sigma_margin = safe_numeric(df.get('pred_margin_sigma', None))

    # Fallback heuristics
    if sigma_total is None or sigma_total.isna().all():
        sigma_total = pd.Series([11.0]*len(df))
    if sigma_margin is None or sigma_margin.isna().all():
        sigma_margin = pd.Series([10.5]*len(df))

    # CRPS calculations
    crps_t = [crps_normal(m, s, x) for m, s, x in zip(mu_total, sigma_total, actual_total)]
    crps_m = [crps_normal(m, s, x) for m, s, x in zip(mu_margin, sigma_margin, actual_margin)]

    # Probability metrics if available
    cover_actual = None
    over_actual = None
    if 'spread_home' in df.columns:
        sp = safe_numeric(df['spread_home'])
        cover_actual = (actual_margin > sp).astype(int)
    if 'market_total' in df.columns:
        mt = safe_numeric(df['market_total'])
        over_actual = (actual_total > mt).astype(int)

    prob_metrics = {}
    def _prob_eval(prob_col, actual):
        pc = safe_numeric(df[prob_col])
        mask = pc.notna() & actual.notna()
        if mask.any():
            obs = actual[mask]
            pred = pc[mask]
            # Brier
            brier = float(np.mean((pred - obs)**2))
            # Log-loss with clipping
            pred_clip = np.clip(pred, 1e-9, 1-1e-9)
            ll = float(-np.mean(obs*np.log(pred_clip) + (1-obs)*np.log(1-pred_clip)))
            return {'brier': brier, 'log_loss': ll, 'samples': int(mask.sum())}
        return None

    if cover_actual is not None:
        for c in ['p_home_cover','p_home_cover_dist','p_home_cover_cdf','p_home_cover_skew','p_home_cover_mix','p_home_cover_piecewise_ext','p_home_cover_kde','p_home_cover_ensemble','p_home_cover_final','p_home_cover_meta','p_home_cover_cal','p_home_cover_mix_cal','p_home_cover_skew_cal','p_home_cover_piecewise_ext_cal','p_home_cover_kde_cal','p_home_cover_cdf_cal']:
            if c in df.columns:
                r = _prob_eval(c, cover_actual)
                if r:
                    prob_metrics[c] = r
    if over_actual is not None:
        for c in ['p_over','p_over_dist','p_over_cdf','p_over_skew','p_over_mix','p_over_piecewise_ext','p_over_kde','p_over_ensemble','p_over_final','p_over_meta','p_over_cal','p_over_mix_cal','p_over_skew_cal','p_over_piecewise_ext_cal','p_over_kde_cal','p_over_cdf_cal']:
            if c in df.columns:
                r = _prob_eval(c, over_actual)
                if r:
                    prob_metrics[c] = r

    crps_total_series = pd.Series(crps_t).replace([np.inf,-np.inf], np.nan)
    crps_margin_series = pd.Series(crps_m).replace([np.inf,-np.inf], np.nan)

    artifact = {
        'date': date_str,
        'crps_total_mean': float(crps_total_series.dropna().mean()) if crps_total_series.notna().any() else None,
        'crps_margin_mean': float(crps_margin_series.dropna().mean()) if crps_margin_series.notna().any() else None,
        'crps_total_median': float(crps_total_series.dropna().median()) if crps_total_series.notna().any() else None,
        'crps_margin_median': float(crps_margin_series.dropna().median()) if crps_margin_series.notna().any() else None,
        'prob_metrics': prob_metrics,
        'count_games': int(len(df))
    }

    (out_dir / f'scoring_{date_str}.json').write_text(json.dumps(artifact, indent=2))
    print('Scoring artifact written:', out_dir / f'scoring_{date_str}.json')

if __name__ == '__main__':
    main()
