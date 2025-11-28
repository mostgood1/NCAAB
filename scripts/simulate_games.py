import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')

# Lightweight Monte Carlo simulation of game totals & margins using possession-based approximation.
# For each game: derive lambda_home, lambda_away scoring rates from projected scores and simulate distributions.
# Outputs summary percentiles for margin & total and writes to simulation_<date>.json.

def simulate_game(mu_home, mu_away, n=5000, seed=None):
    rng = np.random.default_rng(seed)
    # Poisson scoring approximation (simplistic) - could refine using variance inflation
    home_scores = rng.poisson(lam=max(mu_home,0), size=n)
    away_scores = rng.poisson(lam=max(mu_away,0), size=n)
    totals = home_scores + away_scores
    margins = home_scores - away_scores
    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'p05': float(np.quantile(arr,0.05)),
            'p25': float(np.quantile(arr,0.25)),
            'p50': float(np.quantile(arr,0.50)),
            'p75': float(np.quantile(arr,0.75)),
            'p95': float(np.quantile(arr,0.95)),
        }
    return stats(totals), stats(margins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, help='Target date YYYY-MM-DD (default today)')
    ap.add_argument('--samples', type=int, default=5000)
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--enriched-file', type=str, help='Explicit enriched file path for date')
    args = ap.parse_args()

    from datetime import datetime
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.enriched_file:
        enriched_path = Path(args.enriched_file)
    else:
        enriched_path = out_dir / f'predictions_unified_enriched_{date_str}.csv'

    if not enriched_path.exists():
        print('Missing enriched file', enriched_path)
        return

    df = pd.read_csv(enriched_path)
    if df.empty:
        print('Empty enriched file')
        return

    # Projected scores (ensure present)
    if {'proj_home','proj_away'}.issubset(df.columns):
        proj_home = pd.to_numeric(df['proj_home'], errors='coerce')
        proj_away = pd.to_numeric(df['proj_away'], errors='coerce')
    else:
        # Derive from pred_total & pred_margin if not already
        pt = pd.to_numeric(df.get('pred_total_model', df.get('pred_total')), errors='coerce')
        pm = pd.to_numeric(df.get('pred_margin_model', df.get('pred_margin')), errors='coerce')
        proj_home = (pt + pm)/2.0
        proj_away = (pt - pm)/2.0

    artifact = {'date': date_str, 'games': []}

    for idx, row in df.iterrows():
        mh = proj_home.iloc[idx]
        ma = proj_away.iloc[idx]
        if np.isnan(mh) or np.isnan(ma):
            continue
        tot_stats, mar_stats = simulate_game(mh, ma, n=args.samples, seed=idx)
        artifact['games'].append({
            'game_id': row.get('game_id'),
            'home_team': row.get('home_team'),
            'away_team': row.get('away_team'),
            'total_dist': tot_stats,
            'margin_dist': mar_stats
        })

    (out_dir / f'simulation_{date_str}.json').write_text(json.dumps(artifact, indent=2))
    print('Simulation artifact written:', out_dir / f'simulation_{date_str}.json')

if __name__ == '__main__':
    main()
