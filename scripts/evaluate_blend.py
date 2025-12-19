from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate raw vs market-blend predictions against actuals for a date")
    p.add_argument('--date', required=True, help='YYYY-MM-DD')
    return p.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    return df


def compute_metrics(merged: pd.DataFrame) -> dict:
    def metric(col: str) -> dict:
        if col not in merged.columns:
            return {'n': 0}
        s = pd.to_numeric(merged[col], errors='coerce')
        a = pd.to_numeric(merged['actual_total'], errors='coerce')
        mask = s.notna() & a.notna()
        if not mask.any():
            return {'n': 0}
        err = s[mask] - a[mask]
        return {
            'n': int(mask.sum()),
            'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt((err**2).mean())),
            'bias': float(err.mean()),
        }

    base = 'pred_total_calibrated' if 'pred_total_calibrated' in merged.columns else 'pred_total'
    out = {
        'raw_total': metric(base),
        'blend_total': metric('pred_total_market_blend'),
        'market_total': metric('market_total'),
        'closing_total': metric('closing_total'),
    }
    # Distribution and correlation summaries
    def summary(col: str) -> dict:
        if col not in merged.columns:
            return {'n': 0}
        s = pd.to_numeric(merged[col], errors='coerce')
        a = pd.to_numeric(merged['actual_total'], errors='coerce')
        m = s.notna() & a.notna()
        if not m.any():
            return {'n': 0}
        corr = float(np.corrcoef(s[m], a[m])[0,1]) if m.sum() > 1 else float('nan')
        return {
            'n': int(m.sum()),
            'mean': float(s[m].mean()),
            'std': float(s[m].std(ddof=1)),
            'corr_vs_actual': corr,
        }
    out['raw_total_summary'] = summary(base)
    out['blend_total_summary'] = summary('pred_total_market_blend')
    out['market_total_summary'] = summary('market_total')
    return out


def main():
    args = parse_args()
    date = args.date
    pr_path = Path(f'outputs/predictions_unified_enriched_{date}.csv')
    res_path = Path(f'outputs/daily_results/results_{date}.csv')

    if not pr_path.exists() or not res_path.exists():
        print(json.dumps({'status': 'error', 'missing': [str(p) for p in [pr_path, res_path] if not p.exists()]}))
        return

    pr = load_csv(pr_path)
    res = load_csv(res_path)
    keep = [c for c in ['game_id','home_team','away_team','pred_total','pred_total_calibrated','pred_total_market_blend','market_total','closing_total'] if c in pr.columns]
    prv = pr[keep].copy()
    merged = prv.merge(res[['game_id','actual_total','actual_margin']], on='game_id', how='inner')
    metrics = compute_metrics(merged)
    out = {
        'date': date,
        'rows_joined': int(len(merged)),
        'metrics': metrics,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
