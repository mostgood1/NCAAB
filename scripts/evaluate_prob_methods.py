import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Simple ECE (Expected Calibration Error) with fixed bins
def compute_ece(p: pd.Series, y: pd.Series, bins: int = 10) -> float:
    try:
        p = pd.to_numeric(p, errors='coerce').clip(0,1)
        y = pd.to_numeric(y, errors='coerce')
        valid = p.notna() & y.notna()
        p = p[valid]; y = y[valid]
        if p.empty:
            return np.nan
        bin_edges = np.linspace(0,1,bins+1)
        ece = 0.0
        n = len(p)
        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (p >= lo) & (p < hi) if i < bins-1 else (p >= lo) & (p <= hi)
            if not mask.any():
                continue
            avg_p = p[mask].mean()
            avg_y = y[mask].mean()
            ece += (mask.sum()/n) * abs(avg_p - avg_y)
        return float(ece)
    except Exception:
        return np.nan

PROB_COLUMNS = [
    ('p_home_cover','cover'),
    ('p_home_cover_dist','cover'),
    ('p_home_cover_cdf','cover'),
    ('p_home_cover_ensemble','cover'),
    ('p_home_cover_meta','cover'),
    ('p_over','over'),
    ('p_over_dist','over'),
    ('p_over_cdf','over'),
    ('p_over_ensemble','over'),
    ('p_over_meta','over'),
]

TARGET_MAP = {
    'cover': 'ats_home_cover',
    'over': 'ou_over',
}

def load_enriched(limit_days: int | None = None, out: Path = Path('outputs')) -> pd.DataFrame:
    files = sorted(out.glob('predictions_unified_enriched_*.csv'))
    if limit_days:
        files = files[-limit_days:]
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            if 'date' not in df.columns:
                date_part = f.name.replace('predictions_unified_enriched_','').replace('.csv','')
                df['date'] = date_part
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_outcomes(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out.glob('daily_results/results_*.csv')):
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    z = pd.concat(rows, ignore_index=True)
    z['ats_home_cover'] = (z.get('home_cover') if 'home_cover' in z.columns else z.get('ats_home_cover'))
    z['ou_over'] = (z.get('over') if 'over' in z.columns else z.get('ou_over'))
    return z


def evaluate(df: pd.DataFrame, outcomes: pd.DataFrame):
    keys = [k for k in ['game_id','date'] if k in df.columns and k in outcomes.columns]
    merged = df.merge(outcomes[['game_id','date','ats_home_cover','ou_over']], on=keys, how='inner') if keys else pd.DataFrame()
    if merged.empty:
        raise ValueError('No merged rows for evaluation.')
    results = {}
    for col, kind in PROB_COLUMNS:
        if col not in merged.columns:
            continue
        target = TARGET_MAP[kind]
        y = merged[target]
        p = merged[col]
        # Binary conversion (allow numeric already 0/1)
        yb = (pd.to_numeric(y, errors='coerce') > 0.5).astype(int)
        pnum = pd.to_numeric(p, errors='coerce')
        mask = pnum.notna() & yb.notna()
        if mask.sum() < 10:
            continue
        p_clean = pnum[mask].clip(0,1)
        y_clean = yb[mask]
        # Metrics
        try:
            from sklearn.metrics import brier_score_loss, log_loss
            brier = brier_score_loss(y_clean, p_clean)
            ll = log_loss(y_clean, p_clean)
        except Exception:
            brier = np.nan
            ll = np.nan
        ece = compute_ece(p_clean, y_clean, bins=10)
        sharp = float(np.sqrt(np.mean((p_clean - p_clean.mean())**2)))  # probability dispersion
        results[col] = {
            'kind': kind,
            'rows': int(mask.sum()),
            'brier': float(brier),
            'log_loss': float(ll),
            'ece': float(ece),
            'mean_p': float(p_clean.mean()),
            'sharpness_std': sharp,
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=None, help='Limit number of enriched prediction days used')
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--outfile', type=str, default='prob_method_eval.json')
    args = ap.parse_args()
    out_path = Path(args.out_dir)
    preds = load_enriched(args.limit_days, out_path)
    outcomes = load_outcomes(out_path)
    if preds.empty or outcomes.empty:
        print('Missing prediction or outcome data.')
        return
    results = evaluate(preds, outcomes)
    artifact = {
        'methods': results,
        'days_considered': args.limit_days,
    }
    (out_path/args.outfile).write_text(json.dumps(artifact, indent=2))
    print('Probability method evaluation complete:', json.dumps(artifact, indent=2))

if __name__ == '__main__':
    main()
