"""Conference fairness audit.
Computes residual bias and dispersion per conference to detect systematic over/under prediction.

Data sources:
  outputs/daily_results/results_<date>.csv  (needs actual_total, pred_total or pred_total_model)
  data/d1_conferences.csv (team -> conference mapping)

Metrics per conference:
  n_games
  mean_residual_total (pred - actual)
  mae_residual_total
  residual_std_total
  disparity_z_total = (mean_residual_total - global_mean) / global_std
Flags:
  bias_flag if abs(mean_residual_total) > bias_abs_threshold (default 2.0)
  disparity_flag if abs(disparity_z_total) > disparity_z_threshold (default 1.2)

Output JSON: outputs/fairness_<date>.json
Usage:
  python scripts/conference_fairness.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs")
DATA = Path("data")

BIAS_ABS_THRESHOLD = 2.0
DISPARITY_Z_THRESHOLD = 1.2
MIN_GAMES_CONFERENCE = 8


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _load_results(date_str: str) -> pd.DataFrame:
    return _safe_read_csv(OUT / "daily_results" / f"results_{date_str}.csv")


def _load_conference_map() -> dict[str, str]:
    p = DATA / "d1_conferences.csv"
    df = _safe_read_csv(p)
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("team") or cols.get("school") or list(df.columns)[0]
    ccol = cols.get("conference") or cols.get("conf") or cols.get("league") or (list(df.columns)[1] if len(df.columns) > 1 else None)
    if not tcol or not ccol:
        return {}
    out = {}
    for _, r in df.iterrows():
        team = str(r.get(tcol) or '').strip()
        conf = str(r.get(ccol) or '').strip()
        if team and conf:
            out[team.lower()] = conf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default today)')
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime('%Y-%m-%d')

    res = _load_results(date_str)
    cmap = _load_conference_map()

    if res.empty or 'home_team' not in res.columns or 'away_team' not in res.columns:
        payload = {
            'date': date_str,
            'generated_at': dt.datetime.utcnow().isoformat(),
            'status': 'no_data',
            'reason': 'missing results or team columns'
        }
        (OUT / f'fairness_{date_str}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Fairness audit: no data')
        return

    # Determine prediction column (prefer model calibrated > model > pred_total)
    pred_col = 'pred_total_model' if 'pred_total_model' in res.columns else ('pred_total' if 'pred_total' in res.columns else None)
    if pred_col is None or 'actual_total' not in res.columns:
        payload = {
            'date': date_str,
            'generated_at': dt.datetime.utcnow().isoformat(),
            'status': 'missing_prediction_or_actual'
        }
        (OUT / f'fairness_{date_str}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Fairness audit: missing prediction/actual columns')
        return

    res['home_team_low'] = res['home_team'].astype(str).str.lower()
    res['away_team_low'] = res['away_team'].astype(str).str.lower()
    res['conf_home'] = res['home_team_low'].map(cmap)
    res['conf_away'] = res['away_team_low'].map(cmap)

    # Residual per game (use same residual for both conferences participating)
    pred = pd.to_numeric(res[pred_col], errors='coerce')
    actual = pd.to_numeric(res['actual_total'], errors='coerce')
    residual = pred - actual
    res['residual_total'] = residual

    # Expand to conference rows (home + away) for fairness aggregation
    rows = []
    for _, r in res.iterrows():
        for side in ['conf_home', 'conf_away']:
            conf = r.get(side)
            if conf:
                rows.append({'conference': conf, 'residual_total': r.get('residual_total')})
    cdf = pd.DataFrame(rows)
    cdf['residual_total'] = pd.to_numeric(cdf['residual_total'], errors='coerce')
    cdf = cdf.dropna(subset=['residual_total'])

    if cdf.empty:
        payload = {
            'date': date_str,
            'generated_at': dt.datetime.utcnow().isoformat(),
            'status': 'no_conference_residuals'
        }
        (OUT / f'fairness_{date_str}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Fairness audit: no conference residuals')
        return

    global_mean = float(cdf['residual_total'].mean())
    global_std = float(cdf['residual_total'].std()) if cdf['residual_total'].std() > 0 else np.nan
    global_mae = float(cdf['residual_total'].abs().mean())

    records = []
    for conf, grp in cdf.groupby('conference'):
        n = len(grp)
        if n < MIN_GAMES_CONFERENCE:
            continue
        mean_res = float(grp['residual_total'].mean())
        mae_res = float(grp['residual_total'].abs().mean())
        std_res = float(grp['residual_total'].std()) if grp['residual_total'].std() > 0 else 0.0
        if not np.isnan(global_std) and global_std > 0:
            disparity_z = (mean_res - global_mean) / global_std
        else:
            disparity_z = np.nan
        bias_flag = abs(mean_res) > BIAS_ABS_THRESHOLD
        disparity_flag = not np.isnan(disparity_z) and abs(disparity_z) > DISPARITY_Z_THRESHOLD
        records.append({
            'conference': conf,
            'n_games': n,
            'mean_residual_total': mean_res,
            'mae_residual_total': mae_res,
            'residual_std_total': std_res,
            'disparity_z_total': disparity_z,
            'bias_flag': bias_flag,
            'disparity_flag': disparity_flag
        })

    # Summary extremes
    disparity_extreme = None
    bias_extreme = None
    if records:
        try:
            disparity_extreme = max(records, key=lambda r: abs(r['disparity_z_total']) if r['disparity_z_total'] is not None else -1)
        except Exception:
            pass
        try:
            bias_extreme = max(records, key=lambda r: abs(r['mean_residual_total']))
        except Exception:
            pass

    payload = {
        'date': date_str,
        'generated_at': dt.datetime.utcnow().isoformat(),
        'status': 'ok',
        'global': {
            'mean_residual_total': global_mean,
            'std_residual_total': global_std,
            'mae_residual_total': global_mae
        },
        'records': records,
        'extremes': {
            'disparity_extreme': disparity_extreme,
            'bias_extreme': bias_extreme
        }
    }
    OUT.mkdir(exist_ok=True, parents=True)
    out_path = OUT / f'fairness_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Conference fairness audit -> {out_path} (records={len(records)})")

if __name__ == '__main__':
    main()
