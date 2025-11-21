"""Predictability evaluation script.
Computes core predictive quality metrics for the given date and a trailing history window.

Outputs predictability_<date>.json with metrics:
  residual_mean, residual_std, residual_mae
  calibration_slope_total, calibration_intercept_total
  corr_pred_market_total (if market_total present)
  coverage_rows (rows with actual + pred)
  trailing_days_used
  trailing_residual_std, trailing_residual_mae, trailing_calibration_slope_total
  predictability_score (composite: lower residual_std & mae, slope near 1, corr high)

Usage:
  python scripts/predictability_eval.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs")
TRAILING_MAX_DAYS = 45
MIN_ROWS_CALIB = 25


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _load_daily_results(date_str: str) -> pd.DataFrame:
    return _safe_read_csv(OUT / "daily_results" / f"results_{date_str}.csv")


def _collect_trailing(date_str: str, limit_days: int = TRAILING_MAX_DAYS) -> pd.DataFrame:
    dr_dir = OUT / "daily_results"
    if not dr_dir.exists():
        return pd.DataFrame()
    files = sorted(dr_dir.glob("results_*.csv"))
    target_dt = dt.date.fromisoformat(date_str)
    parts = []
    for p in reversed(files):
        stem = p.stem.replace("results_", "")
        try:
            d = dt.date.fromisoformat(stem)
        except Exception:
            continue
        if (target_dt - d).days < 0:
            continue  # future
        if (target_dt - d).days > limit_days:
            break
        df = _safe_read_csv(p)
        if df.empty:
            continue
        df["_date_file"] = stem
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _calibration(actual: pd.Series, pred: pd.Series) -> tuple[float,float] | tuple[None,None]:
    actual = pd.to_numeric(actual, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    df = pd.DataFrame({"a": actual, "p": pred}).dropna()
    if len(df) < MIN_ROWS_CALIB or df["p"].nunique() < 5:
        return None, None
    try:
        vp = float(np.var(df["p"]))
        if vp <= 1e-8:
            return None, None
        cov = float(np.cov(df["p"], df["a"])[0,1])
        slope = cov / vp
        intercept = float(df["a"].mean()) - slope * float(df["p"].mean())
        return float(slope), float(intercept)
    except Exception:
        return None, None


def _predictability_score(residual_std: float | None, residual_mae: float | None, slope: float | None, corr: float | None) -> float | None:
    if None in (residual_std, residual_mae, slope, corr):
        return None
    # Normalize components with heuristic scales
    # Aim: lower std/mae -> higher score, slope near 1 -> higher, corr high -> higher
    std_component = max(0.0, 1.0 - (residual_std / 18.0))  # typical NCAA total residual std ~14-16
    mae_component = max(0.0, 1.0 - (residual_mae / 12.0))
    slope_component = max(0.0, 1.0 - abs(1.0 - slope))  # perfect at 1.0
    corr_component = max(0.0, corr)  # assume corr in [0,1]
    # Weighted blend
    score = 0.30 * std_component + 0.25 * mae_component + 0.25 * slope_component + 0.20 * corr_component
    return round(float(score), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default today)')
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime('%Y-%m-%d')

    today_df = _load_daily_results(date_str)
    if today_df.empty:
        payload = {
            'date': date_str,
            'generated_at': dt.datetime.utcnow().isoformat(),
            'status': 'no_daily_results'
        }
        (OUT / f'predictability_{date_str}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Predictability: no daily results file')
        return

    # Choose prediction column
    pred_col = 'pred_total_model' if 'pred_total_model' in today_df.columns else ('pred_total' if 'pred_total' in today_df.columns else None)
    if pred_col is None or 'actual_total' not in today_df.columns:
        payload = {
            'date': date_str,
            'generated_at': dt.datetime.utcnow().isoformat(),
            'status': 'missing_columns'
        }
        (OUT / f'predictability_{date_str}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Predictability: missing columns')
        return

    actual = pd.to_numeric(today_df['actual_total'], errors='coerce')
    pred = pd.to_numeric(today_df[pred_col], errors='coerce')
    mask = actual.notna() & pred.notna() & (actual > 0)
    residual = pred[mask] - actual[mask]
    residual_mean = float(residual.mean()) if len(residual) else None
    residual_std = float(residual.std()) if len(residual) else None
    residual_mae = float(residual.abs().mean()) if len(residual) else None

    slope, intercept = _calibration(actual[mask], pred[mask])

    corr_pred_market = None
    if 'market_total' in today_df.columns:
        mt = pd.to_numeric(today_df['market_total'], errors='coerce')
        df_corr = pd.DataFrame({'p': pred, 'm': mt}).dropna()
        if len(df_corr) >= 8 and df_corr['p'].nunique() > 3 and df_corr['m'].nunique() > 3:
            try:
                corr_pred_market = float(df_corr.corr().iloc[0,1])
            except Exception:
                pass

    trailing = _collect_trailing(date_str)
    trailing_metrics = {}
    if not trailing.empty and 'actual_total' in trailing.columns:
        pcol_trail = 'pred_total_model' if 'pred_total_model' in trailing.columns else ('pred_total' if 'pred_total' in trailing.columns else None)
        if pcol_trail:
            a_t = pd.to_numeric(trailing['actual_total'], errors='coerce')
            p_t = pd.to_numeric(trailing[pcol_trail], errors='coerce')
            mask_t = a_t.notna() & p_t.notna() & (a_t > 0)
            if mask_t.sum() >= 10:
                resid_t = p_t[mask_t] - a_t[mask_t]
                t_std = float(resid_t.std()) if resid_t.notna().any() else None
                t_mae = float(resid_t.abs().mean()) if resid_t.notna().any() else None
                slope_t, _ = _calibration(a_t[mask_t], p_t[mask_t])
                trailing_metrics = {
                    'trailing_days_used': int(trailing['_date_file'].nunique()),
                    'trailing_residual_std': t_std,
                    'trailing_residual_mae': t_mae,
                    'trailing_calibration_slope_total': slope_t
                }

    predictability_score = _predictability_score(residual_std, residual_mae, slope, corr_pred_market)

    payload = {
        'date': date_str,
        'generated_at': dt.datetime.utcnow().isoformat(),
        'status': 'ok',
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'residual_mae': residual_mae,
        'calibration_slope_total': slope,
        'calibration_intercept_total': intercept,
        'corr_pred_market_total': corr_pred_market,
        'coverage_rows': int(mask.sum()),
        'predictability_score': predictability_score,
        **trailing_metrics
    }
    OUT.mkdir(exist_ok=True, parents=True)
    out_path = OUT / f'predictability_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Predictability metrics -> {out_path}")

if __name__ == '__main__':
    main()
