"""
Run synthetic pipeline end-to-end:
- Load latest enriched predictions
- Apply isotonic calibration and sigma intervals
- Prefer quantile intervals if quantile_model.json or quantiles_history.csv available
- Persist calibrated snapshot and run stake sizing script
"""
from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path
import pandas as pd

OUTPUTS = Path(os.getcwd()) / "outputs"

def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def latest_date_from(df: pd.DataFrame, col: str) -> str | None:
    if df.empty or col not in df.columns:
        return None
    try:
        ser = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        vals = sorted(ser.dropna().unique())
        return vals[-1] if vals else None
    except Exception:
        vals = sorted(df[col].astype(str).dropna().unique())
        return vals[-1] if vals else None

def apply_quantiles(today_df: pd.DataFrame, latest: str) -> pd.DataFrame:
    # Prefer quantiles_history when available; else use quantile_model residuals if present
    qhist = safe_read_csv(OUTPUTS / 'quantiles_history.csv')
    if not qhist.empty and {'date','game_id'}.issubset(qhist.columns):
        qhist['game_id'] = qhist['game_id'].astype(str)
        qlatest = qhist[qhist['date'].astype(str) == str(latest)]
        cols = [c for c in ['date','game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin'] if c in qhist.columns]
        if cols and not qlatest.empty:
            today_df = today_df.merge(qlatest[cols], on=['date','game_id'], how='left')
            # Map to p10/p50/p90 field names
            if {'q10_total','q90_total'}.issubset(today_df.columns):
                today_df['total_p10'] = today_df['q10_total']
                today_df['total_p50'] = today_df.get('q50_total', today_df.get('pred_total'))
                today_df['total_p90'] = today_df['q90_total']
            if {'q10_margin','q90_margin'}.issubset(today_df.columns):
                today_df['margin_p10'] = today_df['q10_margin']
                today_df['margin_p50'] = today_df.get('q50_margin', today_df.get('pred_margin'))
                today_df['margin_p90'] = today_df['q90_margin']
            return today_df
    # Fallback: use residual quantile model if available
    qm_path = OUTPUTS / 'quantile_model.json'
    try:
        if qm_path.exists():
            with open(qm_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
            rq = model.get('residual_quantiles', {})
            qt = rq.get('total', {})
            qm = rq.get('margin', {})
            def _add_resid(df, key, pred_col, p10, p50, p90):
                s = pd.to_numeric(df.get(pred_col), errors='coerce')
                df[f'{key}_p10'] = s + float(p10)
                df[f'{key}_p50'] = s + float(p50)
                df[f'{key}_p90'] = s + float(p90)
                return df
            if {'pred_total'}.issubset(today_df.columns) and qt:
                today_df = _add_resid(today_df, 'total', 'pred_total', qt.get('p10', 0.0), qt.get('p50', 0.0), qt.get('p90', 0.0))
            if {'pred_margin'}.issubset(today_df.columns) and qm:
                today_df = _add_resid(today_df, 'margin', 'pred_margin', qm.get('p10', 0.0), qm.get('p50', 0.0), qm.get('p90', 0.0))
    except Exception:
        pass
    return today_df


def main():
    enriched = safe_read_csv(OUTPUTS / 'predictions_history_enriched.csv')
    if enriched.empty or 'date' not in enriched.columns:
        print('[synthetic] no enriched history found; abort')
        return
    enriched['game_id'] = enriched['game_id'].astype(str)
    latest = latest_date_from(enriched, 'date') or dt.datetime.utcnow().strftime('%Y-%m-%d')
    today_df = enriched[enriched['date'].astype(str) == str(latest)].copy()
    # Calibration and sigma
    try:
        from src.calibration_utils import load_calibration_params, apply_calibration_to_df, apply_sigma_intervals
    except Exception:
        import sys
        sys.path.append(str(Path(os.getcwd()) / 'src'))
        from calibration_utils import load_calibration_params, apply_calibration_to_df, apply_sigma_intervals
    params = load_calibration_params(OUTPUTS / 'calibration_params.json')
    if params:
        today_df = apply_calibration_to_df(today_df, params)
    sigma_df = safe_read_csv(OUTPUTS / 'predictions_history_sigma.csv')
    if not sigma_df.empty and {'date','game_id'}.issubset(sigma_df.columns):
        sigma_df['game_id'] = sigma_df['game_id'].astype(str)
        sigma_df = sigma_df[sigma_df['date'].astype(str) == str(latest)]
        today_df = today_df.merge(sigma_df[['date','game_id','sigma_total','sigma_margin']], on=['date','game_id'], how='left')
    today_df = apply_sigma_intervals(today_df, sigma_total_col='sigma_total')
    # Quantiles preference
    today_df = apply_quantiles(today_df, str(latest))
    out = OUTPUTS / 'predictions_today_calibrated.csv'
    today_df.to_csv(out, index=False)
    print(f'[synthetic] wrote {out} rows={len(today_df)}')
    # Stake sizing
    try:
        import subprocess
        subprocess.run(['python', 'scripts/stake_calibrated_kelly.py'], check=False)
    except Exception:
        pass

if __name__ == '__main__':
    main()
