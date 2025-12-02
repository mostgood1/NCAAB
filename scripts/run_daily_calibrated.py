"""Daily wrapper to apply calibration + refined sigma and generate stake sheet.

Steps:
  - Load today's enriched predictions from outputs/predictions_history_enriched.csv
  - Apply persisted isotonic calibration params
  - Merge refined sigma by date/game_id
  - Recompute intervals
  - Write outputs/predictions_today_calibrated.csv
  - Invoke stake_calibrated_kelly.py to produce outputs/stake_sheet_calibrated.csv
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import subprocess

OUTPUTS = Path('outputs')

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def main(use_quantiles: bool = True, use_conformal: bool = True):
    enriched = _safe_read(OUTPUTS / 'predictions_history_enriched.csv')
    if enriched.empty or 'date' not in enriched.columns:
        print('[daily-cal] Missing enriched predictions; aborting.')
        return
    latest = str(sorted(enriched['date'].dropna().astype(str).unique())[-1])
    enriched['game_id'] = enriched['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    today_df = enriched[enriched['date'].astype(str) == latest].copy()
    # Apply calibration
    sys.path.append(str(Path('.').resolve() / 'src'))
    from calibration_utils import load_calibration_params, apply_calibration_to_df, apply_sigma_intervals
    params = load_calibration_params(OUTPUTS / 'calibration_params.json')
    if params:
        today_df = apply_calibration_to_df(today_df, params)
    # Merge refined sigma if present
    sigma_df = _safe_read(OUTPUTS / 'predictions_history_sigma.csv')
    if not sigma_df.empty and {'date','game_id'}.issubset(sigma_df.columns):
        sigma_df['game_id'] = sigma_df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
        today_df = today_df.merge(sigma_df[['date','game_id','sigma_total','sigma_margin']], on=['date','game_id'], how='left')
    # Recompute intervals with sigma
    today_df = apply_sigma_intervals(today_df, sigma_total_col='sigma_total')
    # Prefer quantile intervals when available
    qdf = _safe_read(OUTPUTS / 'quantiles_history.csv')
    if not qdf.empty and {'date','game_id'}.issubset(qdf.columns):
        qdf['game_id'] = qdf['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
        q_today = qdf[qdf['date'].astype(str) == latest]
        cols = [c for c in ['date','game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin'] if c in qdf.columns]
        if cols:
            today_df = today_df.merge(q_today[cols], on=['date','game_id'], how='left')
            if {'q10_total','q90_total'}.issubset(today_df.columns):
                today_df['total_p10'] = today_df['q10_total']
                today_df['total_p50'] = today_df.get('q50_total', today_df.get('pred_total'))
                today_df['total_p90'] = today_df['q90_total']
            if {'q10_margin','q90_margin'}.issubset(today_df.columns):
                today_df['margin_p10'] = today_df['q10_margin']
                today_df['margin_p50'] = today_df.get('q50_margin', today_df.get('pred_margin'))
                today_df['margin_p90'] = today_df['q90_margin']
    # If conformal-adjusted intervals exist and not disabled, override interval columns
    if use_conformal:
        cpath = OUTPUTS / 'quantiles_conformal_today.csv'
        cdf = _safe_read(cpath)
        if not cdf.empty and {'date','game_id'}.issubset(cdf.columns):
            cdf['game_id'] = cdf['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
            c_today = cdf[cdf['date'].astype(str) == latest]
            cols = [c for c in ['date','game_id','total_c10','total_c50','total_c90','margin_c10','margin_c50','margin_c90'] if c in cdf.columns]
            if cols:
                today_df = today_df.merge(c_today[cols], on=['date','game_id'], how='left')
                today_df['total_p10'] = today_df.get('total_c10', today_df.get('total_p10'))
                today_df['total_p50'] = today_df.get('total_c50', today_df.get('total_p50'))
                today_df['total_p90'] = today_df.get('total_c90', today_df.get('total_p90'))
                today_df['margin_p10'] = today_df.get('margin_c10', today_df.get('margin_p10'))
                today_df['margin_p50'] = today_df.get('margin_c50', today_df.get('margin_p50'))
                today_df['margin_p90'] = today_df.get('margin_c90', today_df.get('margin_p90'))
    # Write calibrated today
    out = OUTPUTS / 'predictions_today_calibrated.csv'
    today_df.to_csv(out, index=False)
    print(f'[daily-cal] Wrote {out} for date {latest}')
    # Invoke staking script
    try:
        cmd = ['python', 'scripts/stake_calibrated_kelly.py']
        if not use_quantiles:
            cmd.append('--no-quantiles')
        # (Optional: future flag for conformal-aware staking adjustments)
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f'[daily-cal] Staking generation failed: {e}')

if __name__ == '__main__':
    use_q = True
    use_conf = True
    args = sys.argv[1:]
    if '--no-quantiles' in args:
        use_q = False
    if '--no-conformal' in args:
        use_conf = False
    main(use_quantiles=use_q, use_conformal=use_conf)
