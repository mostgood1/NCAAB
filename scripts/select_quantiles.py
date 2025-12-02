import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def _coerce_date(s):
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def compute_residual_quantiles(bt: pd.DataFrame, window_days: int, latest_date: str, target_cov: float):
    # Filter by window
    bt = bt.copy()
    if 'date' not in bt.columns:
        raise ValueError('backtest_joined.csv missing date column')
    bt['date_dt'] = pd.to_datetime(bt['date'], errors='coerce')
    ref_date = pd.to_datetime(latest_date, errors='coerce')
    if pd.isna(ref_date):
        ref_date = bt['date_dt'].max()
    start_date = ref_date - pd.Timedelta(days=window_days)
    win = bt[(bt['date_dt'] >= start_date) & (bt['date_dt'] <= ref_date)].copy()

    # Residuals for totals and margins
    def series_if(cols):
        for c in cols:
            if c in win.columns:
                return win[c]
        return pd.Series(dtype=float)

    actual_total = series_if(['actual_total'])
    pred_total = series_if(['pred_total_calibrated', 'pred_total'])
    actual_margin = series_if(['actual_margin'])
    pred_margin = series_if(['pred_margin_calibrated', 'pred_margin'])

    if actual_total.empty or pred_total.empty:
        raise ValueError('backtest_joined missing total columns')
    if actual_margin.empty or pred_margin.empty:
        raise ValueError('backtest_joined missing margin columns')

    resid_total = (actual_total - pred_total).dropna()
    resid_margin = (actual_margin - pred_margin).dropna()

    # Central interval quantile levels for target coverage
    eps = (1.0 - target_cov) / 2.0
    p_low, p_med, p_high = eps, 0.5, 1.0 - eps

    q_total = {
        'r_q_low': float(np.nanquantile(resid_total, p_low)) if len(resid_total) else np.nan,
        'r_q_med': float(np.nanquantile(resid_total, p_med)) if len(resid_total) else np.nan,
        'r_q_high': float(np.nanquantile(resid_total, p_high)) if len(resid_total) else np.nan,
    }
    q_margin = {
        'r_q_low': float(np.nanquantile(resid_margin, p_low)) if len(resid_margin) else np.nan,
        'r_q_med': float(np.nanquantile(resid_margin, p_med)) if len(resid_margin) else np.nan,
        'r_q_high': float(np.nanquantile(resid_margin, p_high)) if len(resid_margin) else np.nan,
    }

    # Realized coverage and width on window (diagnostics)
    def coverage_width(resid: pd.Series, ql: float, qh: float):
        if len(resid) == 0 or not np.isfinite(ql) or not np.isfinite(qh):
            return np.nan, np.nan
        cov = float(((resid >= ql) & (resid <= qh)).mean())
        width = float(qh - ql)
        return cov, width

    cov_t, w_t = coverage_width(resid_total, q_total['r_q_low'], q_total['r_q_high'])
    cov_m, w_m = coverage_width(resid_margin, q_margin['r_q_low'], q_margin['r_q_high'])

    diag = {
        'coverage_total': cov_t,
        'coverage_margin': cov_m,
        'width_total': w_t,
        'width_margin': w_m,
        'n_total': int(len(resid_total)),
        'n_margin': int(len(resid_margin)),
    }
    return q_total, q_margin, diag


def write_quantiles_for_today(pred_hist: pd.DataFrame, q_total: dict, q_margin: dict, out_dir: Path) -> tuple[pd.DataFrame, str]:
    if pred_hist.empty or 'date' not in pred_hist.columns:
        raise ValueError('predictions_history_enriched.csv missing date column')
    # Normalize game_id as string
    pred_hist = pred_hist.copy()
    pred_hist['game_id'] = pred_hist['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    latest = str(sorted(pred_hist['date'].dropna().astype(str).unique())[-1])
    today = pred_hist[pred_hist['date'].astype(str) == latest].copy()

    # Required prediction columns
    for col in ['pred_total', 'pred_margin']:
        if col not in today.columns:
            raise ValueError(f'missing column {col} in today predictions')

    # Apply residual quantiles
    today['q10_total'] = today['pred_total'] + q_total['r_q_low']
    today['q50_total'] = today['pred_total'] + q_total['r_q_med']
    today['q90_total'] = today['pred_total'] + q_total['r_q_high']
    today['q10_margin'] = today['pred_margin'] + q_margin['r_q_low']
    today['q50_margin'] = today['pred_margin'] + q_margin['r_q_med']
    today['q90_margin'] = today['pred_margin'] + q_margin['r_q_high']

    # Enforce monotonicity
    def mono(row, a, b, c):
        x, y, z = row[a], row[b], row[c]
        vals = [x, y, z]
        vals = sorted(vals)
        return pd.Series(vals, index=[a, b, c])

    today[['q10_total', 'q50_total', 'q90_total']] = today.apply(lambda r: mono(r, 'q10_total', 'q50_total', 'q90_total'), axis=1)
    today[['q10_margin', 'q50_margin', 'q90_margin']] = today.apply(lambda r: mono(r, 'q10_margin', 'q50_margin', 'q90_margin'), axis=1)

    # Persist history (append) and selected
    hist_cols = ['date', 'game_id', 'q10_total', 'q50_total', 'q90_total', 'q10_margin', 'q50_margin', 'q90_margin']
    qhist_path = out_dir / 'quantiles_history.csv'
    if qhist_path.exists():
        try:
            old = pd.read_csv(qhist_path)
            old['game_id'] = old['game_id'].astype(str)
        except Exception:
            old = pd.DataFrame(columns=hist_cols)
        # Drop existing rows for latest to avoid duplicates
        old = old[old['date'].astype(str) != latest]
        new_hist = pd.concat([old, today[hist_cols]], ignore_index=True)
    else:
        new_hist = today[hist_cols]
    new_hist.to_csv(qhist_path, index=False)

    qsel_path = out_dir / 'quantiles_selected.csv'
    today[hist_cols].to_csv(qsel_path, index=False)

    return today[hist_cols].copy(), latest


def main(window_days: int = 28, target_cov: float = 0.8):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'outputs'
    bt_path = out_dir / 'backtest_reports' / 'backtest_joined.csv'
    pred_path = out_dir / 'predictions_history_enriched.csv'

    bt = _read_csv_safe(bt_path)
    preds = _read_csv_safe(pred_path)
    if bt.empty:
        raise SystemExit('No backtest_joined.csv found or empty')
    if preds.empty:
        raise SystemExit('No predictions_history_enriched.csv found or empty')

    # Reference date from predictions
    latest = str(sorted(preds['date'].dropna().astype(str).unique())[-1])

    q_total, q_margin, diag = compute_residual_quantiles(bt, window_days, latest, target_cov)
    sel_df, latest_date = write_quantiles_for_today(preds, q_total, q_margin, out_dir)

    # Selection summary
    meta = {
        'method': 'residual_conformal_central',
        'window_days': window_days,
        'target_coverage': target_cov,
        'realized_coverage_total': diag['coverage_total'],
        'realized_coverage_margin': diag['coverage_margin'],
        'median_width_total': float(np.nanmedian(sel_df['q90_total'] - sel_df['q10_total'])) if not sel_df.empty else np.nan,
        'median_width_margin': float(np.nanmedian(sel_df['q90_margin'] - sel_df['q10_margin'])) if not sel_df.empty else np.nan,
        'n_total_window': diag['n_total'],
        'n_margin_window': diag['n_margin'],
        'latest_date': latest_date,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'source_files': {
            'backtest': str(bt_path.relative_to(root)),
            'predictions': str(pred_path.relative_to(root)),
        },
    }
    with open(out_dir / 'quantile_model_selection.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Wrote:', out_dir / 'quantiles_history.csv')
    print('Wrote:', out_dir / 'quantiles_selected.csv')
    print('Wrote:', out_dir / 'quantile_model_selection.json')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--window-days', type=int, default=28)
    p.add_argument('--target-coverage', type=float, default=0.8)
    args = p.parse_args()
    main(window_days=args.window_days, target_cov=args.target_coverage)
