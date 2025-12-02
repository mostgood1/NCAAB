"""Produce drift/stability artifacts over time.

Computes weekly trends for Brier, log-loss, ECE (mean abs gap), interval
coverage, and quantile CRPS/80% coverage. Writes CSVs for monitoring.

Reads:
    - outputs/prob_method_summary.csv
    - outputs/prob_reliability_bins.csv
    - outputs/prob_metrics_history.json
    - outputs/daily_metrics.csv
    - outputs/quantile_metrics.csv

Writes:
    - outputs/drift_summary_weekly.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json

OUTPUTS = Path('outputs')

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def main():
    summary = _safe_read_csv(OUTPUTS / 'prob_method_summary.csv')
    bins = _safe_read_csv(OUTPUTS / 'prob_reliability_bins.csv')
    daily = _safe_read_csv(OUTPUTS / 'daily_metrics.csv')
    qmet = _safe_read_csv(OUTPUTS / 'quantile_metrics.csv')
    # Load conformal metrics if available (latest-only CSVs per date)
    conf_hist = []
    for p in OUTPUTS.glob('conformal_metrics_*.csv'):
        try:
            dfc = pd.read_csv(p)
            if 'date' in dfc.columns:
                conf_hist.append(dfc)
        except Exception:
            pass
    conf = pd.concat(conf_hist, ignore_index=True) if conf_hist else pd.DataFrame()
    metrics_json = OUTPUTS / 'prob_metrics_history.json'
    meta = {}
    if metrics_json.exists():
        meta = json.loads(metrics_json.read_text())
    # ECE approximation: weighted mean abs_gap across bins per method
    if not bins.empty:
        bins['count'] = pd.to_numeric(bins.get('count'), errors='coerce')
        bins['abs_gap'] = pd.to_numeric(bins.get('abs_gap'), errors='coerce')
        ece = bins.groupby('method', observed=False).apply(lambda g: np.average(g['abs_gap'].dropna(), weights=g['count'].fillna(0))).reset_index(name='ece')
    else:
        ece = pd.DataFrame(columns=['method','ece'])
    # Combine method-level summary
    out = summary.copy()
    if not ece.empty:
        out = out.merge(ece, on='method', how='left') if 'method' in out.columns else ece
    cov90 = meta.get('intervals', {}).get('coverage_90')
    cov95 = meta.get('intervals', {}).get('coverage_95')
    out['coverage_90'] = cov90
    out['coverage_95'] = cov95
    # Per-date weekly aggregation: interval coverage + quantile metrics
    # Build date frame
    per_date = pd.DataFrame()
    if not daily.empty and 'date' in daily.columns:
        keep = ['date']
        if 'interval_coverage_total' in daily.columns:
            keep.append('interval_coverage_total')
        if 'interval_coverage_margin' in daily.columns:
            keep.append('interval_coverage_margin')
        per_date = daily[keep].copy()
    if not qmet.empty and 'date' in qmet.columns:
        q_sel = qmet[['date','crps_total','crps_margin','covered_80_total','covered_80_margin']]
        per_date = q_sel if per_date.empty else per_date.merge(q_sel, on='date', how='outer')
    if not per_date.empty:
        per_date['date'] = pd.to_datetime(per_date['date'], errors='coerce')
        per_date['week'] = per_date['date'].dt.to_period('W').astype(str)
        agg_cols = {}
        for c in ('interval_coverage_total','interval_coverage_margin','crps_total','crps_margin','covered_80_total','covered_80_margin'):
            if c in per_date.columns:
                agg_cols[c] = 'mean'
        weekly = per_date.groupby('week', observed=False).agg(agg_cols).reset_index()
        if not conf.empty and 'date' in conf.columns:
            conf['date'] = pd.to_datetime(conf['date'], errors='coerce')
            conf['week'] = conf['date'].dt.to_period('W').astype(str)
            c_week = conf.groupby('week', observed=False).agg({
                'raw_covered_80_total':'mean',
                'conf_covered_80_total':'mean',
                'raw_covered_80_margin':'mean',
                'conf_covered_80_margin':'mean',
                'raw_width_total':'mean',
                'conf_width_total':'mean',
                'raw_width_margin':'mean',
                'conf_width_margin':'mean',
            }).reset_index()
            weekly = weekly.merge(c_week, on='week', how='left')
        # Add simple alert flags
        weekly['alert_low_coverage_total'] = (pd.to_numeric(weekly['covered_80_total'], errors='coerce') < 0.75).astype(int)
        weekly['alert_low_coverage_margin'] = (pd.to_numeric(weekly['covered_80_margin'], errors='coerce') < 0.75).astype(int)
        # Rolling median comparison for CRPS
        weekly['crps_total_med'] = pd.to_numeric(weekly['crps_total'], errors='coerce').rolling(4, min_periods=1).median()
        weekly['crps_margin_med'] = pd.to_numeric(weekly['crps_margin'], errors='coerce').rolling(4, min_periods=1).median()
        weekly['alert_high_crps_total'] = (pd.to_numeric(weekly['crps_total'], errors='coerce') > 1.10 * weekly['crps_total_med']).astype(int)
        weekly['alert_high_crps_margin'] = (pd.to_numeric(weekly['crps_margin'], errors='coerce') > 1.10 * weekly['crps_margin_med']).astype(int)
        # Integrate conformal multi-date metrics if present
        conf_all_path = OUTPUTS / 'conformal_metrics_all.csv'
        if conf_all_path.exists():
            conf_all = pd.read_csv(conf_all_path)
            cols_needed = {"date", "total_cov_delta", "margin_cov_delta", "total_width_ratio", "margin_width_ratio"}
            if cols_needed.issubset(set(conf_all.columns)):
                conf_all['date'] = pd.to_datetime(conf_all['date'], errors='coerce')
                conf_all['week'] = conf_all['date'].dt.to_period('W').astype(str)
                # Aggregate to week level (mean)
                c_week2 = conf_all.groupby('week', observed=False).agg({
                    'total_cov_delta':'mean',
                    'margin_cov_delta':'mean',
                    'total_width_ratio':'mean',
                    'margin_width_ratio':'mean',
                }).reset_index()
                weekly = weekly.merge(c_week2, on='week', how='left')
                # Simple alerts based on conformal deltas/ratios
                alerts = []
                # Create an alerts column combining conditions
                weekly['conformal_alerts'] = ''
                weekly.loc[pd.to_numeric(weekly.get('total_cov_delta'), errors='coerce') < 0, 'conformal_alerts'] = (
                    weekly.loc[pd.to_numeric(weekly.get('total_cov_delta'), errors='coerce') < 0, 'conformal_alerts'] + 'total_cov_down'
                )
                weekly.loc[pd.to_numeric(weekly.get('margin_cov_delta'), errors='coerce') < 0, 'conformal_alerts'] = (
                    weekly.loc[pd.to_numeric(weekly.get('margin_cov_delta'), errors='coerce') < 0, 'conformal_alerts'].astype(str).mask(weekly['conformal_alerts'] == '', '') +
                    (',' if True else '') + 'margin_cov_down'
                )
                weekly.loc[pd.to_numeric(weekly.get('total_width_ratio'), errors='coerce') > 1.25, 'conformal_alerts'] = (
                    weekly.loc[pd.to_numeric(weekly.get('total_width_ratio'), errors='coerce') > 1.25, 'conformal_alerts'].astype(str).mask(weekly['conformal_alerts'] == '', '') +
                    (',' if True else '') + 'total_width_up'
                )
                weekly.loc[pd.to_numeric(weekly.get('margin_width_ratio'), errors='coerce') > 1.25, 'conformal_alerts'] = (
                    weekly.loc[pd.to_numeric(weekly.get('margin_width_ratio'), errors='coerce') > 1.25, 'conformal_alerts'].astype(str).mask(weekly['conformal_alerts'] == '', '') +
                    (',' if True else '') + 'margin_width_up'
                )

        weekly.to_csv(OUTPUTS / 'drift_summary_weekly.csv', index=False)
        print('[drift] Wrote outputs/drift_summary_weekly.csv')
    else:
        # Fallback: write method-level summary if no per-date metrics
        out.to_csv(OUTPUTS / 'drift_summary_weekly.csv', index=False)
        print('[drift] Wrote outputs/drift_summary_weekly.csv (summary only)')

if __name__ == '__main__':
    main()
