"""Aggregate quantile CRPS and coverage trends.

Reads outputs/quantile_metrics.csv and produces:
- outputs/quantile_trend_2w.csv: last 14 days CRPS (total/margin) and 80% coverage
- outputs/quantile_trend_weekly.csv: weekly averages + alerts (low coverage, rising CRPS)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path('outputs')


def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main():
    qmet = _safe_read(OUTPUTS / 'quantile_metrics.csv')
    if qmet.empty or 'date' not in qmet.columns:
        print('[quant-trend] Missing quantile metrics; aborting.')
        return
    qmet['date'] = pd.to_datetime(qmet['date'], errors='coerce')
    qmet = qmet.sort_values('date')
    # Last 14 days
    last_date = qmet['date'].dropna().max()
    if pd.isna(last_date):
        print('[quant-trend] No valid dates.')
        return
    start_2w = last_date - pd.Timedelta(days=14)
    recent = qmet[qmet['date'] >= start_2w].copy()
    cols = ['date','crps_total','crps_margin','covered_80_total','covered_80_margin']
    recent[cols].to_csv(OUTPUTS / 'quantile_trend_2w.csv', index=False)

    # Weekly aggregation + alerts
    qmet['week'] = qmet['date'].dt.to_period('W').astype(str)
    agg = qmet.groupby('week', observed=False).agg({
        'crps_total':'mean',
        'crps_margin':'mean',
        'covered_80_total':'mean',
        'covered_80_margin':'mean',
    }).reset_index()
    # Rolling medians for CRPS
    agg['crps_total_med'] = pd.to_numeric(agg['crps_total'], errors='coerce').rolling(4, min_periods=1).median()
    agg['crps_margin_med'] = pd.to_numeric(agg['crps_margin'], errors='coerce').rolling(4, min_periods=1).median()
    agg['alert_low_cov_total'] = (pd.to_numeric(agg['covered_80_total'], errors='coerce') < 0.75).astype(int)
    agg['alert_low_cov_margin'] = (pd.to_numeric(agg['covered_80_margin'], errors='coerce') < 0.75).astype(int)
    agg['alert_high_crps_total'] = (pd.to_numeric(agg['crps_total'], errors='coerce') > 1.10 * agg['crps_total_med']).astype(int)
    agg['alert_high_crps_margin'] = (pd.to_numeric(agg['crps_margin'], errors='coerce') > 1.10 * agg['crps_margin_med']).astype(int)
    agg.to_csv(OUTPUTS / 'quantile_trend_weekly.csv', index=False)
    print('[quant-trend] Wrote outputs/quantile_trend_2w.csv and outputs/quantile_trend_weekly.csv')


if __name__ == '__main__':
    main()
