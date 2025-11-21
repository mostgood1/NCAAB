"""Auto recalibration trigger script.
Evaluates residual and scoring metrics vs thresholds; writes flag JSON.

Dynamic threshold enhancement:
    - Computes trailing window (up to 14 prior days) of residual means & correlations
    - Derives z-scores for today's metrics vs trailing mean/std
    - Triggers recalibration when z-score magnitude exceeds configurable limits even if raw absolute thresholds not met.

Added metrics keys:
    total_mean_z, margin_mean_z, total_corr_z, margin_corr_z
    trailing_days_used

Z-score thresholds (tunable):
    abs(total_mean_z) > 2.2 -> 'totals_mean_z_extreme'
    abs(margin_mean_z) > 2.2 -> 'margin_mean_z_extreme'
    total_corr_z < -2.0 -> 'totals_corr_z_drop'
    margin_corr_z < -2.0 -> 'margin_corr_z_drop'

Usage:
  python scripts/recalibration_check.py --date YYYY-MM-DD

Output: outputs/recalibration_<date>.json
Structure: {"date":..., "generated_at":..., "recalibration_needed": bool, "reasons": [...], "metrics": {...}}

Heuristics (example thresholds, tune later):
  abs(resid_total_mean) > 2.5 -> reason 'totals_mean_drift'
  abs(resid_margin_mean) > 1.5 -> reason 'margin_mean_drift'
  total_corr < 0.25 -> reason 'totals_corr_low'
  margin_corr < 0.10 -> reason 'margin_corr_low'
  crps_total_last > crps_total_30d_avg * 1.15 -> 'crps_total_degradation'
(Gracefully skips checks when inputs missing.)
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd

OUT = Path("outputs")

def _json(path: Path):
    try:
        if path.exists():
            import json as _j
            return _j.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default: today)')
    args = ap.parse_args()
    date_str = args.date or dt.datetime.now().strftime('%Y-%m-%d')
    residuals = _json(OUT / f'residuals_{date_str}.json')
    scoring = _json(OUT / f'scoring_{date_str}.json')
    # Optionally compute 30d avg CRPS if historical scoring files exist
    crps_total_30 = None
    try:
        crps_vals = []
        for p in sorted(OUT.glob('scoring_*.json')):
            if p.name.endswith('.json'):
                data = _json(p)
                if isinstance(data, dict):
                    tv = data.get('crps_total') or data.get('crps_total_mean')
                    if isinstance(tv, (int,float)):
                        crps_vals.append(float(tv))
        if len(crps_vals) >= 5:
            crps_total_30 = sum(crps_vals[-30:]) / len(crps_vals[-30:])
    except Exception:
        pass
    # Gather trailing history for dynamic thresholds (exclude today)
    trailing_residual_means_total = []
    trailing_residual_means_margin = []
    trailing_corr_total = []
    trailing_corr_margin = []
    try:
        # Look back up to 30 days but only use max 14 most recent with data
        hist_files = sorted([p for p in OUT.glob('residuals_*.json') if p.name != f'residuals_{date_str}.json'])
        # Extract date ordering
        def _dt_from_name(p: Path):
            try:
                return dt.datetime.fromisoformat(p.name.replace('residuals_','').replace('.json',''))
            except Exception:
                return None
        hist_files = [p for p in hist_files if _dt_from_name(p) is not None]
        hist_files = sorted(hist_files, key=_dt_from_name)[-30:]
        for p in reversed(hist_files):  # latest first
            if len(trailing_residual_means_total) >= 14:
                break
            d = _json(p)
            if not isinstance(d, dict):
                continue
            tm_h = d.get('total_stats', {}).get('mean')
            mm_h = d.get('margin_stats', {}).get('mean')
            tc_h = d.get('total_corr')
            mc_h = d.get('margin_corr')
            if isinstance(tm_h, (int,float)):
                trailing_residual_means_total.append(float(tm_h))
            if isinstance(mm_h, (int,float)):
                trailing_residual_means_margin.append(float(mm_h))
            if isinstance(tc_h, (int,float)):
                trailing_corr_total.append(float(tc_h))
            if isinstance(mc_h, (int,float)):
                trailing_corr_margin.append(float(mc_h))
    except Exception:
        pass

    reasons = []
    metrics = {}
    if residuals and isinstance(residuals, dict):
        tm = residuals.get('total_stats', {}).get('mean')
        mm = residuals.get('margin_stats', {}).get('mean')
        tc = residuals.get('total_corr')
        mc = residuals.get('margin_corr')
        metrics.update({'total_mean': tm, 'margin_mean': mm, 'total_corr': tc, 'margin_corr': mc})
        # Raw absolute thresholds
        if isinstance(tm,(int,float)) and abs(tm) > 2.5:
            reasons.append('totals_mean_drift')
        if isinstance(mm,(int,float)) and abs(mm) > 1.5:
            reasons.append('margin_mean_drift')
        if isinstance(tc,(int,float)) and tc < 0.25:
            reasons.append('totals_corr_low')
        if isinstance(mc,(int,float)) and mc < 0.10:
            reasons.append('margin_corr_low')
        # Dynamic z-score thresholds
        def _z(val, series):
            try:
                if not isinstance(val,(int,float)) or len(series) < 5:
                    return None
                import statistics as _st
                mu = _st.fmean(series)
                sd = _st.pstdev(series) if len(series) > 1 else 0.0
                if sd <= 1e-6:
                    return None
                return (val - mu) / sd
            except Exception:
                return None
        total_mean_z = _z(tm, trailing_residual_means_total)
        margin_mean_z = _z(mm, trailing_residual_means_margin)
        total_corr_z = _z(tc, trailing_corr_total)
        margin_corr_z = _z(mc, trailing_corr_margin)
        metrics.update({
            'total_mean_z': total_mean_z,
            'margin_mean_z': margin_mean_z,
            'total_corr_z': total_corr_z,
            'margin_corr_z': margin_corr_z,
            'trailing_days_used': int(max(len(trailing_residual_means_total), len(trailing_residual_means_margin), len(trailing_corr_total), len(trailing_corr_margin)))
        })
        if isinstance(total_mean_z,(int,float)) and abs(total_mean_z) > 2.2:
            reasons.append('totals_mean_z_extreme')
        if isinstance(margin_mean_z,(int,float)) and abs(margin_mean_z) > 2.2:
            reasons.append('margin_mean_z_extreme')
        if isinstance(total_corr_z,(int,float)) and total_corr_z < -2.0:
            reasons.append('totals_corr_z_drop')
        if isinstance(margin_corr_z,(int,float)) and margin_corr_z < -2.0:
            reasons.append('margin_corr_z_drop')
    if scoring and isinstance(scoring, dict):
        crps_total = scoring.get('crps_total') or scoring.get('crps_total_mean')
        metrics['crps_total'] = crps_total
        if isinstance(crps_total,(int,float)) and isinstance(crps_total_30,(int,float)):
            metrics['crps_total_30d_avg'] = crps_total_30
            if crps_total > crps_total_30 * 1.15:
                reasons.append('crps_total_degradation')
    payload = {
        'date': date_str,
        'generated_at': dt.datetime.now().isoformat(),
        'recalibration_needed': bool(reasons),
        'reasons': reasons,
        'metrics': metrics,
    }
    out_path = OUT / f'recalibration_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Wrote recalibration file {out_path} (needed={payload['recalibration_needed']})")

if __name__ == '__main__':
    main()
