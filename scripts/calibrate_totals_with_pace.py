#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

# Ensure local src is importable
sys.path.append(str(ROOT))
try:
    from src.features.pace import attach_pace_features
except Exception:
    attach_pace_features = None


def _select_pred_total_column(df: pd.DataFrame) -> str:
    for c in ['pred_total_calibrated', 'pred_total']:
        if c in df.columns:
            return c
    raise KeyError('Missing pred_total or pred_total_calibrated in enriched file')


def calibrate_with_pace(date: str, baseline_pace_per40: float = 70.0, overwrite_calibrated: bool = True) -> Path:
    p = OUT / f'predictions_unified_enriched_{date}.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing enriched file: {p}')
    df = pd.read_csv(p)
    # Attach pace features if not present
    if attach_pace_features is not None:
        df = attach_pace_features(df)
    # Require pace column
    pace_col = None
    for c in ['pace_game_per40', 'pace_per40', 'pace_game_est']:
        if c in df.columns:
            pace_col = c
            break
    if pace_col is None:
        raise KeyError('No pace column found. Ensure box stats are present and pace features attached.')
    pred_col = _select_pred_total_column(df)
    # Pace-aware calibration: scale predicted total by pace ratio vs baseline
    # Example: if pace is 74 vs baseline 70, adjust up by 74/70.
    df['pred_total_calibrated_pace'] = df[pred_col] * (df[pace_col].astype(float) / float(baseline_pace_per40))
    if overwrite_calibrated and 'pred_total_calibrated' in df.columns:
        df['pred_total_calibrated'] = df['pred_total_calibrated_pace']
    out_path = p
    df.to_csv(out_path, index=False)
    print(json.dumps({
        'date': date,
        'rows': int(df.shape[0]),
        'baseline_pace_per40': baseline_pace_per40,
        'updated_pred_total_calibrated': overwrite_calibrated and ('pred_total_calibrated' in df.columns),
        'path': str(out_path)
    }, indent=2))
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Calibrate totals using pace features (per 40 mins).')
    ap.add_argument('--date', required=True, help='YYYY-MM-DD date of enriched file to calibrate')
    ap.add_argument('--baseline-pace', type=float, default=70.0, help='Baseline pace per 40 mins used for scaling')
    ap.add_argument('--no-overwrite', action='store_true', help='Do not overwrite pred_total_calibrated; only write side column')
    args = ap.parse_args()
    calibrate_with_pace(args.date, baseline_pace_per40=args.baseline_pace, overwrite_calibrated=(not args.no_overwrite))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
