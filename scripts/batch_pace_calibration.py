#!/usr/bin/env python
from __future__ import annotations
import argparse, re, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

# Import helpers from the single-date scripts
sys.path.append(str(ROOT / 'scripts'))
try:
    from fetch_box_scores_espn import enrich_enriched_with_box
    from calibrate_totals_with_pace import calibrate_with_pace
except Exception as e:
    print(f"[err] Failed to import helpers: {e}")
    raise


def find_enriched_dates(max_days: int | None = None) -> list[str]:
    patt = re.compile(r'^predictions_unified_enriched_(\d{4}-\d{2}-\d{2})\.csv$')
    dates = []
    for p in OUT.glob('predictions_unified_enriched_*.csv'):
        m = patt.match(p.name)
        if m:
            dates.append(m.group(1))
    dates.sort()
    if max_days is not None and len(dates) > max_days:
        dates = dates[-max_days:]
    return dates


def run_batch(window_days: int, baseline_pace: float, sleep_sec: float = 0.5) -> dict:
    dates = find_enriched_dates(max_days=window_days)
    summary = { 'window_days': window_days, 'baseline_pace': baseline_pace, 'processed': [], 'errors': [] }
    for d in dates:
        try:
            print(f"[batch] Fetch + calibrate {d}")
            enrich_enriched_with_box(d, sleep_sec=sleep_sec, in_place=True)
            calibrate_with_pace(d, baseline_pace_per40=baseline_pace, overwrite_calibrated=True)
            summary['processed'].append(d)
        except Exception as e:
            summary['errors'].append({'date': d, 'error': str(e)})
    return summary


def main():
    ap = argparse.ArgumentParser(description='Batch ESPN box fetch + pace calibration over a historical window.')
    ap.add_argument('--window-days', type=int, default=60, help='Max historical days to process based on enriched file availability')
    ap.add_argument('--baseline-pace', type=float, default=70.0)
    ap.add_argument('--sleep-sec', type=float, default=0.5)
    args = ap.parse_args()
    res = run_batch(args.window_days, args.baseline_pace, sleep_sec=args.sleep_sec)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
