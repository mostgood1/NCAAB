"""Backfill core artifacts across a date range to unlock calibration and season metrics.

Runs, for each date in [start, end]:
  - closing_join.py (games_with_closing_<date>.csv)
  - residuals_generate.py (residuals_<date>.json)
  - scoring_generate.py (scoring_<date>.json)
  - reliability_calibration.py (reliability_<date>.json)
  - daily_backtest.py (backtest_metrics_<date>.json)

Usage:
  python scripts/backfill_artifacts.py --start 2025-11-01 --end 2025-11-20
If start/end omitted, defaults to last 14 days.
After the loop, runs calibrate_spread_logistic.py and season_aggregate.py to refresh global artifacts.
"""
from __future__ import annotations
import argparse, subprocess, sys, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

SCRIPTS = ROOT / 'scripts'

def run(cmd: list[str]):
    try:
        print('>',' '.join(cmd))
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('Command failed:', e)


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', help='YYYY-MM-DD (default: today-14d)')
    ap.add_argument('--end', help='YYYY-MM-DD (default: today)')
    ap.add_argument('--skip-calibration', action='store_true', help='Skip spread logistic calibration after backfill')
    ap.add_argument('--skip-season', action='store_true', help='Skip season aggregation after backfill')
    args = ap.parse_args()

    today = dt.date.today()
    start = dt.date.fromisoformat(args.start) if args.start else (today - dt.timedelta(days=14))
    end = dt.date.fromisoformat(args.end) if args.end else today

    for d in daterange(start, end):
        ds = d.strftime('%Y-%m-%d')
        print(f'=== Backfill {ds} ===')
        run([PY, str(SCRIPTS / 'closing_join.py'), '--date', ds])
        run([PY, str(SCRIPTS / 'residuals_generate.py'), '--date', ds])
        run([PY, str(SCRIPTS / 'scoring_generate.py'), '--date', ds])
        run([PY, str(SCRIPTS / 'reliability_calibration.py'), '--date', ds])
        run([PY, str(SCRIPTS / 'daily_backtest.py'), '--date', ds])

    # Post-loop global artifacts
    if not args.skip_calibration:
        print('=== Calibrating spread logistic K ===')
        # Prefer writing a provisional calibration when rows are limited, to enable app hints
        run([PY, str(SCRIPTS / 'calibrate_spread_logistic.py'), '--provisional-min-rows', '25', '--min-rows', '200'])
    if not args.skip_season:
        print('=== Refresh season aggregation ===')
        run([PY, str(SCRIPTS / 'season_aggregate.py')])

if __name__ == '__main__':
    main()
