"""Orchestrate post-finalization analytics for a resolved date.

Sequence (idempotent best-effort):
  1. closing_join.py (medians)
  2. residuals_generate.py
  3. scoring_generate.py
  4. predictability_eval.py (existing)
  5. moneyline_prob_derive.py
  6. daily_backtest.py
  7. daily_performance.py (existing aggregator)
  8. season_aggregate.py (rolling season metrics)

Usage:
  python scripts/daily_pipeline_post_finalize.py --date 2025-11-19
"""
from __future__ import annotations
import argparse, subprocess, sys, datetime as dt
from pathlib import Path

SCRIPTS = [
    ('closing medians','closing_join.py'),
    ('residuals','residuals_generate.py'),
    ('scoring','scoring_generate.py'),
    ('predictability','predictability_eval.py'),
    ('moneyline_probs','moneyline_prob_derive.py'),
    ('backtest','daily_backtest.py'),
    ('performance','daily_performance.py'),
    ('season','season_aggregate.py'),
]

ROOT = Path(__file__).resolve().parent

def run(step: str, script: str, date_str: str):
    path = ROOT / script
    if not path.exists():
        print(f"[skip] {step}: missing {script}")
        return
    cmd = [sys.executable, str(path), '--date', date_str]
    if script == 'season_aggregate.py':
        # Provide season window end at date; rely on auto start
        cmd = [sys.executable, str(path), '--season-end', date_str]
    print(f"[run] {step}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[error] {step}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Resolved date YYYY-MM-DD (default: yesterday)')
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
    for step, script in SCRIPTS:
        run(step, script, date_str)

if __name__ == '__main__':
    main()
