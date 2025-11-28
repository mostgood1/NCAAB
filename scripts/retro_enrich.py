"""Retro enrichment script.
Generates enriched unified prediction artifacts for past N days to eliminate
blank teams and ensure coverage parity for audit.

Usage (PowerShell):
  & .venv/Scripts/Activate.ps1; python scripts/retro_enrich.py --days 7
"""
from __future__ import annotations
import argparse, datetime, sys, json, os
from pathlib import Path

# Ensure app import path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # type: ignore

OUT = ROOT / 'outputs'


def enrich_day(date: datetime.date, client, verbose: bool = True) -> dict:
    date_str = date.isoformat()
    enriched_path = OUT / f'predictions_unified_enriched_{date_str}.csv'
    # Skip if already exists and non-empty
    if enriched_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(enriched_path)
            if not df.empty and {'home_team','away_team'}.issubset(df.columns) and (df['home_team'].astype(str).str.strip() != '').all() and (df['away_team'].astype(str).str.strip() != '').all():
                return {'date': date_str, 'skipped': True, 'reason': 'already_enriched', 'rows': int(len(df))}
        except Exception:
            pass
    # Trigger index route for date; relies on existing logic to persist enriched artifact
    resp = client.get(f'/?date={date_str}')
    status = resp.status_code
    if status != 200:
        return {'date': date_str, 'error': f'index_status_{status}'}
    # Verify artifact
    result = {'date': date_str, 'status': 'processed'}
    if enriched_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(enriched_path)
            blanks = 0
            if not df.empty and {'home_team','away_team'}.issubset(df.columns):
                h = df['home_team'].astype(str).str.strip()
                a = df['away_team'].astype(str).str.strip()
                blanks = int(((h == '') | (a == '') | h.str.lower().eq('nan') | a.str.lower().eq('nan')).sum())
            result.update({'rows': int(len(df)), 'blank_team_count': blanks})
        except Exception as e:
            result['artifact_error'] = str(e)
    else:
        result['artifact_missing'] = True
    if verbose:
        print(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser(description='Retro enrich past days artifacts.')
    ap.add_argument('--days', type=int, default=7, help='Number of past days (inclusive of today)')
    ap.add_argument('--verbose', action='store_true', help='Print per-day results')
    args = ap.parse_args()
    days = max(1, min(args.days, 30))
    client = app.test_client()
    today = datetime.date.today()
    results = []
    for offset in range(days):
        d = today - datetime.timedelta(days=offset)
        results.append(enrich_day(d, client, verbose=args.verbose))
    # Persist summary
    OUT.mkdir(exist_ok=True)
    summary_path = OUT / f'retro_enrich_summary_{today.isoformat()}_d{days}.json'
    with open(summary_path, 'w', encoding='utf-8') as fh:
        json.dump({'generated_at': today.isoformat(), 'days': days, 'results': results}, fh, indent=2)
    if args.verbose:
        print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()
