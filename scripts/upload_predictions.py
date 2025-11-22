"""Upload local predictions_<date>.csv to remote deployment.

Usage (PowerShell example):

  $Env:NCAAB_PREDICTIONS_INGEST_TOKEN="YOURTOKEN"
  python scripts/upload_predictions.py --date 2025-11-22 --file outputs/predictions_2025-11-22.csv --url https://ncaab.onrender.com

If --file omitted will attempt outputs/predictions_<date>.csv.
Requires env var NCAAB_PREDICTIONS_INGEST_TOKEN.
"""
from __future__ import annotations
import argparse, os, sys, pathlib, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='Date YYYY-MM-DD of predictions slate')
    ap.add_argument('--file', help='Path to predictions CSV (defaults to outputs/predictions_<date>.csv)')
    ap.add_argument('--url', default='https://ncaab.onrender.com', help='Base URL of remote deployment')
    ap.add_argument('--force', action='store_true', help='Overwrite existing remote predictions_<date>.csv')
    args = ap.parse_args()
    token = os.environ.get('NCAAB_PREDICTIONS_INGEST_TOKEN')
    if not token:
        print('ERROR: NCAAB_PREDICTIONS_INGEST_TOKEN env var required', file=sys.stderr)
        return 2
    path = pathlib.Path(args.file) if args.file else pathlib.Path('outputs') / f'predictions_{args.date}.csv'
    if not path.exists():
        print(f'ERROR: predictions file not found: {path}', file=sys.stderr)
        return 3
    files = {'file': (path.name, path.read_bytes(), 'text/csv')}
    endpoint = args.url.rstrip('/') + f'/api/predictions_ingest?date={args.date}' + ('&force=1' if args.force else '')
    try:
        r = requests.post(endpoint, headers={'X-Ingest-Token': token}, files=files, timeout=30)
        print(f'STATUS {r.status_code}')
        print(r.text[:1000])
        if r.status_code != 200:
            return 4
    except Exception as e:
        print(f'ERROR: request failed: {e}', file=sys.stderr)
        return 5
    return 0

if __name__ == '__main__':
    raise SystemExit(main())