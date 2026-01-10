import os
import sys
import json
import argparse
from pathlib import Path
import urllib.request
import urllib.error
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'


def fetch_display(base_url: str, date_str: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/display_predictions?date={date_str}"
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode('utf-8', errors='ignore')
        try:
            return json.loads(data)
        except Exception:
            raise RuntimeError(f"Invalid JSON from {url}: {data[:200]}")


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    # Normalize keys we care about; keep everything present
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Pull remote display snapshot and save locally as CSV')
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--base-url', default=os.environ.get('NCAAB_BASE_URL', 'https://ncaab.onrender.com'))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    payload = fetch_display(args.base_url, args.date)
    rows = payload.get('rows') or []
    df = rows_to_df(rows)
    out_path = OUT / f"predictions_display_{args.date}.csv"
    df.to_csv(out_path, index=False)
    print(json.dumps({'ok': True, 'date': args.date, 'rows': int(len(df)), 'path': str(out_path)}, indent=2))


if __name__ == '__main__':
    main()
