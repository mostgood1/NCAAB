#!/usr/bin/env python
"""Ensure remote deployment has fresh predictions for a given date.

Workflow:
 1. Compute local hash from outputs/predictions_<date>.csv
 2. Fetch remote integrity (/api/predictions_integrity?date=...) to compare hash & existence
 3. If primary missing, all NaN, or hash mismatch -> upload predictions via /api/predictions_ingest

Usage:
  python scripts/ensure_remote_predictions.py --date YYYY-MM-DD --url https://ncaab.onrender.com \
      --token YOUR_TOKEN [--force]

Exit codes:
  0 success / no action needed
  1 soft error (missing local file)
  2 upload attempted but failed
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import requests
import os
import hashlib

OUT = Path("outputs")

def compute_hash(df: pd.DataFrame) -> str | None:
    try:
        if df.empty or 'game_id' not in df.columns:
            return None
        df = df.copy()
        df['game_id'] = df['game_id'].astype(str)
        for col in ['pred_total','pred_margin']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('game_id')
        parts = []
        for _, r in df.iterrows():
            gid = r['game_id']
            pt = r.get('pred_total')
            pm = r.get('pred_margin')
            pt_str = 'NA' if (pt is None or pd.isna(pt)) else f"{float(pt):.4f}"
            pm_str = 'NA' if (pm is None or pd.isna(pm)) else f"{float(pm):.4f}"
            parts.append(f"{gid}|{pt_str}|{pm_str}")
        return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()
    except Exception:
        return None

def upload(url: str, token: str, date: str, path: Path, force: bool) -> tuple[bool,str]:
    try:
        with path.open('rb') as f:
            resp = requests.post(f"{url}/api/predictions_ingest?date={date}&force={'1' if force else '0'}", headers={'X-Ingest-Token': token}, files={'file': f}, timeout=60)
        ok = resp.status_code == 200 and resp.json().get('ok')
        return ok, resp.text
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--url', default='https://ncaab.onrender.com')
    ap.add_argument('--token', default=os.getenv('NCAAB_PREDICTIONS_INGEST_TOKEN',''))
    ap.add_argument('--file', help='Override predictions file path', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if not args.token:
        print('ERROR: missing ingest token (env NCAAB_PREDICTIONS_INGEST_TOKEN or --token)')
        return 1
    local_path = Path(args.file) if args.file else OUT / f"predictions_{args.date}.csv"
    if not local_path.exists():
        print(f'WARN: local predictions file missing: {local_path}')
        return 1
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        print(f'ERROR: failed reading local predictions: {e}')
        return 1
    local_hash = compute_hash(df)
    if not local_hash:
        print('WARN: unable to compute local hash (empty or missing columns). Uploading anyway.')
        need_upload = True
    else:
        need_upload = True  # default; flip off if healthy
        try:
            r = requests.get(f"{args.url}/api/predictions_integrity?date={args.date}", timeout=30)
            if r.status_code == 200 and 'application/json' in (r.headers.get('Content-Type') or '').lower():
                try:
                    data = r.json()
                except Exception as je:
                    print(f'INFO: JSON parse failed ({je}); will upload.')
                    data = {}
                meta = data.get('meta', {}) if isinstance(data, dict) else {}
                remote_hash = meta.get('predictions_hash_primary')
                exists_primary = meta.get('exists_primary')
                all_nan_primary = meta.get('all_nan_primary')
                rows_primary = meta.get('rows_primary')
                unhealthy = (not exists_primary) or all_nan_primary or rows_primary == 0 or remote_hash != local_hash
                if unhealthy:
                    print('INFO: mismatch or unhealthy remote predictions detected; will upload.')
                else:
                    need_upload = False
            else:
                # Fallback: try /api/health for integrity summary if present
                hr = requests.get(f"{args.url}/api/health", timeout=30)
                if hr.status_code == 200 and 'application/json' in (hr.headers.get('Content-Type') or '').lower():
                    try:
                        hdata = hr.json()
                    except Exception:
                        hdata = {}
                    integ = hdata.get('predictions_integrity') if isinstance(hdata, dict) else None
                    if isinstance(integ, dict):
                        remote_hash = integ.get('predictions_hash_primary')
                        exists_primary = integ.get('exists_primary')
                        all_nan_primary = integ.get('all_nan_primary')
                        rows_primary = integ.get('rows_primary')
                        unhealthy = (not exists_primary) or all_nan_primary or rows_primary == 0 or remote_hash != local_hash
                        if unhealthy:
                            print('INFO: health fallback indicates unhealthy predictions; will upload.')
                        else:
                            need_upload = False
                    else:
                        print(f'INFO: integrity endpoint unavailable (status {r.status_code}); will upload.')
                else:
                    print(f'INFO: integrity endpoint status {r.status_code}; will upload.')
        except Exception as e:
            print(f'INFO: integrity check failed ({e}); will upload.')
    if need_upload:
        ok, resp_text = upload(args.url, args.token, args.date, local_path, args.force)
        if ok:
            print('UPLOAD_OK')
            return 0
        else:
            print('UPLOAD_FAILED', resp_text[:500])
            return 2
    print('NO_ACTION: remote predictions healthy.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
