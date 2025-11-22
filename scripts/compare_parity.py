#!/usr/bin/env python
"""Compare local vs remote NCAAB API results for a given date.

Usage:
  python scripts/compare_parity.py --date 2025-11-22 \
      --local http://127.0.0.1:5000 --remote https://ncaab.onrender.com \
      [--cols game_id,pred_total,pred_margin]

Outputs JSON summary with:
  counts, missing sets, and numeric diffs (tolerance 1e-6).
"""
from __future__ import annotations
import argparse, json, sys
import requests

def fetch(base: str, date: str, cols: str):
    try:
        r = requests.get(f"{base}/api/results", params={'date': date, 'cols': cols}, timeout=45)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--local', default='http://127.0.0.1:5000')
    ap.add_argument('--remote', default='https://ncaab.onrender.com')
    ap.add_argument('--cols', default='game_id,pred_total,pred_margin,pred_total_1h,pred_total_2h,pred_margin_1h,pred_margin_2h,pred_total_canonical,pred_margin_canonical')
    ap.add_argument('--tolerance', type=float, default=1e-6)
    args = ap.parse_args()

    remote = fetch(args.remote, args.date, args.cols)
    local = fetch(args.local, args.date, args.cols)
    r_map = {r['game_id']: r for r in remote.get('rows', []) if 'game_id' in r}
    l_map = {r['game_id']: r for r in local.get('rows', []) if 'game_id' in r}
    all_ids = sorted(set(r_map) | set(l_map))
    missing_remote = [gid for gid in l_map if gid not in r_map]
    missing_local = [gid for gid in r_map if gid not in l_map]
    diffs = []
    for gid in all_ids:
        if gid in missing_remote or gid in missing_local:
            continue
        for field in ['pred_total','pred_margin','pred_total_1h','pred_total_2h','pred_margin_1h','pred_margin_2h','pred_total_canonical','pred_margin_canonical']:
            if field not in r_map[gid] or field not in l_map[gid]:
                continue
            try:
                rv = float(r_map[gid][field])
                lv = float(l_map[gid][field])
                delta = abs(rv - lv)
                if delta > args.tolerance:
                    diffs.append({'game_id': gid, 'field': field, 'local': lv, 'remote': rv, 'delta': delta})
            except Exception:
                continue
    summary = {
        'date': args.date,
        'local_rows': len(l_map),
        'remote_rows': len(r_map),
        'missing_remote_count': len(missing_remote),
        'missing_local_count': len(missing_local),
        'diff_count': len(diffs),
        'missing_remote': missing_remote[:10],
        'missing_local': missing_local[:10],
        'diffs_sample': diffs[:10],
    }
    print(json.dumps(summary, indent=2))
    if diffs:
        print(f"PARITY_FAIL: {len(diffs)} numeric diffs > tolerance {args.tolerance}")
        sys.exit(2)
    if missing_remote or missing_local:
        print("PARITY_INCOMPLETE: missing games detected")
        sys.exit(3)
    print("PARITY_OK")

if __name__ == '__main__':
    main()
