#!/usr/bin/env python
"""Alignment verification between local predictions and remote Render site.

Usage (PowerShell):
  python scripts/align_remote.py -RemoteBase https://ncaab.onrender.com -Date 2025-11-25

If -Date is omitted, the latest local predictions_display_<date>.csv is used.
Outputs:
  outputs/align_remote_<date>_summary.json
  outputs/align_remote_<date>_diff.csv (only if mismatches)
Exit code 0 even if mismatches (for diagnostic use); prints summary.
Set STRICT=1 to exit non-zero on mismatches.
"""
from __future__ import annotations
import argparse, json, sys, os, math
from pathlib import Path
from typing import Any, Dict, List
import urllib.request, urllib.error
import ssl

OUT = Path('outputs')
DEFAULT_REMOTE = 'https://ncaab.onrender.com'

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE  # Render should have valid cert; fallback just in case of local proxies

def _fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={'User-Agent': 'alignment-script/1.0'})
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=25) as resp:  # type: ignore
        data = resp.read()
    try:
        return json.loads(data.decode())
    except Exception as e:
        raise RuntimeError(f'Failed to decode JSON from {url}: {e}')


def _latest_display() -> Path | None:
    files = sorted(OUT.glob('predictions_display_*.csv'))
    return files[-1] if files else None


def _load_local_csv(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        return []
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str)
    return df.to_dict(orient='records')


def _flt(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-RemoteBase', default=os.environ.get('REMOTE_BASE', DEFAULT_REMOTE))
    ap.add_argument('-Date', default='')
    ap.add_argument('-Strict', action='store_true', help='Exit non-zero on any mismatches')
    ap.add_argument('-Tolerance', type=float, default=1e-6, help='Numeric tolerance for prediction equality')
    args = ap.parse_args()

    latest = _latest_display()
    if not latest and not args.Date:
        print('No local predictions_display_<date>.csv files found; abort.', file=sys.stderr)
        return 0
    date_str = args.Date or (latest.stem.replace('predictions_display_','') if latest else '')
    if not date_str:
        print('Could not infer date.', file=sys.stderr)
        return 2

    # Load local rows
    local_rows = _load_local_csv(latest) if latest else []
    local_map = {r.get('game_id'): r for r in local_rows if r.get('game_id') is not None}

    remote_url = f"{args.RemoteBase.rstrip('/')}/api/display_predictions?date={date_str}"
    try:
        remote_data = _fetch_json(remote_url)
    except Exception as e:
        print(f'ERROR fetching remote: {e}', file=sys.stderr)
        return 3
    remote_rows = remote_data.get('rows', [])
    remote_map = {str(r.get('game_id')): r for r in remote_rows if r.get('game_id') is not None}

    # Hash alignment via health (optional)
    health_url = f"{args.RemoteBase.rstrip('/')}/api/health"
    remote_hash = None
    try:
        health = _fetch_json(health_url)
        remote_hash = health.get('display_hash')
    except Exception:
        pass

    mismatches: List[Dict[str, Any]] = []
    missing_remote: List[str] = []
    missing_local: List[str] = []

    # Compare predictions on intersection
    all_ids = set(local_map.keys()) | set(remote_map.keys())
    for gid in sorted(all_ids):
        lr = local_map.get(gid)
        rr = remote_map.get(gid)
        if lr and not rr:
            missing_remote.append(str(gid))
            continue
        if rr and not lr:
            missing_local.append(str(gid))
            continue
        if not lr or not rr:
            continue
        lt = _flt(lr.get('pred_total'))
        rt = _flt(rr.get('pred_total'))
        lm = _flt(lr.get('pred_margin'))
        rm = _flt(rr.get('pred_margin'))
        bt_l = str(lr.get('pred_total_basis')) if lr.get('pred_total_basis') is not None else ''
        bt_r = str(rr.get('pred_total_basis')) if rr.get('pred_total_basis') is not None else ''
        bm_l = str(lr.get('pred_margin_basis')) if lr.get('pred_margin_basis') is not None else ''
        bm_r = str(rr.get('pred_margin_basis')) if rr.get('pred_margin_basis') is not None else ''
        diff_total = None
        diff_margin = None
        if lt is not None and rt is not None:
            diff_total = abs(lt - rt)
        if lm is not None and rm is not None:
            diff_margin = abs(lm - rm)
        basis_mismatch_total = bt_l != bt_r and not (bt_l == 'unknown' and bt_r.startswith(('synthetic','derived','blend_')))  # allow upgrade
        basis_mismatch_margin = bm_l != bm_r and not (bm_l == 'unknown' and bm_r.startswith(('synthetic','derived','blend_')))  # allow upgrade
        numeric_mismatch = (diff_total is not None and diff_total > args.Tolerance) or (diff_margin is not None and diff_margin > args.Tolerance)
        if numeric_mismatch or basis_mismatch_total or basis_mismatch_margin:
            mismatches.append({
                'game_id': gid,
                'pred_total_local': lt,
                'pred_total_remote': rt,
                'diff_total': diff_total,
                'pred_margin_local': lm,
                'pred_margin_remote': rm,
                'diff_margin': diff_margin,
                'basis_total_local': bt_l,
                'basis_total_remote': bt_r,
                'basis_margin_local': bm_l,
                'basis_margin_remote': bm_r,
            })

    summary = {
        'date': date_str,
        'remote_base': args.RemoteBase,
        'remote_hash': remote_hash,
        'local_rows': len(local_rows),
        'remote_rows': len(remote_rows),
        'intersection_rows': len(all_ids) - len(missing_local) - len(missing_remote),
        'missing_remote_count': len(missing_remote),
        'missing_local_count': len(missing_local),
        'mismatch_count': len(mismatches),
        'tolerance': args.Tolerance,
        'strict': args.Strict,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'align_remote_{date_str}_summary.json').write_text(json.dumps({'summary': summary, 'mismatches': mismatches[:200], 'missing_remote': missing_remote[:200], 'missing_local': missing_local[:200]}, indent=2))
    if mismatches:
        import csv
        with open(OUT / f'align_remote_{date_str}_diff.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(mismatches[0].keys()))
            w.writeheader()
            w.writerows(mismatches)

    print(json.dumps(summary, indent=2))
    if args.Strict and mismatches:
        return 4
    return 0

if __name__ == '__main__':
    sys.exit(main())
