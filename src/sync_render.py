"""Compare local calibration diagnostics with remote Render deployment.

Usage:
    python -m src.sync_render --remote-base https://your-render-url --date 2025-11-25

Environment (optional):
    NCAAB_REMOTE_BASE=https://your-render-url

Outputs a diff summary (basis shares, reason counts). Does not modify remote state.
"""
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import Any, Dict
import urllib.request

from datetime import datetime

OUT = Path("outputs")

def _fetch_json(url: str) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def _load_local_diag(date: str | None) -> Dict[str, Any]:
    # Try pipeline_stats snapshot first
    snap = OUT / "pipeline_stats_last.json"
    data: Dict[str, Any] = {}
    if snap.exists():
        try:
            data = json.loads(snap.read_text())
        except Exception:
            data = {}
    # Augment with calibration summary endpoint if running locally
    local_base = os.environ.get("NCAAB_LOCAL_BASE", "http://127.0.0.1:5050")
    diag_url = f"{local_base}/api/calibration_diagnostic?date={date}" if date else f"{local_base}/api/calibration_diagnostic"
    remote_like = _fetch_json(diag_url)
    data['local_endpoint'] = remote_like
    return data

def compare(remote: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Basis shares
    for k in [
        'basis_share_total_cal','basis_share_total_model_missing_cal','basis_share_total_cal_est',
        'basis_share_margin_cal','basis_share_margin_model_missing_cal','basis_share_margin_cal_est'
    ]:
        lv = local.get('local_endpoint', {}).get(k) or local.get(k)
        rv = remote.get(k) or remote.get('summary', {}).get(k)
        if lv is not None or rv is not None:
            out[k] = {'local': lv, 'remote': rv, 'delta': (None if (lv is None or rv is None) else (lv - rv))}
    # Reason counts
    lrc = local.get('local_endpoint', {}).get('summary')
    rrc = remote.get('summary')
    out['reason_counts_local'] = lrc
    out['reason_counts_remote'] = rrc
    return out


def main():
    ap = argparse.ArgumentParser(description="Sync & compare local vs remote calibration diagnostics")
    ap.add_argument('--remote-base', default=os.environ.get('NCAAB_REMOTE_BASE'), help='Remote base URL (e.g., https://service.onrender.com)')
    ap.add_argument('--date', default=None, help='Target date YYYY-MM-DD (optional)')
    args = ap.parse_args()
    if not args.remote_base:
        print('Remote base URL required (--remote-base or NCAAB_REMOTE_BASE).')
        return
    date = args.date or datetime.utcnow().strftime('%Y-%m-%d')
    remote_url = f"{args.remote_base.rstrip('/')}/api/calibration_diagnostic?date={date}"
    remote_json = _fetch_json(remote_url)
    local_json = _load_local_diag(date)
    diff = compare(remote_json, local_json)
    print(json.dumps({'remote': remote_json, 'local': local_json.get('local_endpoint'), 'diff': diff}, indent=2))

if __name__ == '__main__':
    main()
