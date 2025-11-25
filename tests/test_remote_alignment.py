import os, json, subprocess, sys
from pathlib import Path

ENABLE = os.environ.get('ENABLE_REMOTE_ALIGNMENT_TEST') == '1'
REMOTE_BASE = os.environ.get('REMOTE_BASE', 'https://ncaab.onrender.com')

def test_remote_alignment_optional():
    if not ENABLE:
        return  # skipped by default
    # Find latest local display file
    out = Path('outputs')
    files = sorted(out.glob('predictions_display_*.csv'))
    if not files:
        return
    date_str = files[-1].stem.replace('predictions_display_','')
    cmd = [sys.executable, 'scripts/align_remote.py', '-RemoteBase', REMOTE_BASE, '-Date', date_str, '-Tolerance', '1e-6']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"alignment script failed: {proc.stderr}"
    # Parse summary
    try:
        summary = json.loads(proc.stdout)
    except Exception:
        return  # ignore
    assert summary.get('mismatch_count', 0) == 0, f"Remote alignment mismatches detected: {summary.get('mismatch_count')}"  # enforce zero mismatches
