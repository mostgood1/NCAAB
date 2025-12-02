"""Backfill display_date and local start time fields for historical prediction artifacts.

Generates force-filled enriched files with suffix _force_fill.csv used by index route
fallback logic. Safe to run multiple times; idempotent for already filled rows.

Usage (PowerShell examples):
  .venv/Scripts/python.exe scripts/backfill_display_dates.py --days 30
  .venv/Scripts/python.exe scripts/backfill_display_dates.py --since 2025-11-01

Strategy:
- Enumerate predictions_unified_enriched_*.csv (preferred) else predictions_unified_*.csv
- For each date, load rows, derive start_time_iso if missing, apply _backfill_start_fields
  and _correct_midnight_drift using slate date from row['date'] or filename date.
- Ensure display_date is set on every row (fallback to slate date) and that
  start_time_local/start_time_display populated if derivable.
- Write output to predictions_unified_enriched_<date>_force_fill.csv.

Limitations:
- Does not attempt venue timezone inference beyond what enrichment already did;
  relies on _backfill_start_fields logic from app.py which prefers venue-local fields.
- If app.py helpers change, re-run to refresh force_fill artifacts.
"""
from __future__ import annotations
import argparse, re, datetime as dt
from pathlib import Path
import sys
import pandas as pd

# Import helpers from app (avoid running the server)
try:
    import app  # type: ignore
except Exception:
    # Fallback: dynamic load from app.py
    try:
        import importlib.util
        ap = Path('app.py')
        if not ap.exists():
            raise RuntimeError('app.py not found for fallback load')
        spec = importlib.util.spec_from_file_location('app', str(ap))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            app = module  # type: ignore
        else:
            raise RuntimeError('spec loader unavailable for app.py')
    except Exception as e2:
        print(f"ERROR: failed importing app helpers (fallback): {e2}")
        sys.exit(1)
try:
    _derive_start_iso = app._derive_start_iso  # type: ignore
    _backfill_start_fields = app._backfill_start_fields  # type: ignore
    _correct_midnight_drift = app._correct_midnight_drift  # type: ignore
except Exception as e3:
    print(f"ERROR: app helpers missing: {e3}")
    sys.exit(1)

OUT = Path('outputs')
PAT_DATE = re.compile(r'predictions_unified(?:_enriched)?_(\d{4}-\d{2}-\d{2})\.csv$')

def list_candidate_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in OUT.glob('predictions_unified_enriched_*.csv'):
        m = PAT_DATE.match(p.name)
        if m:
            files[m.group(1)] = p
    # Fallback to base if enriched missing
    for p in OUT.glob('predictions_unified_*.csv'):
        if 'enriched' in p.name:
            continue
        m = PAT_DATE.match(p.name)
        if m and m.group(1) not in files:
            files[m.group(1)] = p
    return dict(sorted(files.items()))

def process_file(date_str: str, path: Path, dry_run: bool=False) -> tuple[str,int,int]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return date_str, 0, 0
    if df.empty:
        return date_str, 0, 0
    rows = df.to_dict(orient='records')
    out_rows = []
    changed = False
    for r0 in rows:
        r = dict(r0)
        slate = str(r.get('date') or date_str)
        if not r.get('start_time_iso'):
            r['start_time_iso'] = _derive_start_iso(r)
        r = _backfill_start_fields(r)
        r = _correct_midnight_drift(r, slate_date=slate)
        # Final safety: enforce display_date
        if not r.get('display_date'):
            r['display_date'] = slate
        # Simple change detection: if display_date newly added or differs from slate when slate exists
        if r.get('display_date') and (r.get('display_date') != r0.get('display_date')):
            changed = True
        out_rows.append(r)
    out_df = pd.DataFrame(out_rows)
    # Ensure column ordering: keep originals first then appended fields (best-effort)
    for col in ['display_date','start_time_iso','start_time_local','start_time_local_venue','start_time_display','start_tz_abbr','start_tz_abbr_venue']:
        if col not in out_df.columns:
            continue
    out_path = OUT / f'predictions_unified_enriched_{date_str}_force_fill.csv'
    if not dry_run:
        out_df.to_csv(out_path, index=False)
    return date_str, len(out_df), int(changed)

def main():
    ap = argparse.ArgumentParser(description='Backfill display dates for historical predictions')
    ap.add_argument('--days', type=int, default=None, help='Limit to last N days')
    ap.add_argument('--since', type=str, default=None, help='Process dates >= this YYYY-MM-DD')
    ap.add_argument('--dry-run', action='store_true', help='Do not write files')
    args = ap.parse_args()

    files = list_candidate_files()
    if not files:
        print('No prediction files found.')
        return
    today = dt.date.today()
    selected: list[tuple[str, Path]] = []
    for date_str, p in files.items():
        try:
            d = dt.date.fromisoformat(date_str)
        except Exception:
            continue
        if args.days is not None:
            if (today - d).days > args.days:
                continue
        if args.since:
            try:
                since_d = dt.date.fromisoformat(args.since)
                if d < since_d:
                    continue
            except Exception:
                pass
        selected.append((date_str, p))
    if not selected:
        print('No files match selection criteria.')
        return
    print(f'Processing {len(selected)} files...')
    summary = []
    for date_str, p in selected:
        ds, rows, changed = process_file(date_str, p, dry_run=args.dry_run)
        summary.append((ds, rows, changed))
        print(f'  {ds}: rows={rows} changed_flag={changed}')
    print('Done.')
    # Aggregate stats
    total_rows = sum(r for _, r, _ in summary)
    total_changed = sum(c for _, _, c in summary)
    print(f'Total rows processed: {total_rows}; files with any change flag: {total_changed}')

if __name__ == '__main__':
    main()
