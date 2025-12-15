import os, glob, json, sys, pathlib
import pandas as pd
# Ensure workspace root is importable when running from scripts/
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app import _load_all_daily_results, _accuracy_payload

def main():
    df = _load_all_daily_results()
    payload = _accuracy_payload(df)
    out_dir = os.path.join(os.getcwd(), 'outputs', 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'season_accuracy_summary.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    results_dir = os.path.join(os.getcwd(), 'outputs', 'daily_results')
    files = sorted(glob.glob(os.path.join(results_dir, 'results_*.csv')))
    file_dates = []
    for p in files:
        try:
            d = os.path.basename(p).split('_', 1)[1].split('.csv')[0]
            file_dates.append(d)
        except Exception:
            pass
    payload_dates = sorted(payload.get('daily', {}).keys())
    summary = {
        'status': 'ok' if payload.get('overall') else 'empty',
        'daily_count': len(payload_dates),
        'min_date': payload_dates[0] if payload_dates else None,
        'max_date': payload_dates[-1] if payload_dates else None,
        'files_count': len(file_dates),
        'files_min_date': min(file_dates) if file_dates else None,
        'files_max_date': max(file_dates) if file_dates else None,
        'missing_in_payload': sorted(set(file_dates) - set(payload_dates)),
        'missing_in_files': sorted(set(payload_dates) - set(file_dates)),
        'out_path': out_path,
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
