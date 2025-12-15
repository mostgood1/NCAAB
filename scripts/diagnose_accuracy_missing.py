import os, json, glob, sys, pathlib
import pandas as pd
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app import _load_all_daily_results, _accuracy_missing_by_date

def main():
    df = _load_all_daily_results()
    diag = _accuracy_missing_by_date(df)
    out_dir = os.path.join(os.getcwd(), 'outputs', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    # Write JSON
    json_path = os.path.join(out_dir, 'accuracy_missing_by_date.json')
    with open(json_path, 'w') as f:
        json.dump(diag, f, indent=2)
    # Write CSV
    rows = []
    for d, rec in diag.items():
        rows.append(rec)
    csv_path = os.path.join(out_dir, 'accuracy_missing_by_date.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # Print summary
    assessed = [d for d, r in diag.items() if r.get('ats_rows_complete',0) or r.get('tot_rows_complete',0)]
    zero = [d for d, r in diag.items() if r.get('ats_rows_complete',0)==0 and r.get('tot_rows_complete',0)==0]
    summary = {
        'dates_total': len(diag),
        'assessed_count': len(assessed),
        'zero_count': len(zero),
        'assessed_min': min(assessed) if assessed else None,
        'assessed_max': max(assessed) if assessed else None,
        'json_path': json_path,
        'csv_path': csv_path,
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
