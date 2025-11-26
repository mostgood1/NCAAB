import os
from pathlib import Path
import sys
import pandas as pd

# Target slate date in schedule timezone (default America/New_York)
DATE = pd.Timestamp.now(tz='UTC').tz_convert(os.getenv('SCHEDULE_TZ', 'America/New_York')).strftime('%Y-%m-%d')

try:
    # Ensure repo root is on sys.path to import app.py
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app import app, OUT
except Exception as e:
    print({"ok": False, "step": "import_app", "error": str(e)})
    raise

OUT = Path(OUT)

# Make the request to run the pipeline and generate mismatch report
status = None
try:
    with app.test_client() as c:
        resp = c.get(f"/?date={DATE}")
        status = resp.status_code
except Exception as e:
    print({"ok": False, "step": "request_index", "error": str(e)})
    raise

report_path = OUT / f"time_mismatch_{DATE}.csv"
summary = {"ok": True, "status": status, "date": DATE, "report": str(report_path), "exists": report_path.exists()}

if report_path.exists():
    try:
        df = pd.read_csv(report_path)
        df['diff_minutes'] = pd.to_numeric(df.get('diff_minutes'), errors='coerce')
        mismatches = df[df['diff_minutes'] > 1.0]
        examples = []
        if not mismatches.empty:
            examples = mismatches.head(8)[["home_team","away_team","our_start_display","sched_commence_time","sched_start_time","diff_minutes"]].to_dict(orient='records')
        summary.update({
            "rows": int(len(df)),
            "mismatch_count": int(len(mismatches)),
            "examples": examples,
        })
    except Exception as e:
        summary.update({"read_error": str(e)})

print(summary)
