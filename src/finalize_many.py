from __future__ import annotations
import glob
import sys
from pathlib import Path
from datetime import datetime, timedelta

import subprocess

OUT = Path("outputs")

def collect_enriched_dates(max_days: int = 180) -> list[str]:
    files = sorted(glob.glob(str(OUT / "predictions_unified_enriched_*.csv")))
    dates = [Path(f).stem.replace("predictions_unified_enriched_", "") for f in files]
    # Limit to recent max_days relative to today
    today = datetime.today().date()
    out: list[str] = []
    for d in dates:
        try:
            dd = datetime.fromisoformat(d).date()
            if (today - dd).days <= max_days:
                out.append(d)
        except Exception:
            continue
    return out

def main():
    max_days = 120
    if len(sys.argv) >= 2:
        try:
            max_days = int(sys.argv[1])
        except Exception:
            max_days = 120
    dates = collect_enriched_dates(max_days)
    completed = 0
    for d in dates:
        cmd = [sys.executable, "-m", "ncaab_model.cli", "finalize-day", "--date", d]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                completed += 1
            else:
                print({"date": d, "error": res.stderr.strip() or res.stdout.strip()})
        except Exception as e:
            print({"date": d, "error": str(e)})
    print({"finalized_count": completed, "target": len(dates)})

if __name__ == "__main__":
    main()
