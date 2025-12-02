import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

# Import helpers from app
sys.path.insert(0, str(ROOT))
from app import _fallback_merge_predictions, _correct_midnight_drift  # type: ignore

def regenerate(date_str: str):
    base = OUT / f'predictions_unified_{date_str}.csv'
    enriched = OUT / f'predictions_unified_enriched_{date_str}.csv'
    display = OUT / f'predictions_display_{date_str}.csv'
    if not base.exists():
        print(f"Missing base unified file: {base}")
        return
    uni = pd.read_csv(base)
    enr = pd.read_csv(enriched) if enriched.exists() else uni.copy()
    dis = pd.read_csv(display) if display.exists() else None
    # Apply fallback merge
    enr = _fallback_merge_predictions(enr, uni, dis)
    # Fix midnight drift row-wise using raw context
    if 'start_time_iso' in enr.columns or 'start_time_local' in enr.columns:
        # For row-wise function, apply with slate_date
        rows = []
        for r in enr.to_dict(orient='records'):
            rows.append(_correct_midnight_drift(r, slate_date=date_str))
        enr = pd.DataFrame(rows)
    # Ensure display_date is set (fallback to date field)
    if 'display_date' not in enr.columns:
        enr['display_date'] = None
    enr['display_date'] = enr['display_date'].fillna(enr.get('date'))
    # Write out
    out_p = OUT / f'predictions_unified_enriched_{date_str}.csv'
    enr.to_csv(out_p, index=False)
    print(f"Wrote: {out_p}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/regenerate_enriched.py YYYY-MM-DD")
        sys.exit(1)
    regenerate(sys.argv[1])