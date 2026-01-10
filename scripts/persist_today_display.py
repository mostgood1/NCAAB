import os
import pandas as pd
from datetime import datetime

# Import app functions
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys, importlib
sys.path.append(ROOT)
app = importlib.import_module('app')

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'outputs')

OUT = os.path.join(ROOT, 'outputs')
def persist(date_str: str):
    enr_path = os.path.join(OUT, f'predictions_unified_enriched_{date_str}.csv')
    src = None
    if os.path.exists(enr_path):
        src = enr_path
    else:
        # Fallbacks: use today's calibrated predictions if enriched snapshot is missing
        cal_today = os.path.join(OUT, 'predictions_today_calibrated.csv')
        raw_today = os.path.join(OUT, 'predictions_today.csv')
        if os.path.exists(cal_today):
            src = cal_today
        elif os.path.exists(raw_today):
            src = raw_today
        else:
            print({'error':'missing_sources','enriched': enr_path, 'today_calibrated': cal_today, 'today_raw': raw_today})
            return
    df = pd.read_csv(src)
    # Normalize bases and apply equality-breaker via app helpers
    df_norm = app._normalize_display(df)
    try:
        df_norm = app.apply_pred_total_view(df_norm)
    except Exception:
        pass
    path, digest = app._persist_display(df_norm, date_str)
    print({'date':date_str,'source': src, 'path':str(path),'hash':digest,'rows':len(df_norm)})

if __name__ == '__main__':
    date_str = os.environ.get('DATE') or '2026-01-05'
    persist(date_str)
