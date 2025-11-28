import os, json, datetime
from pathlib import Path
import pandas as pd

OUT = Path('outputs')
CAL_DIR = OUT / 'calibrators'
TODAY = datetime.datetime.utcnow().date().isoformat()
ECE_THRESH = float(os.getenv('AUTO_CAL_ECE_THRESH', '0.035'))
MIN_DAYS = int(os.getenv('AUTO_CAL_MIN_DAYS', '3'))
DRIFT_FLAG_KEYS = ['spread_drift_exceeds','total_drift_exceeds','prob_drift_exceeds','recalibration_required']

ARTIFACT = {
    'date': TODAY,
    'ece_threshold': ECE_THRESH,
    'min_days': MIN_DAYS,
    'triggered': False,
    'reason': [],
    'saved_calibrators_before': [],
    'saved_calibrators_after': [],
}

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def main():
    trend_path = OUT / 'reliability_trend.json'
    trend = load_json(trend_path)
    if trend is None:
        trend = {}
    # Collect recent ECE values if present (supports both list-of-rows or {'rows': [...]} formats)
    recent_ece = []
    rows = []
    if isinstance(trend, list):
        rows = trend
    elif isinstance(trend, dict) and isinstance(trend.get('rows'), list):
        rows = trend.get('rows', [])
    if rows:
        for row in rows[-MIN_DAYS:]:
            if isinstance(row, dict):
                for k, v in row.items():
                    if isinstance(k, str) and k.startswith('ece_'):
                        try:
                            recent_ece.append(float(v))
                        except Exception:
                            continue
    if len(recent_ece) >= MIN_DAYS and max(recent_ece) > ECE_THRESH:
        ARTIFACT['triggered'] = True
        ARTIFACT['reason'].append(f'ece_exceeds_threshold_max={max(recent_ece):.4f}')

    # Drift / recalibration artifact check
    rec_paths = [p for p in OUT.glob(f'recalibration_{TODAY}.json')]
    for rp in rec_paths:
        rec = load_json(rp) or {}
        for key in DRIFT_FLAG_KEYS:
            if rec.get(key):
                ARTIFACT['triggered'] = True
                ARTIFACT['reason'].append(f'drift_flag:{key}')

    # Age-based trigger: calibrators older than 7 days
    if CAL_DIR.exists():
        ages = []
        for f in CAL_DIR.glob('*_iso.joblib'):
            try:
                mtime = datetime.date.fromtimestamp(f.stat().st_mtime)
                age = (datetime.date.today() - mtime).days
                ages.append(age)
            except Exception:
                continue
        if ages and max(ages) >= 7:
            ARTIFACT['triggered'] = True
            ARTIFACT['reason'].append(f'calibrator_age_days_max={max(ages)}')

    # Record existing calibrators
    if CAL_DIR.exists():
        ARTIFACT['saved_calibrators_before'] = sorted([f.name for f in CAL_DIR.glob('*_iso.joblib')])

    if ARTIFACT['triggered']:
        try:
            import calibrate_prob_methods  # type: ignore
            calibrate_prob_methods.main()
        except Exception as e:
            ARTIFACT['reason'].append(f'calibration_error:{e}')
        # Post list
        if CAL_DIR.exists():
            ARTIFACT['saved_calibrators_after'] = sorted([f.name for f in CAL_DIR.glob('*_iso.joblib')])
    out_path = OUT / f'auto_refresh_calibration_{TODAY}.json'
    out_path.write_text(json.dumps(ARTIFACT, indent=2))
    print('Auto calibration refresh complete:', json.dumps(ARTIFACT, indent=2))

if __name__ == '__main__':
    main()
