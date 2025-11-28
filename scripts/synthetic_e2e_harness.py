import json, datetime, subprocess, sys
from pathlib import Path
import pandas as pd

OUT = Path('outputs')
TODAY = datetime.datetime.utcnow().date().isoformat()
SUMMARY = {
    'date': TODAY,
    'steps': {},
}

def check_file(path: Path, key: str):
    try:
        exists = path.exists()
        rows = None
        if exists:
            try:
                df = pd.read_csv(path)
                rows = int(len(df))
            except Exception:
                rows = None
        SUMMARY['steps'][key] = {'exists': exists, 'rows': rows}
        return exists
    except Exception as e:
        SUMMARY['steps'][key] = {'error': str(e)}
        return False


def run_python(script: str, key: str):
    cmd = [sys.executable, script]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        SUMMARY['steps'][key] = {
            'returncode': res.returncode,
            'stdout': (res.stdout[-2000:] if res.stdout else ''),
            'stderr': (res.stderr[-2000:] if res.stderr else '')
        }
        return res.returncode == 0
    except Exception as e:
        SUMMARY['steps'][key] = {'error': str(e)}
        return False


def main():
    # Check core artifacts for today
    check_file(OUT / f'predictions_model_{TODAY}.csv', 'predictions_model')
    check_file(OUT / f'predictions_model_calibrated_{TODAY}.csv', 'predictions_model_calibrated')
    check_file(OUT / f'predictions_model_interval_{TODAY}.csv', 'predictions_model_interval')
    check_file(OUT / f'predictions_unified_enriched_{TODAY}.csv', 'predictions_unified_enriched')
    # Run meta train and explain
    run_python('scripts/train_meta_probs.py', 'train_meta_probs')
    run_python('scripts/explain_meta.py', 'explain_meta')
    # Stake sheets
    check_file(OUT / 'stake_sheet_today.csv', 'stake_sheet_today')
    # Summarize edges and confidence if stake sheet present
    try:
        st = OUT / 'stake_sheet_today.csv'
        if st.exists():
            df = pd.read_csv(st)
            for k in ['edge','confidence','stake','kelly']:
                if k in df.columns:
                    s = pd.to_numeric(df[k], errors='coerce')
                    SUMMARY['steps'][f'stake_{k}_stats'] = {
                        'count': int(s.notna().sum()),
                        'mean': float(s.mean()) if s.notna().any() else None,
                        'min': float(s.min()) if s.notna().any() else None,
                        'max': float(s.max()) if s.notna().any() else None,
                    }
    except Exception:
        pass
    out_path = OUT / f'e2e_summary_{TODAY}.json'
    out_path.write_text(json.dumps(SUMMARY, indent=2))
    print('E2E summary written:', out_path)

if __name__ == '__main__':
    main()
