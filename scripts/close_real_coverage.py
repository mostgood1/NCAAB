"""Orchestrate iterative closure of real coverage gaps (predictions + odds).

Steps:
1. Load today's missing coverage CSV.
2. If missing predictions: build manifest + run target_inference.py (real model hook TBD).
3. If missing odds: run fetch_missing_odds.py then integrate_missing_odds.py.
4. Integrate missing inference predictions.
5. Recompute missing coverage summary and write JSON report.

Non-destructive: creates new enriched artifacts with suffixes; does not overwrite originals.
"""
from pathlib import Path
from datetime import datetime
import subprocess
import json
import pandas as pd

OUT = Path('outputs')
DATE_STR = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = OUT / f'coverage_closure_report_{DATE_STR}.json'


def run(cmd):
    print('>>', cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr:
        print('[stderr]', r.stderr.strip())
    return r.returncode


def file_rows(path):
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return -1


def load_missing():
    p = OUT / f'missing_real_coverage_{DATE_STR}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def summarize_missing(df):
    if df.empty:
        return {'rows': 0}
    summary = {'rows': int(len(df))}
    for col in ['pred_total','pred_margin','market_total','spread_home']:
        if col in df.columns:
            summary[f'{col}_missing'] = int(df[col].isna().sum())
    return summary


def main():
    report = {'date': DATE_STR, 'actions': []}
    miss_df = load_missing()
    initial = summarize_missing(miss_df)
    report['initial'] = initial

    # Predictions pass
    if initial.get('pred_total_missing', 0) > 0 or initial.get('pred_margin_missing', 0) > 0:
        rc = run('python scripts/generate_missing_inference_manifest.py')
        report['actions'].append({'step': 'generate_manifest', 'return_code': rc})
        rc = run('python scripts/target_inference.py')
        report['actions'].append({'step': 'target_inference', 'return_code': rc})
        rc = run('python scripts/integrate_missing_inference.py')
        report['actions'].append({'step': 'integrate_inference', 'return_code': rc})
    else:
        report['actions'].append({'step': 'skip_predictions', 'reason': 'no missing pred_total/pred_margin'})

    # Odds pass
    if initial.get('market_total_missing', 0) > 0 or initial.get('spread_home_missing', 0) > 0:
        rc = run('python scripts/fetch_missing_odds.py')
        report['actions'].append({'step': 'fetch_odds', 'return_code': rc})
        rc = run('python scripts/integrate_missing_odds.py')
        report['actions'].append({'step': 'integrate_odds', 'return_code': rc})
    else:
        report['actions'].append({'step': 'skip_odds', 'reason': 'no missing market_total/spread_home'})

    # Refresh missing coverage export (re-run existing script)
    rc = run('python scripts/list_missing_real_coverage.py')
    report['actions'].append({'step': 'recheck_missing', 'return_code': rc})
    final_missing = load_missing()
    report['final'] = summarize_missing(final_missing)

    # Artifact presence summary
    artifacts = []
    for name in [
        f'predictions_unified_enriched_{DATE_STR}.csv',
        f'predictions_unified_enriched_{DATE_STR}_with_missing_inference.csv',
        f'predictions_unified_enriched_{DATE_STR}_with_missing_odds.csv',
        f'missing_inference_manifest_{DATE_STR}.json',
        f'missing_inference_preds_{DATE_STR}.csv',
        f'missing_odds_fetched_{DATE_STR}.csv'
    ]:
        p = OUT / name
        artifacts.append({'file': name, 'exists': p.exists(), 'rows': file_rows(p)})
    report['artifacts'] = artifacts

    with open(REPORT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print('Coverage closure report written:', REPORT_PATH)
    print(json.dumps(report['final'], indent=2))

if __name__ == '__main__':
    main()
