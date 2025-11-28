import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')

# Simple drift monitor comparing recent window metrics vs long-term baseline.
# Consumes scoring_<date>.json, reliability_<date>.json, performance_<date>.json if present.

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    except Exception:
        return None

def collect_files(pattern: str):
    return sorted(OUT.glob(pattern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, help='Target date (YYYY-MM-DD) default today')
    ap.add_argument('--recent-days', type=int, default=7)
    ap.add_argument('--baseline-days', type=int, default=30)
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--brier-thresh', type=float, default=0.02, help='Allowed absolute Brier delta')
    ap.add_argument('--crps-thresh', type=float, default=0.5, help='Allowed absolute CRPS delta')
    ap.add_argument('--ece-thresh', type=float, default=0.01, help='Allowed absolute ECE delta')
    args = ap.parse_args()

    from datetime import datetime
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    scoring_files = collect_files('scoring_*.json')
    reliability_files = collect_files('reliability_*.json')

    # Helper parse date from filename
    def _date_from_name(p: Path, prefix: str):
        return p.name.replace(prefix,'').replace('.json','')

    recent_cutoff = pd.to_datetime(date_str) - pd.Timedelta(days=args.recent_days)
    baseline_cutoff = pd.to_datetime(date_str) - pd.Timedelta(days=args.baseline_days)

    recent_scoring = []
    baseline_scoring = []
    for f in scoring_files:
        ds = pd.to_datetime(_date_from_name(f,'scoring_'), errors='coerce')
        payload = load_json(f)
        if payload is None:
            continue
        if ds >= recent_cutoff:
            recent_scoring.append(payload)
        if ds >= baseline_cutoff:
            baseline_scoring.append(payload)

    # Aggregate means
    def _mean(payloads, key):
        vals = [p.get(key) for p in payloads if isinstance(p.get(key), (int,float))]
        return float(np.mean(vals)) if vals else None

    rec_crps_total = _mean(recent_scoring,'crps_total_mean')
    base_crps_total = _mean(baseline_scoring,'crps_total_mean')
    rec_crps_margin = _mean(recent_scoring,'crps_margin_mean')
    base_crps_margin = _mean(baseline_scoring,'crps_margin_mean')

    # Reliability (ECE) - assume reliability_<date>.json contains keys ece_cover / ece_over if present
    recent_rel = []
    baseline_rel = []
    for f in reliability_files:
        ds = pd.to_datetime(_date_from_name(f,'reliability_'), errors='coerce')
        payload = load_json(f)
        if payload is None:
            continue
        if ds >= recent_cutoff:
            recent_rel.append(payload)
        if ds >= baseline_cutoff:
            baseline_rel.append(payload)

    rec_ece_cover = _mean(recent_rel,'ece_cover')
    base_ece_cover = _mean(baseline_rel,'ece_cover')
    rec_ece_over = _mean(recent_rel,'ece_over')
    base_ece_over = _mean(baseline_rel,'ece_over')

    def _delta(a,b):
        if a is None or b is None:
            return None
        return float(a - b)

    artifact = {
        'date': date_str,
        'recent_days': args.recent_days,
        'baseline_days': args.baseline_days,
        'crps_total_recent': rec_crps_total,
        'crps_total_baseline': base_crps_total,
        'crps_total_delta': _delta(rec_crps_total, base_crps_total),
        'crps_margin_recent': rec_crps_margin,
        'crps_margin_baseline': base_crps_margin,
        'crps_margin_delta': _delta(rec_crps_margin, base_crps_margin),
        'ece_cover_recent': rec_ece_cover,
        'ece_cover_baseline': base_ece_cover,
        'ece_cover_delta': _delta(rec_ece_cover, base_ece_cover),
        'ece_over_recent': rec_ece_over,
        'ece_over_baseline': base_ece_over,
        'ece_over_delta': _delta(rec_ece_over, base_ece_over)
    }

    # Trigger logic
    triggers = []
    if artifact['crps_total_delta'] and abs(artifact['crps_total_delta']) > args.crps_thresh:
        triggers.append('crps_total')
    if artifact['crps_margin_delta'] and abs(artifact['crps_margin_delta']) > args.crps_thresh:
        triggers.append('crps_margin')
    if artifact['ece_cover_delta'] and abs(artifact['ece_cover_delta']) > args.ece_thresh:
        triggers.append('ece_cover')
    if artifact['ece_over_delta'] and abs(artifact['ece_over_delta']) > args.ece_thresh:
        triggers.append('ece_over')

    artifact['triggers'] = triggers
    artifact['recalibration_recommended'] = bool(triggers)

    (OUT / f'recalibration_{date_str}.json').write_text(json.dumps(artifact, indent=2))
    print('Recalibration artifact written:', OUT / f'recalibration_{date_str}.json')

if __name__ == '__main__':
    main()
