from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load, dump

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore

TARGETS = {
    'p_home_win': 'home_win',
    'p_home_cover': 'ats_home_cover',
    'p_over': 'ou_over',
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit isotonic calibration for existing probability columns using historical outcomes.")
    ap.add_argument('--outputs-dir', default='outputs')
    ap.add_argument('--date-start', default=None)
    ap.add_argument('--date-end', default=None)
    ap.add_argument('--probs-file-pattern', default='model_probs_*.csv', help='Glob pattern for probability files')
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    daily_dir = out_dir / 'daily_results'
    if not daily_dir.exists():
        print('daily_results directory missing; cannot calibrate.')
        return

    # Load historical outcomes
    hist_files = sorted(daily_dir.glob('results_*.csv'))
    frames = []
    for p in hist_files:
        try:
            df = pd.read_csv(p)
            if df is None or df.empty:
                continue
            frames.append(df)
        except Exception:
            continue
    if not frames:
        print('No historical results files.')
        return
    hist = pd.concat(frames, ignore_index=True)
    if 'date' in hist.columns:
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    if args.date_start:
        ds = pd.to_datetime(args.date_start, errors='coerce')
        if pd.notna(ds):
            hist = hist[hist['date'] >= ds]
    if args.date_end:
        de = pd.to_datetime(args.date_end, errors='coerce')
        if pd.notna(de):
            hist = hist[hist['date'] <= de]

    # Build mapping from game_id to outcomes
    if 'game_id' not in hist.columns:
        print('game_id missing in historical results.')
        return
    outcome_map = hist[['game_id'] + list(TARGETS.values())].copy()
    outcome_map['game_id'] = outcome_map['game_id'].astype(str)

    # Load probability files
    prob_files = sorted(out_dir.glob(args.probs_file_pattern))
    if not prob_files:
        print('No probability files found for pattern:', args.probs_file_pattern)
        return
    merged = []
    for pf in prob_files:
        try:
            pr = pd.read_csv(pf)
            if pr.empty or 'game_id' not in pr.columns:
                continue
            pr['game_id'] = pr['game_id'].astype(str)
            merged.append(pr)
        except Exception:
            continue
    if not merged:
        print('No valid probability files.')
        return
    probs = pd.concat(merged, ignore_index=True)

    calib_results = {}
    for p_col, y_col in TARGETS.items():
        if p_col not in probs.columns or y_col not in outcome_map.columns:
            calib_results[p_col] = {'skipped': True}
            continue
        joined = probs.merge(outcome_map[['game_id', y_col]], on='game_id', how='inner')
        joined = joined.dropna(subset=[p_col, y_col])
        if joined.empty:
            calib_results[p_col] = {'skipped': True}
            continue
        y = joined[y_col].astype(int)
        p = joined[p_col].astype(float)
        if IsotonicRegression is None:
            calib_results[p_col] = {'error': 'IsotonicRegression unavailable'}
            continue
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p, y.values)
        p_cal = iso.predict(p)
        # Simple metrics pre/post (ECE approximation, accuracy)
        acc_pre = float(np.mean((p >= 0.5) == (y.values == 1)))
        acc_post = float(np.mean((p_cal >= 0.5) == (y.values == 1)))
        # ECE quick approximation (10 bins)
        def _ece(pt, pr):
            bins = np.linspace(0, 1, 11)
            ece = 0.0
            for i in range(10):
                mask = (pr >= bins[i]) & (pr < bins[i+1]) if i < 9 else (pr >= bins[i]) & (pr <= bins[i+1])
                if not np.any(mask):
                    continue
                acc = np.mean(pt[mask])
                conf = np.mean(pr[mask])
                weight = np.mean(mask)
                ece += weight * abs(acc - conf)
            return float(ece)
        ece_pre = _ece((y.values==1).astype(float), p)
        ece_post = _ece((y.values==1).astype(float), p_cal)
        calib_results[p_col] = {
            'n_rows': int(len(joined)),
            'accuracy_pre': acc_pre,
            'accuracy_post': acc_post,
            'ece_pre': ece_pre,
            'ece_post': ece_post,
        }
        if args.save:
            dump(iso, out_dir / f'calibrator_{p_col}.joblib')

    # Persist summary
    summary_path = out_dir / f'calibration_summary_{Path(args.probs_file_pattern).stem}_{len(prob_files)}files.json'
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(calib_results, f, indent=2)
    print('Calibration summary:', summary_path)


if __name__ == '__main__':
    main()
