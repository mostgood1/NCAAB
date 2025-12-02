import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / 'outputs'


def _load_rel(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if {'type','bin_low','bin_high','confidence','accuracy'}.issubset(df.columns):
            return df
    except Exception:
        return None
    return None


def _build_mapping(rel: pd.DataFrame, which: str) -> dict:
    sub = rel[rel['type'] == which].copy()
    sub = sub.dropna(subset=['confidence','accuracy'])
    if sub.empty:
        return {'x': [], 'y': []}
    # Use average confidence per bin as x and accuracy as y; enforce monotone increasing mapping
    sub = sub.sort_values('confidence')
    x = sub['confidence'].to_numpy()
    y = sub['accuracy'].to_numpy()
    # Monotone isotonic-like smoothing by cumulative max of differences
    y_smooth = np.maximum.accumulate(y)
    x_smooth = np.maximum.accumulate(x)  # ensure non-decreasing x
    # Add endpoints for stability
    x_out = [0.0] + [float(v) for v in x_smooth] + [1.0]
    y_out = [0.0] + [float(v) for v in y_smooth] + [1.0]
    # Deduplicate possible ties
    dedup_x, dedup_y = [], []
    last = None
    for xi, yi in zip(x_out, y_out):
        if last is None or abs(xi - last) > 1e-9:
            dedup_x.append(xi); dedup_y.append(yi)
        last = xi
    return {'x': dedup_x, 'y': dedup_y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reliability-csv', type=str, default=str(OUT / 'meta_reliability.csv'))
    ap.add_argument('--out', type=str, default=str(OUT / 'meta_calibration.json'))
    args = ap.parse_args()

    rel = _load_rel(Path(args.reliability_csv))
    if rel is None or rel.empty:
        print('[meta] No reliability file found; skipping meta calibration')
        return 0
    mapping = {
        'p_home_cover': _build_mapping(rel, 'cover'),
        'p_over': _build_mapping(rel, 'over'),
        'timestamp_utc': datetime.utcnow().isoformat(),
    }
    Path(args.out).write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    print(f"[meta] Wrote isotonic-style meta calibration -> {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
