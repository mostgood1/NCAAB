import sys
import json
import itertools
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

SEG_BOUNDS = [135, 145, 155, 165]  # very_low, low, mid, high, very_high
SEG_NAMES = ["very_low", "low", "mid", "high", "very_high"]


def segment_for_line(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "unknown"
    if v <= SEG_BOUNDS[0]:
        return SEG_NAMES[0]
    if v <= SEG_BOUNDS[1]:
        return SEG_NAMES[1]
    if v <= SEG_BOUNDS[2]:
        return SEG_NAMES[2]
    if v <= SEG_BOUNDS[3]:
        return SEG_NAMES[3]
    return SEG_NAMES[4]


def robust_outcome_over(row: pd.Series) -> float | None:
    # textual
    for oc in ("ou_result_full", "ou_result_full_res"):
        v = row.get(oc)
        if v is not None and str(v).strip() != "":
            s = str(v).strip().lower()
            if s == "over":
                return 1.0
            if s == "under":
                return 0.0
    # numeric
    try:
        actual = row.get("actual_total")
        line = row.get("market_total") if row.get("market_total") is not None else row.get("closing_total")
        if actual is None or line is None:
            return None
        av = float(actual); lv = float(line)
        return 1.0 if av > lv else 0.0
    except Exception:
        return None


def list_candidate_dates(out_dir: Path, days: int) -> list[str]:
    dr = out_dir / 'daily_results'
    dates = []
    if dr.exists():
        for p in sorted(dr.glob('results_*.csv'), reverse=True):
            d = p.stem.replace('results_', '')
            dates.append(d)
    return dates[:days]


def load_join_for_date(out_dir: Path, date: str) -> pd.DataFrame:
    blend_path = out_dir / f"sim_blend_{date}.csv"
    results_path = out_dir / 'daily_results' / f"results_{date}.csv"
    if not blend_path.exists() or not results_path.exists():
        return pd.DataFrame()
    b = pd.read_csv(blend_path)
    r = pd.read_csv(results_path)
    try:
        if 'game_id' in b.columns:
            b['game_id'] = b['game_id'].astype(str)
        if 'game_id' in r.columns:
            r['game_id'] = r['game_id'].astype(str)
    except Exception:
        pass
    if 'game_id' in b.columns and 'game_id' in r.columns:
        m = b.merge(r, on='game_id', how='left')
    else:
        m = b.merge(r, on=['home_team','away_team'], how='left')
    # derive line for segmentation as a Series
    if 'market_total' in m.columns:
        line_ser = pd.to_numeric(m['market_total'], errors='coerce')
    elif 'closing_total' in m.columns:
        line_ser = pd.to_numeric(m['closing_total'], errors='coerce')
    else:
        line_ser = pd.Series([float('nan')] * len(m))
    m['seg'] = line_ser.apply(segment_for_line)
    m['p'] = pd.to_numeric(m.get('p_over_blend'), errors='coerce')
    m['y_over'] = m.apply(robust_outcome_over, axis=1)
    return m


def evaluate_thresholds(df: pd.DataFrame, hi: float, lo: float) -> Tuple[int, float | None]:
    # selected rows
    sel = (df['p'].notna()) & (df['y_over'].notna()) & (
        (df['p'] >= hi) | (df['p'] <= lo)
    )
    picked = df[sel]
    if picked.empty:
        return 0, None
    preds = picked['p'] >= 0.5
    acc = float((preds.astype(float).values == picked['y_over'].astype(float).values).mean())
    return int(len(picked)), acc


def tune_for_segment(df: pd.DataFrame, hi_grid, lo_grid, min_picks: int) -> Tuple[float, float, int, float | None]:
    best = (None, None, 0, None)  # hi, lo, n, acc
    for hi, lo in itertools.product(hi_grid, lo_grid):
        if lo >= 0.5 or hi <= 0.5:
            continue
        n, acc = evaluate_thresholds(df, hi, lo)
        if n < max(1, min_picks):
            # track largest n if we don't meet coverage
            if best[2] < n:
                best = (hi, lo, n, acc)
            continue
        if best[3] is None or (acc is not None and acc > best[3]) or (acc == best[3] and n > best[2]):
            best = (hi, lo, n, acc)
    # if never met min, best carries the largest n candidate
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Tune OU hi/lo per totals segment using trailing days')
    ap.add_argument('--days', type=int, default=28)
    ap.add_argument('--outputs', type=str, default='outputs')
    ap.add_argument('--min-per-segment', type=int, default=50)
    ap.add_argument('--hi-grid', type=str, default='0.56,0.58,0.60,0.62,0.64')
    ap.add_argument('--lo-grid', type=str, default='0.44,0.42,0.40,0.38,0.36,0.34')
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    dates = list_candidate_dates(out_dir, args.days)
    frames = []
    for d in dates:
        m = load_join_for_date(out_dir, d)
        if not m.empty:
            frames.append(m[['seg','p','y_over']].copy())
    if not frames:
        print(json.dumps({"error": "no data for tuning"}))
        return 2
    all_df = pd.concat(frames, ignore_index=True)

    hi_grid = [float(x) for x in args.hi_grid.split(',')]
    lo_grid = [float(x) for x in args.lo_grid.split(',')]

    result: Dict[str, Dict[str, float]] = {}
    summary = {}
    for seg in list(all_df['seg'].dropna().unique()) + ['unknown']:
        sdf = all_df[all_df['seg'] == seg].copy()
        if sdf.empty:
            continue
        hi, lo, n, acc = tune_for_segment(sdf, hi_grid, lo_grid, args.min_per_segment)
        # default to 0.60/0.40 when no candidate
        if hi is None or lo is None:
            hi, lo = 0.60, 0.40
        result[seg] = {"hi": float(hi), "lo": float(lo)}
        summary[seg] = {"n": int(n), "accuracy": (None if acc is None else float(acc))}

    metrics_dir = out_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / 'ou_segment_thresholds.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"wrote": str(out_path), "segments": list(result.keys()), "summary": summary}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
