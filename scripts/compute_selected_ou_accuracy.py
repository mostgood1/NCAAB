import sys
import json
from pathlib import Path
import pandas as pd


def decide_ou(row: pd.Series, p_hi: float, p_lo: float) -> str | None:
    """Return 'Over'/'Under' if row qualifies under thresholds; else None."""
    p = row.get('p_over_blend')
    try:
        if p is None or str(p).strip() == '':
            return None
        pv = float(p)
        if pv >= float(p_hi):
            return 'Over'
        if pv <= float(p_lo):
            return 'Under'
        return None
    except Exception:
        return None


def outcome_over(row: pd.Series) -> float | None:
    """Return 1.0 if final total > line, 0.0 if <= line, else None."""
    # Prefer explicit textual result
    for oc in ('ou_result_full', 'ou_result_full_res'):
        v = row.get(oc)
        if v is not None and str(v).strip() != '':
            s = str(v).strip().lower()
            if s == 'over':
                return 1.0
            if s == 'under':
                return 0.0
    # Compute from actual vs market/closing total
    try:
        actual = row.get('actual_total')
        line = row.get('market_total') if row.get('market_total') is not None else row.get('closing_total')
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


def compute_for_date(date: str, out_dir: Path, p_hi: float, p_lo: float) -> dict:
    blend_path = out_dir / f"sim_blend_{date}.csv"
    results_path = out_dir / 'daily_results' / f"results_{date}.csv"
    if not blend_path.exists() or not results_path.exists():
        return {"date": date, "n_picks": 0, "accuracy": None}
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
    # Decide picks
    m['pick'] = m.apply(lambda row: decide_ou(row, p_hi, p_lo), axis=1)
    m = m[m['pick'].notna()].copy()
    if len(m) == 0:
        return {"date": date, "n_picks": 0, "accuracy": None}
    # Compute outcomes
    m['y_over'] = m.apply(outcome_over, axis=1)
    valid = m[m['y_over'].notna()].copy()
    if len(valid) == 0:
        return {"date": date, "n_picks": int(len(m)), "accuracy": None}
    # Map picks to binary
    preds = valid['pick'].astype(str).str.lower().map({'over':1.0,'under':0.0})
    acc = float((preds.values == valid['y_over'].astype(float).values).mean())
    return {"date": date, "n_picks": int(len(valid)), "accuracy": acc}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: compute_selected_ou_accuracy.py <days> [outputs_dir] [p_hi] [p_lo]"}))
        return 1
    days = int(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('outputs')
    # Thresholds: default from env if present, else 0.60/0.40
    try:
        p_hi = float(sys.argv[3]) if len(sys.argv) > 3 else float((Path('.').exists() and (os.environ.get('NCAAB_P_OVER_THRESHOLD_HIGH') or '0.60')))
    except Exception:
        p_hi = 0.60
    try:
        p_lo = float(sys.argv[4]) if len(sys.argv) > 4 else float((os.environ.get('NCAAB_P_OVER_THRESHOLD_LOW') or '0.40'))
    except Exception:
        p_lo = 0.40
    dates = list_candidate_dates(out_dir, days)
    rows = [compute_for_date(d, out_dir, p_hi, p_lo) for d in dates]
    df = pd.DataFrame(rows)
    # Summaries
    dfv = df[df['accuracy'].notna()].copy()
    total_n = int(dfv['n_picks'].sum()) if len(dfv) else 0
    pooled = float((dfv['accuracy']*dfv['n_picks']).sum()/total_n) if total_n > 0 else None
    mean_daily = float(dfv['accuracy'].mean()) if len(dfv) else None
    out_path = out_dir / 'backtests' / f'selected_ou_summary_last_{days}d.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(json.dumps({"days": days, "n_picks": total_n, "pooled_accuracy": pooled, "mean_daily_accuracy": mean_daily, "wrote": str(out_path)}))
    return 0


if __name__ == '__main__':
    import os
    sys.exit(main())
