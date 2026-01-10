import sys
import json
from pathlib import Path
import pandas as pd
from tune_ou_segment_thresholds import segment_for_line, robust_outcome_over, list_candidate_dates, load_join_for_date


def evaluate_with_thresholds(out_dir: Path, days: int, thresholds: dict) -> dict:
    dates = list_candidate_dates(out_dir, days)
    rows = []
    for d in dates:
        m = load_join_for_date(out_dir, d)
        if m.empty:
            rows.append({"date": d, "n_picks": 0, "accuracy": None}); continue
        # map thresholds per seg
        def decide(row):
            seg = row.get('seg')
            p = row.get('p')
            try:
                if p is None:
                    return None
                t = thresholds.get(seg, thresholds.get('unknown', {"hi":0.60, "lo":0.40}))
                hi = float(t.get('hi', 0.60)); lo = float(t.get('lo', 0.40))
                if p >= hi:
                    return 'over'
                if p <= lo:
                    return 'under'
                return None
            except Exception:
                return None
        m['pick'] = m.apply(decide, axis=1)
        m['y_over'] = m.apply(robust_outcome_over, axis=1)
        valid = m[m['pick'].notna() & m['y_over'].notna()].copy()
        if len(valid) == 0:
            rows.append({"date": d, "n_picks": 0, "accuracy": None}); continue
        preds = valid['pick'].astype(str).str.lower().map({'over':1.0,'under':0.0})
        acc = float((preds.values == valid['y_over'].astype(float).values).mean())
        rows.append({"date": d, "n_picks": int(len(valid)), "accuracy": acc})
    df = pd.DataFrame(rows)
    dfv = df[df['accuracy'].notna()].copy()
    total_n = int(dfv['n_picks'].sum()) if len(dfv) else 0
    pooled = float((dfv['accuracy']*dfv['n_picks']).sum()/total_n) if total_n > 0 else None
    mean_daily = float(dfv['accuracy'].mean()) if len(dfv) else None
    return {"days": days, "n_picks": total_n, "pooled_accuracy": pooled, "mean_daily_accuracy": mean_daily, "daily": rows}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: backtest_segmented_selected_ou.py <days> [outputs_dir]"}))
        return 1
    days = int(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('outputs')
    th_path = out_dir / 'metrics' / 'ou_segment_thresholds.json'
    if not th_path.exists():
        print(json.dumps({"error": f"thresholds missing: {th_path}"}))
        return 2
    thresholds = json.loads(th_path.read_text(encoding='utf-8'))
    res = evaluate_with_thresholds(out_dir, days, thresholds)
    out_path = out_dir / 'backtests' / f'segmented_selected_ou_summary_last_{days}d.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps({"wrote": str(out_path), "n_picks": res['n_picks'], "pooled_accuracy": res['pooled_accuracy']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
