import sys
import json
from pathlib import Path
import pandas as pd


def robust_outcome_over(row: pd.Series) -> float:
    # Returns 1.0 if final total > line, 0.0 if <= line, NaN if unknown
    # Try explicit columns first
    for c in ["outcome_over", "ou_win", "ou_result_over"]:
        if c in row and pd.notna(row[c]):
            try:
                v = float(row[c])
                if v in (0.0, 1.0):
                    return v
            except Exception:
                pass
    # Map textual OU result if present
    for oc in ["ou_result_full", "ou_result_full_res"]:
        if oc in row and pd.notna(row[oc]):
            s = str(row[oc]).strip().lower()
            if s == "over":
                return 1.0
            if s == "under":
                return 0.0
    # Compute from scores and a line
    final_total = None
    for c in ["final_total", "total_final", "score_total", "home_score", "away_score"]:
        if c in row and pd.notna(row[c]):
            if c in ("home_score", "away_score"):
                # need both to compute total
                pass
            else:
                try:
                    final_total = float(row[c])
                    break
                except Exception:
                    continue
    if final_total is None and all(col in row for col in ("home_score", "away_score")):
        try:
            hs = float(row["home_score"]) if pd.notna(row["home_score"]) else None
            as_ = float(row["away_score"]) if pd.notna(row["away_score"]) else None
            if hs is not None and as_ is not None:
                final_total = hs + as_
        except Exception:
            final_total = None

    line = None
    # Prefer explicit market_total, else closing_total
    for c in ["market_total", "market_total_res", "closing_total", "closing_total_res", "total_line", "ou_line"]:
        if c in row and pd.notna(row[c]):
            try:
                line = float(row[c])
                break
            except Exception:
                continue
    # Prefer explicit actual_total if present
    if final_total is None:
        for ac in ["actual_total", "actual_total_res"]:
            if ac in row and pd.notna(row[ac]):
                try:
                    final_total = float(row[ac])
                    break
                except Exception:
                    continue

    if final_total is None or line is None:
        return float("nan")
    return 1.0 if final_total > line else 0.0


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: compute_sim_blend_accuracy.py <date> [outputs_dir] [threshold]"}))
        return 1
    date = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    blend_path = out_dir / f"sim_blend_{date}.csv"
    results_path = out_dir / "daily_results" / f"results_{date}.csv"

    if not blend_path.exists():
        print(json.dumps({"date": date, "error": f"blend file missing: {blend_path}"}))
        return 2
    if not results_path.exists():
        print(json.dumps({"date": date, "error": f"results file missing: {results_path}"}))
        return 3

    blend = pd.read_csv(blend_path)
    results = pd.read_csv(results_path)

    # Try join on game_id first, then fallback to teams
    id_col = "game_id" if "game_id" in blend.columns and "game_id" in results.columns else None
    if id_col:
        merged = blend.merge(results, on=id_col, how="left", suffixes=("_blend","_res"))
    else:
        if all(c in blend.columns for c in ["home_team","away_team"]) and \
           all(c in results.columns for c in ["home_team","away_team"]):
            merged = blend.merge(results, on=["home_team","away_team"], how="left", suffixes=("_blend","_res"))
        else:
            print(json.dumps({"date": date, "error": "Cannot join blend and results; missing keys"}))
            return 4

    merged["outcome_over_bin"] = merged.apply(robust_outcome_over, axis=1)
    valid = merged[merged["outcome_over_bin"].notna()].copy()
    if len(valid) == 0:
        print(json.dumps({"date": date, "error": "No valid outcomes found in results"}))
        return 5

    # All-games accuracy using blended probability as decision (>= thresh => over)
    preds_over = (valid["p_over_blend"].astype(float) >= thresh).astype(float)
    acc = float((preds_over == valid["outcome_over_bin"].astype(float)).mean())

    print(json.dumps({
        "date": date,
        "n": int(len(valid)),
        "threshold": thresh,
        "accuracy_all_games": acc
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
