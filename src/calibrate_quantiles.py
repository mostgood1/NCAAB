import argparse
import json
from pathlib import Path
import pandas as pd

OUT = Path("outputs")
DAILY_RESULTS = OUT / "daily_results"


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def compute_q50_offset(dates):
    rows = []
    for d in dates:
        # Prefer model totals predictions which always include quantiles
        preds = _safe_read_csv(OUT / f"predictions_model_totals_{d}.csv")
        finals = _safe_read_csv(DAILY_RESULTS / f"results_{d}.csv")
        if preds.empty or finals.empty:
            continue
        # Prefer joining by stable game_id if available; else use (date, home, away)
        join_keys = []
        if "game_id" in preds.columns and "game_id" in finals.columns:
            join_keys = ["game_id"]
        else:
            join_keys = ["date","home_team","away_team"]
        merged = preds.merge(finals[join_keys + ["actual_total"]], on=join_keys, how="inner")
        if "pred_total_q50" not in merged.columns:
            continue
        if "actual_total" not in merged.columns:
            continue
        merged["err_q50"] = pd.to_numeric(merged["pred_total_q50"], errors="coerce") - pd.to_numeric(merged["actual_total"], errors="coerce")
        # Accumulate errors; team columns may be absent in model predictions
        rows.append(merged[["date","err_q50"]])
    if not rows:
        return None
    all_err = pd.concat(rows, ignore_index=True)
    all_err = all_err.dropna(subset=["err_q50"]) 
    if all_err.empty:
        return None
    bias = float(all_err["err_q50"].mean())
    return {"q50_offset": bias, "n": int(len(all_err))}


def main():
    ap = argparse.ArgumentParser(description="Calibrate totals quantiles using finals")
    ap.add_argument("--dates", nargs="*", help="Explicit date list YYYY-MM-DD")
    ap.add_argument("--recent", type=int, default=7, help="Use N most recent results")
    args = ap.parse_args()

    if args.dates:
        dates = args.dates
    else:
        # infer available results dates and take recent N
        avail = sorted([p.stem.replace("results_", "") for p in DAILY_RESULTS.glob("results_*.csv")])
        dates = avail[-args.recent:] if avail else []

    res = compute_q50_offset(dates)
    if not res:
        print("No calibration computed; missing data.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_quantiles.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(OUT / "calibration_quantiles.json"), **res}, indent=2))


if __name__ == "__main__":
    main()
