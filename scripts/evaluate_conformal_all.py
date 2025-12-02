import argparse
import os
from pathlib import Path
import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compute_coverage(df: pd.DataFrame, low_col: str, high_col: str, actual_col: str) -> float:
    if df.empty or any(c not in df.columns for c in [low_col, high_col, actual_col]):
        return float('nan')
    valid = df[[low_col, high_col, actual_col]].dropna()
    if valid.empty:
        return float('nan')
    covered = (valid[actual_col] >= valid[low_col]) & (valid[actual_col] <= valid[high_col])
    return covered.mean()


def compute_width(df: pd.DataFrame, low_col: str, high_col: str) -> float:
    if df.empty or any(c not in df.columns for c in [low_col, high_col]):
        return float('nan')
    valid = df[[low_col, high_col]].dropna()
    if valid.empty:
        return float('nan')
    return (valid[high_col] - valid[low_col]).mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw vs conformal coverage/width across all dates")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing artifacts")
    parser.add_argument("--results-file", default="daily_results/results_latest.csv", help="Fallback results file if per-date missing")
    args = parser.parse_args()

    out_dir = Path(args.outputs_dir)
    results_latest = Path(args.results_file)

    quant_hist = safe_read_csv(out_dir / "quantiles_history.csv")
    conf_hist = safe_read_csv(out_dir / "quantiles_conformal_history.csv")
    results_hist = safe_read_csv(out_dir / "results_history.csv")
    results_fallback = safe_read_csv(results_latest)

    if quant_hist.empty or conf_hist.empty:
        print("[conformal-all-eval] Missing quantile or conformal history; nothing to evaluate.")
        return

    # Expect columns: date, game_id, q10_total/q50_total/q90_total, q10_margin/q50_margin/q90_margin
    # Conformal history columns: total_c10/c50/c90, margin_c10/c50/c90
    # Results history should contain: date, game_id, actual_total, actual_margin
    needed_quant_cols = {"date", "game_id", "q10_total", "q90_total", "q10_margin", "q90_margin"}
    needed_conf_cols = {"date", "game_id", "total_c10", "total_c90", "margin_c10", "margin_c90"}
    if not needed_quant_cols.issubset(set(quant_hist.columns)) or not needed_conf_cols.issubset(set(conf_hist.columns)):
        print("[conformal-all-eval] Required columns not found in history files.")
        return

    # Prepare results by date
    if results_hist.empty:
        results_hist = pd.DataFrame(columns=["date", "game_id", "total_actual", "margin_actual"])

    # If results_history is empty, try using fallback latest results for any overlapping date present in histories
    if results_hist.empty and not results_fallback.empty:
        # Attempt to infer date from histories
        candidate_dates = sorted(set(quant_hist["date"]).intersection(set(conf_hist["date"])))
        if candidate_dates:
            results_fallback = results_fallback.copy()
            # Ensure columns exist
            for col in ["date", "game_id", "total_actual", "margin_actual"]:
                if col not in results_fallback.columns:
                    results_fallback[col] = pd.NA
            # If date missing, fill with most recent
            if results_fallback["date"].isna().all():
                results_fallback["date"] = candidate_dates[-1]
            results_hist = results_fallback[["date", "game_id", "total_actual", "margin_actual"]]

    # Normalize ids
    for df in (quant_hist, conf_hist, results_hist, results_fallback):
        if not df.empty:
            for col in ("game_id", "date"):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)

    # Aggregate per date
    dates = sorted(set(quant_hist["date"]).intersection(set(conf_hist["date"])))
    rows = []
    for d in dates:
        qd = quant_hist[quant_hist["date"] == d]
        cd = conf_hist[conf_hist["date"] == d]
        rd = results_hist[results_hist["date"] == d]

        # Merge for aligned game_id
        base = qd.merge(rd, on=["date", "game_id"], how="inner")
        conf = cd.merge(rd, on=["date", "game_id"], how="inner")

        total_cov_raw = compute_coverage(base, "q10_total", "q90_total", "actual_total")
        total_cov_conf = compute_coverage(conf, "total_c10", "total_c90", "actual_total")
        margin_cov_raw = compute_coverage(base, "q10_margin", "q90_margin", "actual_margin")
        margin_cov_conf = compute_coverage(conf, "margin_c10", "margin_c90", "actual_margin")

        total_width_raw = compute_width(qd, "q10_total", "q90_total")
        total_width_conf = compute_width(cd, "total_c10", "total_c90")
        margin_width_raw = compute_width(qd, "q10_margin", "q90_margin")
        margin_width_conf = compute_width(cd, "margin_c10", "margin_c90")

        rows.append({
            "date": d,
            "total_cov_raw": total_cov_raw,
            "total_cov_conf": total_cov_conf,
            "total_cov_delta": (total_cov_conf - total_cov_raw) if pd.notna(total_cov_raw) and pd.notna(total_cov_conf) else float('nan'),
            "total_width_raw": total_width_raw,
            "total_width_conf": total_width_conf,
            "total_width_ratio": (total_width_conf / total_width_raw) if pd.notna(total_width_raw) and total_width_raw not in (0, 0.0) else float('nan'),
            "margin_cov_raw": margin_cov_raw,
            "margin_cov_conf": margin_cov_conf,
            "margin_cov_delta": (margin_cov_conf - margin_cov_raw) if pd.notna(margin_cov_raw) and pd.notna(margin_cov_conf) else float('nan'),
            "margin_width_raw": margin_width_raw,
            "margin_width_conf": margin_width_conf,
            "margin_width_ratio": (margin_width_conf / margin_width_raw) if pd.notna(margin_width_raw) and margin_width_raw not in (0, 0.0) else float('nan'),
        })

    res = pd.DataFrame(rows)
    out_path = out_dir / "conformal_metrics_all.csv"
    res.to_csv(out_path, index=False)
    print(f"[conformal-all-eval] Wrote {out_path}")


if __name__ == "__main__":
    main()
