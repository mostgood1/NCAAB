import pandas as pd
import numpy as np
from pathlib import Path
import json

OUT = Path("outputs")
REPORT_DIR = OUT / "backtest_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def main():
    # Load joined backtest if present; else attempt to construct from daily_results and predictions
    joined_path = REPORT_DIR / "backtest_joined.csv"
    df = _safe_read_csv(joined_path)
    if df.empty:
        # Fallback assembly
        dr_all = _safe_read_csv(OUT / "daily_results" / "results_all.csv")
        preds_all = _safe_read_csv(OUT / "predictions_all.csv")
        if not dr_all.empty and not preds_all.empty:
            try:
                dr_all["game_id"] = dr_all.get("game_id", pd.Series()).astype(str)
                preds_all["game_id"] = preds_all.get("game_id", pd.Series()).astype(str)
                df = dr_all.merge(preds_all, on="game_id", how="left", suffixes=("_res","_pred"))
            except Exception:
                df = dr_all.copy()
        else:
            df = dr_all.copy()

    if df.empty:
        print("No backtest data available.")
        return

    # Derive actual totals/margins from scores
    if {"home_score","away_score"}.issubset(df.columns):
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        as_ = pd.to_numeric(df["away_score"], errors="coerce")
        df["actual_total"] = hs + as_
        df["actual_margin"] = hs - as_
    else:
        df["actual_total"] = np.nan
        df["actual_margin"] = np.nan

    # Choose prediction columns (prefer calibrated)
    pred_total = pd.to_numeric(df.get("pred_total_calibrated"), errors="coerce") if "pred_total_calibrated" in df.columns else pd.to_numeric(df.get("pred_total"), errors="coerce")
    pred_margin = pd.to_numeric(df.get("pred_margin_calibrated"), errors="coerce") if "pred_margin_calibrated" in df.columns else pd.to_numeric(df.get("pred_margin"), errors="coerce")
    df["_pred_total"] = pred_total
    df["_pred_margin"] = pred_margin

    # Market/closing lines
    def _num_series(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)
    closing_total = _num_series("closing_total")
    market_total = _num_series("market_total")
    df["_line_total"] = np.where(closing_total.notna(), closing_total, market_total)
    closing_spread = _num_series("closing_spread_home")
    spread_home = _num_series("spread_home")
    df["_line_spread"] = np.where(closing_spread.notna(), closing_spread, spread_home)

    # Cohort filter: finals-only rows with valid actuals and lines
    finals_mask = True
    if "status" in df.columns:
        finals_mask = df["status"].astype(str).str.lower().isin(["final","completed","finished","done"]) | (pd.to_numeric(df.get("home_score"), errors="coerce").notna() & pd.to_numeric(df.get("away_score"), errors="coerce").notna())
    valid_actuals = df["actual_total"].notna()
    valid_lines = df["_line_total"].notna() | df["_line_spread"].notna()
    cohort = df[finals_mask & valid_actuals & valid_lines].copy()

    if cohort.empty:
        print("No valid cohort for backtest.")
        return

    # Metrics
    # OU hit rate: compare prediction vs line with actual outcome over/under
    ou_mask = cohort["_pred_total"].notna() & cohort["_line_total"].notna()
    ou_pred_over = cohort.loc[ou_mask, "_pred_total"] > cohort.loc[ou_mask, "_line_total"]
    ou_actual_over = cohort.loc[ou_mask, "actual_total"] > cohort.loc[ou_mask, "_line_total"]
    ou_hit = (ou_pred_over == ou_actual_over).mean() if len(ou_pred_over) else np.nan

    # ATS hit rate: model margin vs spread direction
    ats_mask = cohort["_pred_margin"].notna() & cohort["_line_spread"].notna()
    # Spread convention: negative means home favored; win ATS if pred_margin - spread has same sign as actual_margin - spread
    pred_edge = cohort.loc[ats_mask, "_pred_margin"] - cohort.loc[ats_mask, "_line_spread"]
    actual_edge = cohort.loc[ats_mask, "actual_margin"] - cohort.loc[ats_mask, "_line_spread"]
    ats_hit = (np.sign(pred_edge) == np.sign(actual_edge)).mean() if len(pred_edge) else np.nan

    # Bias & RMSE
    totals_bias = (cohort["_pred_total"] - cohort["actual_total"]).dropna()
    totals_rmse = np.sqrt(((cohort["_pred_total"] - cohort["actual_total"]) ** 2).dropna().mean()) if totals_bias.size else np.nan
    margin_bias = (cohort["_pred_margin"] - cohort["actual_margin"]).dropna()
    margin_rmse = np.sqrt(((cohort["_pred_margin"] - cohort["actual_margin"]) ** 2).dropna().mean()) if margin_bias.size else np.nan

    summary = {
        "n_games": int(len(cohort)),
        "ou_hit_rate": None if pd.isna(ou_hit) else float(ou_hit),
        "ats_hit_rate": None if pd.isna(ats_hit) else float(ats_hit),
        "totals_bias_mean": None if totals_bias.empty else float(totals_bias.mean()),
        "totals_rmse": None if pd.isna(totals_rmse) else float(totals_rmse),
        "margin_bias_mean": None if margin_bias.empty else float(margin_bias.mean()),
        "margin_rmse": None if pd.isna(margin_rmse) else float(margin_rmse),
    }

    # Save
    cohort.to_csv(REPORT_DIR / "backtest_cohort.csv", index=False)
    with open(REPORT_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
