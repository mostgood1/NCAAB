"""Daily residual distribution & summary generator.

Usage:
  python scripts/residuals_daily.py --date YYYY-MM-DD
If --date omitted, defaults to yesterday (local).

Outputs:
  outputs/residuals_<date>.json : distribution summary & histogram buckets
  outputs/residuals_history.csv : appended per-game residuals (cumulative)

Logic:
  - Load predictions_unified_<date>.csv (fallback: predictions_model_<date>.csv)
  - Load daily_results/results_<date>.csv or games_<date>.csv for actual scores
  - Compute actual_total = home_score + away_score; actual_margin = home_score - away_score
  - Use pred_total_model (or pred_total) and pred_margin_model (or pred_margin) for residuals
  - Produce stats: mean, std, skew, kurtosis, quantiles, tail counts, calibration slope proxy.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs")

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _preds_for(date_str: str) -> pd.DataFrame:
    uni = _safe_csv(OUT / f"predictions_unified_{date_str}.csv")
    if not uni.empty:
        uni["game_id"] = uni["game_id"].astype(str)
        return uni
    mp = _safe_csv(OUT / f"predictions_model_{date_str}.csv")
    if not mp.empty:
        mp["game_id"] = mp.get("game_id", mp.get("id", pd.Series(range(len(mp)))))
        mp["game_id"] = mp["game_id"].astype(str)
        # normalize name columns
        if "pred_total_model" in mp.columns and "pred_total" not in mp.columns:
            mp["pred_total"] = mp["pred_total_model"]
        if "pred_margin_model" in mp.columns and "pred_margin" not in mp.columns:
            mp["pred_margin"] = mp["pred_margin_model"]
        return mp
    return pd.DataFrame()

def _results_for(date_str: str) -> pd.DataFrame:
    dr = _safe_csv(OUT / "daily_results" / f"results_{date_str}.csv")
    if not dr.empty and {"home_score","away_score"}.issubset(dr.columns):
        dr["game_id"] = dr.get("game_id", dr.get("id", pd.Series(range(len(dr)))))
        dr["game_id"] = dr["game_id"].astype(str)
        return dr
    g = _safe_csv(OUT / f"games_{date_str}.csv")
    if not g.empty and {"home_score","away_score"}.issubset(g.columns):
        g["game_id"] = g.get("game_id", g.get("id", pd.Series(range(len(g)))))
        g["game_id"] = g["game_id"].astype(str)
        return g
    return pd.DataFrame()

def _moment_stats(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"count":0}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    # Sample skew/kurtosis (Fisher) manual to avoid scipy dep
    m3 = float(np.mean((vals - mean)**3))
    m2 = std**2
    m4 = float(np.mean((vals - mean)**4))
    skew = m3 / (m2**1.5) if m2 > 1e-12 else 0.0
    kurtosis = m4 / (m2**2) - 3.0 if m2 > 1e-12 else 0.0
    q05, q25, q50, q75, q95 = [float(np.quantile(vals, q)) for q in [0.05,0.25,0.5,0.75,0.95]]
    tails_gt10 = int(np.sum(np.abs(vals) > 10))
    tails_gt15 = int(np.sum(np.abs(vals) > 15))
    return {
        "count": int(vals.size),
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "p05": q05,
        "p25": q25,
        "p50": q50,
        "p75": q75,
        "p95": q95,
        "tails_gt10": tails_gt10,
        "tails_gt15": tails_gt15,
    }

def _hist(vals: np.ndarray, bin_width: int = 2, limit: int = 30) -> list:
    if vals.size == 0:
        return []
    bins = list(range(-limit, limit + bin_width, bin_width))
    hist = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (vals >= lo) & (vals < hi)
        hist.append({"lo": lo, "hi": hi, "count": int(mask.sum())})
    return hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    preds = _preds_for(date_str)
    results = _results_for(date_str)

    payload = {"date": date_str, "generated_at": dt.datetime.now().isoformat()}
    if preds.empty or results.empty:
        payload["status"] = "missing_inputs"
    else:
        merged = preds.merge(results[["game_id","home_score","away_score"]], on="game_id", how="inner")
        merged["actual_total"] = merged["home_score"] + merged["away_score"]
        merged.loc[(merged["home_score"]==0) & (merged["away_score"]==0), "actual_total"] = np.nan
        merged["actual_margin"] = merged["home_score"] - merged["away_score"]
        merged.loc[(merged["home_score"]==0) & (merged["away_score"]==0), "actual_margin"] = np.nan
        # prediction columns
        pt_col = "pred_total_model" if "pred_total_model" in merged.columns else "pred_total"
        pm_col = "pred_margin_model" if "pred_margin_model" in merged.columns else "pred_margin"
        merged["residual_total"] = merged[pt_col] - merged["actual_total"]
        merged["residual_margin"] = merged[pm_col] - merged["actual_margin"]
        # Filter only resolved rows
        resolved = merged[merged["actual_total"].notna()]
        rtot = resolved["residual_total"].to_numpy(dtype=float)
        rmar = resolved["residual_margin"].to_numpy(dtype=float)
        payload["total_stats"] = _moment_stats(rtot)
        payload["margin_stats"] = _moment_stats(rmar)
        payload["total_hist"] = _hist(rtot)
        payload["margin_hist"] = _hist(rmar, bin_width=1, limit=20)
        # Simple calibration slope proxy: corr(pred, actual)
        try:
            pred_vals = resolved[pt_col].astype(float)
            act_vals = resolved["actual_total"].astype(float)
            if len(resolved) > 2:
                corr = float(np.corrcoef(pred_vals, act_vals)[0,1])
            else:
                corr = None
            payload["total_corr"] = corr
        except Exception:
            payload["total_corr_error"] = True
        try:
            pred_m = resolved[pm_col].astype(float)
            act_m = resolved["actual_margin"].astype(float)
            if len(resolved) > 2:
                corr_m = float(np.corrcoef(pred_m, act_m)[0,1])
            else:
                corr_m = None
            payload["margin_corr"] = corr_m
        except Exception:
            payload["margin_corr_error"] = True
        # Append per-game residuals to history CSV
        hist_rows = resolved[["game_id", pt_col, pm_col, "actual_total", "actual_margin", "residual_total", "residual_margin"]].copy()
        hist_rows["date"] = date_str
        hist_path = OUT / "residuals_history.csv"
        try:
            if hist_path.exists():
                hist_rows.to_csv(hist_path, mode="a", header=False, index=False)
            else:
                hist_rows.to_csv(hist_path, index=False)
        except Exception:
            payload["history_append_error"] = True

    out_path = OUT / f"residuals_{date_str}.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote residuals summary to {out_path}")
    except Exception as e:
        print(f"Failed writing residuals JSON: {e}")

if __name__ == "__main__":
    main()
