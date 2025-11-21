"""Daily prediction accuracy and drift metrics.

Computes evaluation metrics for a given date using daily_results (preferred) or
unified predictions export as fallback. Persists JSON artifact under outputs.
"""
from __future__ import annotations
import json, math, datetime as dt
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

OUT = Path("outputs")

def _safe_read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def compute_daily_metrics(date: str) -> Dict[str, Any]:
    date = date.strip()
    results_path = OUT / "daily_results" / f"results_{date}.csv"
    unified_path = OUT / f"predictions_unified_{date}.csv"
    df = _safe_read(results_path)
    source = "daily_results"
    if df.empty:
        df = _safe_read(unified_path)
        source = "unified_predictions"
    if df.empty:
        return {"date": date, "source": None, "error": "no_data"}
    # Normalize date
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            df["date"] = df["date"].astype(str)
    # Actual total
    if {"home_score","away_score"}.issubset(df.columns):
        try:
            hs = pd.to_numeric(df["home_score"], errors="coerce")
            as_ = pd.to_numeric(df["away_score"], errors="coerce")
            df["actual_total"] = hs + as_
            df["actual_margin"] = hs - as_
        except Exception:
            pass
    # Metrics require actual + predicted
    metrics: Dict[str, Any] = {"date": date, "source": source}
    # Totals error
    if {"pred_total","actual_total"}.issubset(df.columns):
        pt = pd.to_numeric(df["pred_total"], errors="coerce")
        at = pd.to_numeric(df["actual_total"], errors="coerce")
        mask = pt.notna() & at.notna()
        if mask.any():
            diff = pt[mask] - at[mask]
            metrics.update({
                "totals_mae": float(diff.abs().mean()),
                "totals_rmse": float(math.sqrt((diff**2).mean())),
                "totals_bias": float(diff.mean()),
                "totals_n": int(mask.sum()),
            })
    # Margin error
    if {"pred_margin","actual_margin"}.issubset(df.columns):
        pm = pd.to_numeric(df["pred_margin"], errors="coerce")
        am = pd.to_numeric(df["actual_margin"], errors="coerce")
        mask = pm.notna() & am.notna()
        if mask.any():
            diff = pm[mask] - am[mask]
            metrics.update({
                "margin_mae": float(diff.abs().mean()),
                "margin_rmse": float(math.sqrt((diff**2).mean())),
                "margin_bias": float(diff.mean()),
                "margin_n": int(mask.sum()),
            })
    # Calibration slope (simple): regressing actual_total on pred_total
    try:
        if {"pred_total","actual_total"}.issubset(df.columns):
            pt = pd.to_numeric(df["pred_total"], errors="coerce")
            at = pd.to_numeric(df["actual_total"], errors="coerce")
            mask = pt.notna() & at.notna()
            if mask.sum() >= 3:
                # Simple slope = Cov / Var
                cov = float(((pt[mask] - pt[mask].mean()) * (at[mask] - at[mask].mean())).mean())
                var = float(((pt[mask] - pt[mask].mean())**2).mean())
                slope = cov / var if var else np.nan
                metrics["totals_calibration_slope"] = float(slope)
    except Exception:
        pass
    # Distribution drift: compare today's pred_total mean vs last 7 days unified mean
    try:
        hist_means = []
        today_mean = None
        if "pred_total" in df.columns:
            pt = pd.to_numeric(df["pred_total"], errors="coerce")
            today_mean = float(pt.dropna().mean()) if pt.notna().any() else None
        # Iterate prior 7 days
        if today_mean is not None:
            base_date = dt.date.fromisoformat(date)
            for i in range(1, 8):
                d_prev = (base_date - dt.timedelta(days=i)).isoformat()
                prev_df = _safe_read(OUT / f"predictions_unified_{d_prev}.csv")
                if prev_df.empty or "pred_total" not in prev_df.columns:
                    continue
                pm = pd.to_numeric(prev_df["pred_total"], errors="coerce")
                if pm.notna().any():
                    hist_means.append(float(pm.dropna().mean()))
        if today_mean is not None and hist_means:
            metrics["totals_mean_today"] = today_mean
            metrics["totals_mean_hist7"] = float(np.mean(hist_means))
            metrics["totals_drift_abs"] = float(today_mean - np.mean(hist_means))
    except Exception:
        pass
    return metrics

def persist_daily_metrics(date: str) -> Path | None:
    m = compute_daily_metrics(date)
    out_dir = OUT / "eval_metrics"; out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"metrics_{date}.json"
    try:
        p.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return p
    except Exception:
        return None

if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    path = persist_daily_metrics(date_arg)
    print(json.dumps({"written": bool(path), "path": str(path) if path else None}, indent=2))
