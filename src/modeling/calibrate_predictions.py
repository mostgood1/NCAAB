"""Calibration script for model predictions.

Uses recent finalized game results to fit a simple linear correction layer
for total points and margin predictions:

    y_true ≈ a + b * y_pred

If insufficient historical results are available, falls back to bias-only or
identity mapping.

Outputs a calibrated predictions CSV with added columns:
  pred_total_calibrated, pred_margin_calibrated

Usage:
  python -m src.modeling.calibrate_predictions --date 2025-11-19 \
      --predictions-file outputs/predictions_model_2025-11-19.csv \
      --results-dir outputs/daily_results --window-days 14
"""
from __future__ import annotations
import argparse, pathlib, datetime as dt, json
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def _load_recent_results(results_dir: pathlib.Path, cutoff_date: dt.date, window_days: int) -> pd.DataFrame:
    frames = []
    for p in results_dir.glob("results_*.csv"):
        try:
            # Extract date from filename
            date_str = p.stem.replace("results_", "")
            d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if (cutoff_date - d).days <= window_days and (cutoff_date - d).days >= 0:
            try:
                df = pd.read_csv(p)
                if {"game_id","total_points","margin"}.issubset(df.columns):
                    frames.append(df[["game_id","total_points","margin"]].copy())
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["game_id","total_points","margin"])
    return pd.concat(frames, ignore_index=True)


def _fit_linear(y_true: np.ndarray, y_pred: np.ndarray):
    # Handle degenerate cases
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if len(y_true) < 10 or np.std(y_pred) < 1e-6:
        # Only bias adjust
        bias = float(np.mean(y_true - y_pred)) if len(y_true) else 0.0
        return {"a": bias, "b": 1.0, "mode": "bias"}
    # Ordinary least squares for slope/intercept
    X = np.vstack([np.ones_like(y_pred), y_pred]).T
    try:
        coeffs, *_ = np.linalg.lstsq(X, y_true, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    except Exception:
        a = float(np.mean(y_true - y_pred)); b = 1.0
        return {"a": a, "b": b, "mode": "fallback"}
    return {"a": a, "b": b, "mode": "linear"}


def calibrate(pred_path: pathlib.Path, date_str: str, window_days: int, results_dir: pathlib.Path, out_path: pathlib.Path | None) -> pathlib.Path:
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions file missing: {pred_path}")
    preds = pd.read_csv(pred_path)
    if "game_id" not in preds.columns:
        raise ValueError("predictions CSV missing game_id column")
    # Load recent results
    cutoff_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    results_df = _load_recent_results(results_dir, cutoff_date, window_days)
    # Merge historical residuals (exclude today's games if present)
    hist = preds.merge(results_df, on="game_id", how="inner")
    # Fit calibrators
    total_cal = _fit_linear(hist.get("total_points", pd.Series(dtype=float)).to_numpy(), hist.get("pred_total_model", pd.Series(dtype=float)).to_numpy()) if not hist.empty else {"a":0.0,"b":1.0,"mode":"none"}
    margin_cal = _fit_linear(hist.get("margin", pd.Series(dtype=float)).to_numpy(), hist.get("pred_margin_model", pd.Series(dtype=float)).to_numpy()) if not hist.empty else {"a":0.0,"b":1.0,"mode":"none"}
    # Apply
    # Detect systematic scale compression (compare against historical typical NCAA D1 totals ~135-155).
    raw_tot = preds["pred_total_model"].to_numpy()
    scale_factor = 1.0
    median_raw = float(np.nanmedian(raw_tot)) if raw_tot.size else 0.0
    # If median raw < 90 and > 20, assume totals are roughly half and scale using historical results if available, else heuristic 2.2.
    if median_raw and median_raw < 90:
        # Derive scale factor from historical residual set if present
        if not hist.empty and "total_points" in hist.columns and "pred_total_model" in hist.columns:
            # Use robust median ratio
            hist_ratio = np.nanmedian(hist["total_points"].to_numpy() / np.maximum(hist["pred_total_model"].to_numpy(), 1e-6))
            if np.isfinite(hist_ratio) and 1.5 < hist_ratio < 3.5:
                scale_factor = float(hist_ratio)
            else:
                scale_factor = 2.2
        else:
            # Fall back to median market vs model if a joined market column exists (rare here). Placeholder constant.
            scale_factor = 2.2
    preds["pred_total_calibrated"] = (total_cal["a"] + total_cal["b"] * raw_tot) * scale_factor
    preds["pred_margin_calibrated"] = margin_cal["a"] + margin_cal["b"] * preds["pred_margin_model"].to_numpy()
    # Adaptive clamp: use empirical quantiles if enough rows else fallback 80-220
    try:
        cal_vals = preds["pred_total_calibrated"].to_numpy()
        if cal_vals.size >= 25:
            low_q = float(np.nanpercentile(cal_vals, 2))
            high_q = float(np.nanpercentile(cal_vals, 98))
            # Expand slightly to avoid edge hard-cuts
            lower_bound = max(70.0, low_q - 5)
            upper_bound = min(240.0, high_q + 5)
            preds["pred_total_calibrated"] = np.clip(cal_vals, lower_bound, upper_bound)
        else:
            preds["pred_total_calibrated"] = np.clip(cal_vals, 80, 220)
    except Exception:
        preds["pred_total_calibrated"] = preds["pred_total_calibrated"].clip(lower=80, upper=220)
    # Persist
    out_path = out_path or (OUT / f"predictions_model_calibrated_{date_str}.csv")
    preds.to_csv(out_path, index=False)
    # Meta JSON
    meta = {
        "date": date_str,
        "window_days": window_days,
    "total_calibration": {**total_cal, "scale_factor": scale_factor, "median_raw": median_raw},
        "margin_calibration": margin_cal,
        "rows_used": int(len(hist)),
        "pred_rows": int(len(preds)),
        "timestamp_utc": dt.datetime.utcnow().isoformat()
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote calibrated predictions -> {out_path}")
    return out_path


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date for slate (YYYY-MM-DD)")
    ap.add_argument("--predictions-file", required=True, help="Path to raw predictions_model_<date>.csv")
    ap.add_argument("--results-dir", default=str(OUT / "daily_results"))
    ap.add_argument("--window-days", type=int, default=14, help="Lookback window for calibration residuals")
    ap.add_argument("--out", default=None, help="Optional output path for calibrated CSV")
    args = ap.parse_args()
    pred_path = pathlib.Path(args.predictions_file)
    results_dir = pathlib.Path(args.results_dir)
    out_path = pathlib.Path(args.out) if args.out else None
    calibrate(pred_path, args.date, args.window_days, results_dir, out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
