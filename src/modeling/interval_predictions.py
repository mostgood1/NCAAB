"""Generate simple prediction intervals for calibrated (or raw) model outputs.

Strategy:
  1. Collect historical residuals from recent finalized results joined to historical
     prediction artifacts (prefers calibrated predictions if present).
  2. Compute RMSE (root mean squared error) per target (total, margin). If insufficient
     rows (<15) fall back to a conservative default stdev guess.
  3. Produce symmetric normal-based intervals at selected confidence levels using
     z multipliers: 75% (~1.15), 90% (~1.645), 95% (~1.96). We keep 75 and 90 to avoid
     overprecision early season; 95 added for reference/meta only.
  4. Clamp total interval bounds to basketball plausible range (60–240) and margin bounds
     to (-60, 60) to avoid pathological extremes when RMSE is large or defaulted.

Outputs:
  predictions_model_interval_<date>.csv with columns:
    game_id, pred_total_model, pred_margin_model, pred_total_calibrated?, pred_margin_calibrated?,
    pred_total_ci75_low, pred_total_ci75_high, pred_total_ci90_low, pred_total_ci90_high,
    pred_margin_ci75_low, pred_margin_ci75_high, pred_margin_ci90_low, pred_margin_ci90_high
  (Calibrated columns included if source file had them.)

Meta JSON (same stem .json) includes RMSE stats and row counts.

Usage:
  python -m src.modeling.interval_predictions --date YYYY-MM-DD \
      --predictions-file outputs/predictions_model_YYYY-MM-DD.csv \
      [--calibrated-file outputs/predictions_model_calibrated_YYYY-MM-DD.csv] \
      [--results-dir outputs/daily_results] [--window-days 30]

Notes / Assumptions:
  - Residual join key is game_id; predictions must contain game_id.
  - If calibrated file provided and exists, intervals use calibrated predictions as point estimate;
    otherwise fall back to raw model outputs.
  - Symmetric intervals assume approximate normality of residuals; early season this may be crude.
    Future improvement: empirical quantiles or quantile regression.
"""
from __future__ import annotations
import argparse, pathlib, datetime as dt, json
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

Z = {
    "75": 1.150349,  # two-sided central 75% interval (~12.5% tails)
    "90": 1.644854,  # two-sided central 90% interval
    "95": 1.959964,  # not output as columns, meta only
}

def _collect_residuals(results_dir: pathlib.Path, cutoff_date: dt.date, window_days: int) -> pd.DataFrame:
    frames = []
    for p in results_dir.glob("results_*.csv"):
        try:
            d = dt.datetime.strptime(p.stem.replace("results_", ""), "%Y-%m-%d").date()
        except Exception:
            continue
        delta = (cutoff_date - d).days
        if delta < 0 or delta > window_days:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if {"game_id","total_points","margin"}.issubset(df.columns):
            frames.append(df[["game_id","total_points","margin"]].copy())
    if not frames:
        return pd.DataFrame(columns=["game_id","total_points","margin"])
    return pd.concat(frames, ignore_index=True)

def _rmse(residuals: np.ndarray) -> float:
    # Coerce to float array to avoid object/string issues from mixed dtypes
    try:
        residuals = residuals.astype(float)
    except Exception:
        residuals = np.array([float(x) if isinstance(x,(int,float)) else np.nan for x in residuals])
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(residuals**2)))

def build_intervals(date_str: str, pred_path: pathlib.Path, calibrated_path: pathlib.Path | None,
                    results_dir: pathlib.Path, window_days: int, out_path: pathlib.Path | None) -> pathlib.Path:
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    base_preds = pd.read_csv(pred_path)
    if "game_id" not in base_preds.columns:
        raise ValueError("predictions file missing game_id column")
    # Use calibrated file if provided and exists
    use_calibrated = False
    preds = base_preds.copy()
    if calibrated_path and calibrated_path.exists():
        cal = pd.read_csv(calibrated_path)
        # Ensure same game_id set
        if set(cal.game_id) == set(preds.game_id):
            # Merge calibrated columns in case not already present
            for col in ["pred_total_calibrated","pred_margin_calibrated"]:
                if col in cal.columns:
                    preds[col] = cal[col]
            use_calibrated = True
    point_total_col = "pred_total_calibrated" if use_calibrated and "pred_total_calibrated" in preds.columns else "pred_total_model"
    point_margin_col = "pred_margin_calibrated" if use_calibrated and "pred_margin_calibrated" in preds.columns else "pred_margin_model"

    cutoff_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    hist = _collect_residuals(results_dir, cutoff_date, window_days)
    # Join residuals with historical predictions (prefer calibrated historical if available)
    residual_rows = []
    if not hist.empty:
        # Attempt to locate historical prediction artifacts for each date inside window
        for gid, total_true, margin_true in hist.itertuples(index=False):
            # Heuristic: search both calibrated and raw predictions containing gid across window
            found = False
            # Raw
            for pred_art in OUT.glob("predictions_model_*.csv"):
                try:
                    dfp = pd.read_csv(pred_art, usecols=["game_id", "pred_total_model", "pred_margin_model"], dtype={"game_id": str})
                except Exception:
                    continue
                if str(gid) in dfp.game_id.values:
                    row = dfp[dfp.game_id == str(gid)].iloc[0]
                    pt = float(row.get("pred_total_model", np.nan))
                    pm = float(row.get("pred_margin_model", np.nan))
                    found = True
                    break
            # Calibrated (override raw if exists)
            for pred_art in OUT.glob("predictions_model_calibrated_*.csv"):
                try:
                    dfc = pd.read_csv(pred_art, usecols=["game_id", "pred_total_calibrated", "pred_margin_calibrated"], dtype={"game_id": str})
                except Exception:
                    continue
                if str(gid) in dfc.game_id.values:
                    crow = dfc[dfc.game_id == str(gid)].iloc[0]
                    pt = float(crow.get("pred_total_calibrated", np.nan))
                    pm = float(crow.get("pred_margin_calibrated", np.nan))
                    found = True
                    break
            if found:
                residual_rows.append((gid, total_true - pt, margin_true - pm))
    residual_df = pd.DataFrame(residual_rows, columns=["game_id","resid_total","resid_margin"]) if residual_rows else pd.DataFrame(columns=["game_id","resid_total","resid_margin"])
    rmse_total = _rmse(residual_df.get("resid_total", pd.Series(dtype=float)).to_numpy())
    rmse_margin = _rmse(residual_df.get("resid_margin", pd.Series(dtype=float)).to_numpy())
    rows_used = int(len(residual_df))

    # Fallback stdev guesses if insufficient residual rows
    if rows_used < 15 or not np.isfinite(rmse_total):
        rmse_total = 12.0  # typical pace-driven total residual early season
    if rows_used < 15 or not np.isfinite(rmse_margin):
        rmse_margin = 10.0

    # Build intervals
    for level, z in Z.items():
        if level == "95":
            continue  # skip 95% columns (meta only)
        total_low = preds[point_total_col] - z * rmse_total
        total_high = preds[point_total_col] + z * rmse_total
        margin_low = preds[point_margin_col] - z * rmse_margin
        margin_high = preds[point_margin_col] + z * rmse_margin
        # Clamp
        total_low = total_low.clip(lower=60, upper=240)
        total_high = total_high.clip(lower=60, upper=240)
        margin_low = margin_low.clip(lower=-60, upper=60)
        margin_high = margin_high.clip(lower=-60, upper=60)
        preds[f"pred_total_ci{level}_low"] = total_low
        preds[f"pred_total_ci{level}_high"] = total_high
        preds[f"pred_margin_ci{level}_low"] = margin_low
        preds[f"pred_margin_ci{level}_high"] = margin_high

    out_path = out_path or (OUT / f"predictions_model_interval_{date_str}.csv")
    preds.to_csv(out_path, index=False)
    meta = {
        "date": date_str,
        "window_days": window_days,
        "rows_used": rows_used,
        "rmse_total": rmse_total,
        "rmse_margin": rmse_margin,
        "point_total_column": point_total_col,
        "point_margin_column": point_margin_col,
        "z_values": Z,
        "timestamp_utc": dt.datetime.utcnow().isoformat()
    }
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))
    print(f"Wrote interval predictions -> {out_path}")
    return out_path

def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    ap.add_argument("--predictions-file", required=True, help="Path to raw predictions_model_<date>.csv")
    ap.add_argument("--calibrated-file", default=None, help="Optional calibrated predictions file path")
    ap.add_argument("--results-dir", default=str(OUT / "daily_results"))
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--out", default=None, help="Optional output path override for interval predictions")
    args = ap.parse_args()
    pred_path = pathlib.Path(args.predictions_file)
    cal_path = pathlib.Path(args.calibrated_file) if args.calibrated_file else None
    results_dir = pathlib.Path(args.results_dir)
    out_path = pathlib.Path(args.out) if args.out else None
    build_intervals(args.date, pred_path, cal_path, results_dir, args.window_days, out_path)

if __name__ == "__main__":  # pragma: no cover
    main()
