"""
Fit a simple linear fallback calibration for totals.

Uses recent daily_results and corresponding predictions to derive a slope/intercept
mapping from raw model `pred_total` (basis starts with 'model' but not calibrated)
to actual final totals. Writes parameters to outputs/calibration_totals_fallback.json.

Run:
  python src/calibrate_totals_fallback.py --days 30
"""

from __future__ import annotations

import argparse
import json
import os
import datetime as dt
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _list_recent_results(days: int) -> list[Path]:
    dr = OUT / "daily_results"
    if not dr.exists():
        return []
    files = sorted(dr.glob("results_*.csv"))
    # Filter by last N days if possible
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    selected: list[Path] = []
    for p in reversed(files):
        try:
            ds = p.stem.split("_")[1]
            d = dt.datetime.strptime(ds, "%Y-%m-%d")
            if d >= cutoff:
                selected.append(p)
            if len(selected) >= days:
                break
        except Exception:
            continue
    return list(reversed(selected))


def _find_preds_for_date(date_str: str) -> Path | None:
    cands = [
        OUT / f"predictions_unified_enriched_{date_str}.csv",
        OUT / f"predictions_unified_{date_str}.csv",
        OUT / f"predictions_display_{date_str}.csv",
        OUT / f"predictions_enriched_{date_str}.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _actual_total_from_results(df: pd.DataFrame) -> pd.Series:
    # Best-effort actual total extraction
    for col in [
        "total_actual",
        "final_total",
        "game_total",
        "actual_total",
    ]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    # Derive from scores if present
    if "home_score" in df.columns and "away_score" in df.columns:
        return pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")
    return pd.Series([np.nan] * len(df))


def collect_pairs(days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rp in _list_recent_results(days):
        try:
            ds = rp.stem.split("_")[1]
        except Exception:
            continue
        preds_p = _find_preds_for_date(ds)
        if not preds_p:
            continue
        res = _safe_read_csv(rp)
        preds = _safe_read_csv(preds_p)
        if res.empty or preds.empty:
            continue
        # Normalize keys
        res["game_id"] = res.get("game_id", pd.Series()).astype(str)
        preds["game_id"] = preds.get("game_id", pd.Series()).astype(str)
        # Select relevant prediction columns (prefer model-specific fields when available)
        keep_cols = [
            "game_id",
            "date",
            "pred_total",
            "pred_total_basis",
            "pred_total_model",
            "pred_total_model_basis",
            "pred_total_calibrated",
        ]
        preds_small = preds[[c for c in keep_cols if c in preds.columns]].copy()
        # Merge
        m = pd.merge(res, preds_small, on="game_id", how="inner", suffixes=("_res", "_pred"))
        m["date_use"] = ds
        # Actual totals
        m["actual_total"] = _actual_total_from_results(m)
        # Choose the prediction series to calibrate: prefer pred_total_model, else pred_total
        if "pred_total_model" in m.columns:
            m["_pred_for_cal"] = pd.to_numeric(m["pred_total_model"], errors="coerce")
            basis_series = m.get("pred_total_model_basis", pd.Series()).fillna("").astype(str)
        else:
            m["_pred_for_cal"] = pd.to_numeric(m.get("pred_total"), errors="coerce")
            basis_series = m.get("pred_total_basis", pd.Series()).fillna("").astype(str)
        # Filter to model-like rows needing calibration when possible; otherwise keep any non-null predictions
        model_like = basis_series.str.startswith("model") & (~basis_series.str.contains("cal", case=False))
        # If model-like yields none, fall back to any rows with predictions
        if not bool(model_like.any()):
            model_like = m["_pred_for_cal"].notna()
        m = m[model_like].copy()
        m["pred_total"] = pd.to_numeric(m.get("_pred_for_cal"), errors="coerce")
        m["actual_total"] = pd.to_numeric(m.get("actual_total"), errors="coerce")
        m = m.dropna(subset=["pred_total", "actual_total"])  # ensure usable pairs
        if not m.empty:
            for _, r in m.iterrows():
                rows.append({
                    "date": ds,
                    "game_id": r.get("game_id"),
                    "pred_total": float(r.get("pred_total")),
                    "actual_total": float(r.get("actual_total")),
                })
    return pd.DataFrame(rows)


def fit_calibration(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"status": "empty", "n": 0}
    x = pairs["pred_total"].to_numpy()
    y = pairs["actual_total"].to_numpy()
    # Linear fit y ≈ a + b*x
    try:
        b, a = np.polyfit(x, y, 1)
    except Exception:
        a, b = float(np.nan), float(np.nan)
    # Metrics
    y_hat = a + b * x
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    bias = float(np.mean(y_hat - y))
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else None
    return {
        "status": "ok",
        "n": int(len(x)),
        "slope": float(b),
        "intercept": float(a),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr_raw_pred_actual": corr,
        "date_min": str(pairs["date"].min()) if "date" in pairs.columns else None,
        "date_max": str(pairs["date"].max()) if "date" in pairs.columns else None,
    }


def main(days: int = 30, out_path: Path | None = None) -> dict[str, Any]:
    pairs = collect_pairs(days)
    calib = fit_calibration(pairs)
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "days": int(days),
        "pairs": int(len(pairs)),
        "calibration": calib,
    }
    if out_path is None:
        out_path = OUT / "calibration_totals_fallback.json"
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=int(os.getenv("CALIB_FALLBACK_DAYS", "30")))
    parser.add_argument("--out", type=str, default=str(OUT / "calibration_totals_fallback.json"))
    args = parser.parse_args()
    main(days=args.days, out_path=Path(args.out))
