from __future__ import annotations
import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DAILY_DIR = OUT / "daily_results"

# ---------- Helpers ----------

def _parse_date(s: str | None) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _list_prediction_files_for_dates(dates: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for d in dates:
        for pat in [
            OUT / f"predictions_unified_enriched_{d}.csv",
            OUT / f"predictions_unified_{d}.csv",
            OUT / f"predictions_enriched_{d}.csv",
            OUT / f"predictions_display_{d}.csv",
        ]:
            if pat.exists():
                files.append(pat)
                break
    return files


def _list_result_files_for_dates(dates: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for d in dates:
        p = DAILY_DIR / f"results_{d}.csv"
        if p.exists():
            files.append(p)
    return files


def _resolve_dates(start: Optional[str], end: Optional[str], recent: Optional[int]) -> list[str]:
    if recent:
        # Recent by available predictions files (filter for YYYY-MM-DD)
        cand_preds = sorted([p for p in OUT.glob("predictions_unified_enriched_*.csv")])
        if not cand_preds:
            cand_preds = sorted([p for p in OUT.glob("predictions_unified_*.csv")])
        tokens = [p.stem.split("_")[-1] for p in cand_preds]
        dates = [t for t in tokens if len(t) == 10 and t[4] == "-" and t[7] == "-"]
        return dates[-recent:]
    if start and end:
        dt_start = dt.datetime.strptime(start, "%Y-%m-%d")
        dt_end = dt.datetime.strptime(end, "%Y-%m-%d")
        step = (dt_end - dt_start).days
        if step < 0:
            dt_start, dt_end = dt_end, dt_start
            step = -step
        return [(dt_start + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(step + 1)]
    if start and not end:
        return [start]
    # default: try today
    today = dt.datetime.now().strftime("%Y-%m-%d")
    return [today]


def _coerce_actual_total(df: pd.DataFrame) -> pd.Series:
    # Accept several column name variants; fallback to sum of scores
    # Prefer results-side totals first when present
    for cand in ("actual_total_res", "final_total_res", "actual_total", "final_total", "total_final"):
        if cand in df.columns:
            try:
                return pd.to_numeric(df[cand], errors="coerce")
            except Exception:
                pass
    # Try score columns (prefer results-side columns when merged)
    home_candidates = [
        c for c in df.columns if str(c).lower() in ("home_score_res", "score_home_res", "home_points_res")
    ] or [
        c for c in df.columns if str(c).lower() in ("score_home", "home_score", "home_points")
    ]
    away_candidates = [
        c for c in df.columns if str(c).lower() in ("away_score_res", "score_away_res", "away_points_res")
    ] or [
        c for c in df.columns if str(c).lower() in ("score_away", "away_score", "away_points")
    ]
    if home_candidates and away_candidates:
        try:
            sh = pd.to_numeric(df[home_candidates[0]], errors="coerce")
            sa = pd.to_numeric(df[away_candidates[0]], errors="coerce")
            return sh + sa
        except Exception:
            pass
    return pd.Series([np.nan] * len(df))


def _metrics(errors: np.ndarray, preds: np.ndarray, actuals: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {"count": 0}
    mae = float(np.nanmean(np.abs(errors)))
    rmse = float(np.sqrt(np.nanmean(np.square(errors))))
    bias = float(np.nanmean(errors))
    # correlation safe-check
    try:
        corr = float(np.corrcoef(preds, actuals)[0, 1]) if np.isfinite(preds).all() and np.isfinite(actuals).all() else None
    except Exception:
        corr = None
    return {
        "count": int(errors.size),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "bias": round(bias, 3),
        "corr": corr,
    }


def _pinball_loss(actual: float, q_pred: float, q: float) -> float:
    if math.isnan(actual) or math.isnan(q_pred):
        return np.nan
    e = actual - q_pred
    return max(q * e, (q - 1) * e)


@dataclasses.dataclass
class BacktestConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    recent: Optional[int] = None
    out_prefix: Optional[str] = None


def run_backtest(cfg: BacktestConfig) -> dict[str, Any]:
    dates = _resolve_dates(cfg.start, cfg.end, cfg.recent)
    pred_files = _list_prediction_files_for_dates(dates)
    res_files = _list_result_files_for_dates(dates)
    if not pred_files:
        return {"error": "No prediction files found for dates", "dates": dates}
    # Load & concat predictions
    preds_list = []
    for pf in pred_files:
        df = _safe_read_csv(pf)
        if not df.empty:
            df["_src"] = str(pf)
            preds_list.append(df)
    preds = pd.concat(preds_list, ignore_index=True) if preds_list else pd.DataFrame()
    if preds.empty:
        return {"error": "Predictions empty after load", "dates": dates}
    # Load & concat results
    res_list = []
    for rf in res_files:
        df = _safe_read_csv(rf)
        if not df.empty:
            df["_src"] = str(rf)
            res_list.append(df)
    results = pd.concat(res_list, ignore_index=True) if res_list else pd.DataFrame()
    if results.empty:
        return {"error": "Results empty; run finalize-day first for these dates", "dates": dates}
    # Normalize keys
    for d in (preds, results):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
    # Join on game_id+date when available
    join_keys = [k for k in ("game_id", "date") if k in preds.columns and k in results.columns]
    if not join_keys:
        join_keys = [k for k in ("game_id",) if k in preds.columns and k in results.columns]
    df = preds.merge(results, on=join_keys, how="inner", suffixes=("", "_res"))
    if df.empty:
        return {"error": "Join produced no rows; key mismatch", "dates": dates}
    # Actual total
    actual_total = _coerce_actual_total(df)
    df["actual_total"] = actual_total
    # Predictions: prefer model totals explicitly when present, else fallback
    # Support multiple canonical column names produced by integration
    model_candidates = [
        "pred_total_model",
        "pred_total_model_unified",
        "pred_total_model_raw",
        "pred_total_model_x",
        "pred_total_model_y",
    ]
    # Build per-row evaluation from available columns: prefer model where present
    model_series = None
    model_source = None
    for mc in model_candidates:
        if mc in df.columns:
            s = pd.to_numeric(df[mc], errors="coerce")
            if s.notna().any():
                model_series = s
                model_source = mc
                break
    cal_series = pd.to_numeric(df["pred_total_calibrated"], errors="coerce") if "pred_total_calibrated" in df.columns else None
    base_series = pd.to_numeric(df["pred_total"], errors="coerce") if "pred_total" in df.columns else None
    # Compose final eval series
    if model_series is not None:
        pred_total = model_series.copy()
        # fill missing with calibrated, then base
        if cal_series is not None:
            pred_total = pred_total.where(pred_total.notna(), cal_series)
        if base_series is not None:
            pred_total = pred_total.where(pred_total.notna(), base_series)
        eval_source = model_source
    else:
        # No model series: use calibrated then base
        if cal_series is not None:
            pred_total = cal_series.copy()
            eval_source = "pred_total_calibrated"
        elif base_series is not None:
            pred_total = base_series.copy()
            eval_source = "pred_total"
        else:
            pred_total = pd.to_numeric(df.get("pred_total"), errors="coerce")
            eval_source = ("pred_total" if "pred_total" in df.columns else None)
    df["pred_total_eval"] = pred_total
    # Per-row eval basis mask
    basis = pd.Series([None] * len(df))
    if model_series is not None:
        basis = np.where(model_series.notna(), "model", basis)
    if cal_series is not None:
        basis = np.where((pd.isna(basis)) & (cal_series.notna()), "cal", basis)
    if base_series is not None:
        basis = np.where(pd.isna(basis) & base_series.notna(), df.get("pred_total_basis", "raw"), basis)
    df["pred_total_eval_basis"] = basis
    # Source column used for evaluation (note: if backfilled from cal/base, this will still show model_source; use more detailed source if needed)
    df["pred_total_eval_source"] = eval_source
    # Market/closing for divergence checks
    df["closing_total_eval"] = pd.to_numeric(df.get("closing_total"), errors="coerce")
    df["market_total_eval"] = pd.to_numeric(df.get("market_total"), errors="coerce")
    # Errors
    df["error_total"] = df["pred_total_eval"] - df["actual_total"]
    df["error_market"] = df["market_total_eval"] - df["actual_total"]
    df["error_closing"] = df["closing_total_eval"] - df["actual_total"]
    # Quantile metrics (pinball loss) if available
    for q, col in [(0.1, "pred_total_q10"), (0.5, "pred_total_q50"), (0.9, "pred_total_q90")]:
        if col in df.columns:
            df[f"pinball_{int(q*100)}"] = [
                _pinball_loss(at, qp, q) for at, qp in zip(df["actual_total"], pd.to_numeric(df[col], errors="coerce"))
            ]
    # Summary metrics
    errors = df["error_total"].to_numpy(dtype=float)
    metrics_overall = _metrics(errors, df["pred_total_eval"].to_numpy(dtype=float), df["actual_total"].to_numpy(dtype=float))
    # Group metrics by basis
    by_basis: dict[str, Any] = {}
    basis_col = "pred_total_eval_basis" if "pred_total_eval_basis" in df.columns else ("pred_total_basis" if "pred_total_basis" in df.columns else None)
    if basis_col:
        for basis, g in df.groupby(df[basis_col].fillna("")):
            err_g = g["error_total"].to_numpy(dtype=float)
            by_basis[str(basis)] = _metrics(err_g, g["pred_total_eval"].to_numpy(dtype=float), g["actual_total"].to_numpy(dtype=float))
    # Divergence vs market
    non_market = int(((df["pred_total_eval"].round(1) != df["market_total_eval"].round(1)) & df["pred_total_eval"].notna()).sum()) if "market_total_eval" in df.columns else None
    # Pinball summaries
    pinball_summary: dict[str, Any] = {}
    for qlab in ("10", "50", "90"):
        col = f"pinball_{qlab}"
        if col in df.columns:
            pinball_summary[col] = float(pd.Series(df[col]).mean(skipna=True))
    # Write outputs
    start = dates[0]
    end = dates[-1]
    prefix = cfg.out_prefix or f"backtest_totals_{start}_{end}"
    row_out = OUT / f"{prefix}.csv"
    sum_out = OUT / f"{prefix}_summary.json"
    # Keep a compact set of columns for row-level output
    keep_cols = [
        c for c in [
            "date",
            "game_id",
            "home_team",
            "away_team",
            "pred_total_eval",
            "pred_total_eval_basis",
            "pred_total_eval_source",
            "pred_total_basis",
            "actual_total",
            "market_total_eval",
            "closing_total_eval",
            "error_total",
            "error_market",
            "error_closing",
            "quotes_count",
            "seg_n_rows",
            "blend_weight",
            "conference",
        ]
        if c in df.columns
    ]
    try:
        df[keep_cols].to_csv(row_out, index=False)
    except Exception:
        # Fallback: write everything
        df.to_csv(row_out, index=False)
    summary_payload = {
        "dates": dates,
        "rows": int(len(df)),
        "overall": metrics_overall,
        "by_basis": by_basis,
        "non_market_total_rows": non_market,
        "pinball": pinball_summary,
        "sources": {
            "predictions": [str(p) for p in pred_files],
            "results": [str(r) for r in res_files],
        },
    }
    sum_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


def main():
    ap = argparse.ArgumentParser(description="Backtest totals predictions vs actuals")
    ap.add_argument("--start", type=str, help="Start date YYYY-MM-DD", default=None)
    ap.add_argument("--end", type=str, help="End date YYYY-MM-DD", default=None)
    ap.add_argument("--recent", type=int, help="Use N most recent dates by predictions files", default=None)
    ap.add_argument("--out-prefix", type=str, help="Output file prefix (without extension)", default=None)
    args = ap.parse_args()
    cfg = BacktestConfig(start=_parse_date(args.start), end=_parse_date(args.end), recent=args.recent, out_prefix=args.out_prefix)
    payload = run_backtest(cfg)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
