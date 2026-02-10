from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from joblib import load
except Exception:  # pragma: no cover
    load = None  # type: ignore

from sklearn.metrics import mean_absolute_error

from ..train.live_correction import DEFAULT_FEATURE_COLS, LiveCorrectionTrainConfig, load_live_feature_days


@dataclass(frozen=True)
class BiasBucket:
    lo: float
    hi: float
    n: int
    bias: float
    resid_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": float(self.lo),
            "hi": float(self.hi),
            "n": int(self.n),
            "bias": float(self.bias),
            "resid_std": float(self.resid_std),
        }


@dataclass(frozen=True)
class BiasCalibrator:
    kind: str  # global_bias_v1 | elapsed_bucket_bias_v1
    global_bias: float
    buckets: list[BiasBucket]
    bucket_step: float
    min_elapsed: float
    max_elapsed: float
    shrink_k: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "global_bias": float(self.global_bias),
            "bucket_step": float(self.bucket_step),
            "min_elapsed": float(self.min_elapsed),
            "max_elapsed": float(self.max_elapsed),
            "shrink_k": float(self.shrink_k),
            "buckets": [b.to_dict() for b in self.buckets],
        }


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s).strip())


def _chronological_date_split(dates: pd.Series, test_frac: float) -> tuple[set[str], set[str]]:
    uniq_dates = sorted(pd.Series(dates).astype(str).dropna().unique().tolist())
    if len(uniq_dates) <= 1:
        return (set(uniq_dates), set())
    cut = int(max(1, np.floor((1.0 - float(test_frac)) * float(len(uniq_dates)))))
    cut = min(max(cut, 1), max(len(uniq_dates) - 1, 1))
    train_dates = set(uniq_dates[:cut])
    test_dates = set(uniq_dates[cut:])
    return (train_dates, test_dates)


def fit_elapsed_bias_calibrator(
    *,
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    elapsed_col: str = "elapsed_min",
    bucket_min: float = 4.0,
    bucket_max: float = 36.0,
    bucket_step: float = 2.0,
    min_bucket_n: int = 50,
    shrink_k: float = 0.0,
) -> list[BiasBucket]:
    """Fit an additive bias correction: calibrated_pred = pred - bias(bucket)."""
    if df.empty:
        return []
    e = pd.to_numeric(df.get(elapsed_col), errors="coerce")
    y = pd.to_numeric(df.get(actual_col), errors="coerce")
    p = pd.to_numeric(df.get(pred_col), errors="coerce")
    m = e.notna() & y.notna() & p.notna()
    if int(m.sum()) == 0:
        return []
    tmp = pd.DataFrame({"elapsed": e[m], "y": y[m], "p": p[m]}).copy()
    tmp = tmp[(tmp["elapsed"] >= float(bucket_min)) & (tmp["elapsed"] <= float(bucket_max))].copy()
    if tmp.empty:
        return []

    # Global bias across all included rows.
    resid_all = pd.to_numeric(tmp["p"], errors="coerce") - pd.to_numeric(tmp["y"], errors="coerce")
    global_bias = float(resid_all.mean())

    buckets: list[BiasBucket] = []
    lo = float(bucket_min)
    while lo < float(bucket_max) - 1e-9:
        hi = min(float(bucket_max), lo + float(bucket_step))
        sel = tmp[(tmp["elapsed"] >= lo) & (tmp["elapsed"] < hi)].copy()
        n = int(len(sel))
        if n >= int(min_bucket_n):
            resid = pd.to_numeric(sel["p"], errors="coerce") - pd.to_numeric(sel["y"], errors="coerce")
            raw_bias = float(resid.mean())
            k = float(max(0.0, shrink_k))
            # Shrink bucket bias toward global bias to reduce overfit.
            w = float(n) / float(n + k) if (k > 0) else 1.0
            bias = w * raw_bias + (1.0 - w) * global_bias
            resid_std = float(resid.std(ddof=0)) if n > 1 else 0.0
            buckets.append(BiasBucket(lo=lo, hi=hi, n=n, bias=bias, resid_std=resid_std))
        lo = hi
    return buckets


def apply_elapsed_bias_calibrator(
    *,
    df: pd.DataFrame,
    pred_col: str,
    out_col: str,
    buckets: list[BiasBucket],
    elapsed_col: str = "elapsed_min",
) -> pd.DataFrame:
    if df.empty:
        df[out_col] = np.nan
        return df
    e = pd.to_numeric(df.get(elapsed_col), errors="coerce")
    p = pd.to_numeric(df.get(pred_col), errors="coerce")
    out = p.copy()
    out[:] = np.nan

    if not buckets:
        df[out_col] = p
        return df

    # Default: pass through if elapsed missing
    out.loc[p.notna()] = p.loc[p.notna()]

    for b in buckets:
        mask = e.notna() & p.notna() & (e >= float(b.lo)) & (e < float(b.hi))
        if int(mask.sum()) > 0:
            out.loc[mask] = p.loc[mask] - float(b.bias)

    df[out_col] = out
    return df


def fit_global_bias_calibrator(
    *,
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
) -> float:
    if df.empty:
        return 0.0
    y = pd.to_numeric(df.get(actual_col), errors="coerce")
    p = pd.to_numeric(df.get(pred_col), errors="coerce")
    m = y.notna() & p.notna()
    if int(m.sum()) == 0:
        return 0.0
    resid = p.loc[m] - y.loc[m]
    return float(resid.mean())


def apply_global_bias_calibrator(
    *,
    df: pd.DataFrame,
    pred_col: str,
    out_col: str,
    global_bias: float,
) -> pd.DataFrame:
    p = pd.to_numeric(df.get(pred_col), errors="coerce")
    df[out_col] = p - float(global_bias)
    return df


def _eval_block(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, Any]:
    if df.empty:
        return {"n": 0, "mae": None, "bias": None}
    y = pd.to_numeric(df.get(actual_col), errors="coerce")
    p = pd.to_numeric(df.get(pred_col), errors="coerce")
    m = y.notna() & p.notna()
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "mae": None, "bias": None}
    resid = p.loc[m] - y.loc[m]
    return {
        "n": n,
        "mae": float(mean_absolute_error(y.loc[m], p.loc[m])),
        "bias": float(resid.mean()),
        "resid_std": float(resid.std(ddof=0)) if n > 1 else 0.0,
    }


def _group_eval_blocks(
    *,
    df: pd.DataFrame,
    group_col: str,
    pred_cols: list[str],
    actual_col: str,
) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_col, dropna=False):
        for pc in pred_cols:
            if pc not in g.columns:
                continue
            m = _eval_block(g, pc, actual_col)
            rows.append({group_col: key, "pred_col": pc, **m})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values([group_col, "pred_col"], kind="stable").reset_index(drop=True)


def _add_elapsed_bucket_col(
    *,
    df: pd.DataFrame,
    elapsed_col: str,
    min_elapsed: float,
    max_elapsed: float,
    bucket_step: float,
    out_col: str = "elapsed_bucket",
) -> pd.DataFrame:
    if df.empty:
        df[out_col] = pd.Series(dtype="object")
        return df
    e = pd.to_numeric(df.get(elapsed_col), errors="coerce")
    step = float(bucket_step)
    if step <= 0:
        df[out_col] = pd.Series([None] * len(df))
        return df
    bins = np.arange(float(min_elapsed), float(max_elapsed) + step + 1e-9, step)
    cats = pd.cut(e, bins=bins, right=False, include_lowest=True)
    # Convert intervals to stable strings like "[4.0, 6.0)".
    df[out_col] = cats.astype(str)
    return df


def eval_live_correction(
    *,
    model_path: Path,
    cfg: LiveCorrectionTrainConfig,
    outputs_dir: Path,
    out_dir: Path,
    feature_cols: Optional[list[str]] = None,
    bucket_step: float = 2.0,
    min_bucket_n: int = 50,
    calibration_mode: str = "bucket_shrink",
    shrink_k: float = 200.0,
    calibration_fit_last_days: int | None = None,
    rolling_calibration: bool = False,
    write_scored_csv: bool = True,
    write_diagnostics_csv: bool = True,
) -> dict[str, Any]:
    if load is None:
        raise RuntimeError("joblib is required to load live correction models")
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    model_obj = load(model_path)
    if not isinstance(model_obj, dict) or "pipeline" not in model_obj:
        raise ValueError("Invalid model artifact (expected dict with 'pipeline')")
    pipe = model_obj["pipeline"]

    df, meta_load = load_live_feature_days(outputs_dir=outputs_dir, start_date=cfg.start_date, end_date=cfg.end_date)
    if df.empty:
        return {"status": "empty", "message": "No live_features found", "load": meta_load}

    df = df.copy()
    df["date"] = df.get("date").astype(str)
    df["elapsed_min"] = pd.to_numeric(df.get("elapsed_min"), errors="coerce")
    df["actual_total"] = pd.to_numeric(df.get("actual_total"), errors="coerce")
    df = df.dropna(subset=["elapsed_min", "actual_total"]).copy()
    df = df[(df["elapsed_min"] >= float(cfg.min_elapsed)) & (df["elapsed_min"] <= float(cfg.max_elapsed))].copy()
    if df.empty:
        return {"status": "empty", "message": "No rows after filters", "load": meta_load}

    cols = feature_cols[:] if feature_cols else DEFAULT_FEATURE_COLS[:]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {"status": "empty", "message": "No feature columns present"}
    for c in cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    train_dates, test_dates = _chronological_date_split(df["date"], cfg.test_frac)
    df["split"] = np.where(df["date"].isin(sorted(list(test_dates))), "test", "train")

    # Predict
    X = df[cols].copy()
    try:
        df["pred_total_livecor_raw"] = pipe.predict(X)
    except Exception as e:
        raise RuntimeError(f"predict failed: {e}")

    # Fit calibrator on train only (optionally only the last N train days to reduce drift)
    train_df = df[df["split"] == "train"].copy()
    fit_df = train_df
    if calibration_fit_last_days is not None:
        n_days = int(calibration_fit_last_days)
        if n_days > 0:
            uniq = sorted(set(str(x) for x in fit_df["date"].dropna().astype(str).tolist()))
            if uniq:
                keep = set(uniq[-n_days:])
                fit_df = fit_df[fit_df["date"].isin(keep)].copy()
    mode = str(calibration_mode or "").strip().lower()
    if mode not in {"none", "global", "bucket", "bucket_shrink"}:
        raise ValueError("calibration_mode must be one of none|global|bucket|bucket_shrink")

    buckets: list[BiasBucket] = []
    global_bias = fit_global_bias_calibrator(df=fit_df, pred_col="pred_total_livecor_raw", actual_col="actual_total")

    if mode == "none":
        df["pred_total_livecor_cal"] = df["pred_total_livecor_raw"]
        cal = BiasCalibrator(
            kind="none",
            global_bias=float(global_bias),
            buckets=[],
            bucket_step=float(bucket_step),
            min_elapsed=float(cfg.min_elapsed),
            max_elapsed=float(cfg.max_elapsed),
            shrink_k=float(shrink_k),
        )
    elif mode == "global":
        df = apply_global_bias_calibrator(df=df, pred_col="pred_total_livecor_raw", out_col="pred_total_livecor_cal", global_bias=float(global_bias))
        cal = BiasCalibrator(
            kind="global_bias_v1",
            global_bias=float(global_bias),
            buckets=[],
            bucket_step=float(bucket_step),
            min_elapsed=float(cfg.min_elapsed),
            max_elapsed=float(cfg.max_elapsed),
            shrink_k=float(shrink_k),
        )
    else:
        buckets = fit_elapsed_bias_calibrator(
            df=fit_df,
            pred_col="pred_total_livecor_raw",
            actual_col="actual_total",
            elapsed_col="elapsed_min",
            bucket_min=float(cfg.min_elapsed),
            bucket_max=float(cfg.max_elapsed),
            bucket_step=float(bucket_step),
            min_bucket_n=int(min_bucket_n),
            shrink_k=(0.0 if mode == "bucket" else float(shrink_k)),
        )
        df = apply_elapsed_bias_calibrator(
            df=df,
            pred_col="pred_total_livecor_raw",
            out_col="pred_total_livecor_cal",
            buckets=buckets,
            elapsed_col="elapsed_min",
        )
        cal = BiasCalibrator(
            kind=("elapsed_bucket_bias_v1" if mode == "bucket" else "elapsed_bucket_bias_shrink_v1"),
            global_bias=float(global_bias),
            buckets=buckets,
            bucket_step=float(bucket_step),
            min_elapsed=float(cfg.min_elapsed),
            max_elapsed=float(cfg.max_elapsed),
            shrink_k=(0.0 if mode == "bucket" else float(shrink_k)),
        )

    # Optional: rolling calibration (fit on all prior dates, apply to each test date)
    if bool(rolling_calibration) and mode != "none":
        df["pred_total_livecor_cal_rolling"] = np.nan
        test_dates_sorted = sorted(set(df.loc[df["split"] == "test", "date"].astype(str).tolist()))
        for d in test_dates_sorted:
            hist = df[df["date"].astype(str) < str(d)].copy()
            if hist.empty:
                continue
            if calibration_fit_last_days is not None:
                n_days = int(calibration_fit_last_days)
                if n_days > 0:
                    uniq = sorted(set(str(x) for x in hist["date"].dropna().astype(str).tolist()))
                    keep = set(uniq[-n_days:]) if uniq else set()
                    hist = hist[hist["date"].isin(keep)].copy()

            if mode == "global":
                gb = fit_global_bias_calibrator(df=hist, pred_col="pred_total_livecor_raw", actual_col="actual_total")
                sel = df["date"].astype(str) == str(d)
                df.loc[sel, "pred_total_livecor_cal_rolling"] = pd.to_numeric(df.loc[sel, "pred_total_livecor_raw"], errors="coerce") - float(gb)
            else:
                gb = fit_global_bias_calibrator(df=hist, pred_col="pred_total_livecor_raw", actual_col="actual_total")
                bks = fit_elapsed_bias_calibrator(
                    df=hist,
                    pred_col="pred_total_livecor_raw",
                    actual_col="actual_total",
                    elapsed_col="elapsed_min",
                    bucket_min=float(cfg.min_elapsed),
                    bucket_max=float(cfg.max_elapsed),
                    bucket_step=float(bucket_step),
                    min_bucket_n=int(min_bucket_n),
                    shrink_k=(0.0 if mode == "bucket" else float(shrink_k)),
                )
                sel_df = df[df["date"].astype(str) == str(d)].copy()
                sel_df = apply_elapsed_bias_calibrator(
                    df=sel_df,
                    pred_col="pred_total_livecor_raw",
                    out_col="_tmp_cal",
                    buckets=bks,
                    elapsed_col="elapsed_min",
                )
                df.loc[df["date"].astype(str) == str(d), "pred_total_livecor_cal_rolling"] = sel_df["_tmp_cal"].values

    # Overall + split metrics
    metrics = {
        "raw_all": _eval_block(df, "pred_total_livecor_raw", "actual_total"),
        "cal_all": _eval_block(df, "pred_total_livecor_cal", "actual_total"),
        "raw_train": _eval_block(df[df["split"] == "train"], "pred_total_livecor_raw", "actual_total"),
        "raw_test": _eval_block(df[df["split"] == "test"], "pred_total_livecor_raw", "actual_total"),
        "cal_train": _eval_block(df[df["split"] == "train"], "pred_total_livecor_cal", "actual_total"),
        "cal_test": _eval_block(df[df["split"] == "test"], "pred_total_livecor_cal", "actual_total"),
    }

    if bool(rolling_calibration) and "pred_total_livecor_cal_rolling" in df.columns:
        metrics["cal_test_rolling"] = _eval_block(df[df["split"] == "test"], "pred_total_livecor_cal_rolling", "actual_total")

    # Baselines
    base = {}
    for base_col in ["live_line_total", "proj_model_total", "proj_blend_total"]:
        if base_col in df.columns:
            base[f"{base_col}_all"] = _eval_block(df, base_col, "actual_total")
            base[f"{base_col}_test"] = _eval_block(df[df["split"] == "test"], base_col, "actual_total")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scored_path = out_dir / f"live_correction_scored_{stamp}.csv"
    eval_path = out_dir / f"live_correction_eval_{stamp}.json"
    cal_path = out_dir / f"live_correction_calibration_{stamp}.json"

    diag_by_date_path = out_dir / f"live_correction_diag_by_date_{stamp}.csv"
    diag_by_bucket_path = out_dir / f"live_correction_diag_by_elapsed_bucket_{stamp}.csv"

    cal_art = {
        **cal.to_dict(),
        "fit_split": "train",
        "fit_last_days": int(calibration_fit_last_days) if calibration_fit_last_days is not None else None,
        "rolling_calibration": bool(rolling_calibration),
        "model_path": str(model_path),
        "calibration_mode": mode,
    }

    payload = {
        "status": "ok",
        "model_path": str(model_path),
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "rows": int(len(df)),
        "feature_cols": cols,
        "load": meta_load,
        "metrics": metrics,
        "baselines": base,
        "diagnostics": {
            "by_date_csv": str(diag_by_date_path) if write_diagnostics_csv else None,
            "by_elapsed_bucket_csv": str(diag_by_bucket_path) if write_diagnostics_csv else None,
        },
        "calibration_path": str(cal_path),
        "scored_csv": str(scored_path) if write_scored_csv else None,
        "eval_json": str(eval_path),
    }

    cal_path.write_text(json.dumps(cal_art, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if write_scored_csv:
        keep = [
            c
            for c in [
                "date",
                "ts",
                "game_id",
                "elapsed_min",
                "remaining_min",
                "total_points",
                "actual_total",
                "live_line_total",
                "proj_model_total",
                "proj_blend_total",
                "pred_total_livecor_raw",
                "pred_total_livecor_cal",
                "pred_total_livecor_cal_rolling",
                "split",
            ]
            if c in df.columns
        ]
        df_out = df[keep].copy()
        df_out.to_csv(scored_path, index=False)

    if write_diagnostics_csv:
        pred_cols = ["pred_total_livecor_raw", "pred_total_livecor_cal", "pred_total_livecor_cal_rolling"]
        pred_cols = [c for c in pred_cols if c in df.columns]
        test_df = df[df["split"] == "test"].copy()

        by_date = _group_eval_blocks(df=test_df, group_col="date", pred_cols=pred_cols, actual_col="actual_total")
        if not by_date.empty:
            by_date.to_csv(diag_by_date_path, index=False)

        test_df = _add_elapsed_bucket_col(
            df=test_df,
            elapsed_col="elapsed_min",
            min_elapsed=float(cfg.min_elapsed),
            max_elapsed=float(cfg.max_elapsed),
            bucket_step=float(bucket_step),
            out_col="elapsed_bucket",
        )
        by_bucket = _group_eval_blocks(df=test_df, group_col="elapsed_bucket", pred_cols=pred_cols, actual_col="actual_total")
        if not by_bucket.empty:
            by_bucket.to_csv(diag_by_bucket_path, index=False)

    return payload
