from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from joblib import dump
except Exception as e:  # pragma: no cover
    dump = None  # type: ignore

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore


DATE_RE = re.compile(r"live_features_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass(frozen=True)
class LiveCorrectionTrainConfig:
    start_date: dt.date
    end_date: dt.date
    algo: str = "hgb"  # hgb|ridge
    test_frac: float = 0.2
    min_elapsed: float = 4.0
    max_elapsed: float = 36.0
    alpha: float = 1.0  # ridge
    random_state: int = 42


DEFAULT_FEATURE_COLS = [
    "elapsed_min",
    "remaining_min",
    "total_points",
    "home_score",
    "away_score",
    "pbp_poss_est",
    "pbp_poss_per_min",
    "pbp_ppp",
    "pbp_to",
    "pbp_orb",
    "pbp_drb",
    "live_line_total",
    "proj_model_total",
    "proj_blend_total",
    "exp_total_at_elapsed",
    "sim_final_total",
]


def _iter_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        return []
    out: list[dt.date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + dt.timedelta(days=1)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_live_feature_days(
    *,
    outputs_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    missing = 0
    read = 0
    for d in _iter_dates(start_date, end_date):
        p = outputs_dir / f"live_features_{d.isoformat()}.csv"
        if not p.exists() or p.stat().st_size < 10:
            missing += 1
            continue
        try:
            df = pd.read_csv(p, low_memory=True)
            if df.empty:
                missing += 1
                continue
            if "date" not in df.columns:
                df["date"] = d.isoformat()
            frames.append(df)
            read += 1
        except Exception:
            missing += 1
            continue

    if not frames:
        return pd.DataFrame(), {"days_read": 0, "days_missing": int(missing)}

    out = pd.concat(frames, ignore_index=True)
    return out, {"days_read": int(read), "days_missing": int(missing), "rows": int(len(out))}


def train_live_correction(
    *,
    cfg: LiveCorrectionTrainConfig,
    outputs_dir: Path,
    out_dir: Path,
    feature_cols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Train a simple live totals correction model.

    Target: `actual_total`.
    Features: snapshot + pbp proxies + live total line (when available).
    """
    if dump is None:
        raise RuntimeError("joblib is required to save live correction models")

    df, meta_load = load_live_feature_days(outputs_dir=outputs_dir, start_date=cfg.start_date, end_date=cfg.end_date)
    if df.empty:
        return {"status": "empty", "message": "No live_features files found", "load": meta_load}

    df = df.copy()
    df["date"] = df.get("date").astype(str)

    # Basic filters
    df["elapsed_min"] = pd.to_numeric(df.get("elapsed_min"), errors="coerce")
    df["actual_total"] = pd.to_numeric(df.get("actual_total"), errors="coerce")
    df = df.dropna(subset=["elapsed_min", "actual_total"])
    df = df[(df["elapsed_min"] >= float(cfg.min_elapsed)) & (df["elapsed_min"] <= float(cfg.max_elapsed))].copy()
    if df.empty:
        return {"status": "empty", "message": "No rows after filters", "load": meta_load}

    cols = feature_cols[:] if feature_cols else DEFAULT_FEATURE_COLS[:]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {"status": "empty", "message": "No feature columns found in data", "available": list(df.columns)}

    df = _coerce_numeric(df, cols)
    X = df[cols].copy()
    y = pd.to_numeric(df["actual_total"], errors="coerce")
    dates = df["date"].astype(str)

    uniq_dates = sorted(dates.dropna().unique().tolist())
    if len(uniq_dates) < 5:
        # Still train, but warn about tiny range
        pass
    cut = int(max(1, np.floor((1.0 - float(cfg.test_frac)) * float(len(uniq_dates)))))
    cut = min(max(cut, 1), max(len(uniq_dates) - 1, 1))
    train_dates = set(uniq_dates[:cut])
    test_dates = set(uniq_dates[cut:])

    train_mask = dates.isin(train_dates)
    test_mask = dates.isin(test_dates)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    algo = str(cfg.algo or "").strip().lower()
    if algo not in {"ridge", "hgb"}:
        raise ValueError("algo must be ridge or hgb")

    if algo == "ridge":
        model = Ridge(alpha=float(cfg.alpha), random_state=int(cfg.random_state))
        pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
    else:
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("HistGradientBoostingRegressor unavailable in this sklearn")
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=6,
            max_iter=500,
            learning_rate=0.05,
            random_state=int(cfg.random_state),
        )
        pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("model", model)])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))

    # Baselines (if present)
    b = {}
    for base_col in ["live_line_total", "proj_model_total", "proj_blend_total"]:
        if base_col in df.columns:
            base = pd.to_numeric(df.loc[test_mask, base_col], errors="coerce")
            m = base.notna() & y_test.notna()
            if int(m.sum()) > 0:
                b[base_col + "_mae"] = float(mean_absolute_error(y_test.loc[m], base.loc[m]))
            else:
                b[base_col + "_mae"] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"live_correction_{algo}_{stamp}.joblib"
    meta_path = out_dir / f"live_correction_{algo}_{stamp}.json"

    artifact = {
        "status": "ok",
        "algo": algo,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "test_frac": float(cfg.test_frac),
        "min_elapsed": float(cfg.min_elapsed),
        "max_elapsed": float(cfg.max_elapsed),
        "feature_cols": cols,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_dates": sorted(list(train_dates))[:10],
        "test_dates": sorted(list(test_dates))[:10],
        "mae": mae,
        "baselines": b,
        "load": meta_load,
        "model_path": str(model_path),
    }

    dump({"cfg": artifact, "pipeline": pipe}, model_path)
    meta_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact
