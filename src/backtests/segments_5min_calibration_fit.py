from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitSegments5MinCalibrationConfig:
    backtest_csv: Path
    out_path: Path
    start: Optional[str] = None
    end: Optional[str] = None
    pred_col: str = "pred_q50"
    actual_col: str = "actual_total"
    end_min_col: str = "end_min"
    date_col: str = "date"
    # Guardrails against overfit / pathological fits
    min_a: float = 0.0
    max_a: float = 1.30
    min_rows_per_end_min: int = 250


def _fit_affine(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Fit actual ~= a*pred + b via least squares."""
    x = pred.astype(float)
    y = actual.astype(float)
    A = np.column_stack([x, np.ones_like(x)])
    # Solve min ||A*[a,b]-y||
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])
    return a, b


def fit_segments_5min_calibration(cfg: FitSegments5MinCalibrationConfig) -> dict:
    if not cfg.backtest_csv.exists():
        raise FileNotFoundError(f"Missing backtest CSV: {cfg.backtest_csv}")

    df = pd.read_csv(cfg.backtest_csv)
    needed = {cfg.pred_col, cfg.actual_col, cfg.end_min_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest CSV missing required columns: {missing}")

    if cfg.start:
        df = df[df.get(cfg.date_col).astype(str) >= str(cfg.start)]
    if cfg.end:
        df = df[df.get(cfg.date_col).astype(str) <= str(cfg.end)]

    df[cfg.end_min_col] = pd.to_numeric(df[cfg.end_min_col], errors="coerce")
    df[cfg.pred_col] = pd.to_numeric(df[cfg.pred_col], errors="coerce")
    df[cfg.actual_col] = pd.to_numeric(df[cfg.actual_col], errors="coerce")
    df = df.dropna(subset=[cfg.end_min_col, cfg.pred_col, cfg.actual_col])

    # Fit per endpoint
    a_by_end_min: dict[str, float] = {}
    b_by_end_min: dict[str, float] = {}
    rows_used = 0
    rows_by_end_min: dict[str, int] = {}

    for end_min, g in df.groupby(cfg.end_min_col):
        n = int(len(g))
        key = str(int(end_min))
        rows_by_end_min[key] = n
        if n < int(cfg.min_rows_per_end_min):
            continue

        x = g[cfg.pred_col].to_numpy(dtype=float)
        y = g[cfg.actual_col].to_numpy(dtype=float)
        a, b = _fit_affine(x, y)
        if not np.isfinite(a) or not np.isfinite(b):
            continue

        a_clipped = float(np.clip(a, float(cfg.min_a), float(cfg.max_a)))
        # If we clip slope, recompute intercept so we still match the mean level.
        if a_clipped != float(a):
            b = float(np.mean(y - a_clipped * x))
        a = a_clipped
        if not np.isfinite(a) or not np.isfinite(b):
            continue

        a_by_end_min[key] = a
        b_by_end_min[key] = float(b)
        rows_used += n

    payload = {
        "kind": "affine_calibration_by_end_min",
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "source_backtest_csv": str(cfg.backtest_csv),
        "start": cfg.start,
        "end": cfg.end,
        "pred_col": cfg.pred_col,
        "actual_col": cfg.actual_col,
        "a_by_end_min": a_by_end_min,
        "b_by_end_min": b_by_end_min,
        "rows_used": int(rows_used),
        "rows_by_end_min": rows_by_end_min,
        "guardrails": {
            "min_a": float(cfg.min_a),
            "max_a": float(cfg.max_a),
            "min_rows_per_end_min": int(cfg.min_rows_per_end_min),
        },
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "out_path": str(cfg.out_path),
        "rows_used": int(rows_used),
        "a_by_end_min": a_by_end_min,
        "b_by_end_min": b_by_end_min,
    }
