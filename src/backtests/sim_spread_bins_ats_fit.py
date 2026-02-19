from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitSimSpreadBinsATSConfig:
    backtest_csv: Path
    out_path: Path
    spread_edges: list[float]
    start: str | None = None
    end: str | None = None
    pred_margin_col: str | None = None
    min_games_bin: int = 30
    shrink_k: float = 200.0
    max_abs_delta: float = 12.0


def _best_delta_for_ats(score: pd.Series, actual_cover: pd.Series) -> float | None:
    """Return additive delta to apply to `score` to maximize ATS correctness.

    `score` is the cover decision score (e.g., pred_margin + spread_home).
    Predict cover when (score + delta) > 0.

    This is equivalent to choosing a threshold t where predict cover iff score > t,
    and delta = -t.
    """

    s = pd.to_numeric(score, errors="coerce")
    y = actual_cover.astype("boolean")
    m = s.notna() & y.notna()
    s = s[m].astype(float)
    y = y[m].astype(bool)

    n = int(len(s))
    if n < 25:
        return None

    order = np.argsort(s.to_numpy(dtype=float))
    s_sorted = s.to_numpy(dtype=float)[order]
    y_sorted = y.to_numpy(dtype=bool)[order]

    # If we pick threshold t, we predict cover when score > t.
    # For a split at i (0..n):
    # - lower indices [0, i) predicted non-cover
    # - upper indices [i, n) predicted cover
    # correct = (# non-covers in lower) + (# covers in upper)
    covers = y_sorted.astype(int)
    prefix_covers = np.cumsum(covers)
    total_covers = int(prefix_covers[-1])

    best_correct = -1
    best_threshold: float | None = None

    for i in range(n + 1):
        covers_lower = int(prefix_covers[i - 1]) if i > 0 else 0
        noncovers_lower = i - covers_lower
        covers_upper = total_covers - covers_lower
        correct = noncovers_lower + covers_upper

        if correct > best_correct:
            best_correct = correct
            if i == 0:
                # Predict cover for all rows
                best_threshold = float(s_sorted[0]) - 1e-6
            elif i == n:
                # Predict cover for no rows
                best_threshold = float(s_sorted[-1]) + 1e-6
            else:
                best_threshold = float(0.5 * (s_sorted[i - 1] + s_sorted[i]))

    if best_threshold is None or (not np.isfinite(float(best_threshold))):
        return None

    return float(-best_threshold)


def fit_sim_spread_bins_ats(cfg: FitSimSpreadBinsATSConfig) -> dict:
    df = pd.read_csv(cfg.backtest_csv)
    if df.empty:
        raise ValueError(f"Empty backtest CSV: {cfg.backtest_csv}")

    for col in ("date", "spread_home", "actual_margin"):
        if col not in df.columns:
            raise ValueError(f"Backtest CSV missing required column: {col}")

    pred_col = cfg.pred_margin_col
    if pred_col is None:
        for c in ("q50_margin", "mu_margin", "pred_margin"):
            if c in df.columns:
                pred_col = c
                break
    if not pred_col or pred_col not in df.columns:
        raise ValueError(
            "Backtest CSV missing prediction margin column; expected one of q50_margin/mu_margin/pred_margin "
            f"(or provide --pred-margin-col). Columns seen: {list(df.columns)[:20]}..."
        )

    if cfg.start:
        df = df[df["date"] >= cfg.start]
    if cfg.end:
        df = df[df["date"] <= cfg.end]

    sp = pd.to_numeric(df["spread_home"], errors="coerce")
    am = pd.to_numeric(df["actual_margin"], errors="coerce")
    pm = pd.to_numeric(df[pred_col], errors="coerce")

    # Exclude pushes based on actual result vs spread.
    actual_score = am + sp
    non_push = actual_score.abs() >= 1e-9

    base = pd.DataFrame(
        {
            "date": df.get("date"),
            "spread_home": sp,
            "actual_margin": am,
            "pred_margin": pm,
            "actual_score": actual_score,
            "non_push": non_push,
        }
    )
    base = base.dropna(subset=["spread_home", "actual_margin", "pred_margin", "actual_score"])
    base = base[base["non_push"]]

    edges = sorted({float(x) for x in cfg.spread_edges})
    if len(edges) < 2:
        raise ValueError("Need at least 2 spread edges to form bins")

    bins_out: list[dict] = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (base["spread_home"] >= float(lo)) & (base["spread_home"] < float(hi))
        dfi = base[m].copy()
        n = int(len(dfi))
        if n < int(cfg.min_games_bin):
            continue

        score0 = dfi["pred_margin"] + dfi["spread_home"]
        actual_cover = dfi["actual_score"] > 0

        delta_best = _best_delta_for_ats(score0, actual_cover)
        if delta_best is None:
            continue

        shrink = float(n / (n + float(cfg.shrink_k)))
        shrink = float(np.clip(shrink, 0.0, 1.0))

        delta_add = float(shrink * float(delta_best))
        if np.isfinite(delta_add):
            delta_add = float(np.clip(delta_add, -float(cfg.max_abs_delta), float(cfg.max_abs_delta)))
        else:
            continue

        bins_out.append(
            {
                "min": float(lo),
                "max": float(hi),
                "n_games": int(n),
                "delta_margin_add": float(delta_add),
                "sigma_margin_mult_mult": 1.0,
                "margin_scale_mult": 1.0,
            }
        )

    out = {
        "source": "fit-sim-spread-bins-ats",
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "source_backtest_csv": str(cfg.backtest_csv),
        "start": cfg.start,
        "end": cfg.end,
        "pred_margin_col": str(pred_col),
        "rows_used": int(len(base)),
        "min_games_bin": int(cfg.min_games_bin),
        "shrink_k": float(cfg.shrink_k),
        "max_abs_delta": float(cfg.max_abs_delta),
        "spread_edges": [float(x) for x in edges],
        "spread_bins": bins_out,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    out["out_path"] = str(cfg.out_path)
    return out


def apply_spread_bins_to_default_sim_calibration(
    out_dir: Path,
    spread_bins: list[dict],
    generated_at: str,
    source: str,
) -> dict:
    out_dir = Path(out_dir)
    default_path = out_dir / "sim_calibration.json"

    backup_path = None
    if default_path.exists():
        ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = out_dir / f"sim_calibration.backup_{ts}.json"
        try:
            backup_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            backup_path = None

    merged: dict = {}
    try:
        if default_path.exists():
            obj = json.loads(default_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                merged.update(obj)
    except Exception:
        merged = {}

    merged["spread_bins"] = spread_bins
    merged["_updated_by"] = str(source)
    merged["_updated_at"] = str(generated_at)

    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "applied": str(default_path),
        "backup": str(backup_path) if backup_path else None,
    }
