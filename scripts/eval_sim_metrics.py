"""Evaluate simulation quantile quality vs realized results.

Joins outputs/sim_quantiles_<date>.csv with outputs/daily_results/results_<date>.csv
and reports summary metrics for:
  - Full game: total, margin
  - 1H: total_1h, margin_1h

Metrics:
  - MAE / RMSE for q50 (median)
  - Pinball loss for q10/q50/q90
  - 80% interval coverage for [q10, q90]

Usage:
  python scripts/eval_sim_metrics.py --start 2026-01-01 --end 2026-01-14
  python scripts/eval_sim_metrics.py --last 21
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date(s: str) -> _date:
    return datetime.strptime(str(s), "%Y-%m-%d").date()


def _daterange(start: _date, end: _date) -> list[str]:
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    mask = np.isfinite(y) & np.isfinite(q)
    if not mask.any():
        return float("nan")
    y2 = y[mask]
    q2 = q[mask]
    diff = y2 - q2
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


@dataclass
class Metrics:
    n: int
    mae_q50: float
    rmse_q50: float
    pinball_q10: float
    pinball_q50: float
    pinball_q90: float
    cov_80: float


def _metrics(df: pd.DataFrame, y_col: str, q10_col: str, q50_col: str, q90_col: str) -> Metrics:
    y = _to_num(df[y_col]).to_numpy(dtype=float)
    q10 = _to_num(df[q10_col]).to_numpy(dtype=float)
    q50 = _to_num(df[q50_col]).to_numpy(dtype=float)
    q90 = _to_num(df[q90_col]).to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(q50)
    n = int(mask.sum())
    if n == 0:
        return Metrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    err = y[mask] - q50[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    pb10 = _pinball(y, q10, 0.10)
    pb50 = _pinball(y, q50, 0.50)
    pb90 = _pinball(y, q90, 0.90)

    mask_cov = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q90)
    if mask_cov.any():
        cov = float(np.mean((y[mask_cov] >= q10[mask_cov]) & (y[mask_cov] <= q90[mask_cov])))
    else:
        cov = float("nan")

    return Metrics(n=n, mae_q50=mae, rmse_q50=rmse, pinball_q10=pb10, pinball_q50=pb50, pinball_q90=pb90, cov_80=cov)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.4f}" if abs(x) < 1000 else f"{x:.2f}"


def eval_window(out_dir: Path, dates: list[str]) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows: list[dict] = []

    for d in dates:
        sq_path = out_dir / f"sim_quantiles_{d}.csv"
        res_path = out_dir / "daily_results" / f"results_{d}.csv"

        sq = _safe_read_csv(sq_path)
        res = _safe_read_csv(res_path)
        if sq.empty or res.empty:
            continue

        if "game_id" not in sq.columns or "game_id" not in res.columns:
            continue

        sq = sq.copy()
        res = res.copy()
        sq["game_id"] = sq["game_id"].astype(str).str.replace(r"\\.0$", "", regex=True)
        res["game_id"] = res["game_id"].astype(str).str.replace(r"\\.0$", "", regex=True)

        merged = res.merge(sq, on="game_id", how="left", suffixes=("", "_sim"))

        # Full game metrics
        have_full = {"actual_total", "q10_total", "q50_total", "q90_total"}.issubset(merged.columns)
        have_mar = {"actual_margin", "q10_margin", "q50_margin", "q90_margin"}.issubset(merged.columns)

        m_total = _metrics(merged, "actual_total", "q10_total", "q50_total", "q90_total") if have_full else Metrics(0, *([float("nan")] * 6))
        m_margin = _metrics(merged, "actual_margin", "q10_margin", "q50_margin", "q90_margin") if have_mar else Metrics(0, *([float("nan")] * 6))

        # 1H metrics (new sims)
        have_1h = {"actual_total_1h", "q10_total_1h", "q50_total_1h", "q90_total_1h"}.issubset(merged.columns)
        have_1h_mar = {"actual_margin_1h", "q10_margin_1h", "q50_margin_1h", "q90_margin_1h"}.issubset(merged.columns)

        m_total_1h = _metrics(merged, "actual_total_1h", "q10_total_1h", "q50_total_1h", "q90_total_1h") if have_1h else Metrics(0, *([float("nan")] * 6))
        m_margin_1h = _metrics(merged, "actual_margin_1h", "q10_margin_1h", "q50_margin_1h", "q90_margin_1h") if have_1h_mar else Metrics(0, *([float("nan")] * 6))

        rows.append(
            {
                "date": d,
                "n_games": int(len(res)),
                "n_sim_rows": int(len(sq)),
                "full_total_n": m_total.n,
                "full_total_mae": m_total.mae_q50,
                "full_total_rmse": m_total.rmse_q50,
                "full_total_cov80": m_total.cov_80,
                "full_margin_n": m_margin.n,
                "full_margin_mae": m_margin.mae_q50,
                "full_margin_rmse": m_margin.rmse_q50,
                "full_margin_cov80": m_margin.cov_80,
                "h1_total_n": m_total_1h.n,
                "h1_total_mae": m_total_1h.mae_q50,
                "h1_total_rmse": m_total_1h.rmse_q50,
                "h1_total_cov80": m_total_1h.cov_80,
                "h1_margin_n": m_margin_1h.n,
                "h1_margin_mae": m_margin_1h.mae_q50,
                "h1_margin_rmse": m_margin_1h.rmse_q50,
                "h1_margin_cov80": m_margin_1h.cov_80,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=str, default=str(Path("outputs")))
    ap.add_argument("--start", type=str, default="")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--last", type=int, default=0, help="Evaluate last N days ending yesterday")
    ap.add_argument("--csv", type=str, default="", help="Optional path to write per-day metrics CSV")
    args = ap.parse_args()

    out_dir = Path(args.outputs)

    if args.last and args.last > 0:
        end = datetime.utcnow().date() - timedelta(days=1)
        start = end - timedelta(days=int(args.last) - 1)
    else:
        if not args.start or not args.end:
            ap.error("Provide --start/--end or --last")
        start = _parse_date(args.start)
        end = _parse_date(args.end)

    dates = _daterange(start, end)
    df = eval_window(out_dir, dates)

    if df.empty:
        print("No joined rows found (missing sim_quantiles or results files for window).")
        return 2

    # Aggregate (weighted by n where possible)
    def _wavg(col_val: str, col_n: str) -> float:
        v = pd.to_numeric(df[col_val], errors="coerce")
        n = pd.to_numeric(df[col_n], errors="coerce").fillna(0)
        mask = v.notna() & (n > 0)
        if not mask.any():
            return float("nan")
        return float((v[mask] * n[mask]).sum() / n[mask].sum())

    summary = {
        "days": int(len(df)),
        "full_total_mae": _wavg("full_total_mae", "full_total_n"),
        "full_total_rmse": _wavg("full_total_rmse", "full_total_n"),
        "full_total_cov80": _wavg("full_total_cov80", "full_total_n"),
        "full_margin_mae": _wavg("full_margin_mae", "full_margin_n"),
        "full_margin_rmse": _wavg("full_margin_rmse", "full_margin_n"),
        "full_margin_cov80": _wavg("full_margin_cov80", "full_margin_n"),
        "h1_total_mae": _wavg("h1_total_mae", "h1_total_n"),
        "h1_total_rmse": _wavg("h1_total_rmse", "h1_total_n"),
        "h1_total_cov80": _wavg("h1_total_cov80", "h1_total_n"),
        "h1_margin_mae": _wavg("h1_margin_mae", "h1_margin_n"),
        "h1_margin_rmse": _wavg("h1_margin_rmse", "h1_margin_n"),
        "h1_margin_cov80": _wavg("h1_margin_cov80", "h1_margin_n"),
    }

    print("Sim metrics summary")
    print(f"  window: {start} -> {end} ({summary['days']} days)")
    print(f"  full total:   MAE={_fmt(summary['full_total_mae'])} RMSE={_fmt(summary['full_total_rmse'])} cov80={_fmt(summary['full_total_cov80'])}")
    print(f"  full margin:  MAE={_fmt(summary['full_margin_mae'])} RMSE={_fmt(summary['full_margin_rmse'])} cov80={_fmt(summary['full_margin_cov80'])}")
    print(f"  1H total:     MAE={_fmt(summary['h1_total_mae'])} RMSE={_fmt(summary['h1_total_rmse'])} cov80={_fmt(summary['h1_total_cov80'])}")
    print(f"  1H margin:    MAE={_fmt(summary['h1_margin_mae'])} RMSE={_fmt(summary['h1_margin_rmse'])} cov80={_fmt(summary['h1_margin_cov80'])}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Wrote per-day metrics: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
