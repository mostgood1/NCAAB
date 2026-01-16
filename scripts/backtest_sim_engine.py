"""Backtest the simulation engine (sim_quantiles_*.csv) against actual results.

Inputs:
  - outputs/sim_quantiles_<date>.csv
  - outputs/daily_results/results_<date>.csv

Outputs (default):
  - outputs/backtests/sim_engine_backtest_<start>_<end>.csv (per-date metrics)
  - outputs/backtests/sim_engine_backtest_<start>_<end>.json (overall summary)

This focuses on totals/margins numeric accuracy and quantile calibration:
  - MAE/RMSE/Bias for q50
  - 80% interval coverage for [q10, q90]
  - tail rates (below q10 / above q90)

Usage:
  python scripts/backtest_sim_engine.py --start 2025-12-08 --end 2026-01-13
    python scripts/backtest_sim_engine.py --start 2025-12-08 --end 2026-01-13 --rebuild-sims
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


OUTPUTS = Path("outputs")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _normalize_game_id(series: pd.Series) -> pd.Series:
    # Some CSVs can stringify numeric ids as "123.0".
    s = series.astype(str)
    return s.str.replace(r"\.0$", "", regex=True).str.strip()


def _parse_date(s: str) -> str | None:
    s = str(s).strip()
    return s if DATE_RE.match(s) else None


def _date_from_stem(stem: str, prefix: str) -> str | None:
    if not stem.startswith(prefix):
        return None
    return _parse_date(stem[len(prefix) :])


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@dataclass(frozen=True)
class IntervalMetrics:
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None
    covered_80: float | None
    below_q10: float | None
    above_q90: float | None
    mean_width_80: float | None


def _metrics_from_quantiles(
    *,
    y: pd.Series,
    q10: pd.Series,
    q50: pd.Series,
    q90: pd.Series,
) -> IntervalMetrics:
    yv = pd.to_numeric(y, errors="coerce")
    q10v = pd.to_numeric(q10, errors="coerce")
    q50v = pd.to_numeric(q50, errors="coerce")
    q90v = pd.to_numeric(q90, errors="coerce")

    valid = yv.notna() & q50v.notna()
    n = int(valid.sum())
    if n == 0:
        return IntervalMetrics(
            n=0,
            mae=None,
            rmse=None,
            bias=None,
            covered_80=None,
            below_q10=None,
            above_q90=None,
            mean_width_80=None,
        )

    err = (q50v[valid] - yv[valid]).astype(float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))

    valid_interval = valid & q10v.notna() & q90v.notna()
    if int(valid_interval.sum()) > 0:
        covered_80 = float(((yv[valid_interval] >= q10v[valid_interval]) & (yv[valid_interval] <= q90v[valid_interval])).mean())
        below_q10 = float((yv[valid_interval] < q10v[valid_interval]).mean())
        above_q90 = float((yv[valid_interval] > q90v[valid_interval]).mean())
        mean_width_80 = float(np.mean((q90v[valid_interval] - q10v[valid_interval]).astype(float)))
    else:
        covered_80 = None
        below_q10 = None
        above_q90 = None
        mean_width_80 = None

    return IntervalMetrics(
        n=n,
        mae=mae,
        rmse=rmse,
        bias=bias,
        covered_80=covered_80,
        below_q10=below_q10,
        above_q90=above_q90,
        mean_width_80=mean_width_80,
    )


def _list_sim_quantiles(outputs_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in outputs_dir.glob("sim_quantiles_*.csv"):
        d = _date_from_stem(p.stem, "sim_quantiles_")
        if d:
            out[d] = p
    return out


def _list_results(outputs_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    dr = outputs_dir / "daily_results"
    for p in dr.glob("results_*.csv"):
        d = _date_from_stem(p.stem, "results_")
        if d:
            out[d] = p
    return out


def _apply_date_filter(dates: list[str], start: str | None, end: str | None) -> list[str]:
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    return dates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(OUTPUTS), help="Path to outputs directory")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--rebuild-sims",
        action="store_true",
        help="Re-run the current simulator to (re)write outputs/sim_quantiles_<date>.csv before scoring",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=None,
        help="If --rebuild-sims is set, override simulator sample count (default: simulator's internal default)",
    )
    ap.add_argument("--write-games", action="store_true", help="Write per-game merged rows")
    ap.add_argument("--include-nonfinal", action="store_true", help="Include rows without actual totals/margins")
    ap.add_argument("--require-sim-ok", action="store_true", default=False, help="Filter to sim_ok==True where available")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    sim_map = _list_sim_quantiles(outputs_dir)
    res_map = _list_results(outputs_dir)

    dates = sorted(set(sim_map).intersection(res_map))
    start = _parse_date(args.start) if args.start else None
    end = _parse_date(args.end) if args.end else None
    dates = _apply_date_filter(dates, start, end)

    if not dates:
        print(json.dumps({"error": "No overlapping sim_quantiles/results dates found", "sim_dates": len(sim_map), "result_dates": len(res_map)}))
        return 2

    if args.rebuild_sims:
        # Add repo root for local imports, matching other scripts.
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.simulation.game_sim import run_simulations_for_date

        for d in dates:
            try:
                kwargs = {}
                if args.samples is not None:
                    kwargs["samples"] = int(args.samples)
                run_simulations_for_date(outputs_dir, d, **kwargs)
            except Exception as e:
                print(json.dumps({"warn": "rebuild_sims_failed", "date": d, "error": str(e)}))
        # Refresh sim map after rebuild.
        sim_map = _list_sim_quantiles(outputs_dir)

    per_date_rows: list[dict] = []
    all_rows: list[pd.DataFrame] = []

    for d in dates:
        sim = _safe_read_csv(sim_map[d])
        res = _safe_read_csv(res_map[d])
        if sim.empty or res.empty:
            continue

        if "game_id" not in sim.columns or "game_id" not in res.columns:
            continue

        sim = sim.copy()
        res = res.copy()
        sim["game_id"] = _normalize_game_id(sim["game_id"])
        res["game_id"] = _normalize_game_id(res["game_id"])

        # Normalize date columns if present; otherwise pin to filename date.
        if "date" in sim.columns:
            sim["date"] = pd.to_datetime(sim["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            sim = sim[sim["date"] == d]
        else:
            sim["date"] = d

        if "date" in res.columns:
            res["date"] = pd.to_datetime(res["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            res = res[res["date"] == d]
        else:
            res["date"] = d

        # Deduplicate just in case.
        sim = sim.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
        res = res.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

        keep_cols = [c for c in [
            "game_id",
            "date",
            "sim_ok",
            "sim_method",
            "q10_total",
            "q50_total",
            "q90_total",
            "q10_margin",
            "q50_margin",
            "q90_margin",
            "pace_mu",
            "market_total",
        ] if c in sim.columns]
        sim_small = sim[keep_cols]

        merged = res.merge(sim_small, on=["date", "game_id"], how="left", suffixes=("", "_sim"))

        if args.require_sim_ok and "sim_ok" in merged.columns:
            merged = merged[merged["sim_ok"].astype(str).str.lower().isin({"true", "1", "yes"})]

        if not args.include_nonfinal:
            if "actual_total" in merged.columns:
                merged = merged[pd.to_numeric(merged["actual_total"], errors="coerce").notna()]

        total_metrics = _metrics_from_quantiles(
            y=merged.get("actual_total"),
            q10=merged.get("q10_total"),
            q50=merged.get("q50_total"),
            q90=merged.get("q90_total"),
        )
        margin_metrics = _metrics_from_quantiles(
            y=merged.get("actual_margin"),
            q10=merged.get("q10_margin"),
            q50=merged.get("q50_margin"),
            q90=merged.get("q90_margin"),
        )

        per_date_rows.append({
            "date": d,
            "n_total": total_metrics.n,
            "mae_total": total_metrics.mae,
            "rmse_total": total_metrics.rmse,
            "bias_total": total_metrics.bias,
            "covered80_total": total_metrics.covered_80,
            "below10_total": total_metrics.below_q10,
            "above90_total": total_metrics.above_q90,
            "mean_width80_total": total_metrics.mean_width_80,
            "n_margin": margin_metrics.n,
            "mae_margin": margin_metrics.mae,
            "rmse_margin": margin_metrics.rmse,
            "bias_margin": margin_metrics.bias,
            "covered80_margin": margin_metrics.covered_80,
            "below10_margin": margin_metrics.below_q10,
            "above90_margin": margin_metrics.above_q90,
            "mean_width80_margin": margin_metrics.mean_width_80,
        })

        all_rows.append(merged)

    if not per_date_rows or not all_rows:
        print(json.dumps({"error": "No usable rows after loading/merging", "dates": len(dates)}))
        return 3

    per_date = pd.DataFrame(per_date_rows).sort_values("date").reset_index(drop=True)
    merged_all = pd.concat(all_rows, ignore_index=True)

    overall_total = _metrics_from_quantiles(
        y=merged_all.get("actual_total"),
        q10=merged_all.get("q10_total"),
        q50=merged_all.get("q50_total"),
        q90=merged_all.get("q90_total"),
    )
    overall_margin = _metrics_from_quantiles(
        y=merged_all.get("actual_margin"),
        q10=merged_all.get("q10_margin"),
        q50=merged_all.get("q50_margin"),
        q90=merged_all.get("q90_margin"),
    )

    start_out = per_date["date"].iloc[0]
    end_out = per_date["date"].iloc[-1]
    out_dir = outputs_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"sim_engine_backtest_{start_out}_{end_out}.csv"
    per_date.to_csv(out_csv, index=False)

    out_json = out_dir / f"sim_engine_backtest_{start_out}_{end_out}.json"
    out_json.write_text(
        json.dumps(
            {
                "start": start_out,
                "end": end_out,
                "dates": len(per_date),
                "overall": {
                    "total": asdict(overall_total),
                    "margin": asdict(overall_margin),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    wrote = {
        "per_date_csv": str(out_csv),
        "overall_json": str(out_json),
        "dates": int(per_date.shape[0]),
        "n_games": int(merged_all.shape[0]),
    }

    if args.write_games:
        out_games = out_dir / f"sim_engine_backtest_games_{start_out}_{end_out}.csv"
        merged_all.to_csv(out_games, index=False)
        wrote["games_csv"] = str(out_games)

    print(json.dumps(wrote, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
