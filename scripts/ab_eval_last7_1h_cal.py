"""A/B evaluate 1H-only calibration over the most recent finalized window.

This script is intentionally self-contained and restores outputs/sim_calibration.json
after running.

It:
  1) picks last N result dates from outputs/daily_results/results_*.csv
  2) runs baseline sims to sim_quantiles_base1h7_*/sim_segments_base1h7_*
  3) fits delta_total_1h / delta_margin_1h from baseline outputs vs actuals
  4) writes those 1H keys into sim_calibration.json (preserving full-game keys)
  5) runs calibrated sims to sim_quantiles_cal1h7_*/sim_segments_cal1h7_*
  6) computes MAE/bias for FG, 1H, and seg endpoint 20 (vs actual_total_1h)

Usage:
  python scripts/ab_eval_last7_1h_cal.py --n 7 --samples 600

Prefixes:
    By default, this writes:
        outputs/sim_quantiles_base1h7_<date>.csv
        outputs/sim_segments_base1h7_<date>.csv
    You can override tags to avoid overwriting artifacts, e.g.:
        python scripts/ab_eval_last7_1h_cal.py --base-tag base1h7_bias --cal-tag cal1h7_bias --bias-only-segments
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path so `src.*` imports work when invoked as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.game_sim import run_simulations_for_date

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _normalize_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.str.replace(r"\.0$", "", regex=True).str.strip()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _mae_bias(pred: pd.Series, actual: pd.Series) -> dict:
    p = pd.to_numeric(pred, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    df = pd.DataFrame({"p": p, "a": a}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return {"n": 0, "mae": None, "bias": None}
    err = (df["p"] - df["a"]).astype(float)
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _collect_last_dates(outputs_dir: Path, n: int) -> list[str]:
    res_dir = outputs_dir / "daily_results"
    dates = []
    for p in res_dir.glob("results_*.csv"):
        d = p.stem.replace("results_", "")
        if DATE_RE.match(d):
            dates.append(d)
    dates = sorted(set(dates))
    return dates[-n:]


def _fit_1h_deltas(outputs_dir: Path, dates: list[str], quant_prefix: str) -> dict:
    rows = []
    for d in dates:
        sim = _read_csv(outputs_dir / f"{quant_prefix}{d}.csv")
        res = _read_csv(outputs_dir / "daily_results" / f"results_{d}.csv")
        if sim.empty or res.empty:
            continue
        if "game_id" not in sim.columns or "game_id" not in res.columns:
            continue
        if not {"mu_total_1h", "mu_margin_1h"}.issubset(sim.columns):
            continue
        need = {"home_score_1h", "away_score_1h"}
        if not need.issubset(res.columns) and not {"actual_total_1h", "actual_margin_1h"}.issubset(res.columns):
            continue

        sim = sim.copy()
        res = res.copy()
        sim["game_id"] = _normalize_game_id(sim["game_id"])
        res["game_id"] = _normalize_game_id(res["game_id"])
        sim = sim.drop_duplicates(subset=["game_id"])
        res = res.drop_duplicates(subset=["game_id"])

        if "home_score_1h" in res.columns and "away_score_1h" in res.columns:
            hs1 = pd.to_numeric(res["home_score_1h"], errors="coerce")
            as1 = pd.to_numeric(res["away_score_1h"], errors="coerce")
            res["actual_total_1h"] = hs1 + as1
            res["actual_margin_1h"] = hs1 - as1

        merged = res[["game_id", "actual_total_1h", "actual_margin_1h"]].merge(
            sim[["game_id", "mu_total_1h", "mu_margin_1h"]], on="game_id", how="inner"
        )
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        merged = merged[(merged["actual_total_1h"] > 0) & (merged["mu_total_1h"] > 0)]
        if not merged.empty:
            rows.append(merged)

    if not rows:
        return {"n": 0, "delta_total_1h": 0.0, "delta_margin_1h": 0.0}

    df = pd.concat(rows, ignore_index=True)
    d_total = float(np.mean((df["actual_total_1h"] - df["mu_total_1h"]).astype(float)))
    d_margin = float(np.mean((df["actual_margin_1h"] - df["mu_margin_1h"]).astype(float)))
    return {"n": int(len(df)), "delta_total_1h": d_total, "delta_margin_1h": d_margin}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--n", type=int, default=7, help="How many most recent result dates")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--base-tag", default="base1h7", help="Tag used for baseline output prefixes")
    ap.add_argument("--cal-tag", default="cal1h7", help="Tag used for calibrated output prefixes")
    ap.add_argument("--cap-abs-delta-1h", type=float, default=15.0)
    ap.add_argument("--bias-only-segments", action="store_true", help="Set NCAAB_SEGMENT_BIAS_ONLY=1 during runs")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    dates = _collect_last_dates(outputs_dir, int(args.n))
    if not dates:
        print(json.dumps({"error": "no_dates"}))
        return 2

    calib_path = outputs_dir / "sim_calibration.json"
    orig_text = calib_path.read_text(encoding="utf-8") if calib_path.exists() else None
    orig_obj = {}
    if orig_text:
        try:
            orig_obj = json.loads(orig_text)
            if not isinstance(orig_obj, dict):
                orig_obj = {}
        except Exception:
            orig_obj = {}

    # Run baseline
    if args.bias_only_segments:
        import os

        os.environ["NCAAB_SEGMENT_BIAS_ONLY"] = "1"

    base_quant_prefix = f"sim_quantiles_{args.base_tag}_"
    base_seg_prefix = f"sim_segments_{args.base_tag}_"
    base_meta_prefix = f"sim_meta_{args.base_tag}_"
    cal_quant_prefix = f"sim_quantiles_{args.cal_tag}_"
    cal_seg_prefix = f"sim_segments_{args.cal_tag}_"
    cal_meta_prefix = f"sim_meta_{args.cal_tag}_"

    for d in dates:
        run_simulations_for_date(
            outputs_dir,
            d,
            samples=int(args.samples),
            mean_source="features_strict",
            use_pace=True,
            engine="auto",
            quantiles_out_prefix=base_quant_prefix,
            segments_out_prefix=base_seg_prefix,
            meta_out_prefix=base_meta_prefix,
        )

    fit = _fit_1h_deltas(outputs_dir, dates, base_quant_prefix)
    # IMPORTANT: baseline sims already include whatever 1H calibration is in the current
    # sim_calibration.json. So the fitted deltas here are *residual updates*.
    prior_total_1h = float(orig_obj.get("delta_total_1h", 0.0) or 0.0)
    prior_margin_1h = float(orig_obj.get("delta_margin_1h", 0.0) or 0.0)
    upd_total = float(fit.get("delta_total_1h", 0.0) or 0.0)
    upd_margin = float(fit.get("delta_margin_1h", 0.0) or 0.0)

    d_total = float(np.clip(prior_total_1h + upd_total, -float(args.cap_abs_delta_1h), float(args.cap_abs_delta_1h)))
    d_margin = float(np.clip(prior_margin_1h + upd_margin, -float(args.cap_abs_delta_1h), float(args.cap_abs_delta_1h)))

    cand = dict(orig_obj)
    cand["delta_total_1h"] = d_total
    cand["delta_margin_1h"] = d_margin

    # Write candidate calibration
    calib_path.write_text(json.dumps(cand, indent=2, sort_keys=True), encoding="utf-8")

    try:
        # Run calibrated
        for d in dates:
            run_simulations_for_date(
                outputs_dir,
                d,
                samples=int(args.samples),
                mean_source="features_strict",
                use_pace=True,
                engine="auto",
                quantiles_out_prefix=cal_quant_prefix,
                segments_out_prefix=cal_seg_prefix,
                meta_out_prefix=cal_meta_prefix,
            )
    finally:
        # Restore original calibration
        if orig_text is not None:
            calib_path.write_text(orig_text, encoding="utf-8")
        else:
            try:
                if calib_path.exists():
                    calib_path.unlink()
            except Exception:
                pass

    # Metrics
    fg_rows = []
    fg_rows_cal = []
    seg20_rows = []
    seg20_rows_cal = []

    for d in dates:
        res = _read_csv(outputs_dir / "daily_results" / f"results_{d}.csv")
        if res.empty or "game_id" not in res.columns:
            continue
        res = res.copy()
        res["game_id"] = _normalize_game_id(res["game_id"])

        if "home_score" in res.columns and "away_score" in res.columns:
            hs = pd.to_numeric(res["home_score"], errors="coerce")
            aw = pd.to_numeric(res["away_score"], errors="coerce")
            res["actual_total"] = hs + aw
        if "home_score_1h" in res.columns and "away_score_1h" in res.columns:
            hs1 = pd.to_numeric(res["home_score_1h"], errors="coerce")
            as1 = pd.to_numeric(res["away_score_1h"], errors="coerce")
            res["actual_total_1h"] = hs1 + as1

        base = _read_csv(outputs_dir / f"{base_quant_prefix}{d}.csv")
        cal = _read_csv(outputs_dir / f"{cal_quant_prefix}{d}.csv")
        if not base.empty and "game_id" in base.columns:
            base["game_id"] = _normalize_game_id(base["game_id"])
            fg_rows.append(res[["game_id", "actual_total", "actual_total_1h"]].merge(
                base[["game_id", "q50_total", "q50_total_1h"]], on="game_id", how="inner"
            ))
        if not cal.empty and "game_id" in cal.columns:
            cal["game_id"] = _normalize_game_id(cal["game_id"])
            fg_rows_cal.append(res[["game_id", "actual_total", "actual_total_1h"]].merge(
                cal[["game_id", "q50_total", "q50_total_1h"]], on="game_id", how="inner"
            ))

        segb = _read_csv(outputs_dir / f"{base_seg_prefix}{d}.csv")
        segc = _read_csv(outputs_dir / f"{cal_seg_prefix}{d}.csv")
        if not segb.empty and {"game_id", "end_min", "q50_total_score_end"}.issubset(segb.columns):
            segb["game_id"] = _normalize_game_id(segb["game_id"])
            seg20 = segb[pd.to_numeric(segb["end_min"], errors="coerce") == 20]
            seg20_rows.append(res[["game_id", "actual_total_1h"]].merge(seg20[["game_id", "q50_total_score_end"]], on="game_id", how="inner"))
        if not segc.empty and {"game_id", "end_min", "q50_total_score_end"}.issubset(segc.columns):
            segc["game_id"] = _normalize_game_id(segc["game_id"])
            seg20 = segc[pd.to_numeric(segc["end_min"], errors="coerce") == 20]
            seg20_rows_cal.append(res[["game_id", "actual_total_1h"]].merge(seg20[["game_id", "q50_total_score_end"]], on="game_id", how="inner"))

    out = {
        "dates": dates,
        "samples": int(args.samples),
        "fit": {"n": int(fit.get("n", 0)), "delta_total_1h": d_total, "delta_margin_1h": d_margin},
    }

    if fg_rows:
        dfb = pd.concat(fg_rows, ignore_index=True)
        out["baseline_fg"] = _mae_bias(dfb["q50_total"], dfb["actual_total"])
        out["baseline_1h"] = _mae_bias(dfb["q50_total_1h"], dfb["actual_total_1h"])
    if fg_rows_cal:
        dfc = pd.concat(fg_rows_cal, ignore_index=True)
        out["cal_fg"] = _mae_bias(dfc["q50_total"], dfc["actual_total"])
        out["cal_1h"] = _mae_bias(dfc["q50_total_1h"], dfc["actual_total_1h"])
    if seg20_rows:
        ds = pd.concat(seg20_rows, ignore_index=True)
        out["baseline_seg20"] = _mae_bias(ds["q50_total_score_end"], ds["actual_total_1h"])
    if seg20_rows_cal:
        ds = pd.concat(seg20_rows_cal, ignore_index=True)
        out["cal_seg20"] = _mae_bias(ds["q50_total_score_end"], ds["actual_total_1h"])

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
