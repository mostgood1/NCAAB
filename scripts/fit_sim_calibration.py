"""Fit a lightweight global calibration for the sim engine.

This writes a small JSON file that `src/simulation/game_sim.py` will apply (if present)
when generating `outputs/sim_quantiles_<date>.csv`.

Calibration parameters:
  - delta_total, delta_margin: additive mean shifts
  - sigma_total_mult, sigma_margin_mult: multiplicative uncertainty inflation
    - rho: correlation between home and away scores (used when sigma_margin is missing)

Method:
  1) Merge sim outputs (mu_total/mu_margin/sigma_total/sigma_margin) with actuals.
  2) Choose delta_* as mean residual (actual - mu).
  3) Choose sigma_*_mult so that the empirical 80% central coverage for a normal
     interval matches the target z for q10/q90 (z=1.28155).

Usage:
  python scripts/fit_sim_calibration.py --start 2026-01-01 --end 2026-01-13
    python scripts/fit_sim_calibration.py --start 2026-01-01 --end 2026-01-13 --rebuild-sims --samples 2000

Writes:
  outputs/sim_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
Z_80 = 1.2815515655446004  # central 80% => q10/q90


def _parse_date(s: str) -> str | None:
    s = str(s).strip()
    return s if DATE_RE.match(s) else None


def _normalize_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.str.replace(r"\.0$", "", regex=True).str.strip()


def _list_dates(outputs_dir: Path) -> list[str]:
    sim_dates = {p.stem.replace("sim_quantiles_", "") for p in outputs_dir.glob("sim_quantiles_*.csv")}
    res_dates = {p.stem.replace("results_", "") for p in (outputs_dir / "daily_results").glob("results_*.csv")}
    dates = sorted(d for d in sim_dates.intersection(res_dates) if _parse_date(d))
    return dates


def _apply_date_filter(dates: list[str], start: str | None, end: str | None) -> list[str]:
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    return dates


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    try:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        df = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 10:
            return None
        c = float(df["a"].corr(df["b"]))
        if math.isnan(c):
            return None
        return c
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Outputs directory")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--fit-1h-only",
        action="store_true",
        help="Only update 1H calibration keys (delta_*_1h, sigma_*_1h_mult). Full-game keys are preserved from any prior calibration file.",
    )
    ap.add_argument(
        "--no-accumulate",
        action="store_true",
        help="Do not accumulate full-game updates on top of any prior calibration. Useful when fitting from scratch on an already-calibrated sim output.",
    )
    ap.add_argument(
        "--rebuild-sims",
        action="store_true",
        help="Re-run the simulator to (re)write outputs/sim_quantiles_<date>.csv before fitting calibration. "
        "When enabled, any existing sim_calibration.json is temporarily ignored during rebuild.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=None,
        help="If --rebuild-sims is set, override simulator sample count (default: simulator's internal default)",
    )
    ap.add_argument("--min-games", type=int, default=50, help="Minimum merged games required")
    ap.add_argument(
        "--write-market-total-bins",
        action="store_true",
        help="Also fit and write market-total bucket calibration under 'market_total_bins'.",
    )
    ap.add_argument(
        "--market-total-bins",
        default="0,135,145,155,999",
        help="Comma-separated bin edges for market_total bucket calibration (e.g. '0,135,145,155,999').",
    )
    ap.add_argument(
        "--min-games-bin",
        type=int,
        default=60,
        help="Minimum games required per market_total bin to write that bin's calibration.",
    )
    ap.add_argument("--cap-abs-delta", type=float, default=25.0, help="Clamp |delta_total| and |delta_margin| to this value")
    ap.add_argument("--cap-abs-delta-1h", type=float, default=15.0, help="Clamp |delta_total_1h| and |delta_margin_1h| to this value")
    ap.add_argument("--cap-sigma-mult", type=float, default=3.0, help="Clamp sigma_total_mult and sigma_margin_mult to [0.5, cap]")
    ap.add_argument("--cap-sigma-1h-mult", type=float, default=1.5, help="Clamp sigma_total_1h_mult and sigma_margin_1h_mult to [0.5, cap]")
    ap.add_argument("--out", default=None, help="Calibration JSON path (default: <outputs>/sim_calibration.json)")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    out_path = Path(args.out) if args.out else (outputs_dir / "sim_calibration.json")

    # If sims were produced with an existing calibration file and we are NOT rebuilding
    # sims without it, then the mu/sigma columns already reflect that prior calibration.
    # For full-game fitting, we can optionally accumulate on top of the prior.
    prior = {}
    if not args.rebuild_sims:
        try:
            if out_path.exists():
                prior = json.loads(out_path.read_text(encoding="utf-8"))
                if not isinstance(prior, dict):
                    prior = {}
        except Exception:
            prior = {}

    prior_for_preserve = dict(prior) if isinstance(prior, dict) else {}
    if args.no_accumulate:
        prior_delta_total = 0.0
        prior_delta_margin = 0.0
        prior_sigma_total_mult = 1.0
        prior_sigma_margin_mult = 1.0
        prior_delta_total_1h = 0.0
        prior_delta_margin_1h = 0.0
    else:
        prior_delta_total = float(prior.get("delta_total", 0.0) or 0.0) if prior else 0.0
        prior_delta_margin = float(prior.get("delta_margin", 0.0) or 0.0) if prior else 0.0
        prior_sigma_total_mult = float(prior.get("sigma_total_mult", 1.0) or 1.0) if prior else 1.0
        prior_sigma_margin_mult = float(prior.get("sigma_margin_mult", 1.0) or 1.0) if prior else 1.0
        prior_delta_total_1h = float(prior.get("delta_total_1h", 0.0) or 0.0) if prior else 0.0
        prior_delta_margin_1h = float(prior.get("delta_margin_1h", 0.0) or 0.0) if prior else 0.0

    dates = _list_dates(outputs_dir)
    start = _parse_date(args.start) if args.start else None
    end = _parse_date(args.end) if args.end else None
    dates = _apply_date_filter(dates, start, end)

    if not dates:
        print(json.dumps({"error": "no_dates", "outputs": str(outputs_dir)}))
        return 2

    if args.rebuild_sims:
        # Ensure repository root is on sys.path so `src` imports work.
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.simulation.game_sim import run_simulations_for_date

        # Temporarily ignore any existing sim calibration during rebuild.
        calib_path = outputs_dir / "sim_calibration.json"
        prior_calib_text: str | None = None
        if calib_path.exists():
            try:
                prior_calib_text = calib_path.read_text(encoding="utf-8")
                calib_path.unlink()
            except Exception:
                prior_calib_text = None

        try:
            for d in dates:
                kwargs = {}
                if args.samples is not None:
                    kwargs["samples"] = int(args.samples)
                run_simulations_for_date(outputs_dir, d, **kwargs)
        finally:
            # Restore prior calibration file if we removed it.
            if prior_calib_text is not None and not calib_path.exists():
                try:
                    calib_path.write_text(prior_calib_text, encoding="utf-8")
                except Exception:
                    pass

    all_rows: list[pd.DataFrame] = []
    all_rows_1h: list[pd.DataFrame] = []

    for d in dates:
        sim_path = outputs_dir / f"sim_quantiles_{d}.csv"
        res_path = outputs_dir / "daily_results" / f"results_{d}.csv"
        sim = _read_csv(sim_path)
        res = _read_csv(res_path)
        if sim.empty or res.empty:
            continue
        if "game_id" not in sim.columns or "game_id" not in res.columns:
            continue

        sim = sim.copy()
        res = res.copy()
        sim["game_id"] = _normalize_game_id(sim["game_id"])
        res["game_id"] = _normalize_game_id(res["game_id"])

        # Ensure each game contributes once. Some artifacts can contain duplicate rows
        # per game_id (e.g., joins/snapshots); calibration should not overweight those.
        sim = sim.drop_duplicates(subset=["game_id"])
        res = res.drop_duplicates(subset=["game_id"])

        keep_sim = [c for c in [
            "date",
            "game_id",
            "mu_total",
            "mu_margin",
            "sigma_total",
            "sigma_margin",
            "market_total",
            "q50_total",
            "q50_margin",
            "mu_total_1h",
            "mu_margin_1h",
            "sigma_total_1h",
            "sigma_margin_1h",
        ] if c in sim.columns]

        keep_res = [c for c in [
            "date",
            "game_id",
            "home_score",
            "away_score",
            "actual_total",
            "actual_margin",
            "actual_total_1h",
            "actual_margin_1h",
        ] if c in res.columns]

        if "date" not in sim.columns:
            sim["date"] = d
        if "date" not in res.columns:
            res["date"] = d

        merged = res[keep_res].merge(sim[keep_sim], on=["date", "game_id"], how="inner")
        if merged.empty:
            continue
        all_rows.append(merged)

        # Keep a dedicated 1H calibration frame only when inputs exist.
        needed_1h = {"actual_total_1h", "actual_margin_1h", "mu_total_1h", "mu_margin_1h", "sigma_total_1h"}
        if needed_1h.issubset(set(merged.columns)):
            all_rows_1h.append(merged[[c for c in merged.columns if c in needed_1h.union({"date", "game_id", "sigma_margin_1h"})]].copy())

    if not all_rows:
        print(json.dumps({"error": "no_merged_rows"}))
        return 2

    df = pd.concat(all_rows, ignore_index=True)

    # numeric
    for c in ["home_score", "away_score", "actual_total", "actual_margin", "mu_total", "mu_margin", "sigma_total", "sigma_margin"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional quantile centers (prefer these for totals calibration when present)
    for c in ["q50_total", "q50_margin"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to real finals and derive actual_total/actual_margin from the final score columns.
    # Some results rows can contain placeholder 0/0 scores; those must not influence calibration.
    if "home_score" in df.columns and "away_score" in df.columns:
        hs = df["home_score"]
        as_ = df["away_score"]
        final_mask = hs.notna() & as_.notna() & ((hs > 0) | (as_ > 0))
        df = df.loc[final_mask].copy()
        df["actual_total"] = (hs.loc[final_mask].to_numpy() + as_.loc[final_mask].to_numpy())
        df["actual_margin"] = (hs.loc[final_mask].to_numpy() - as_.loc[final_mask].to_numpy())
    else:
        if "actual_total" in df.columns:
            df = df[df["actual_total"].notna() & (df["actual_total"] > 0)].copy()

    # Prefer q50_* as the center if present; fallback to mu_*.
    center_total_col = "q50_total" if "q50_total" in df.columns else "mu_total"
    center_margin_col = "q50_margin" if "q50_margin" in df.columns else "mu_margin"

    df = df.dropna(subset=["actual_total", "actual_margin", center_total_col, center_margin_col, "sigma_total"])
    df = df[df["actual_total"] > 0]
    df = df[df["sigma_total"] > 0]

    # Optional market_total (for bucket calibration)
    if "market_total" in df.columns:
        df["market_total"] = pd.to_numeric(df["market_total"], errors="coerce")

    if len(df) < int(args.min_games):
        print(json.dumps({"error": "too_few_games", "n_games": int(len(df)), "min_games": int(args.min_games)}))
        return 2

    def _clamp_abs(x: float, cap: float) -> float:
        try:
            x = float(x)
            cap = float(cap)
            if cap <= 0:
                return x
            return float(np.clip(x, -cap, cap))
        except Exception:
            return float(x)

    def _clamp_sigma(x: float, cap: float) -> float:
        try:
            x = float(x)
            cap = float(cap)
            lo = 0.5
            hi = max(lo, cap)
            return float(np.clip(x, lo, hi))
        except Exception:
            return float(x)

    calib = {
        "start": dates[0],
        "end": dates[-1],
    }

    if args.fit_1h_only:
        # Preserve full-game params (and rho) from prior calibration.
        calib.update({
            "n_games": int(len(df)),
            "delta_total": float(prior_for_preserve.get("delta_total", 0.0) or 0.0),
            "delta_margin": float(prior_for_preserve.get("delta_margin", 0.0) or 0.0),
            "sigma_total_mult": float(prior_for_preserve.get("sigma_total_mult", 1.0) or 1.0),
            "sigma_margin_mult": float(prior_for_preserve.get("sigma_margin_mult", 1.0) or 1.0),
        })
        if isinstance(prior_for_preserve, dict) and "rho" in prior_for_preserve and prior_for_preserve.get("rho") is not None:
            try:
                calib["rho"] = float(prior_for_preserve.get("rho"))
                calib["rho_method"] = prior_for_preserve.get("rho_method")
                calib["n_games_rho"] = prior_for_preserve.get("n_games_rho")
            except Exception:
                pass
    else:
        # Mean shifts (residual-on-current outputs); optionally accumulate on top of any prior calibration.
        delta_total_resid = float(np.mean((df["actual_total"] - df[center_total_col]).astype(float)))
        delta_margin_resid = float(np.mean((df["actual_margin"] - df[center_margin_col]).astype(float)))
        delta_total = float(prior_delta_total + delta_total_resid)
        delta_margin = float(prior_delta_margin + delta_margin_resid)
        delta_total = _clamp_abs(delta_total, float(args.cap_abs_delta))
        delta_margin = _clamp_abs(delta_margin, float(args.cap_abs_delta))

        # Sigma inflation: use absolute normalized residuals
        # Compute z on *current* sigma columns (may already include prior inflation)
        # and then scale multipliers on top of prior multipliers.
        z_total = ((df["actual_total"] - (df[center_total_col] + delta_total_resid)) / df["sigma_total"]).astype(float)
        z_total = z_total.replace([np.inf, -np.inf], np.nan).dropna()
        if len(z_total) == 0:
            sigma_total_mult_upd = 1.0
        else:
            q = float(np.quantile(np.abs(z_total), 0.80))
            sigma_total_mult_upd = float(max(0.5, min(5.0, q / Z_80)))

        sigma_total_mult = float(prior_sigma_total_mult * sigma_total_mult_upd)
        sigma_total_mult = _clamp_sigma(sigma_total_mult, float(args.cap_sigma_mult))

        # Margin inflation only if sigma_margin exists
        sigma_margin_mult_upd = 1.0
        if "sigma_margin" in df.columns:
            dm = df.dropna(subset=["sigma_margin"]).copy()
            dm = dm[dm["sigma_margin"] > 0]
            if len(dm) > 0:
                z_margin = ((dm["actual_margin"] - (dm["mu_margin"] + delta_margin_resid)) / dm["sigma_margin"]).astype(float)
                z_margin = z_margin.replace([np.inf, -np.inf], np.nan).dropna()
                if len(z_margin) > 0:
                    q = float(np.quantile(np.abs(z_margin), 0.80))
                    sigma_margin_mult_upd = float(max(0.5, min(5.0, q / Z_80)))

        sigma_margin_mult = float(prior_sigma_margin_mult * sigma_margin_mult_upd)
        sigma_margin_mult = _clamp_sigma(sigma_margin_mult, float(args.cap_sigma_mult))

        calib.update({
            "n_games": int(len(df)),
            "delta_total": delta_total,
            "delta_margin": delta_margin,
            "sigma_total_mult": sigma_total_mult,
            "sigma_margin_mult": sigma_margin_mult,
        })

        # Optional: market_total bucket calibration (totals only)
        if args.write_market_total_bins and "market_total" in df.columns:
            try:
                edges = [float(x.strip()) for x in str(args.market_total_bins).split(",") if str(x).strip()]
                edges = sorted(set(edges))
            except Exception:
                edges = []
            if len(edges) >= 2:
                bins_out: list[dict] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    try:
                        dfi = df[(df["market_total"].notna()) & (df["market_total"] >= float(lo)) & (df["market_total"] < float(hi))].copy()
                    except Exception:
                        continue
                    if len(dfi) < int(args.min_games_bin):
                        continue

                    # Bin residuals vs the same center used for global.
                    dt_resid = float(np.mean((dfi["actual_total"] - dfi[center_total_col]).astype(float)))
                    # Shrink bin adjustment toward global residual to reduce noise.
                    shrink = float(len(dfi) / (len(dfi) + 200.0))
                    shrink = float(np.clip(shrink, 0.0, 1.0))
                    delta_add = float(shrink * (dt_resid - delta_total_resid))
                    delta_add = _clamp_abs(delta_add, 12.0)

                    zt = ((dfi["actual_total"] - (dfi[center_total_col] + dt_resid)) / dfi["sigma_total"]).astype(float)
                    zt = zt.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(zt) == 0:
                        mult_upd = 1.0
                    else:
                        q = float(np.quantile(np.abs(zt), 0.80))
                        mult_upd = float(max(0.5, min(5.0, q / Z_80)))

                    # Express sigma adjustment as a multiplicative factor on top of global update.
                    try:
                        ratio = float(mult_upd) / float(sigma_total_mult_upd if sigma_total_mult_upd else 1.0)
                    except Exception:
                        ratio = 1.0
                    if not np.isfinite(ratio) or ratio <= 0:
                        ratio = 1.0
                    sigma_mult_mult = float(ratio ** shrink)
                    sigma_mult_mult = float(np.clip(sigma_mult_mult, 0.75, 1.35))

                    bins_out.append(
                        {
                            "min": float(lo),
                            "max": float(hi),
                            "n_games": int(len(dfi)),
                            "delta_total_add": float(delta_add),
                            "sigma_total_mult_mult": float(sigma_mult_mult),
                        }
                    )

                if bins_out:
                    calib["market_total_bins"] = bins_out

    # Estimate global rho from historical games by correlating centered per-team residuals.
    # Use the calibrated mean shifts so we don't bake in simple bias.
    if (not args.fit_1h_only) and "home_score" in df.columns and "away_score" in df.columns:
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        mu_total_cal = pd.to_numeric(df["mu_total"], errors="coerce") + float(delta_total)
        mu_margin_cal = pd.to_numeric(df["mu_margin"], errors="coerce") + float(delta_margin)
        mu_home_cal = (mu_total_cal + mu_margin_cal) / 2.0
        mu_away_cal = (mu_total_cal - mu_margin_cal) / 2.0
        res_home = hs - mu_home_cal
        res_away = aw - mu_away_cal
        rho_hat = _safe_corr(res_home, res_away)
        if rho_hat is not None:
            # Guardrails: game-to-game estimate can be noisy; keep within sensible bounds.
            rho_hat = float(max(-0.25, min(0.85, rho_hat)))
            calib.update({
                "rho": rho_hat,
                "rho_method": "corr(actual_home - mu_home_cal, actual_away - mu_away_cal)",
                "n_games_rho": int(pd.DataFrame({"a": res_home, "b": res_away}).dropna().shape[0]),
            })

    # Optional 1H calibration (only if we have both actuals and sim 1H moments).
    if all_rows_1h:
        df1 = pd.concat(all_rows_1h, ignore_index=True)
        for c in [
            "actual_total_1h",
            "actual_margin_1h",
            "mu_total_1h",
            "mu_margin_1h",
            "sigma_total_1h",
            "sigma_margin_1h",
        ]:
            if c in df1.columns:
                df1[c] = pd.to_numeric(df1[c], errors="coerce")

        df1 = df1.dropna(subset=["actual_total_1h", "actual_margin_1h", "mu_total_1h", "mu_margin_1h", "sigma_total_1h"])
        df1 = df1[df1["sigma_total_1h"] > 0]

        if len(df1) >= int(args.min_games):
            # Residual updates on top of existing 1H deltas when present.
            delta_total_1h_resid = float(np.mean((df1["actual_total_1h"] - df1["mu_total_1h"]).astype(float)))
            delta_margin_1h_resid = float(np.mean((df1["actual_margin_1h"] - df1["mu_margin_1h"]).astype(float)))
            delta_total_1h = float(prior_delta_total_1h + delta_total_1h_resid)
            delta_margin_1h = float(prior_delta_margin_1h + delta_margin_1h_resid)
            delta_total_1h = _clamp_abs(delta_total_1h, float(args.cap_abs_delta_1h))
            delta_margin_1h = _clamp_abs(delta_margin_1h, float(args.cap_abs_delta_1h))

            z_total_1h = ((df1["actual_total_1h"] - (df1["mu_total_1h"] + delta_total_1h_resid)) / df1["sigma_total_1h"]).astype(float)
            z_total_1h = z_total_1h.replace([np.inf, -np.inf], np.nan).dropna()
            if len(z_total_1h) == 0:
                sigma_total_1h_mult = 1.0
            else:
                q = float(np.quantile(np.abs(z_total_1h), 0.80))
                sigma_total_1h_mult = float(max(0.5, min(5.0, q / Z_80)))
            sigma_total_1h_mult = _clamp_sigma(float(sigma_total_1h_mult), float(args.cap_sigma_1h_mult))

            sigma_margin_1h_mult = 1.0
            if "sigma_margin_1h" in df1.columns:
                dm1 = df1.dropna(subset=["sigma_margin_1h"]).copy()
                dm1 = dm1[dm1["sigma_margin_1h"] > 0]
                if len(dm1) > 0:
                    z_margin_1h = ((dm1["actual_margin_1h"] - (dm1["mu_margin_1h"] + delta_margin_1h_resid)) / dm1["sigma_margin_1h"]).astype(float)
                    z_margin_1h = z_margin_1h.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(z_margin_1h) > 0:
                        q = float(np.quantile(np.abs(z_margin_1h), 0.80))
                        sigma_margin_1h_mult = float(max(0.5, min(5.0, q / Z_80)))
            sigma_margin_1h_mult = _clamp_sigma(float(sigma_margin_1h_mult), float(args.cap_sigma_1h_mult))

            calib.update({
                "delta_total_1h": delta_total_1h,
                "delta_margin_1h": delta_margin_1h,
                "sigma_total_1h_mult": sigma_total_1h_mult,
                "sigma_margin_1h_mult": sigma_margin_1h_mult,
                "n_games_1h": int(len(df1)),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calib, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"wrote": str(out_path), **calib}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
