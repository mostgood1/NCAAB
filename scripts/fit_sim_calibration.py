"""Fit a lightweight global calibration for the sim engine.

This writes a small JSON file that `src/simulation/game_sim.py` will apply (if present)
when generating `outputs/sim_quantiles_<date>.csv`.

Calibration parameters:
  - delta_total, delta_margin: additive mean shifts
    - margin_scale: multiplicative scaling of the predicted margin mean
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


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF for numpy arrays without SciPy."""
    try:
        erf = getattr(np, "erf", None)
        if erf is not None:
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    except Exception:
        pass
    # Fallback: vectorize math.erf
    import math

    v_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + v_erf(x / np.sqrt(2.0)))


def _brier(p: np.ndarray, y: np.ndarray) -> float | None:
    try:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(p) & np.isfinite(y)
        if int(m.sum()) < 25:
            return None
        p = np.clip(p[m], 0.0, 1.0)
        y = y[m]
        return float(np.mean((p - y) ** 2))
    except Exception:
        return None


def _fit_sigma_mult_for_ats_brier(
    mean_score: pd.Series,
    sigma: pd.Series,
    actual_cover: pd.Series,
    *,
    grid_lo: float = 0.70,
    grid_hi: float = 1.50,
    grid_step: float = 0.02,
) -> float | None:
    """Fit a multiplicative sigma factor to minimize Brier on ATS cover probability.

    Uses a normal approximation: P(cover) = Phi(mean_score / (sigma * s)).
    """
    try:
        mu = pd.to_numeric(mean_score, errors="coerce").astype(float)
        sg = pd.to_numeric(sigma, errors="coerce").astype(float)
        y = actual_cover.astype(bool)
        df = pd.DataFrame({"mu": mu, "sg": sg, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        df = df[df["sg"] > 0]
        if len(df) < 50:
            return None

        mu_v = df["mu"].to_numpy(dtype=float)
        sg_v = df["sg"].to_numpy(dtype=float)
        y_v = df["y"].to_numpy(dtype=bool).astype(float)

        best_s = None
        best_loss = None
        grid = np.arange(float(grid_lo), float(grid_hi) + 1e-9, float(grid_step), dtype=float)
        for s in grid:
            z = mu_v / (sg_v * float(s))
            p = _norm_cdf(z)
            loss = _brier(p, y_v)
            if loss is None:
                continue
            if best_loss is None or loss < best_loss:
                best_loss = float(loss)
                best_s = float(s)
        return best_s
    except Exception:
        return None


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


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(v), float(lo), float(hi)))
    except Exception:
        return float(max(lo, min(hi, v)))


def _fit_slope(x: pd.Series, y: pd.Series) -> float | None:
    """Return slope b for y ~ b*x (no intercept), or None if unstable."""
    try:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        y = pd.to_numeric(y, errors="coerce").astype(float)
        df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 25:
            return None
        vx = float(np.var(df["x"], ddof=0))
        if not np.isfinite(vx) or vx <= 1e-9:
            return None
        cov = float(np.mean((df["x"] - float(df["x"].mean())) * (df["y"] - float(df["y"].mean()))))
        b = float(cov / vx)
        if not np.isfinite(b):
            return None
        return b
    except Exception:
        return None


def _best_delta_for_ats(score: pd.Series, actual_cover: pd.Series) -> float | None:
    """Find delta to add to score such that sign(score+delta) best matches actual_cover.

    score: predicted (margin + spread) BEFORE delta
    actual_cover: boolean, True if actual (margin + spread) > 0
    """
    try:
        s = pd.to_numeric(score, errors="coerce").astype(float)
        y = actual_cover.astype(bool)
        df = pd.DataFrame({"s": s, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 25:
            return None

        df = df.sort_values("s", kind="mergesort")
        svals = df["s"].to_numpy(dtype=float)
        yvals = df["y"].to_numpy(dtype=bool)

        # Threshold t = -delta; predict True if s > t.
        # Start with t < min(s): all predicted True.
        correct = int(yvals.sum())
        best_correct = correct
        # choose t very small initially
        best_t = float(svals[0]) - 1.0

        # As t crosses each s_i, that item flips from True to False.
        for i in range(len(svals)):
            if yvals[i]:
                correct -= 1
            else:
                correct += 1
            # candidate threshold just above this value
            t = float(svals[i]) + 1e-9
            if correct > best_correct:
                best_correct = correct
                best_t = t
            elif correct == best_correct:
                # tie-break: prefer delta closer to 0
                if abs(-t) < abs(-best_t):
                    best_t = t

        # Also consider t >= max(s): all predicted False.
        correct_all_false = int((~yvals).sum())
        t = float(svals[-1]) + 1.0
        if correct_all_false > best_correct or (correct_all_false == best_correct and abs(-t) < abs(-best_t)):
            best_t = t

        return float(-best_t)
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
    ap.add_argument(
        "--fit-margin-scale",
        action="store_true",
        help="Fit and write 'margin_scale' (slope) so that scaled predicted margins better match actual margins.",
    )
    ap.add_argument(
        "--cap-margin-scale",
        type=float,
        default=2.0,
        help="Clamp margin_scale to [1/cap, cap] to avoid extreme slopes.",
    )
    ap.add_argument(
        "--write-margin-abs-bins",
        action="store_true",
        help="Also fit and write abs(predicted margin) bucket calibration under 'margin_abs_bins'.",
    )
    ap.add_argument(
        "--margin-abs-bins",
        default="0,2,4,6,8,10,12,16,24,999",
        help="Comma-separated bin edges for abs(predicted margin) bucket calibration (e.g. '0,2,4,6,8,12,999').",
    )
    ap.add_argument(
        "--write-spread-abs-bins",
        action="store_true",
        help="Also fit and write abs(market spread_home) bucket calibration under 'spread_abs_bins'.",
    )
    ap.add_argument(
        "--spread-abs-bins",
        default="0,2,4,6,8,10,12,16,24,999",
        help="Comma-separated bin edges for abs(spread_home) bucket calibration (e.g. '0,2,4,6,8,12,999').",
    )
    ap.add_argument(
        "--write-spread-bins",
        action="store_true",
        help="Also fit and write signed spread_home bucket calibration under 'spread_bins'.",
    )
    ap.add_argument(
        "--spread-bins",
        default="-40,-12,-8,-6,-4,-2,0,2,4,6,8,12,40",
        help="Comma-separated bin edges for signed spread_home bucket calibration.",
    )
    ap.add_argument(
        "--spread-bins-objective",
        default="resid",
        choices=["resid", "ats"],
        help="Objective for spread_bins: 'resid' fits margin residuals; 'ats' chooses delta_margin_add to maximize ATS correctness in-bin.",
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
        prior_margin_scale = 1.0
        prior_delta_total_1h = 0.0
        prior_delta_margin_1h = 0.0
    else:
        prior_delta_total = float(prior.get("delta_total", 0.0) or 0.0) if prior else 0.0
        prior_delta_margin = float(prior.get("delta_margin", 0.0) or 0.0) if prior else 0.0
        prior_sigma_total_mult = float(prior.get("sigma_total_mult", 1.0) or 1.0) if prior else 1.0
        prior_sigma_margin_mult = float(prior.get("sigma_margin_mult", 1.0) or 1.0) if prior else 1.0
        prior_margin_scale = float(prior.get("margin_scale", 1.0) or 1.0) if prior else 1.0
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
            "spread_home",
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

    # Preserve existing spread_bins unless we're explicitly re-fitting them.
    # This avoids accidentally dropping spread-bin ATS corrections when running
    # a global calibration fit (common in daily automation).
    if (not bool(args.write_spread_bins)) and isinstance(prior_for_preserve, dict):
        try:
            prior_bins = prior_for_preserve.get("spread_bins")
            if isinstance(prior_bins, list) and len(prior_bins) > 0:
                calib["spread_bins"] = prior_bins
        except Exception:
            pass

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
        # Fit margin mean scaling (slope) first, then compute residual deltas.
        margin_scale = float(prior_margin_scale)
        if bool(args.fit_margin_scale):
            b = _fit_slope(df[center_margin_col], df["actual_margin"])
            if b is not None:
                cap = float(args.cap_margin_scale)
                cap = float(max(1.01, cap))
                margin_scale = _clamp(float(b), 1.0 / cap, cap)

        # Mean shifts (residual-on-current outputs); optionally accumulate on top of any prior calibration.
        delta_total_resid = float(np.mean((df["actual_total"] - df[center_total_col]).astype(float)))
        delta_margin_resid = float(np.mean((df["actual_margin"] - (margin_scale * df[center_margin_col])).astype(float)))
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
                z_margin = ((dm["actual_margin"] - (margin_scale * dm[center_margin_col] + delta_margin_resid)) / dm["sigma_margin"]).astype(float)
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
            "margin_scale": float(margin_scale),
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

        # Optional: abs(predicted margin) bucket calibration (winner/margin)
        if args.write_margin_abs_bins:
            try:
                edges = [float(x.strip()) for x in str(args.margin_abs_bins).split(",") if str(x).strip()]
                edges = sorted(set(edges))
            except Exception:
                edges = []
            if len(edges) >= 2:
                bins_out: list[dict] = []
                abs_center = pd.to_numeric(df[center_margin_col], errors="coerce").abs()
                for lo, hi in zip(edges[:-1], edges[1:]):
                    try:
                        m = (abs_center.notna()) & (abs_center >= float(lo)) & (abs_center < float(hi))
                        dfi = df[m].copy()
                    except Exception:
                        continue
                    if len(dfi) < int(args.min_games_bin):
                        continue

                    # Bin residuals vs same center+scale used for global.
                    dm_resid = float(np.mean((dfi["actual_margin"] - (margin_scale * dfi[center_margin_col])).astype(float)))
                    shrink = float(len(dfi) / (len(dfi) + 200.0))
                    shrink = float(np.clip(shrink, 0.0, 1.0))
                    delta_add = float(shrink * (dm_resid - delta_margin_resid))
                    delta_add = _clamp_abs(delta_add, 12.0)

                    # Sigma adjustment ratio relative to global update.
                    sigma_mult_mult = 1.0
                    if "sigma_margin" in dfi.columns:
                        dm2 = dfi.dropna(subset=["sigma_margin"]).copy()
                        dm2 = dm2[dm2["sigma_margin"] > 0]
                        if len(dm2) > 0:
                            z = ((dm2["actual_margin"] - (margin_scale * dm2[center_margin_col] + dm_resid)) / dm2["sigma_margin"]).astype(float)
                            z = z.replace([np.inf, -np.inf], np.nan).dropna()
                            if len(z) > 0:
                                q = float(np.quantile(np.abs(z), 0.80))
                                mult_upd = float(max(0.5, min(5.0, q / Z_80)))
                            else:
                                mult_upd = 1.0
                            try:
                                ratio = float(mult_upd) / float(sigma_margin_mult_upd if sigma_margin_mult_upd else 1.0)
                            except Exception:
                                ratio = 1.0
                            if not np.isfinite(ratio) or ratio <= 0:
                                ratio = 1.0
                            sigma_mult_mult = float(ratio ** shrink)
                            sigma_mult_mult = float(np.clip(sigma_mult_mult, 0.75, 1.35))

                    # Optional: bin-specific slope ratio (kept conservative).
                    margin_scale_mult = 1.0
                    b_bin = _fit_slope(dfi[center_margin_col], dfi["actual_margin"])
                    if b_bin is not None and float(margin_scale) != 0:
                        r = float(b_bin) / float(margin_scale)
                        if np.isfinite(r) and r > 0:
                            margin_scale_mult = float((r ** shrink))
                            margin_scale_mult = float(np.clip(margin_scale_mult, 0.85, 1.15))

                    bins_out.append(
                        {
                            "min": float(lo),
                            "max": float(hi),
                            "n_games": int(len(dfi)),
                            "delta_margin_add": float(delta_add),
                            "sigma_margin_mult_mult": float(sigma_mult_mult),
                            "margin_scale_mult": float(margin_scale_mult),
                        }
                    )

                if bins_out:
                    calib["margin_abs_bins"] = bins_out

        # Optional: abs(spread_home) bucket calibration (ATS-aligned)
        if args.write_spread_abs_bins and "spread_home" in df.columns:
            try:
                edges = [float(x.strip()) for x in str(args.spread_abs_bins).split(",") if str(x).strip()]
                edges = sorted(set(edges))
            except Exception:
                edges = []
            if len(edges) >= 2:
                bins_out: list[dict] = []
                abs_sp = pd.to_numeric(df["spread_home"], errors="coerce").abs()
                for lo, hi in zip(edges[:-1], edges[1:]):
                    try:
                        m = (abs_sp.notna()) & (abs_sp >= float(lo)) & (abs_sp < float(hi))
                        dfi = df[m].copy()
                    except Exception:
                        continue
                    if len(dfi) < int(args.min_games_bin):
                        continue

                    dm_resid = float(np.mean((dfi["actual_margin"] - (margin_scale * dfi[center_margin_col])).astype(float)))
                    shrink = float(len(dfi) / (len(dfi) + 200.0))
                    shrink = float(np.clip(shrink, 0.0, 1.0))
                    delta_add = float(shrink * (dm_resid - delta_margin_resid))
                    delta_add = _clamp_abs(delta_add, 12.0)

                    sigma_mult_mult = 1.0
                    if "sigma_margin" in dfi.columns:
                        dm2 = dfi.dropna(subset=["sigma_margin"]).copy()
                        dm2 = dm2[dm2["sigma_margin"] > 0]
                        if len(dm2) > 0:
                            z = ((dm2["actual_margin"] - (margin_scale * dm2[center_margin_col] + dm_resid)) / dm2["sigma_margin"]).astype(float)
                            z = z.replace([np.inf, -np.inf], np.nan).dropna()
                            if len(z) > 0:
                                q = float(np.quantile(np.abs(z), 0.80))
                                mult_upd = float(max(0.5, min(5.0, q / Z_80)))
                            else:
                                mult_upd = 1.0
                            try:
                                ratio = float(mult_upd) / float(sigma_margin_mult_upd if sigma_margin_mult_upd else 1.0)
                            except Exception:
                                ratio = 1.0
                            if not np.isfinite(ratio) or ratio <= 0:
                                ratio = 1.0
                            sigma_mult_mult = float(ratio ** shrink)
                            sigma_mult_mult = float(np.clip(sigma_mult_mult, 0.75, 1.35))

                    margin_scale_mult = 1.0
                    b_bin = _fit_slope(dfi[center_margin_col], dfi["actual_margin"])
                    if b_bin is not None and float(margin_scale) != 0:
                        r = float(b_bin) / float(margin_scale)
                        if np.isfinite(r) and r > 0:
                            margin_scale_mult = float((r ** shrink))
                            margin_scale_mult = float(np.clip(margin_scale_mult, 0.85, 1.15))

                    bins_out.append(
                        {
                            "min": float(lo),
                            "max": float(hi),
                            "n_games": int(len(dfi)),
                            "delta_margin_add": float(delta_add),
                            "sigma_margin_mult_mult": float(sigma_mult_mult),
                            "margin_scale_mult": float(margin_scale_mult),
                        }
                    )

                if bins_out:
                    calib["spread_abs_bins"] = bins_out

        # Optional: signed spread_home bucket calibration (ATS-aligned)
        if args.write_spread_bins and "spread_home" in df.columns:
            try:
                edges = [float(x.strip()) for x in str(args.spread_bins).split(",") if str(x).strip()]
                edges = sorted(set(edges))
            except Exception:
                edges = []
            if len(edges) >= 2:
                bins_out: list[dict] = []
                sp = pd.to_numeric(df["spread_home"], errors="coerce")
                for lo, hi in zip(edges[:-1], edges[1:]):
                    try:
                        m = (sp.notna()) & (sp >= float(lo)) & (sp < float(hi))
                        dfi = df[m].copy()
                    except Exception:
                        continue
                    if len(dfi) < int(args.min_games_bin):
                        continue

                    shrink = float(len(dfi) / (len(dfi) + 200.0))
                    shrink = float(np.clip(shrink, 0.0, 1.0))

                    objective = str(getattr(args, "spread_bins_objective", "resid") or "resid").strip().lower()
                    if objective == "ats":
                        sh = pd.to_numeric(dfi["spread_home"], errors="coerce")
                        actual_score = pd.to_numeric(dfi["actual_margin"], errors="coerce") + sh
                        # Match backtest behavior: exclude pushes (actual_margin + spread == 0)
                        non_push = actual_score.abs() >= 1e-9

                        base_col = "mu_margin" if "mu_margin" in dfi.columns else center_margin_col
                        score0 = (margin_scale * pd.to_numeric(dfi[base_col], errors="coerce")) + float(delta_margin) + sh
                        score0 = score0.where(non_push)
                        actual_cover = (actual_score.where(non_push)) > 0
                        delta_best = _best_delta_for_ats(score0, actual_cover)
                        if delta_best is None:
                            continue
                        delta_add = float(shrink * float(delta_best))
                        delta_add = _clamp_abs(delta_add, 12.0)
                        # For sigma estimation, center on the unshrunken best delta.
                        mu_bin = (margin_scale * pd.to_numeric(dfi[base_col], errors="coerce")) + float(delta_margin) + float(delta_best)
                    else:
                        dm_resid = float(np.mean((dfi["actual_margin"] - (margin_scale * dfi[center_margin_col])).astype(float)))
                        delta_add = float(shrink * (dm_resid - delta_margin_resid))
                        delta_add = _clamp_abs(delta_add, 12.0)
                        mu_bin = (margin_scale * pd.to_numeric(dfi[center_margin_col], errors="coerce")) + float(dm_resid)

                    sigma_mult_mult = 1.0
                    if "sigma_margin" in dfi.columns:
                        # Prefer ATS-aligned dispersion fit when objective=ats.
                        if objective == "ats":
                            try:
                                sh = pd.to_numeric(dfi["spread_home"], errors="coerce")
                                actual_score = pd.to_numeric(dfi["actual_margin"], errors="coerce") + sh
                                non_push = actual_score.abs() >= 1e-9
                                base_col2 = "mu_margin" if "mu_margin" in dfi.columns else center_margin_col
                                # Use the SHRUNKEN delta that will actually be applied.
                                mean_score = (margin_scale * pd.to_numeric(dfi[base_col2], errors="coerce")) + float(delta_margin) + float(delta_add) + sh
                                actual_cover = (actual_score.where(non_push)) > 0
                                # sigma already has the global sigma_margin_mult baked in downstream; we fit a relative multiplier.
                                sigma_series = pd.to_numeric(dfi["sigma_margin"], errors="coerce") * float(sigma_margin_mult)
                                s_best = _fit_sigma_mult_for_ats_brier(mean_score.where(non_push), sigma_series.where(non_push), actual_cover)
                                if s_best is not None and np.isfinite(float(s_best)) and float(s_best) > 0:
                                    # Shrink toward 1.0 for stability.
                                    sigma_mult_mult = float(1.0 + shrink * (float(s_best) - 1.0))
                                    sigma_mult_mult = float(np.clip(sigma_mult_mult, 0.75, 1.35))
                            except Exception:
                                sigma_mult_mult = 1.0

                        # Fallback: central-coverage-based relative sigma fit.
                        if float(sigma_mult_mult) == 1.0:
                            dm2 = dfi.dropna(subset=["sigma_margin"]).copy()
                            dm2 = dm2[dm2["sigma_margin"] > 0]
                            if len(dm2) > 0:
                                mu_for_sigma = mu_bin.loc[dm2.index] if hasattr(mu_bin, "loc") else mu_bin
                                z = ((pd.to_numeric(dm2["actual_margin"], errors="coerce") - pd.to_numeric(mu_for_sigma, errors="coerce")) / pd.to_numeric(dm2["sigma_margin"], errors="coerce")).astype(float)
                                z = z.replace([np.inf, -np.inf], np.nan).dropna()
                                if len(z) > 0:
                                    q = float(np.quantile(np.abs(z), 0.80))
                                    mult_upd = float(max(0.5, min(5.0, q / Z_80)))
                                else:
                                    mult_upd = 1.0
                                try:
                                    ratio = float(mult_upd) / float(sigma_margin_mult_upd if sigma_margin_mult_upd else 1.0)
                                except Exception:
                                    ratio = 1.0
                                if not np.isfinite(ratio) or ratio <= 0:
                                    ratio = 1.0
                                sigma_mult_mult = float(ratio ** shrink)
                                sigma_mult_mult = float(np.clip(sigma_mult_mult, 0.75, 1.35))

                    margin_scale_mult = 1.0
                    b_bin = _fit_slope(dfi[center_margin_col], dfi["actual_margin"])
                    if b_bin is not None and float(margin_scale) != 0:
                        r = float(b_bin) / float(margin_scale)
                        if np.isfinite(r) and r > 0:
                            margin_scale_mult = float((r ** shrink))
                            margin_scale_mult = float(np.clip(margin_scale_mult, 0.85, 1.15))

                    bins_out.append(
                        {
                            "min": float(lo),
                            "max": float(hi),
                            "n_games": int(len(dfi)),
                            "delta_margin_add": float(delta_add),
                            "sigma_margin_mult_mult": float(sigma_mult_mult),
                            "margin_scale_mult": float(margin_scale_mult),
                        }
                    )

                if bins_out:
                    calib["spread_bins"] = bins_out

    # Estimate global rho from historical games by correlating centered per-team residuals.
    # Use the calibrated mean shifts so we don't bake in simple bias.
    if (not args.fit_1h_only) and "home_score" in df.columns and "away_score" in df.columns:
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        mu_total_cal = pd.to_numeric(df["mu_total"], errors="coerce") + float(delta_total)
        try:
            ms = float(calib.get("margin_scale", 1.0) or 1.0)
        except Exception:
            ms = 1.0
        mu_margin_cal = (pd.to_numeric(df["mu_margin"], errors="coerce") * float(ms)) + float(delta_margin)
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
