"""Evaluate simulation outputs vs finalized results.

Inputs (per date):
  - outputs/sim_quantiles_<date>.csv
  - outputs/daily_results/results_<date>.csv

Outputs:
  - outputs/metrics/sim_metrics_<date>.csv (per-game)
  - outputs/metrics/sim_metrics_by_date.csv (aggregated)

Metrics:
  - Quantile scoring (pinball-based CRPS surrogate) + 80% interval coverage for totals/margins
  - Brier + log loss for:
      * full-game OU (p_over_market vs closing_total)
      * full-game ATS (p_cover_home vs closing_spread_home)
      * full-game winner (p_home_win vs home win)

This is intentionally lightweight (no extra deps) and robust to missing columns.
"""

from __future__ import annotations

import argparse
import math
import json
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "daily_results"


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(str(s))


def _iter_dates(start: str, end: str) -> list[str]:
    s = _parse_date(start)
    e = _parse_date(end)
    if e < s:
        raise ValueError(f"end < start: {start}..{end}")
    out: list[str] = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d = d + timedelta(days=1)
    return out


def _safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _norm_game_id(df: pd.DataFrame) -> pd.DataFrame:
    if "game_id" in df.columns:
        df = df.copy()
        df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    u = y_true - y_pred
    return float(np.nanmean(np.maximum(q * u, (q - 1) * u)))


def _crps_from_q10_q50_q90(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> float:
    # Approximate CRPS by integrating pinball loss over a small quantile grid.
    # Using {0.1, 0.5, 0.9} only keeps this cheap/robust.
    return float(
        (1.0 / 3.0)
        * (
            _pinball_loss(y, q10, 0.1)
            + _pinball_loss(y, q50, 0.5)
            + _pinball_loss(y, q90, 0.9)
        )
    )


def _clip_prob(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    return p.clip(lower=1e-6, upper=1 - 1e-6)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    # binary log loss
    return float(np.nanmean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def evaluate_date(date: str, outputs_dir: Path = OUT) -> dict:
    sim_path = outputs_dir / f"sim_quantiles_{date}.csv"
    res_path = outputs_dir / "daily_results" / f"results_{date}.csv"
    meta_path = outputs_dir / f"sim_meta_{date}.json"

    sim = _norm_game_id(_safe_read_csv(sim_path))
    res = _norm_game_id(_safe_read_csv(res_path))

    frac_missing_sigma_margin = None
    if not sim.empty and "sigma_margin" in sim.columns:
        s = pd.to_numeric(sim["sigma_margin"], errors="coerce")
        try:
            frac_missing_sigma_margin = float(s.isna().mean())
        except Exception:
            frac_missing_sigma_margin = None

    meta = _safe_read_json(meta_path)
    rho_effective = meta.get("rho_effective")
    try:
        rho_effective = float(rho_effective) if rho_effective is not None else None
    except Exception:
        rho_effective = None
    rho_raw = meta.get("rho")
    try:
        rho_raw = float(rho_raw) if rho_raw is not None else None
    except Exception:
        rho_raw = None

    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if sim.empty or res.empty:
        return {"date": date, "ok": False, "reason": "missing_inputs"}

    # Canonical actuals
    if "actual_total" not in res.columns and {"home_score", "away_score"}.issubset(res.columns):
        res = res.copy()
        res["actual_total"] = pd.to_numeric(res["home_score"], errors="coerce") + pd.to_numeric(res["away_score"], errors="coerce")
    if "actual_margin" not in res.columns and {"home_score", "away_score"}.issubset(res.columns):
        res = res.copy()
        res["actual_margin"] = pd.to_numeric(res["home_score"], errors="coerce") - pd.to_numeric(res["away_score"], errors="coerce")

    df = res.merge(sim, on=["game_id"], how="left", suffixes=("", "_sim"))
    df["date"] = date

    # Quantile scoring
    actual_total = pd.to_numeric(df.get("actual_total"), errors="coerce").to_numpy()
    actual_margin = pd.to_numeric(df.get("actual_margin"), errors="coerce").to_numpy()

    q10_t = pd.to_numeric(df.get("q10_total"), errors="coerce").to_numpy()
    q50_t = pd.to_numeric(df.get("q50_total"), errors="coerce").to_numpy()
    q90_t = pd.to_numeric(df.get("q90_total"), errors="coerce").to_numpy()

    q10_m = pd.to_numeric(df.get("q10_margin"), errors="coerce").to_numpy()
    q50_m = pd.to_numeric(df.get("q50_margin"), errors="coerce").to_numpy()
    q90_m = pd.to_numeric(df.get("q90_margin"), errors="coerce").to_numpy()

    cov80_total = np.nan
    cov80_margin = np.nan
    try:
        cov80_total = float(np.nanmean((actual_total >= q10_t) & (actual_total <= q90_t)))
    except Exception:
        pass
    try:
        cov80_margin = float(np.nanmean((actual_margin >= q10_m) & (actual_margin <= q90_m)))
    except Exception:
        pass

    crps_total = np.nan
    crps_margin = np.nan
    try:
        crps_total = _crps_from_q10_q50_q90(actual_total, q10_t, q50_t, q90_t)
    except Exception:
        pass
    try:
        crps_margin = _crps_from_q10_q50_q90(actual_margin, q10_m, q50_m, q90_m)
    except Exception:
        pass

    # OU probability
    closing_total = pd.to_numeric(df.get("closing_total", df.get("market_total")), errors="coerce")
    p_over = df.get("p_over_market")
    brier_over = logloss_over = np.nan
    if p_over is not None and closing_total.notna().any() and df.get("actual_total") is not None:
        y_over = (pd.to_numeric(df.get("actual_total"), errors="coerce") > closing_total).astype(float)
        p = _clip_prob(pd.Series(p_over))
        mask = y_over.notna() & p.notna() & closing_total.notna()
        if mask.any():
            y = y_over[mask].to_numpy(dtype=float)
            pp = p[mask].to_numpy(dtype=float)
            brier_over = _brier(y, pp)
            logloss_over = _log_loss(y, pp)

    # ATS probability
    closing_spread_home = pd.to_numeric(df.get("closing_spread_home", df.get("spread_home")), errors="coerce")
    p_cover_home = df.get("p_cover_home")
    brier_cover = logloss_cover = np.nan
    if p_cover_home is not None and closing_spread_home.notna().any() and df.get("actual_margin") is not None:
        y_cover = (pd.to_numeric(df.get("actual_margin"), errors="coerce") + closing_spread_home > 0).astype(float)
        p = _clip_prob(pd.Series(p_cover_home))
        mask = y_cover.notna() & p.notna() & closing_spread_home.notna()
        if mask.any():
            y = y_cover[mask].to_numpy(dtype=float)
            pp = p[mask].to_numpy(dtype=float)
            brier_cover = _brier(y, pp)
            logloss_cover = _log_loss(y, pp)

    # Winner probability
    p_home_win = df.get("p_home_win")
    brier_win = logloss_win = np.nan
    if p_home_win is not None and df.get("actual_margin") is not None:
        y_win = (pd.to_numeric(df.get("actual_margin"), errors="coerce") > 0).astype(float)
        p = _clip_prob(pd.Series(p_home_win))
        mask = y_win.notna() & p.notna()
        if mask.any():
            y = y_win[mask].to_numpy(dtype=float)
            pp = p[mask].to_numpy(dtype=float)
            brier_win = _brier(y, pp)
            logloss_win = _log_loss(y, pp)

    # Write per-game file for inspection
    per_game_path = metrics_dir / f"sim_metrics_{date}.csv"
    df.to_csv(per_game_path, index=False)

    try:
        per_game_csv = str(per_game_path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        per_game_csv = str(per_game_path).replace("\\", "/")

    summary = {
        "date": date,
        "ok": True,
        "n": int(len(df)),
        "frac_missing_sigma_margin": frac_missing_sigma_margin,
        "rho": rho_raw,
        "rho_effective": rho_effective,
        "crps_total": float(crps_total) if np.isfinite(crps_total) else None,
        "crps_margin": float(crps_margin) if np.isfinite(crps_margin) else None,
        "cov80_total": float(cov80_total) if np.isfinite(cov80_total) else None,
        "cov80_margin": float(cov80_margin) if np.isfinite(cov80_margin) else None,
        "brier_over": float(brier_over) if np.isfinite(brier_over) else None,
        "logloss_over": float(logloss_over) if np.isfinite(logloss_over) else None,
        "brier_cover": float(brier_cover) if np.isfinite(brier_cover) else None,
        "logloss_cover": float(logloss_cover) if np.isfinite(logloss_cover) else None,
        "brier_win": float(brier_win) if np.isfinite(brier_win) else None,
        "logloss_win": float(logloss_win) if np.isfinite(logloss_win) else None,
        "per_game_csv": per_game_csv,
    }

    # Append/update aggregate-by-date file
    by_date_path = metrics_dir / "sim_metrics_by_date.csv"
    row = pd.DataFrame([
        {
            "date": date,
            "n": summary["n"],
            "frac_missing_sigma_margin": summary["frac_missing_sigma_margin"],
            "rho": summary["rho"],
            "rho_effective": summary["rho_effective"],
            "crps_total": summary["crps_total"],
            "crps_margin": summary["crps_margin"],
            "cov80_total": summary["cov80_total"],
            "cov80_margin": summary["cov80_margin"],
            "brier_over": summary["brier_over"],
            "logloss_over": summary["logloss_over"],
            "brier_cover": summary["brier_cover"],
            "logloss_cover": summary["logloss_cover"],
            "brier_win": summary["brier_win"],
            "logloss_win": summary["logloss_win"],
        }
    ])
    if by_date_path.exists():
        old = _safe_read_csv(by_date_path)
        if not old.empty and "date" in old.columns:
            old["date"] = old["date"].astype(str)
            old = old[old["date"] != date]
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row
    out = out.sort_values("date")
    out.to_csv(by_date_path, index=False)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", help="Date to evaluate (YYYY-MM-DD)")
    ap.add_argument("--start", help="Start date (YYYY-MM-DD) for range evaluation")
    ap.add_argument("--end", help="End date (YYYY-MM-DD) for range evaluation")
    ap.add_argument("--outputs", default=str(OUT), help="Outputs directory (default: outputs)")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    dates: list[str]
    if args.date:
        dates = [args.date]
    else:
        if not args.start or not args.end:
            ap.error("Provide either DATE or both --start and --end")
        dates = _iter_dates(args.start, args.end)

    ok_any = False
    summaries: list[dict] = []
    for d in dates:
        s = evaluate_date(d, outputs_dir=outputs_dir)
        summaries.append(s)
        ok_any = ok_any or bool(s.get("ok"))

    if len(summaries) == 1:
        print(summaries[0])
        return 0 if summaries[0].get("ok") else 2

    # Compact range summary
    ok_count = sum(1 for s in summaries if s.get("ok"))
    print({
        "ok": ok_count,
        "total": len(summaries),
        "start": dates[0],
        "end": dates[-1],
        "outputs": str(outputs_dir),
    })
    return 0 if ok_any else 2


if __name__ == "__main__":
    raise SystemExit(main())
