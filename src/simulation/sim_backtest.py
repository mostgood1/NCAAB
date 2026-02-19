from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .game_sim import run_simulations_for_date


def _parse_date(s: str | None) -> Optional[str]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s).strip()).isoformat()
    except Exception:
        return None


def _norm_gid(v: Any) -> str:
    try:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s
    except Exception:
        try:
            return str(int(v))
        except Exception:
            return str(v)


def _pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    y = np.asarray(y, dtype=float)
    qhat = np.asarray(qhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(qhat)
    if not mask.any():
        return float("nan")
    e = y[mask] - qhat[mask]
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def _crps_from_q10_q50_q90(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, grid_n: int = 101) -> dict[str, Any]:
    """Approximate CRPS via the quantile representation using a piecewise-linear quantile function.

    Uses identity: CRPS(F, y) = 2 * \int_0^1 rho_tau(y - Q(tau)) dtau.
    We approximate Q(tau) with two linear segments meeting at q50 and integrate numerically.
    """
    y = np.asarray(y, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    mask = np.isfinite(y) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
    if not mask.any():
        return {"count": 0, "crps": None, "grid_n": int(grid_n)}

    yy = y[mask]
    a = q10[mask]
    b = q50[mask]
    c = q90[mask]

    # Enforce monotonicity defensively.
    lo = np.minimum(a, np.minimum(b, c))
    hi = np.maximum(a, np.maximum(b, c))
    mid = np.clip(b, lo, hi)
    a = np.minimum(lo, mid)
    c = np.maximum(hi, mid)
    b = mid

    tau = np.linspace(0.0, 1.0, int(max(11, grid_n)), dtype=float)
    tau = np.clip(tau, 0.0, 1.0)
    tau2 = tau[None, :]

    # Slopes for left/right segments (0.1->0.5 and 0.5->0.9). Extrapolate linearly outside.
    sL = (b - a) / 0.4
    sR = (c - b) / 0.4

    # Build Q(tau) with continuity at q50.
    # For tau <= 0.5 use left segment anchored at (0.5, q50).
    # For tau > 0.5 use right segment anchored at (0.5, q50).
    Q = np.where(
        tau2 <= 0.5,
        b[:, None] + (tau2 - 0.5) * sL[:, None],
        b[:, None] + (tau2 - 0.5) * sR[:, None],
    )

    e = yy[:, None] - Q
    pin = np.maximum(tau2 * e, (tau2 - 1.0) * e)
    # Trapezoidal integration over tau for each row, then average.
    try:
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        crps_per = 2.0 * trapz(pin, tau, axis=1)
    except Exception:
        # Extremely defensive fallback: simple average spacing.
        dt = float(1.0 / float(max(tau.size - 1, 1)))
        crps_per = 2.0 * dt * np.sum(pin, axis=1)
    return {"count": int(mask.sum()), "crps": float(np.mean(crps_per)), "grid_n": int(tau.size)}


def _mae_rmse(y: np.ndarray, yhat: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    err = yhat[mask] - y[mask]
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
    }


def _brier(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return {"count": 0, "brier": None}
    pp = np.clip(p[mask], 1e-6, 1.0 - 1e-6)
    return {"count": int(mask.sum()), "brier": float(np.mean(np.square(pp - y[mask])))}


def _logloss(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return {"count": 0, "logloss": None}
    pp = np.clip(p[mask], 1e-6, 1.0 - 1e-6)
    return {"count": int(mask.sum()), "logloss": float(np.mean(-(y[mask] * np.log(pp) + (1.0 - y[mask]) * np.log(1.0 - pp))))}


def _resolve_dates_from_results(out_dir: Path, start: Optional[str], end: Optional[str], recent: Optional[int]) -> list[str]:
    dr = out_dir / "daily_results"
    if not dr.exists():
        return []

    dates: list[str] = []
    for p in sorted(dr.glob("results_*.csv")):
        token = p.stem.replace("results_", "")
        d = _parse_date(token)
        if d:
            dates.append(d)

    dates = sorted(set(dates))

    if recent and recent > 0:
        return dates[-int(recent) :]

    if start and end:
        if start > end:
            start, end = end, start
        return [d for d in dates if start <= d <= end]

    if start and not end:
        return [d for d in dates if d == start]

    return dates


def _load_results_for_date(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / "daily_results" / f"results_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)

    # Filter out placeholder rows that are not truly finalized.
    # Some results files can contain 0/0 scores (missing finals); those should not be treated as outcomes.
    hs = pd.to_numeric(df.get("home_score"), errors="coerce") if "home_score" in df.columns else None
    as_ = pd.to_numeric(df.get("away_score"), errors="coerce") if "away_score" in df.columns else None
    if hs is not None and as_ is not None:
        final_mask = hs.notna() & as_.notna() & ((hs > 0) | (as_ > 0))
        df = df.loc[final_mask].copy()
        # Derive actuals from final scores for consistency.
        df["actual_total"] = (hs.loc[final_mask].to_numpy() + as_.loc[final_mask].to_numpy())
        df["actual_margin"] = (hs.loc[final_mask].to_numpy() - as_.loc[final_mask].to_numpy())
    else:
        at = pd.to_numeric(df.get("actual_total"), errors="coerce") if "actual_total" in df.columns else None
        if at is not None:
            df = df.loc[at.notna() & (at > 0)].copy()
    return df


def _load_sim_for_date(out_dir: Path, date: str, sim_quantiles_prefix: str = "sim_quantiles_") -> pd.DataFrame:
    p = out_dir / f"{sim_quantiles_prefix}{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df = df[df["date"].astype(str) == str(date)]
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


def _join_sim_results(sim: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    if sim.empty or res.empty:
        return pd.DataFrame()

    sim = sim.copy()
    res = res.copy()

    def _dedup(df: pd.DataFrame) -> pd.DataFrame:
        if "game_id" in df.columns:
            gid = df["game_id"].astype(str).fillna("")
            nonempty = gid.str.len().gt(0)
            df_gid = df.loc[nonempty].drop_duplicates(subset=["game_id"]) if nonempty.any() else df.iloc[0:0]
            df_nogid = df.loc[~nonempty]
            if not df_nogid.empty and {"home_team", "away_team"}.issubset(df_nogid.columns):
                df_nogid = df_nogid.drop_duplicates(subset=["home_team", "away_team"])
            return pd.concat([df_gid, df_nogid], ignore_index=True)
        if {"home_team", "away_team"}.issubset(df.columns):
            return df.drop_duplicates(subset=["home_team", "away_team"])
        return df

    sim = _dedup(sim)
    res = _dedup(res)

    # Primary join: game_id when present.
    merged_parts: list[pd.DataFrame] = []
    if "game_id" in sim.columns and "game_id" in res.columns:
        gid = sim["game_id"].astype(str).fillna("")
        sim_gid = sim.loc[gid.str.len().gt(0)].copy()
        sim_nogid = sim.loc[gid.str.len().eq(0)].copy()

        if not sim_gid.empty:
            merged_parts.append(sim_gid.merge(res, on=["game_id"], how="left", suffixes=("", "_res")))
        # Secondary join for missing ids: fall back to home/away team.
        if not sim_nogid.empty and {"home_team", "away_team"}.issubset(sim_nogid.columns) and {"home_team", "away_team"}.issubset(res.columns):
            merged_parts.append(sim_nogid.merge(res, on=["home_team", "away_team"], how="left", suffixes=("", "_res")))
    else:
        if not {"home_team", "away_team"}.issubset(sim.columns) or not {"home_team", "away_team"}.issubset(res.columns):
            return pd.DataFrame()
        merged_parts.append(sim.merge(res, on=["home_team", "away_team"], how="left", suffixes=("", "_res")))

    if not merged_parts:
        return pd.DataFrame()
    return pd.concat(merged_parts, ignore_index=True)


def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["actual_total"] = pd.to_numeric(out.get("actual_total"), errors="coerce")
    out["actual_margin"] = pd.to_numeric(out.get("actual_margin"), errors="coerce")

    out["market_total"] = pd.to_numeric(out.get("market_total"), errors="coerce")
    out["spread_home"] = pd.to_numeric(out.get("spread_home"), errors="coerce")

    # 1H actuals when available
    out["actual_total_1h"] = pd.to_numeric(out.get("actual_total_1h"), errors="coerce")
    if "home_score_1h" in out.columns and "away_score_1h" in out.columns:
        hs1 = pd.to_numeric(out.get("home_score_1h"), errors="coerce")
        as1 = pd.to_numeric(out.get("away_score_1h"), errors="coerce")
        out["actual_margin_1h"] = hs1 - as1
    else:
        out["actual_margin_1h"] = np.nan

    # Binary labels (drop pushes)
    def _over_label(actual: pd.Series, line: pd.Series) -> pd.Series:
        y = pd.Series(np.nan, index=actual.index, dtype=float)
        mask = actual.notna() & line.notna()
        y.loc[mask & (actual > line)] = 1.0
        y.loc[mask & (actual < line)] = 0.0
        return y

    def _cover_home_label(margin: pd.Series, spread_home: pd.Series) -> pd.Series:
        # Home covers if home_score + spread_home > away_score
        y = pd.Series(np.nan, index=margin.index, dtype=float)
        mask = margin.notna() & spread_home.notna()
        # push if equality
        lhs = margin + spread_home
        y.loc[mask & (lhs > 0)] = 1.0
        y.loc[mask & (lhs < 0)] = 0.0
        return y

    def _win_home_label(margin: pd.Series) -> pd.Series:
        y = pd.Series(np.nan, index=margin.index, dtype=float)
        mask = margin.notna()
        y.loc[mask & (margin > 0)] = 1.0
        y.loc[mask & (margin < 0)] = 0.0
        return y

    out["y_over"] = _over_label(out["actual_total"], out["market_total"])
    out["y_cover_home"] = _cover_home_label(out["actual_margin"], out["spread_home"])
    out["y_home_win"] = _win_home_label(out["actual_margin"])

    # 1H market labels if those lines exist
    out["market_total_1h"] = pd.to_numeric(out.get("market_total_1h"), errors="coerce")
    out["spread_home_1h"] = pd.to_numeric(out.get("spread_home_1h"), errors="coerce")
    out["y_over_1h"] = _over_label(out["actual_total_1h"], out["market_total_1h"])
    out["y_cover_home_1h"] = _cover_home_label(out["actual_margin_1h"], out["spread_home_1h"])

    return out


@dataclasses.dataclass
class SimBacktestConfig:
    out_dir: Path
    start: Optional[str] = None
    end: Optional[str] = None
    recent: Optional[int] = None
    engine: str = "events"
    samples: int = 5000
    rho: float = 0.25
    recompute: bool = False
    out_prefix: str = "sim_engine"
    sim_quantiles_prefix: str = "sim_quantiles_"


def run_sim_backtest(cfg: SimBacktestConfig) -> dict[str, Any]:
    cfg.out_dir = Path(cfg.out_dir)
    start = _parse_date(cfg.start)
    end = _parse_date(cfg.end)

    dates = _resolve_dates_from_results(cfg.out_dir, start, end, cfg.recent)
    if not dates:
        return {"error": "No results_*.csv dates found", "out_dir": str(cfg.out_dir)}

    bt_dir = cfg.out_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    per_date_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for d in dates:
        sim_path = cfg.out_dir / f"{cfg.sim_quantiles_prefix}{d}.csv"
        if cfg.recompute or (not sim_path.exists()):
            try:
                run_simulations_for_date(
                    cfg.out_dir,
                    d,
                    samples=int(cfg.samples),
                    rho=float(cfg.rho),
                    engine=str(cfg.engine),
                    quantiles_out_prefix=str(cfg.sim_quantiles_prefix),
                )
            except Exception as e:
                skipped.append({"date": d, "stage": "recompute", "error": str(e)})

        try:
            sim = _load_sim_for_date(cfg.out_dir, d, sim_quantiles_prefix=str(cfg.sim_quantiles_prefix))
            res = _load_results_for_date(cfg.out_dir, d)
            merged = _join_sim_results(sim, res)
        except Exception as e:
            skipped.append({"date": d, "stage": "load/join", "error": str(e)})
            per_date_rows.append({"date": d, "n": 0})
            continue
        if merged.empty:
            per_date_rows.append({"date": d, "n": 0})
            continue

        merged = _build_labels(merged)

        # Basic continuous errors
        mu_total = pd.to_numeric(merged.get("mu_total"), errors="coerce")
        mu_margin = pd.to_numeric(merged.get("mu_margin"), errors="coerce")
        mu_total_1h = pd.to_numeric(merged.get("mu_total_1h"), errors="coerce")
        mu_margin_1h = pd.to_numeric(merged.get("mu_margin_1h"), errors="coerce")

        merged["err_total"] = mu_total - merged["actual_total"]
        merged["err_margin"] = mu_margin - merged["actual_margin"]
        merged["err_total_1h"] = mu_total_1h - merged["actual_total_1h"]
        merged["err_margin_1h"] = mu_margin_1h - merged["actual_margin_1h"]

        merged["date"] = d
        all_rows.append(merged)

        # Per-date summary
        per_date_rows.append(
            {
                "date": d,
                "n": int(len(merged)),
                "mae_total": _mae_rmse(merged["actual_total"].to_numpy(), mu_total.to_numpy()).get("mae"),
                "mae_margin": _mae_rmse(merged["actual_margin"].to_numpy(), mu_margin.to_numpy()).get("mae"),
            }
        )

    if not all_rows:
        return {"error": "No joins produced any rows", "dates": dates}

    df = pd.concat(all_rows, ignore_index=True)

    finals_mask = pd.to_numeric(df.get("actual_total"), errors="coerce").notna() & (pd.to_numeric(df.get("actual_total"), errors="coerce") > 0)

    # Overall metrics
    metrics: dict[str, Any] = {
        "range": {"start": dates[0], "end": dates[-1], "n_dates": int(len(dates))},
        "engine": cfg.engine,
        "samples": int(cfg.samples),
        "rho": float(cfg.rho),
        "finals_rows": int(finals_mask.sum()) if hasattr(finals_mask, "sum") else None,
    }
    if skipped:
        metrics["skipped"] = {
            "n": int(len(skipped)),
            "details": skipped,
        }

    metrics["totals"] = _mae_rmse(df["actual_total"].to_numpy(), pd.to_numeric(df.get("mu_total"), errors="coerce").to_numpy())
    metrics["margins"] = _mae_rmse(df["actual_margin"].to_numpy(), pd.to_numeric(df.get("mu_margin"), errors="coerce").to_numpy())
    metrics["totals_1h"] = _mae_rmse(df["actual_total_1h"].to_numpy(), pd.to_numeric(df.get("mu_total_1h"), errors="coerce").to_numpy())
    metrics["margins_1h"] = _mae_rmse(df["actual_margin_1h"].to_numpy(), pd.to_numeric(df.get("mu_margin_1h"), errors="coerce").to_numpy())

    # Target-quantile diagnostics (only if simulator emitted these flags)
    try:
        for col, key in [
            ("target_quantiles_total_applied", "total"),
            ("target_quantiles_margin_applied", "margin"),
            ("target_quantiles_total_1h_applied", "total_1h"),
            ("target_quantiles_margin_1h_applied", "margin_1h"),
        ]:
            if col not in df.columns:
                continue
            applied = df[col].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
            applied_finals = applied & finals_mask
            metrics.setdefault("targeting", {})[key] = {
                "count": int(applied_finals.sum()) if hasattr(applied_finals, "sum") else None,
                "rate": float(applied_finals.mean()) if len(df.loc[finals_mask]) else None,
            }
    except Exception:
        pass

    # Pinball losses (quantiles)
    for target, actual_col, q_cols in [
        ("total", "actual_total", ["q10_total", "q50_total", "q90_total"]),
        ("margin", "actual_margin", ["q10_margin", "q50_margin", "q90_margin"]),
        ("total_1h", "actual_total_1h", ["q10_total_1h", "q50_total_1h", "q90_total_1h"]),
        ("margin_1h", "actual_margin_1h", ["q10_margin_1h", "q50_margin_1h", "q90_margin_1h"]),
    ]:
        actual = pd.to_numeric(df.get(actual_col), errors="coerce").to_numpy()
        q10 = pd.to_numeric(df.get(q_cols[0]), errors="coerce").to_numpy()
        q50 = pd.to_numeric(df.get(q_cols[1]), errors="coerce").to_numpy()
        q90 = pd.to_numeric(df.get(q_cols[2]), errors="coerce").to_numpy()
        metrics[f"pinball_{target}"] = {
            "q10": _pinball(actual, q10, 0.10),
            "q50": _pinball(actual, q50, 0.50),
            "q90": _pinball(actual, q90, 0.90),
        }

        # CRPS (quantile-function approximation from q10/q50/q90)
        metrics[f"crps_{target}"] = _crps_from_q10_q50_q90(actual, q10, q50, q90)

    # Probability scoring
    metrics["probs"] = {
        "over": {**_brier(df["y_over"].to_numpy(), pd.to_numeric(df.get("p_over_market"), errors="coerce").to_numpy()),
                 **_logloss(df["y_over"].to_numpy(), pd.to_numeric(df.get("p_over_market"), errors="coerce").to_numpy())},
        "cover_home": {**_brier(df["y_cover_home"].to_numpy(), pd.to_numeric(df.get("p_cover_home"), errors="coerce").to_numpy()),
                       **_logloss(df["y_cover_home"].to_numpy(), pd.to_numeric(df.get("p_cover_home"), errors="coerce").to_numpy())},
        "home_win": {**_brier(df["y_home_win"].to_numpy(), pd.to_numeric(df.get("p_home_win"), errors="coerce").to_numpy()),
                     **_logloss(df["y_home_win"].to_numpy(), pd.to_numeric(df.get("p_home_win"), errors="coerce").to_numpy())},
        "over_1h": {**_brier(df["y_over_1h"].to_numpy(), pd.to_numeric(df.get("p_over_market_1h"), errors="coerce").to_numpy()),
                    **_logloss(df["y_over_1h"].to_numpy(), pd.to_numeric(df.get("p_over_market_1h"), errors="coerce").to_numpy())},
        "cover_home_1h": {**_brier(df["y_cover_home_1h"].to_numpy(), pd.to_numeric(df.get("p_cover_home_1h"), errors="coerce").to_numpy()),
                          **_logloss(df["y_cover_home_1h"].to_numpy(), pd.to_numeric(df.get("p_cover_home_1h"), errors="coerce").to_numpy())},
    }

    out_stem = f"{cfg.out_prefix}_{dates[0]}_{dates[-1]}"
    out_csv = bt_dir / f"{out_stem}.csv"
    out_json = bt_dir / f"{out_stem}_summary.json"
    out_per_date = bt_dir / f"{out_stem}_per_date.csv"
    out_worst = bt_dir / f"{out_stem}_worst.csv"

    # Keep output size reasonable: drop raw score duplicates if present
    df.to_csv(out_csv, index=False, na_rep="")
    pd.DataFrame(per_date_rows).to_csv(out_per_date, index=False, na_rep="")
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # Quick diagnostic: top worst misses among finalized games
    try:
        dff = df.loc[finals_mask].copy() if len(df) else df
        if not dff.empty:
            cols = [c for c in [
                "date",
                "game_id",
                "home_team",
                "away_team",
                "actual_total",
                "mu_total",
                "err_total",
                "market_total",
                "actual_margin",
                "mu_margin",
                "err_margin",
                "spread_home",
            ] if c in dff.columns]

            worst_total = dff.loc[pd.to_numeric(dff.get("err_total"), errors="coerce").abs().sort_values(ascending=False).head(50).index, cols]
            worst_margin = dff.loc[pd.to_numeric(dff.get("err_margin"), errors="coerce").abs().sort_values(ascending=False).head(50).index, cols]
            out_diag = pd.concat(
                [
                    worst_total.assign(kind="total"),
                    worst_margin.assign(kind="margin"),
                ],
                ignore_index=True,
            )
            out_diag.to_csv(out_worst, index=False, na_rep="")
    except Exception:
        pass

    return {
        "wrote": {
            "per_game": str(out_csv),
            "per_date": str(out_per_date),
            "summary": str(out_json),
            "worst": str(out_worst),
        },
        "n_rows": int(len(df)),
        "n_dates": int(len(dates)),
    }
