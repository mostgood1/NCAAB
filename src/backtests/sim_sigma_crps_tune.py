from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * (1.0 + np.asarray([math.erf(float(v) * inv_sqrt2) for v in z], dtype=float))


def _crps_normal_vec(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector CRPS for Normal(mu, sigma) vs y. Returns NaN where invalid."""
    out = np.full_like(mu, fill_value=np.nan, dtype=float)
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y) & (sigma > 1e-9)
    if not np.any(mask):
        return out
    z = (y[mask] - mu[mask]) / sigma[mask]
    phi = _norm_pdf(z)
    Phi = _norm_cdf(z)
    out[mask] = sigma[mask] * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return out


@dataclass(frozen=True)
class TuneSigmaCRPSConfig:
    out_dir: Path
    start: str
    end: str
    sim_quantiles_prefix: str = "sim_quantiles_"
    interval_actuals_prefix: str = "interval_actuals_5min_"
    use_regulation_40: bool = True
    grid_min: float = 0.6
    grid_max: float = 1.8
    grid_steps: int = 25
    min_games: int = 200
    # Baseline sigma multipliers already embedded in outputs/sim_quantiles_<date>.csv.
    # If None, these are loaded from outputs/sim_calibration.json.
    baseline_sigma_total_mult: Optional[float] = None
    baseline_sigma_margin_mult: Optional[float] = None


def _load_baseline_sigma_multipliers(out_dir: Path) -> tuple[float, float]:
    """Return (sigma_total_mult, sigma_margin_mult) from outputs/sim_calibration.json if present."""
    try:
        p = Path(out_dir) / "sim_calibration.json"
        if not p.exists():
            return (1.0, 1.0)
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return (1.0, 1.0)
        st = float(obj.get("sigma_total_mult", 1.0) or 1.0)
        sm = float(obj.get("sigma_margin_mult", 1.0) or 1.0)
        if not math.isfinite(st) or st <= 0:
            st = 1.0
        if not math.isfinite(sm) or sm <= 0:
            sm = 1.0
        return (st, sm)
    except Exception:
        return (1.0, 1.0)


def _date_range(start_iso: str, end_iso: str) -> list[str]:
    s = dt.date.fromisoformat(start_iso)
    e = dt.date.fromisoformat(end_iso)
    cur = s
    one = dt.timedelta(days=1)
    out: list[str] = []
    while cur <= e:
        out.append(cur.isoformat())
        cur += one
    return out


def _load_quantiles(out_dir: Path, date_iso: str, prefix: str) -> pd.DataFrame:
    path = Path(out_dir) / f"{prefix}{date_iso}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "game_id" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)
    for c in ["mu_total", "sigma_total", "mu_margin", "sigma_margin"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_interval_actuals(out_dir: Path, date_iso: str, prefix: str) -> pd.DataFrame:
    path = Path(out_dir) / f"{prefix}{date_iso}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "game_id" not in df.columns or "end_min" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    for c in ["actual_total_score_end", "actual_home_score_end", "actual_away_score_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _collect_training_frame(cfg: TuneSigmaCRPSConfig) -> pd.DataFrame:
    out_dir = Path(cfg.out_dir)
    rows: list[pd.DataFrame] = []
    for d in _date_range(cfg.start, cfg.end):
        q = _load_quantiles(out_dir, d, cfg.sim_quantiles_prefix)
        if q.empty:
            continue

        if bool(cfg.use_regulation_40):
            ia = _load_interval_actuals(out_dir, d, cfg.interval_actuals_prefix)
            if ia.empty:
                continue
            if "actual_total_score_end" not in ia.columns:
                continue
            ia40 = ia[ia["end_min"] == 40].copy()
            if ia40.empty:
                continue
            keep = ["game_id", "actual_total_score_end"]
            if "actual_home_score_end" in ia40.columns and "actual_away_score_end" in ia40.columns:
                ia40["actual_margin_reg40"] = ia40["actual_home_score_end"] - ia40["actual_away_score_end"]
                keep.append("actual_margin_reg40")
            ia40 = ia40[keep].drop_duplicates(subset=["game_id"], keep="last")
            ia40 = ia40.rename(columns={"actual_total_score_end": "y_total", "actual_margin_reg40": "y_margin"})

            m = q.merge(ia40, on="game_id", how="inner")
        else:
            # Fall back: cannot align with regulation targets without interval actuals.
            continue

        if m.empty:
            continue
        m = m[[c for c in ["game_id", "mu_total", "sigma_total", "mu_margin", "sigma_margin", "y_total", "y_margin"] if c in m.columns]].copy()
        m["date"] = d
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    for c in ["mu_total", "sigma_total", "mu_margin", "sigma_margin", "y_total", "y_margin"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def tune_sigma_crps(cfg: TuneSigmaCRPSConfig) -> dict[str, Any]:
    df = _collect_training_frame(cfg)
    if df.empty:
        return {"error": "No training rows found", "start": cfg.start, "end": cfg.end}

    baseline_total_mult, baseline_margin_mult = _load_baseline_sigma_multipliers(cfg.out_dir)
    if cfg.baseline_sigma_total_mult is not None:
        try:
            v = float(cfg.baseline_sigma_total_mult)
            if math.isfinite(v) and v > 0:
                baseline_total_mult = v
        except Exception:
            pass
    if cfg.baseline_sigma_margin_mult is not None:
        try:
            v = float(cfg.baseline_sigma_margin_mult)
            if math.isfinite(v) and v > 0:
                baseline_margin_mult = v
        except Exception:
            pass

    grid = np.linspace(float(cfg.grid_min), float(cfg.grid_max), int(cfg.grid_steps))

    out: dict[str, Any] = {
        "start": cfg.start,
        "end": cfg.end,
        "use_regulation_40": bool(cfg.use_regulation_40),
        "grid": {"min": float(cfg.grid_min), "max": float(cfg.grid_max), "steps": int(cfg.grid_steps)},
        "n_rows": int(len(df)),
        "baseline_multipliers": {
            "sigma_total_mult": float(baseline_total_mult),
            "sigma_margin_mult": float(baseline_margin_mult),
        },
    }

    # Total sigma tuning
    best_total = None
    best_total_crps = None
    base_total_crps = None

    if {"mu_total", "sigma_total", "y_total"}.issubset(df.columns):
        mu = df["mu_total"].to_numpy(dtype=float)
        sig = df["sigma_total"].to_numpy(dtype=float)
        y = df["y_total"].to_numpy(dtype=float)
        if np.isfinite(mu).any() and np.isfinite(sig).any() and np.isfinite(y).any():
            sig_base = sig / float(baseline_total_mult)
            base = _crps_normal_vec(mu, sig_base * float(baseline_total_mult), y)
            base_total_crps = float(np.nanmean(base)) if np.isfinite(base).any() else None

            for m in grid:
                # Tune *absolute* multiplier (M) to use in sim_calibration.json.
                crps = _crps_normal_vec(mu, sig_base * float(m), y)
                v = float(np.nanmean(crps)) if np.isfinite(crps).any() else None
                if v is None:
                    continue
                if (best_total_crps is None) or (v < best_total_crps):
                    best_total_crps = v
                    best_total = float(m)

    # Margin sigma tuning (only if y_margin present)
    best_margin = None
    best_margin_crps = None
    base_margin_crps = None

    if {"mu_margin", "sigma_margin", "y_margin"}.issubset(df.columns) and df["y_margin"].notna().any():
        mu = df["mu_margin"].to_numpy(dtype=float)
        sig = df["sigma_margin"].to_numpy(dtype=float)
        y = df["y_margin"].to_numpy(dtype=float)
        sig_base = sig / float(baseline_margin_mult)
        base = _crps_normal_vec(mu, sig_base * float(baseline_margin_mult), y)
        base_margin_crps = float(np.nanmean(base)) if np.isfinite(base).any() else None

        for m in grid:
            crps = _crps_normal_vec(mu, sig_base * float(m), y)
            v = float(np.nanmean(crps)) if np.isfinite(crps).any() else None
            if v is None:
                continue
            if (best_margin_crps is None) or (v < best_margin_crps):
                best_margin_crps = v
                best_margin = float(m)

    out["totals"] = {
        "n": int(pd.to_numeric(df.get("y_total"), errors="coerce").notna().sum()) if "y_total" in df.columns else 0,
        "sigma_total_mult_best": best_total,
        "crps_mean_baseline": base_total_crps,
        "crps_mean_best": best_total_crps,
    }
    out["margins"] = {
        "n": int(pd.to_numeric(df.get("y_margin"), errors="coerce").notna().sum()) if "y_margin" in df.columns else 0,
        "sigma_margin_mult_best": best_margin,
        "crps_mean_baseline": base_margin_crps,
        "crps_mean_best": best_margin_crps,
    }

    return out


def evaluate_sigma_crps(
    cfg: TuneSigmaCRPSConfig,
    sigma_total_mult: float,
    sigma_margin_mult: float,
) -> dict[str, Any]:
    """Evaluate mean CRPS for fixed absolute multipliers over cfg's date range."""
    df = _collect_training_frame(cfg)
    if df.empty:
        return {"error": "No training rows found", "start": cfg.start, "end": cfg.end}

    baseline_total_mult, baseline_margin_mult = _load_baseline_sigma_multipliers(cfg.out_dir)
    if cfg.baseline_sigma_total_mult is not None:
        try:
            v = float(cfg.baseline_sigma_total_mult)
            if math.isfinite(v) and v > 0:
                baseline_total_mult = v
        except Exception:
            pass
    if cfg.baseline_sigma_margin_mult is not None:
        try:
            v = float(cfg.baseline_sigma_margin_mult)
            if math.isfinite(v) and v > 0:
                baseline_margin_mult = v
        except Exception:
            pass
    try:
        sigma_total_mult = float(sigma_total_mult)
    except Exception:
        sigma_total_mult = float(baseline_total_mult)
    try:
        sigma_margin_mult = float(sigma_margin_mult)
    except Exception:
        sigma_margin_mult = float(baseline_margin_mult)

    out: dict[str, Any] = {
        "start": cfg.start,
        "end": cfg.end,
        "use_regulation_40": bool(cfg.use_regulation_40),
        "n_rows": int(len(df)),
        "baseline_multipliers": {
            "sigma_total_mult": float(baseline_total_mult),
            "sigma_margin_mult": float(baseline_margin_mult),
        },
        "eval_multipliers": {
            "sigma_total_mult": float(sigma_total_mult),
            "sigma_margin_mult": float(sigma_margin_mult),
        },
    }

    if {"mu_total", "sigma_total", "y_total"}.issubset(df.columns):
        mu = df["mu_total"].to_numpy(dtype=float)
        sig = df["sigma_total"].to_numpy(dtype=float)
        y = df["y_total"].to_numpy(dtype=float)
        sig_base = sig / float(baseline_total_mult)
        base = _crps_normal_vec(mu, sig_base * float(baseline_total_mult), y)
        ev = _crps_normal_vec(mu, sig_base * float(sigma_total_mult), y)
        out["totals"] = {
            "n": int(pd.to_numeric(df.get("y_total"), errors="coerce").notna().sum()) if "y_total" in df.columns else 0,
            "crps_mean_baseline": float(np.nanmean(base)) if np.isfinite(base).any() else None,
            "crps_mean_eval": float(np.nanmean(ev)) if np.isfinite(ev).any() else None,
        }

    if {"mu_margin", "sigma_margin", "y_margin"}.issubset(df.columns) and df["y_margin"].notna().any():
        mu = df["mu_margin"].to_numpy(dtype=float)
        sig = df["sigma_margin"].to_numpy(dtype=float)
        y = df["y_margin"].to_numpy(dtype=float)
        sig_base = sig / float(baseline_margin_mult)
        base = _crps_normal_vec(mu, sig_base * float(baseline_margin_mult), y)
        ev = _crps_normal_vec(mu, sig_base * float(sigma_margin_mult), y)
        out["margins"] = {
            "n": int(pd.to_numeric(df.get("y_margin"), errors="coerce").notna().sum()) if "y_margin" in df.columns else 0,
            "crps_mean_baseline": float(np.nanmean(base)) if np.isfinite(base).any() else None,
            "crps_mean_eval": float(np.nanmean(ev)) if np.isfinite(ev).any() else None,
        }

    return out


def write_sim_calibration_update(
    out_dir: Path,
    tuned: dict[str, Any],
    out_path: Path,
    apply_to_default: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_path = Path(out_path)

    payload: dict[str, Any] = {
        "source": "tune-sim-sigma-crps",
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
    }
    payload.update(tuned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    applied_path = None
    backup_path = None
    if apply_to_default:
        default_path = out_dir / "sim_calibration.json"
        # Backup existing file if present
        if default_path.exists():
            ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = out_dir / f"sim_calibration.backup_{ts}.json"
            try:
                backup_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                backup_path = None

        merged: dict[str, Any] = {}
        try:
            if default_path.exists():
                merged_obj = json.loads(default_path.read_text(encoding="utf-8"))
                if isinstance(merged_obj, dict):
                    merged.update(merged_obj)
        except Exception:
            merged = {}

        # Update only sigma multipliers when present in tuned output
        try:
            best_total = tuned.get("totals", {}).get("sigma_total_mult_best")
            if best_total is not None:
                merged["sigma_total_mult"] = float(best_total)
        except Exception:
            pass
        try:
            best_margin = tuned.get("margins", {}).get("sigma_margin_mult_best")
            if best_margin is not None:
                merged["sigma_margin_mult"] = float(best_margin)
        except Exception:
            pass

        merged["_updated_by"] = "tune-sim-sigma-crps"
        merged["_updated_at"] = payload.get("generated_at")

        default_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        applied_path = str(default_path)

    return {
        "wrote": str(out_path),
        "applied": applied_path,
        "backup": str(backup_path) if backup_path else None,
    }
