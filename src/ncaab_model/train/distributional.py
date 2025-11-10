from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from .baseline import _feature_matrix, _ridge_fit


def train_distributional_totals(
    features_csv: Path,
    out_dir: Path,
    alpha_mu: float = 1.0,
    alpha_sigma: float = 1.0,
    min_sigma: float = 6.0,
    sigma_mode: str = "log",
    baseline_preds_csv: Path | None = None,
    baseline_pred_col: str = "pred_total",
) -> dict:
    """Train a distributional totals model with built-in linear calibration and optional baseline feature.

    Optional: if baseline_preds_csv provided and contains baseline_pred_col keyed by game_id, we merge
    that single numeric column into features before fitting (serves as an anchor feature).

    Steps:
      1. (Optional) Merge baseline prediction as extra feature column `baseline_pred_total`.
      2. Fit ridge mean model -> raw yhat.
      3. Calibrate yhat to targets via y ≈ a*yhat + b absorbed into W_mu, b_mu.
      4. Fit sigma model on calibrated absolute residuals (linear or log space).
      5. Save parameter packs including calibration factors.
    """
    df = pd.read_csv(features_csv)
    if baseline_preds_csv is not None and baseline_preds_csv.exists():
        try:
            bp = pd.read_csv(baseline_preds_csv)
            if "game_id" in bp.columns and baseline_pred_col in bp.columns and "game_id" in df.columns:
                bp["game_id"] = bp["game_id"].astype(str)
                df["game_id"] = df["game_id"].astype(str)
                df = df.merge(bp[["game_id", baseline_pred_col]].rename(columns={baseline_pred_col: "baseline_pred_total"}), on="game_id", how="left")
        except Exception as e:
            print(f"[yellow]Baseline predictions merge failed; continuing without them: {e}[/yellow]")
    df = df.dropna(subset=["target_total"])  # require totals target
    X, cols = _feature_matrix(df)
    y = df["target_total"].to_numpy(dtype=np.float32)

    # Mean model
    W_mu, b_mu, mu_mu, sig_mu = _ridge_fit(X, y, alpha=alpha_mu)
    yhat_raw = ((X - mu_mu) / sig_mu) @ W_mu + b_mu
    # Calibration (a,b)
    var_raw = float(np.var(yhat_raw))
    if var_raw <= 1e-8:
        cal_a, cal_b = 1.0, 0.0
    else:
        cov = float(np.cov(yhat_raw, y, bias=True)[0,1])
        cal_a = cov / var_raw
        cal_b = float(np.mean(y) - cal_a * np.mean(yhat_raw))
    W_mu = W_mu * cal_a
    b_mu = cal_a * b_mu + cal_b
    yhat = ((X - mu_mu) / sig_mu) @ W_mu + b_mu  # calibrated predictions

    # Sigma model target
    sigma_mode_lc = (sigma_mode or "linear").lower()
    if sigma_mode_lc not in {"linear", "log"}:
        raise ValueError("sigma_mode must be 'linear' or 'log'")
    abs_res = np.abs(y - yhat)
    abs_res = np.clip(abs_res, 1e-3, None)
    sigma_offset = 1.0
    target_sigma = np.log(abs_res + sigma_offset) if sigma_mode_lc == "log" else abs_res
    W_s, b_s, mu_s, sig_s = _ridge_fit(X, target_sigma.astype(np.float32), alpha=alpha_sigma)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "dist_total_mu.npz", W=W_mu, b=b_mu, mu=mu_mu, sigma=sig_mu, feature_columns=np.array(cols, dtype=object), cal_a=cal_a, cal_b=cal_b)
    np.savez(out_dir / "dist_total_sigma.npz", W=W_s, b=b_s, mu=mu_s, sigma=sig_s, feature_columns=np.array(cols, dtype=object), min_sigma=min_sigma, sigma_mode=sigma_mode_lc, sigma_offset=sigma_offset)

    mae_mu = float(np.mean(np.abs(y - yhat)))
    raw_pred_sigma = ((X - mu_s) / sig_s) @ W_s + b_s
    recon_sigma = (np.exp(raw_pred_sigma) - sigma_offset) if sigma_mode_lc == "log" else raw_pred_sigma
    recon_sigma = np.clip(recon_sigma, min_sigma, None)
    mae_sigma = float(np.mean(np.abs(abs_res - recon_sigma)))

    return {
        "mae_mu": mae_mu,
        "mae_sigma": mae_sigma,
        "cal_a": cal_a,
        "cal_b": cal_b,
        "n_rows": int(len(df)),
        "feature_columns": cols,
        "paths": {
            "mu": str((out_dir / "dist_total_mu.npz").resolve()),
            "sigma": str((out_dir / "dist_total_sigma.npz").resolve()),
        },
    }


def predict_distributional_totals(
    features_csv: Path,
    models_dir: Path,
    baseline_preds: pd.DataFrame | None = None,
    blend_weight: float = 0.0,
    global_shift: float = 0.0,
    calibrate_to_baseline: bool = False,
    calibration_max_ratio: float = 3.0,
    sigma_cap: float | None = 25.0,
    add_debug: bool = True,
    prob_calibrate: bool = False,
    calibration_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    mu_pack = np.load(models_dir / "dist_total_mu.npz", allow_pickle=True)
    sg_pack = np.load(models_dir / "dist_total_sigma.npz", allow_pickle=True)
    W_mu, b_mu, mu_mu, sig_mu = mu_pack["W"], float(mu_pack["b"]), mu_pack["mu"], mu_pack["sigma"]
    W_s, b_s, mu_s, sig_s = sg_pack["W"], float(sg_pack["b"]), sg_pack["mu"], sg_pack["sigma"]
    sigma_mode = str(sg_pack.get("sigma_mode", "linear"))
    sigma_offset = float(sg_pack.get("sigma_offset", 1.0))
    # Enforce feature alignment with training columns
    feat_cols = [str(c) for c in mu_pack.get("feature_columns", [])]
    if not feat_cols:
        # Fallback to baseline feature construction
        X, _ = _feature_matrix(df)
    else:
        # Build X by selecting required columns in order; fill missing with 0
        X_parts = []
        for c in feat_cols:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
            else:
                col = pd.Series(0.0, index=df.index, dtype=np.float32)
            X_parts.append(col)
        X = np.column_stack([p.to_numpy(dtype=np.float32) for p in X_parts])
    min_sigma = float(sg_pack.get("min_sigma", 6.0))
    y_mu = ((X - mu_mu) / sig_mu) @ W_mu + b_mu
    y_s_raw = ((X - mu_s) / sig_s) @ W_s + b_s
    if sigma_mode == "log":
        y_s = np.exp(y_s_raw) - sigma_offset
    else:
        y_s = y_s_raw
    y_s = np.clip(y_s, min_sigma, None)
    # Degeneracy guard: if all mu predictions nearly identical or sigma variance ~0, flag in output
    mu_var = float(np.var(y_mu)) if y_mu.size else 0.0
    sig_var = float(np.var(y_s)) if y_s.size else 0.0
    degenerate = bool(mu_var < 1e-6 or sig_var < 1e-6)
    out = df[[c for c in ["game_id","date","home_team","away_team"] if c in df.columns]].copy()
    out["pred_total_mu_raw"] = y_mu.copy()
    # Optional blending with baseline predictions
    baseline_series = None
    if baseline_preds is not None:
        try:
            bp = baseline_preds.copy()
            if "game_id" in bp.columns:
                bp["game_id"] = bp["game_id"].astype(str)
            if "game_id" in out.columns:
                out["game_id"] = out["game_id"].astype(str)
            key = "pred_total" if "pred_total" in bp.columns else None
            if key:
                out = out.merge(bp[["game_id", key]], on="game_id", how="left")
                baseline_series = pd.to_numeric(out[key], errors="coerce")
        except Exception:
            baseline_series = None
    if baseline_series is not None and blend_weight > 0:
        base_vals = baseline_series.to_numpy(dtype=np.float32)
        # Where baseline is NaN, fall back to model y_mu
        if base_vals.shape != y_mu.shape:
            base_vals = np.resize(base_vals, y_mu.shape)
        mask = ~np.isfinite(base_vals)
        if mask.any():
            base_vals[mask] = y_mu[mask]
        y_mu = (1 - blend_weight) * y_mu + blend_weight * base_vals
    # Global shift applied after blending
    if global_shift != 0:
        y_mu = y_mu + float(global_shift)
    # Optional calibration: align scale of mu to baseline predictions mean
    if calibrate_to_baseline and baseline_series is not None and baseline_series.notna().any():
        mean_base = float(baseline_series.mean())
        mean_mu = float(np.mean(y_mu)) if y_mu.size else 0.0
        if mean_mu > 0 and mean_base > 0:
            ratio = mean_base / mean_mu
            # Cap excessive scaling
            ratio = float(min(ratio, calibration_max_ratio))
            y_mu = y_mu * ratio
            out["calibration_ratio"] = ratio
        else:
            out["calibration_ratio"] = np.nan
    # Sigma cap if requested
    if sigma_cap is not None and sigma_cap > 0:
        y_s = np.clip(y_s, None, float(sigma_cap))
    # Probability calibration placeholder (maps raw z to adjusted tail probs if table provided)
    # Only applied in downstream usage; here we attach both raw and (optionally) calibrated mu for transparency.
    out["pred_total_mu"] = y_mu
    out["pred_total_sigma"] = y_s
    if prob_calibrate and calibration_table is not None and {"z", "p_under"}.issubset(calibration_table.columns):
        try:
            # Build z-scores per game relative to line if available (requires a 'total' column)
            if "total" in df.columns:
                z = (df["total"].astype(float) - y_mu) / y_s
                # Simple nearest-bin mapping
                calib_bins = calibration_table.sort_values("z")
                z_arr = z.to_numpy()
                mapped = []
                zbins = calib_bins["z"].to_numpy()
                p_under_bins = calib_bins["p_under"].to_numpy()
                for val in z_arr:
                    idx = int(np.searchsorted(zbins, val, side="left"))
                    if idx <= 0:
                        mapped.append(p_under_bins[0])
                    elif idx >= len(zbins):
                        mapped.append(p_under_bins[-1])
                    else:
                        mapped.append(p_under_bins[idx])
                out["p_under_calibrated"] = mapped
        except Exception as e:
            out["p_under_calibrated_error"] = str(e)
    out["pred_dist_degenerate"] = degenerate
    if add_debug:
        out["mu_mean"] = float(np.mean(y_mu)) if y_mu.size else np.nan
        out["mu_std"] = float(np.std(y_mu)) if y_mu.size else np.nan
        out["sigma_mean"] = float(np.mean(y_s)) if y_s.size else np.nan
        out["sigma_std"] = float(np.std(y_s)) if y_s.size else np.nan
        if baseline_series is not None:
            out["baseline_mean"] = float(baseline_series.mean())
            out["baseline_std"] = float(baseline_series.std())
    return out
