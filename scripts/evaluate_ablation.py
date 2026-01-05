import json
import math
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# Helper: load recent daily results and join with features

def _list_recent_results(days: int = 14) -> List[Path]:
    dr = OUT / "daily_results"
    if not dr.exists():
        return []
    files = sorted(dr.glob("results_*.csv"))
    return files[-days:]


def _load_features() -> pd.DataFrame:
    for name in ["features_all.csv", "features_curr.csv", "features_last2.csv"]:
        p = OUT / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                pass
    return pd.DataFrame()


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _shooting_cols(df: pd.DataFrame) -> List[str]:
    keys = [
        "3pt_rate", "3pt_pct", "2pt_rate", "2pt_pct", "pip", "fbp", "scp",
    ]
    cand = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            cand.append(c)
    return cand


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    # Closed-form ridge: w = (X^T X + lam I)^{-1} X^T y
    # Add bias via column of ones
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0  # don't regularize bias
    A = Xb.T @ Xb + lam * I
    b = Xb.T @ y
    w = np.linalg.pinv(A) @ b
    return w


def _ridge_pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return Xb @ w


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _crps_normal(mu: np.ndarray, sigma: float, x: np.ndarray) -> float:
    # CRPS for normal N(mu, sigma) averaged over samples
    # Formula: E[|X - x|] - 0.5 E[|X - X'|], for normal closed form
    # Equivalent closed form used here: see Gneiting & Raftery (2007)
    if sigma <= 0:
        sigma = 1e-6
    z = (x - mu) / sigma
    from math import sqrt
    from scipy.stats import norm  # optional; if missing, fallback approximation
    try:
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(math.pi))
        return float(np.mean(crps))
    except Exception:
        # Fallback numeric approximation via sampling
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=mu, scale=sigma, size=(1000, len(mu)))
        term1 = np.mean(np.abs(samples - x), axis=0)
        term2 = np.mean(np.abs(samples - samples.mean(axis=0)), axis=0)  # crude
        return float(np.mean(term1 - 0.5 * term2))


def evaluate(days: int = 14, lam: float = 2.0) -> dict:
    feats = _load_features()
    result_files = _list_recent_results(days)
    if feats.empty or not result_files:
        return {
            "ok": False,
            "error": "features_or_results_missing",
            "features_rows": int(len(feats)),
            "result_files": len(result_files),
        }
    # Build results frame
    res_parts = []
    for pf in result_files:
        try:
            df = pd.read_csv(pf)
            df["date"] = df.get("date", df.get("display_date", "")).astype(str)
            res_parts.append(df)
        except Exception:
            pass
    results = pd.concat(res_parts, ignore_index=True) if res_parts else pd.DataFrame()
    if results.empty:
        return {"ok": False, "error": "no_results_rows"}
    # Targets: total_actual and margin_actual (robust derivation)
    if {"home_score", "away_score"}.issubset(set(results.columns)):
        results["total_actual"] = results["home_score"].astype(float) + results["away_score"].astype(float)
        results["margin_actual"] = results["home_score"].astype(float) - results["away_score"].astype(float)
    else:
        # Fallbacks
        results["total_actual"] = results.get("final_total", np.nan)
        results["margin_actual"] = results.get("final_margin", np.nan)
    # Join features by game_id
    key = "game_id"
    if key in feats.columns and key in results.columns:
        df = results.merge(feats, on=key, suffixes=("_res", ""), how="inner")
    else:
        # Try by unordered team pair + date
        for pair_cols in [("home_team", "away_team"), ("home_team_name", "away_team_name")]:
            h, a = pair_cols
            if h in feats.columns and a in feats.columns and h in results.columns and a in results.columns and "date" in feats.columns and "date" in results.columns:
                df = results.merge(feats, on=["date", h, a], suffixes=("_res", ""), how="inner")
                break
        else:
            return {"ok": False, "error": "join_keys_missing"}
    # Drop rows missing targets
    df = df.dropna(subset=["total_actual", "margin_actual"])
    num_cols = _numeric_cols(df)
    # Remove obvious IDs/leakage
    drop_like = ["game_id", "date", "home_score", "away_score", "total_actual", "margin_actual",
                 "pred_total", "pred_margin", "final_total", "final_margin"]
    X_cols = [c for c in num_cols if c not in drop_like]
    if not X_cols:
        # Fallback: compute baseline MAE from predictions_display vs actuals
        base = _baseline_eval(days)
        return {"ok": True, "fallback_baseline": base, "message": "no_numeric_features; reported baseline only"}
    shoot_cols = [c for c in _shooting_cols(df) if c in X_cols]
    base_cols = [c for c in X_cols if c not in shoot_cols]
    if not base_cols:
        base = _baseline_eval(days)
        return {"ok": True, "fallback_baseline": base, "message": "no_base_features; reported baseline only"}
    # Train/test split by date (last 4 days as test if available)
    dates = sorted(list(set(df["date"].astype(str))))
    test_dates = set(dates[-min(4, len(dates)) :])
    train = df[~df["date"].astype(str).isin(test_dates)].copy()
    test = df[df["date"].astype(str).isin(test_dates)].copy()
    if len(train) == 0 or len(test) == 0:
        base = _baseline_eval(days)
        return {
            "ok": True,
            "fallback_baseline": base,
            "message": "insufficient train/test after join; reported baseline only",
            "joined_rows": int(len(df)),
        }
    def _eval_target(target: str) -> dict:
        y_tr = train[target].to_numpy(dtype=float)
        y_te = test[target].to_numpy(dtype=float)
        # Base
        Xb_tr, mu_b, sd_b = _standardize(train[base_cols].to_numpy(dtype=float))
        Xb_te = (test[base_cols].to_numpy(dtype=float) - mu_b) / sd_b
        wb = _ridge_fit(Xb_tr, y_tr, lam=lam)
        pb = _ridge_pred(Xb_te, wb)
        mae_b = float(np.mean(np.abs(pb - y_te))) if len(y_te) else float("nan")
        sigma_b = float(np.std(y_tr - _ridge_pred(Xb_tr, wb))) if len(y_tr) else 1.0
        crps_b = _crps_normal(pb, sigma_b, y_te) if len(y_te) else float("nan")
        # Enriched (base + shooting)
        Xe_tr_full = train[base_cols + shoot_cols].to_numpy(dtype=float)
        Xe_te_full = test[base_cols + shoot_cols].to_numpy(dtype=float)
        Xe_tr, mu_e, sd_e = _standardize(Xe_tr_full)
        Xe_te = (Xe_te_full - mu_e) / sd_e
        we = _ridge_fit(Xe_tr, y_tr, lam=lam)
        pe = _ridge_pred(Xe_te, we)
        mae_e = float(np.mean(np.abs(pe - y_te))) if len(y_te) else float("nan")
        sigma_e = float(np.std(y_tr - _ridge_pred(Xe_tr, we))) if len(y_tr) else 1.0
        crps_e = _crps_normal(pe, sigma_e, y_te) if len(y_te) else float("nan")
        return {
            "target": target,
            "mae_base": mae_b,
            "mae_enriched": mae_e,
            "mae_delta": mae_b - mae_e if (not math.isnan(mae_b) and not math.isnan(mae_e)) else None,
            "crps_base": crps_b,
            "crps_enriched": crps_e,
            "crps_delta": crps_b - crps_e if (not math.isnan(crps_b) and not math.isnan(crps_e)) else None,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_dates": sorted(list(test_dates)),
            "n_shooting_features": int(len(shoot_cols)),
        }
    out = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "days": days,
        "results_files": [p.name for p in result_files],
        "targets": [
            _eval_target("total_actual"),
            _eval_target("margin_actual"),
        ],
        "shooting_cols_sample": shoot_cols[:15],
    }
    return out


def _baseline_eval(days: int = 14) -> dict:
    """Compute MAE for predictions_display (pred_total_calibrated/pred_margin_calibrated preferred) vs actuals."""
    files = sorted((OUT.glob('predictions_display_*.csv')))
    files = files[-days:]
    preds_parts = []
    for pf in files:
        try:
            df = pd.read_csv(pf)
            df['date'] = df.get('date', df.get('display_date', '')).astype(str)
            preds_parts.append(df)
        except Exception:
            pass
    preds = pd.concat(preds_parts, ignore_index=True) if preds_parts else pd.DataFrame()
    res_files = _list_recent_results(days)
    res_parts = []
    for rf in res_files:
        try:
            df = pd.read_csv(rf)
            df['date'] = df.get('date', df.get('display_date', '')).astype(str)
            res_parts.append(df)
        except Exception:
            pass
    results = pd.concat(res_parts, ignore_index=True) if res_parts else pd.DataFrame()
    if preds.empty or results.empty:
        return {"ok": False, "error": "baseline_sources_missing", "pred_rows": int(len(preds)), "res_rows": int(len(results))}
    # Join by game_id when available
    key = 'game_id'
    if key in preds.columns and key in results.columns:
        df = results.merge(preds, on=key, suffixes=("_res", ""), how="inner")
    else:
        # fallback by date + teams
        for cols in [("home_team","away_team"), ("home_team_name","away_team_name")]:
            h,a = cols
            if h in preds.columns and a in preds.columns and h in results.columns and a in results.columns and 'date' in preds.columns and 'date' in results.columns:
                df = results.merge(preds, on=['date', h, a], suffixes=("_res", ""), how="inner")
                break
        else:
            return {"ok": False, "error": "baseline_join_failed"}
    # Actuals
    if {'home_score','away_score'}.issubset(set(df.columns)):
        df['total_actual'] = df['home_score'].astype(float) + df['away_score'].astype(float)
        df['margin_actual'] = df['home_score'].astype(float) - df['away_score'].astype(float)
    else:
        df['total_actual'] = df.get('final_total', np.nan)
        df['margin_actual'] = df.get('final_margin', np.nan)
    # Predictions preference: calibrated -> model -> displayed pred_total/pred_margin
    def pick_pred(col_cand: List[str]) -> np.ndarray:
        for c in col_cand:
            if c in df.columns:
                v = df[c].to_numpy(dtype=float)
                if np.isfinite(v).any():
                    return v
        return np.full((len(df),), np.nan)
    p_total = pick_pred(['pred_total_calibrated','pred_total_model','pred_total'])
    p_margin = pick_pred(['pred_margin_calibrated','pred_margin_model','pred_margin'])
    mask_t = np.isfinite(df['total_actual'].to_numpy(dtype=float)) & np.isfinite(p_total)
    mask_m = np.isfinite(df['margin_actual'].to_numpy(dtype=float)) & np.isfinite(p_margin)
    mae_t = float(np.mean(np.abs(p_total[mask_t] - df['total_actual'].to_numpy(dtype=float)[mask_t]))) if mask_t.any() else float('nan')
    mae_m = float(np.mean(np.abs(p_margin[mask_m] - df['margin_actual'].to_numpy(dtype=float)[mask_m]))) if mask_m.any() else float('nan')
    return {
        "ok": True,
        "files": [p.name for p in files],
        "mae_total": mae_t,
        "mae_margin": mae_m,
        "rows_total": int(mask_t.sum()),
        "rows_margin": int(mask_m.sum()),
    }


if __name__ == "__main__":
    report = evaluate(days=14, lam=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"eval_ablation_{dt.datetime.utcnow().strftime('%Y-%m-%d')}.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
