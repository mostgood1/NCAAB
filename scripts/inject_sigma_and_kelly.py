import os
import json
import math
import pandas as pd

# Minimal injector: fill pred_total_sigma, pred_margin_sigma, and kelly_fraction_total_adj
# Sources:
# - Prefer predictions_model_interval_<date>.csv for per-game sigma if present
# - Fallback to variance JSON or conformal defaults (rmse_total, rmse_margin)
# - kelly_fraction_total_adj: apply a confidence cap based on sigma

def load_interval_sigmas(outputs_dir: str, date_str: str):
    p = os.path.join(outputs_dir, f"predictions_model_interval_{date_str}.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    key = "game_id" if "game_id" in df.columns else None
    sig_total_col = None
    sig_margin_col = None
    for c in df.columns:
        lc = c.lower()
        if sig_total_col is None and ("total" in lc and "sigma" in lc):
            sig_total_col = c
        if sig_margin_col is None and ("margin" in lc and "sigma" in lc):
            sig_margin_col = c
    if key is None or sig_total_col is None or sig_margin_col is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        gid = str(r[key])
        out[gid] = {
            "pred_total_sigma": r[sig_total_col],
            "pred_margin_sigma": r[sig_margin_col],
        }
    return out

def load_variance_fallback(outputs_dir: str, date_str: str):
    vt = os.path.join(outputs_dir, "variance", f"variance_total_{date_str}.json")
    vm = os.path.join(outputs_dir, "variance", f"variance_margin_{date_str}.json")
    rmse_total = None
    rmse_margin = None
    try:
        if os.path.exists(vt):
            with open(vt, "r") as f:
                j = json.load(f)
                rmse_total = j.get("pred_total_sigma_bootstrap_global") or j.get("rmse_total")
        if os.path.exists(vm):
            with open(vm, "r") as f:
                j = json.load(f)
                rmse_margin = j.get("pred_margin_sigma_bootstrap_global") or j.get("rmse_margin")
    except Exception:
        pass
    # Final fallback constants if absent
    if rmse_total is None:
        rmse_total = 12.0
    if rmse_margin is None:
        rmse_margin = 10.0
    return rmse_total, rmse_margin

def confidence_cap_from_sigma(sigma: float, baseline: float) -> float:
    # Simple inverse relation: higher sigma -> lower cap
    # Map sigma to [0.15, 0.35] range using a soft clamp
    if sigma is None or not math.isfinite(sigma):
        return max(0.15, min(0.35, baseline))
    cap = 0.35 - min(0.20, sigma / 100.0 * 0.20)
    return max(0.15, min(0.35, cap))

def inject(outputs_dir: str, date_str: str):
    path = os.path.join(outputs_dir, f"predictions_unified_enriched_{date_str}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Build lookup for interval sigmas
    interval = load_interval_sigmas(outputs_dir, date_str)
    rmse_total, rmse_margin = load_variance_fallback(outputs_dir, date_str)

    # Ensure columns exist
    for col in ["pred_total_sigma", "pred_margin_sigma", "kelly_fraction_total_adj"]:
        if col not in df.columns:
            df[col] = pd.NA

    key = "game_id" if "game_id" in df.columns else None
    for i, r in df.iterrows():
        gid = str(r[key]) if key else None
        # Fill total sigma
        sig_total = r.get("pred_total_sigma")
        if pd.isna(sig_total):
            if gid in interval:
                sig_total = interval[gid]["pred_total_sigma"]
            else:
                sig_total = rmse_total
            df.at[i, "pred_total_sigma"] = sig_total
        # Fill margin sigma
        sig_margin = r.get("pred_margin_sigma")
        if pd.isna(sig_margin):
            if gid in interval:
                sig_margin = interval[gid]["pred_margin_sigma"]
            else:
                sig_margin = rmse_margin
            df.at[i, "pred_margin_sigma"] = sig_margin
        # Adjust Kelly fraction for totals
        kelly_total = r.get("kelly_fraction_total")
        adj = r.get("kelly_fraction_total_adj")
        if pd.isna(adj) and kelly_total is not None and pd.notna(kelly_total):
            cap = confidence_cap_from_sigma(sig_total, baseline=float(kelly_total))
            df.at[i, "kelly_fraction_total_adj"] = min(float(kelly_total), cap)

    outp = path  # in place
    df.to_csv(outp, index=False)
    summary = {
        "date": date_str,
        "rows": int(df.shape[0]),
        "filled_pred_total_sigma": int(df["pred_total_sigma"].notna().sum()),
        "filled_pred_margin_sigma": int(df["pred_margin_sigma"].notna().sum()),
        "filled_kelly_fraction_total_adj": int(df["kelly_fraction_total_adj"].notna().sum()),
        "interval_used": len(interval),
        "rmse_fallback_total": rmse_total,
        "rmse_fallback_margin": rmse_margin,
        "path": outp,
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--outputs", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = ap.parse_args()
    inject(os.path.abspath(args.outputs), args.date)
