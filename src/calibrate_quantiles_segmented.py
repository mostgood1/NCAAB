import pandas as pd
import json
import glob
from pathlib import Path

OUT = Path("outputs")

def _read_results(date: str) -> pd.DataFrame:
    p = OUT / f"daily_results/results_{date}.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def _read_predictions(date: str) -> pd.DataFrame:
    # Prefer model totals file that contains quantiles
    p = OUT / f"predictions_model_totals_{date}.csv"
    if p.exists():
        return pd.read_csv(p)
    # Fallback to unified enriched if available
    p2 = OUT / f"predictions_unified_enriched_{date}.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return pd.DataFrame()

def _collect_recent_dates(n_days: int = 14) -> list[str]:
    # Use available results files to determine recent dates
    files = sorted(glob.glob(str(OUT / "daily_results" / "results_*.csv")))
    dates = [Path(f).stem.replace("results_", "") for f in files]
    return dates[-n_days:] if dates else []

def _join_preds_results(df_preds: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    # Join on available robust keys
    keys = [c for c in ["game_id","home_team","away_team","date"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        # Minimal fallback keys
        keys = [c for c in ["home_team","away_team"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        return pd.DataFrame()
    # Use only minimal result columns to avoid suffix collisions
    res_cols = list(set(keys + [c for c in ["final_total","total_final","finals_total","actual_total"] if c in df_res.columns]))
    df_res_min = df_res[res_cols].copy()
    # Normalize final total column name
    if "final_total" not in df_res_min.columns:
        for cand in ["total_final","finals_total","actual_total"]:
            if cand in df_res_min.columns:
                df_res_min = df_res_min.rename(columns={cand: "final_total"})
                break
    merged = pd.merge(df_preds, df_res_min, on=keys, how="inner")
    return merged

def _compute_alpha(df: pd.DataFrame, target: float = 0.10) -> tuple[float,int]:
    # Require quantiles
    req = ["pred_total_q10","pred_total_q50","pred_total_q90","final_total"]
    if not set(req).issubset(df.columns):
        return 1.0, 0
    df = df.copy()
    df["pred_total_q10"] = pd.to_numeric(df["pred_total_q10"], errors="coerce")
    df["pred_total_q50"] = pd.to_numeric(df["pred_total_q50"], errors="coerce")
    df["pred_total_q90"] = pd.to_numeric(df["pred_total_q90"], errors="coerce")
    df["final_total"] = pd.to_numeric(df["final_total"], errors="coerce")
    df = df.dropna(subset=["pred_total_q10","pred_total_q50","pred_total_q90","final_total"]) 
    if df.empty:
        return 1.0, 0
    # Binary search alpha to match target tail coverage
    lo, hi = 0.6, 1.6
    best_alpha = 1.0
    best_err = 1e9
    for _ in range(20):
        mid = (lo + hi) / 2.0
        anchor = df.get("pred_total_calibrated", df["pred_total_q50"])  # use calibrated q50 if present
        d10 = (df["pred_total_q50"] - df["pred_total_q10"]) * mid
        d90 = (df["pred_total_q90"] - df["pred_total_q50"]) * mid
        q10c = anchor - d10
        q90c = anchor + d90
        below = (df["final_total"] < q10c).mean()
        above = (df["final_total"] > q90c).mean()
        err = abs(below - target) + abs(above - target)
        if err < best_err:
            best_err, best_alpha = err, mid
        # Adjust search bounds toward increasing coverage if below/above are low
        cov = (below + above) / 2.0
        if cov < target:
            lo = mid
        else:
            hi = mid
    return float(round(best_alpha, 4)), int(len(df))

def main(n_days: int = 14, target: float = 0.10):
    dates = _collect_recent_dates(n_days)
    frames = []
    for d in dates:
        preds = _read_predictions(d)
        res = _read_results(d)
        if preds.empty or res.empty:
            continue
        merged = _join_preds_results(preds, res)
        if not merged.empty:
            frames.append(merged)
    if not frames:
        print("No calibration computed; missing data")
        return
    df = pd.concat(frames, ignore_index=True)
    alpha, n = _compute_alpha(df, target)
    payload = {"spread_alpha": alpha, "n": n, "target_tail": target}
    (OUT / "calibration_quantiles_segmented.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))

if __name__ == "__main__":
    main()