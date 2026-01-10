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
    p = OUT / f"predictions_model_totals_{date}.csv"
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    p2 = OUT / f"predictions_unified_enriched_{date}.csv"
    try:
        if p2.exists():
            return pd.read_csv(p2)
    except Exception:
        pass
    return pd.DataFrame()

def _collect_recent_dates(n_days: int = 60) -> list[str]:
    files = sorted(glob.glob(str(OUT / "daily_results" / "results_*.csv")))
    dates = [Path(f).stem.replace("results_", "") for f in files]
    return dates[-n_days:] if dates else []

def _join(df_preds: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["game_id","home_team","away_team","date"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        keys = [c for c in ["home_team","away_team"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        return pd.DataFrame()
    keep = list(set(keys + [c for c in ["actual_total","final_total"] if c in df_res.columns]))
    res_min = df_res[keep].copy()
    if "actual_total" not in res_min.columns and "final_total" in res_min.columns:
        res_min = res_min.rename(columns={"final_total": "actual_total"})
    merged = pd.merge(df_preds, res_min, on=keys, how="inner")
    return merged

def _pinball_loss(q: float, y: pd.Series, yhat: pd.Series) -> float:
    e = y - yhat
    return float(((q - (e < 0).astype(float)) * e).abs().mean())

def _evaluate_params(df: pd.DataFrame, shift: float, scale: float) -> float:
    # Apply shift to q50 and scale deviations for q10/q90
    q10 = pd.to_numeric(df["pred_total_q10"], errors="coerce")
    q50 = pd.to_numeric(df["pred_total_q50"], errors="coerce")
    q90 = pd.to_numeric(df["pred_total_q90"], errors="coerce")
    y = pd.to_numeric(df["actual_total"], errors="coerce")
    mask = q10.notna() & q50.notna() & q90.notna() & y.notna()
    if not mask.any():
        return 1e9
    q10c = (q50 + shift) - scale * (q50 - q10)
    q50c = (q50 + shift)
    q90c = (q50 + shift) + scale * (q90 - q50)
    # Pinball losses at 0.1, 0.5, 0.9
    l10 = _pinball_loss(0.1, y[mask], q10c[mask])
    l50 = _pinball_loss(0.5, y[mask], q50c[mask])
    l90 = _pinball_loss(0.9, y[mask], q90c[mask])
    return l10 + l50 + l90

def main(n_days: int = 60):
    dates = _collect_recent_dates(n_days)
    frames = []
    for d in dates:
        preds = _read_predictions(d)
        res = _read_results(d)
        if preds.empty or res.empty:
            continue
        joined = _join(preds, res)
        if not joined.empty:
            frames.append(joined)
    if not frames:
        print(json.dumps({"error": "No data"}))
        return
    df = pd.concat(frames, ignore_index=True)
    best = {"loss": 1e9, "shift": 0.0, "scale": 1.0}
    # Grid search
    for shift in [x / 2.0 for x in range(-40, 41)]:  # -20..20 step 0.5
        for scale in [0.5 + 0.05 * k for k in range(0, 21)]:  # 0.5..1.5 step 0.05
            loss = _evaluate_params(df, shift, scale)
            if loss < best["loss"]:
                best = {"loss": loss, "shift": round(shift, 3), "scale": round(scale, 3)}
    payload = {"shift": best["shift"], "scale": best["scale"], "loss": best["loss"], "n": int(len(df))}
    (OUT / "calibration_quantiles_crps.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))

if __name__ == "__main__":
    main()