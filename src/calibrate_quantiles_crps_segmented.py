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
    p = OUT / f"predictions_unified_enriched_{date}.csv"
    try:
        if p.exists():
            return pd.read_csv(p)
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
    keep = list(set(keys + [c for c in ["actual_total","final_total","closing_total","market_total"] if c in df_res.columns or c in df_preds.columns]))
    res_min = df_res[[c for c in keep if c in df_res.columns]].copy()
    if "actual_total" not in res_min.columns and "final_total" in res_min.columns:
        res_min = res_min.rename(columns={"final_total": "actual_total"})
    # Attach market/closing totals from preds side when present
    preds_min = df_preds[[c for c in [*keys, "closing_total", "market_total", "pred_total_q10", "pred_total_q50", "pred_total_q90"] if c in df_preds.columns]].copy()
    merged = pd.merge(preds_min, res_min, on=keys, how="inner")
    return merged

def _pinball_loss(q: float, y: pd.Series, yhat: pd.Series) -> float:
    e = y - yhat
    return float(((q - (e < 0).astype(float)) * e).abs().mean())

def _evaluate_params(df: pd.DataFrame, shift: float, scale: float) -> float:
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
    l10 = _pinball_loss(0.1, y[mask], q10c[mask])
    l50 = _pinball_loss(0.5, y[mask], q50c[mask])
    l90 = _pinball_loss(0.9, y[mask], q90c[mask])
    return l10 + l50 + l90

def _tempo_value(df: pd.DataFrame) -> pd.Series:
    # Prefer closing_total, then market_total, else q50 as proxy
    ct = pd.to_numeric(df.get("closing_total"), errors="coerce") if "closing_total" in df.columns else None
    mt = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else None
    q50 = pd.to_numeric(df.get("pred_total_q50"), errors="coerce") if "pred_total_q50" in df.columns else None
    s = pd.Series(index=df.index, dtype=float)
    if ct is not None:
        s = ct
    elif mt is not None:
        s = mt
    elif q50 is not None:
        s = q50
    return s

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
    tempo = _tempo_value(df)
    tempo = tempo.dropna()
    if tempo.empty:
        print(json.dumps({"error": "No tempo values"}))
        return
    # Compute tertile edges
    e1 = float(tempo.quantile(0.33))
    e2 = float(tempo.quantile(0.66))
    def bin_label(x: float) -> str:
        if x <= e1:
            return "low"
        elif x <= e2:
            return "mid"
        else:
            return "high"
    df["tempo_bin"] = _tempo_value(df).apply(lambda v: bin_label(v) if pd.notna(v) else None)
    bins = {"edges": [e1, e2], "bins": {}, "global": {}}
    # Global best as fallback
    best_g = {"loss": 1e9, "shift": 0.0, "scale": 1.0}
    for shift in [x / 2.0 for x in range(-40, 41)]:
        for scale in [0.5 + 0.05 * k for k in range(0, 21)]:
            loss = _evaluate_params(df, shift, scale)
            if loss < best_g["loss"]:
                best_g = {"loss": loss, "shift": round(shift, 3), "scale": round(scale, 3)}
    bins["global"] = {"shift": best_g["shift"], "scale": best_g["scale"], "loss": best_g["loss"], "n": int(len(df))}
    # Per-bin calibration
    for bl in ("low","mid","high"):
        sub = df[df["tempo_bin"] == bl]
        if sub.empty or len(sub) < 50:
            bins["bins"][bl] = {"shift": best_g["shift"], "scale": best_g["scale"], "loss": None, "n": int(len(sub))}
            continue
        best = {"loss": 1e9, "shift": 0.0, "scale": 1.0}
        for shift in [x / 2.0 for x in range(-40, 41)]:
            for scale in [0.5 + 0.05 * k for k in range(0, 21)]:
                loss = _evaluate_params(sub, shift, scale)
                if loss < best["loss"]:
                    best = {"loss": loss, "shift": round(shift, 3), "scale": round(scale, 3)}
        bins["bins"][bl] = {"shift": best["shift"], "scale": best["scale"], "loss": best["loss"], "n": int(len(sub))}
    (OUT / "calibration_quantiles_crps_segmented.json").write_text(json.dumps(bins, indent=2))
    print(json.dumps(bins))

if __name__ == "__main__":
    main()