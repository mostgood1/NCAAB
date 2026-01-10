from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

OUT = Path("outputs")

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _collect_dates(n_days: int = 60) -> list[str]:
    files = sorted(glob.glob(str(OUT / "daily_results" / "results_*.csv")))
    dates = [Path(f).stem.replace("results_", "") for f in files]
    return dates[-n_days:] if dates else []

def _join_features_results(date: str) -> pd.DataFrame:
    # Prefer augmented per-date features when available; fallback to enriched-derived minimal features
    feats_aug = _safe_read_csv(OUT / f"features_{date}_augmented.csv")
    feats_base = _safe_read_csv(OUT / f"features_{date}.csv")
    res = _safe_read_csv(OUT / f"daily_results/results_{date}.csv")
    enrich = _safe_read_csv(OUT / f"predictions_unified_enriched_{date}.csv")
    aug_recent = _safe_read_csv(OUT / "features_augmented_recent.csv")
    feats = feats_aug if not feats_aug.empty else feats_base
    if (feats is None or feats.empty) and not enrich.empty:
        e = enrich.copy()
        for c in ("game_id","date"):
            if c in e.columns:
                e[c] = e[c].astype(str)
        cols = [c for c in ("game_id","date","market_total","closing_total","q10","q50","q90","pred_total_calibrated","neutral_site","start_tz_abbr") if c in e.columns]
        feats = e[cols].copy()
        if "q50" in feats.columns and "market_total" in feats.columns:
            feats["delta_q50_market"] = pd.to_numeric(feats["q50"], errors="coerce") - pd.to_numeric(feats["market_total"], errors="coerce")
        if "pred_total_calibrated" in feats.columns and "market_total" in feats.columns:
            feats["delta_cal_market"] = pd.to_numeric(feats["pred_total_calibrated"], errors="coerce") - pd.to_numeric(feats["market_total"], errors="coerce")
    if feats is None or feats.empty or res.empty:
        return pd.DataFrame()
    for d in (feats, res):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
    keys = [k for k in ("game_id","date") if k in feats.columns and k in res.columns]
    if not keys:
        keys = ["game_id"]
    keep = keys + [c for c in ("actual_total","final_total","market_total","closing_total") if c in res.columns]
    res_min = res[keep].copy()
    if "actual_total" not in res_min.columns and "final_total" in res_min.columns:
        res_min = res_min.rename(columns={"final_total": "actual_total"})
    df = feats.merge(res_min, on=keys, how="inner")
    # Merge augmented pace/possession and related features
    if not aug_recent.empty:
        d = aug_recent.copy()
        if "date" in d.columns:
            try:
                d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                d["date"] = d["date"].astype(str)
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        aug_d = d[d.get("date").astype(str) == date]
        use_cols = [c for c in ["game_id","date","home_pace","home_ts","home_3p_rate","home_to_rate","home_drb_rate",
                                 "away_pace","away_ts","away_3p_rate","away_to_rate","away_drb_rate",
                                 "home_b2b","away_b2b","tz_offset_hours","home_adv"] if c in aug_d.columns]
        if use_cols:
            df = df.merge(aug_d[use_cols], on=[k for k in keys if k in ["game_id","date"]], how="left")
    if not enrich.empty:
        for d in (enrich,):
            if "game_id" in d.columns:
                d["game_id"] = d["game_id"].astype(str)
            if "date" in d.columns:
                d["date"] = d["date"].astype(str)
        use_cols = keys + [c for c in ("q10","q50","q90","pred_total_model","pred_total_calibrated","pred_total_basis") if c in enrich.columns]
        df = df.merge(enrich[use_cols], on=keys, how="left")
        if "q50" in df.columns and "market_total" in df.columns:
            df["delta_q50_market"] = pd.to_numeric(df["q50"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
        if "pred_total_calibrated" in df.columns and "market_total" in df.columns:
            df["delta_cal_market"] = pd.to_numeric(df["pred_total_calibrated"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
    return df

def evaluate(n_days: int = 60, thresholds=(0.6, 0.65, 0.7), edge_requirements=(5,7,10), model: str = "auto") -> dict:
    # Prefer boosted calibrated model if available unless overridden
    if model == "lr":
        model_path = OUT / "ou_classifier_lr.joblib"
        meta_path = OUT / "ou_classifier_lr.json"
    elif model == "hgb":
        model_path = OUT / "ou_classifier_hgb.joblib"
        meta_path = OUT / "ou_classifier_hgb.json"
    else:
        model_path = OUT / "ou_classifier_hgb.joblib"
        meta_path = OUT / "ou_classifier_hgb.json"
        if not model_path.exists() or not meta_path.exists():
            model_path = OUT / "ou_classifier_lr.joblib"
            meta_path = OUT / "ou_classifier_lr.json"
    if not model_path.exists() or not meta_path.exists():
        return {"error": "Classifier not trained"}
    pipe = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    feat_num = meta.get("features_num", [])
    feat_cat = meta.get("features_cat", [])
    dates = _collect_dates(n_days)
    frames = []
    for d in dates:
        j = _join_features_results(d)
        if not j.empty:
            frames.append(j)
    if not frames:
        return {"error": "No evaluation data"}
    df = pd.concat(frames, ignore_index=True)
    # Prefer market_total; fallback to closing_total when market_total is missing
    if "market_total" not in df.columns and "closing_total" in df.columns:
        df["market_total"] = pd.to_numeric(df.get("closing_total"), errors="coerce")
    elif "market_total" in df.columns and "closing_total" in df.columns:
        mt_raw = pd.to_numeric(df.get("market_total"), errors="coerce")
        ct_raw = pd.to_numeric(df.get("closing_total"), errors="coerce")
        df["market_total"] = mt_raw.where(mt_raw.notna(), ct_raw)
    mt = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else pd.Series(dtype=float)
    at = pd.to_numeric(df.get("actual_total"), errors="coerce") if "actual_total" in df.columns else pd.Series(dtype=float)
    mask = mt.notna() & at.notna()
    df = df.loc[mask].copy()
    if df.empty:
        return {"error": "No evaluation rows after mask"}
    # Ensure all expected feature columns exist
    for c in feat_num + feat_cat:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feat_num + feat_cat]
    proba = pipe.predict_proba(X)[:, 1]
    df["p_over"] = proba
    y = (df["actual_total"] > df["market_total"]).astype(int)
    out = {"count": int(len(df))}
    for t in thresholds:
        pick_mask = df["p_over"].ge(t) | df["p_over"].le(1.0 - t)
        picks = df.loc[pick_mask]
        if picks.empty:
            out[f"thr_{t}"] = {"count": 0, "correct": 0, "pct": None}
        else:
            # Over picks when p>=t, Under picks when p<=1-t
            pred = np.where(picks["p_over"].ge(t), 1, 0)
            correct = int(np.sum(pred == ((picks["actual_total"] > picks["market_total"]).astype(int)).values))
            out[f"thr_{t}"] = {"count": int(len(picks)), "correct": correct, "pct": round(correct / len(picks), 4)}
            # Combined gating with edge on calibrated vs market if available
            if "delta_cal_market" in df.columns:
                for e in edge_requirements:
                    emask = picks["delta_cal_market"].abs().ge(e)
                    subset = picks.loc[emask]
                    if subset.empty:
                        out[f"thr_{t}_edge_{e}"] = {"count": 0, "correct": 0, "pct": None}
                    else:
                        pred2 = np.where(subset["p_over"].ge(t), 1, 0)
                        corr2 = int(np.sum(pred2 == ((subset["actual_total"] > subset["market_total"]).astype(int)).values))
                        out[f"thr_{t}_edge_{e}"] = {"count": int(len(subset)), "correct": corr2, "pct": round(corr2 / len(subset), 4)}
    return out

def main():
    import sys
    n_days = 60
    model = "auto"
    if len(sys.argv) >= 2:
        try:
            n_days = int(sys.argv[1])
        except Exception:
            n_days = 60
    if len(sys.argv) >= 3:
        model_arg = sys.argv[2].strip().lower()
        if model_arg in ("lr","hgb"):
            model = model_arg
    res = evaluate(n_days, thresholds=(0.6, 0.65, 0.7), edge_requirements=(5,7,10), model=model)
    OUT.joinpath("eval_ou_classifier.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))

if __name__ == "__main__":
    main()