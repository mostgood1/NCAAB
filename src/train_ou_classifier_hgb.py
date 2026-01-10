from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
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
    # Build minimal features from enriched when per-date features missing
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
    for d in (feats, res, enrich):
        if isinstance(d, pd.DataFrame) and not d.empty:
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
    # Merge augmented pace/poss/TO/DRB features
    if not aug_recent.empty:
        for d in (aug_recent,):
            # normalize date to str for join
            if "date" in d.columns:
                try:
                    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                except Exception:
                    d["date"] = d["date"].astype(str)
            if "game_id" in d.columns:
                d["game_id"] = d["game_id"].astype(str)
        # filter to this date
        aug_d = aug_recent[aug_recent.get("date").astype(str) == date]
        use_cols = [c for c in ["game_id","date","home_pace","home_ts","home_3p_rate","home_to_rate","home_drb_rate",
                                 "away_pace","away_ts","away_3p_rate","away_to_rate","away_drb_rate",
                                 "home_b2b","away_b2b","tz_offset_hours","home_adv"] if c in aug_d.columns]
        if use_cols:
            df = df.merge(aug_d[use_cols], on=[k for k in keys if k in ["game_id","date"]], how="left")
    if not enrich.empty:
        use_cols = keys + [c for c in ("q10","q50","q90","pred_total_model","pred_total_calibrated","pred_total_basis") if c in enrich.columns]
        df = df.merge(enrich[use_cols], on=keys, how="left")
        if "q50" in df.columns and "market_total" in df.columns:
            df["delta_q50_market"] = pd.to_numeric(df["q50"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
        if "pred_total_calibrated" in df.columns and "market_total" in df.columns:
            df["delta_cal_market"] = pd.to_numeric(df["pred_total_calibrated"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
    return df

def main():
    import sys
    exclude_last = 0
    n_days = 120
    if len(sys.argv) >= 2:
        try:
            exclude_last = int(sys.argv[1])
        except Exception:
            exclude_last = 0
    if len(sys.argv) >= 3:
        try:
            n_days = int(sys.argv[2])
        except Exception:
            n_days = 120
    dates = _collect_dates(n_days)
    if exclude_last > 0 and len(dates) > exclude_last:
        dates = dates[:-exclude_last]
    frames = []
    for d in dates:
        j = _join_features_results(d)
        if not j.empty:
            frames.append(j)
    if not frames:
        print(json.dumps({"error": "No training data"}))
        return
    df = pd.concat(frames, ignore_index=True)
    mt = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else pd.Series(dtype=float)
    at = pd.to_numeric(df.get("actual_total"), errors="coerce") if "actual_total" in df.columns else pd.Series(dtype=float)
    mask = mt.notna() & at.notna()
    df = df.loc[mask].copy()
    y = (at.loc[mask] > mt.loc[mask]).astype(int)
    feature_cols_num = [c for c in [
        "away_tempo_rating","home_tempo_rating","tempo_rating_sum",
        "away_off_rating","home_off_rating","off_rating_diff","def_rating_diff",
        "away_efg5","home_efg5","away_ftr5","home_ftr5",
        "away_orb_rate5","home_orb_rate5",
        "rest_home","rest_away",
        "market_total","closing_total",
        "q10","q50","q90","delta_q50_market","delta_cal_market",
        # augmented pace/TO/DRB and shooting proxies
        "home_pace","home_ts","home_3p_rate","home_to_rate","home_drb_rate",
        "away_pace","away_ts","away_3p_rate","away_to_rate","away_drb_rate",
    ] if c in df.columns]
    feature_cols_cat = [c for c in ["neutral_site","start_tz_abbr"] if c in df.columns]
    if not feature_cols_num:
        exclude = set(["actual_total","market_total","closing_total","date","game_id"])
        feature_cols_num = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:64]
    # Drop categorical if no observed values
    if feature_cols_cat:
        has_values = any(df[c].notna().any() for c in feature_cols_cat)
        if not has_values:
            feature_cols_cat = []
    # Preprocessor
    transformers = []
    if feature_cols_num:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_cols_num))
    if feature_cols_cat:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), feature_cols_cat))
    preprocessor = ColumnTransformer(transformers=transformers)
    base = HistGradientBoostingClassifier(loss="log_loss", max_depth=None, learning_rate=0.06, max_iter=300)
    # Newer sklearn uses 'estimator' instead of 'base_estimator'
    try:
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    X = df[feature_cols_num + feature_cols_cat]
    pipe.fit(X, y)
    # In-sample accuracy (baseline visibility only)
    acc = float(accuracy_score(y, pipe.predict(X)))
    payload = {"train_count": int(len(df)), "train_acc": acc, "features_num": feature_cols_num, "features_cat": feature_cols_cat, "model": "hgb_isotonic"}
    joblib.dump(pipe, OUT / "ou_classifier_hgb.joblib")
    OUT.joinpath("ou_classifier_hgb.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))

if __name__ == "__main__":
    main()