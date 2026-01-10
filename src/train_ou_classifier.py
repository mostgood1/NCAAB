from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
    feats = _safe_read_csv(OUT / f"features_{date}.csv")
    res = _safe_read_csv(OUT / f"daily_results/results_{date}.csv")
    enrich = _safe_read_csv(OUT / f"predictions_unified_enriched_{date}.csv")
    if feats.empty or res.empty:
        return pd.DataFrame()
    for d in (feats, res):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
    keys = [k for k in ("game_id","date") if k in feats.columns and k in res.columns]
    if not keys:
        keys = ["game_id"]
    # Select needed result columns
    keep = keys + [c for c in ("actual_total","final_total","market_total","closing_total") if c in res.columns]
    res_min = res[keep].copy()
    if "actual_total" not in res_min.columns and "final_total" in res_min.columns:
        res_min = res_min.rename(columns={"final_total": "actual_total"})
    df = feats.merge(res_min, on=keys, how="inner")
    if not enrich.empty:
        for d in (enrich,):
            if "game_id" in d.columns:
                d["game_id"] = d["game_id"].astype(str)
            if "date" in d.columns:
                d["date"] = d["date"].astype(str)
        use_cols = keys + [c for c in ("q10","q50","q90","pred_total_model","pred_total_calibrated","pred_total_basis") if c in enrich.columns]
        df = df.merge(enrich[use_cols], on=keys, how="left")
        # Derived deltas
        if "q50" in df.columns and "market_total" in df.columns:
            df["delta_q50_market"] = pd.to_numeric(df["q50"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
        if "pred_total_calibrated" in df.columns and "market_total" in df.columns:
            df["delta_cal_market"] = pd.to_numeric(df["pred_total_calibrated"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
    return df

def main():
    import sys
    exclude_last = 0
    n_days = 60
    if len(sys.argv) >= 2:
        try:
            exclude_last = int(sys.argv[1])
        except Exception:
            exclude_last = 0
    if len(sys.argv) >= 3:
        try:
            n_days = int(sys.argv[2])
        except Exception:
            n_days = 60
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
    # Target: actual_total > market_total -> 1 else 0
    y = None
    mt = pd.to_numeric(df.get("market_total"), errors="coerce") if "market_total" in df.columns else pd.Series(dtype=float)
    at = pd.to_numeric(df.get("actual_total"), errors="coerce") if "actual_total" in df.columns else pd.Series(dtype=float)
    mask = mt.notna() & at.notna()
    df = df.loc[mask].copy()
    y = (at.loc[mask] > mt.loc[mask]).astype(int)
    # Feature columns (present in audit)
    feature_cols_num = [c for c in [
        "away_tempo_rating","home_tempo_rating","tempo_rating_sum",
        "away_off_rating","home_off_rating","off_rating_diff","def_rating_diff",
        "away_efg5","home_efg5","away_ftr5","home_ftr5",
        "away_orb_rate5","home_orb_rate5",
        "rest_home","rest_away",
        "market_total","closing_total",
        "q10","q50","q90","delta_q50_market","delta_cal_market",
    ] if c in df.columns]
    feature_cols_cat = [c for c in ["neutral_site","start_tz_abbr"] if c in df.columns]
    # Fallback: select generic numeric features if lists are empty
    if not feature_cols_num:
        exclude = set(["actual_total","market_total","closing_total","date","game_id"])
        feature_cols_num = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])][:32]
    # Build transformers only for non-empty sets
    X_num = df[feature_cols_num]
    X_cat = df[feature_cols_cat]
    # If categorical columns have no observed values, drop them
    if feature_cols_cat:
        has_values = any(df[c].notna().any() for c in feature_cols_cat)
        if not has_values:
            feature_cols_cat = []
            X_cat = pd.DataFrame(index=df.index)
    # Preprocess
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if feature_cols_num:
        transformers.append(("num", num_pipeline, feature_cols_num))
    if feature_cols_cat:
        transformers.append(("cat", cat_pipeline, feature_cols_cat))
    preprocessor = ColumnTransformer(transformers=transformers)
    base_clf = LogisticRegression(max_iter=200, class_weight="balanced")
    try:
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3)
    except Exception:
        clf = base_clf
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    pipe.fit(df[feature_cols_num + feature_cols_cat], y)
    # Simple holdout accuracy using training set (for baseline visibility)
    y_pred = pipe.predict(df[feature_cols_num + feature_cols_cat])
    acc = float(accuracy_score(y, y_pred))
    payload = {"train_count": int(len(df)), "train_acc": acc, "features_num": feature_cols_num, "features_cat": feature_cols_cat}
    joblib.dump(pipe, OUT / "ou_classifier_lr.joblib")
    OUT.joinpath("ou_classifier_lr.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))

if __name__ == "__main__":
    main()