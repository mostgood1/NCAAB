"""Inference harness: generates model-only predictions for an upcoming slate.

Usage:
  python -m src.modeling.infer --date 2025-11-19 --models-dir outputs/models

Writes: outputs/predictions_model_<date>.csv with columns:
  game_id, pred_total_model, pred_margin_model

If models are missing, exits gracefully.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import datetime as dt
import pandas as pd
import numpy as np
from joblib import load
from . import train_total as _train_total_mod  # noqa: F401  # Ensure custom wrapper classes are registered
from . import train_margin as _train_margin_mod  # noqa: F401

from .data import load_features

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

# Wrapper classes reproduced to satisfy pickle resolution when training occurred under __main__
class _BoosterWrapper:
    def __init__(self, booster):
        self._b = booster
    def predict(self, X_any):  # noqa: ANN001
        return self._b.predict(X_any)

class _XGBWrapper:
    def __init__(self, booster):
        self._b = booster
    def predict(self, X_any):  # noqa: ANN001
        import xgboost as _xgb  # type: ignore
        return self._b.predict(_xgb.DMatrix(X_any))

class _MeanBaseline:
    def __init__(self, value: float):
        self.value = value
    def predict(self, X_any):  # noqa: ANN001
        import numpy as _np
        return _np.full(len(X_any), self.value)

class _ZeroBaseline:
    def predict(self, X_any):  # noqa: ANN001
        import numpy as _np
        return _np.zeros(len(X_any))


def _latest_model_path(name: str, models_root: pathlib.Path) -> pathlib.Path | None:
    candidates = sorted(models_root.glob("*/" + name), reverse=True)
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--models-dir", default=str(OUT / "models"))
    ap.add_argument("--features-file", default="features_curr.csv", help="Feature file for upcoming slate")
    args = ap.parse_args()
    date_str = args.date
    models_root = pathlib.Path(args.models_dir)
    feat_file = OUT / args.features_file if not pathlib.Path(args.features_file).is_absolute() else pathlib.Path(args.features_file)

    if not models_root.exists():
        print("Models directory missing", file=sys.stderr)
        sys.exit(0)
    total_p = _latest_model_path("total_model.pkl", models_root)
    margin_p = _latest_model_path("margin_model.pkl", models_root)
    if not (total_p and total_p.exists() and margin_p and margin_p.exists()):
        print("Model artifacts not found", file=sys.stderr)
        sys.exit(0)
    try:
        total_model = load(total_p)
    except Exception as e:
        print(f"Failed loading total model: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        margin_model = load(margin_p)
    except Exception as e:
        print(f"Failed loading margin model: {e}", file=sys.stderr)
        sys.exit(1)

    feat_df = load_features()
    if feat_df.empty:
        print("Features empty", file=sys.stderr)
        sys.exit(0)
    # Filter to requested date if a date column exists
    if "date" in feat_df.columns:
        feat_df["date"] = pd.to_datetime(feat_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        slate_df = feat_df[feat_df["date"] == date_str].copy()
    else:
        slate_df = feat_df.copy()
    if slate_df.empty:
        # Attempt fallback: explicitly load features_curr.csv if initial source lacked date rows
        try:
            fallback_path = OUT / "features_curr.csv"
            if fallback_path.exists():
                f2 = pd.read_csv(fallback_path)
                if "date" in f2.columns:
                    f2["date"] = pd.to_datetime(f2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                slate_df = f2[f2.get("date").astype(str) == date_str] if "date" in f2.columns else f2
        except Exception:
            slate_df = pd.DataFrame()
        if slate_df.empty:
            print("No feature rows for date", file=sys.stderr)
            sys.exit(0)
    slate_df["game_id"] = slate_df.get("game_id").astype(str)
    # Build model input features (mirror training selection logic)
    cols_needed = [
        "home_off_rating","away_off_rating","home_def_rating","away_def_rating",
        "home_tempo_rating","away_tempo_rating","tempo_rating_sum"
    ]
    for c in cols_needed:
        if c not in slate_df.columns:
            slate_df[c] = np.nan
    X = pd.DataFrame({
        "home_off_rating": slate_df["home_off_rating"],
        "away_off_rating": slate_df["away_off_rating"],
        "home_def_rating": slate_df["home_def_rating"],
        "away_def_rating": slate_df["away_def_rating"],
        "off_diff": slate_df["home_off_rating"] - slate_df["away_off_rating"],
        "def_diff": slate_df["home_def_rating"] - slate_df["away_def_rating"],
        "tempo_avg": (pd.to_numeric(slate_df["home_tempo_rating"], errors="coerce") + pd.to_numeric(slate_df["away_tempo_rating"], errors="coerce")) / 2.0,
        "tempo_rating_sum": slate_df["tempo_rating_sum"],
    })
    X = X.apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    try:
        pred_total = total_model.predict(X)
    except Exception:
        pred_total = np.full(len(X), np.nan)
    try:
        pred_margin = margin_model.predict(X)
    except Exception:
        pred_margin = np.full(len(X), np.nan)
    out_df = pd.DataFrame({
        "game_id": slate_df["game_id"],
        "pred_total_model": pred_total,
        "pred_margin_model": pred_margin,
        "pred_total_model_basis": "model_v1"
    })
    out_path = OUT / f"predictions_model_{date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
