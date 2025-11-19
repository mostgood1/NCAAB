"""Training script for margin (home - away) prediction model.
Usage:
  python -m src.modeling.train_margin --algo auto
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any, Dict
import numpy as np

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    import numpy as _np
    def mean_absolute_error(y_true, y_pred):  # type: ignore
        return float(_np.mean(_np.abs(_np.asarray(y_true) - _np.asarray(y_pred))))
    def mean_squared_error(y_true, y_pred, squared=True):  # type: ignore
        mse = float(_np.mean((_np.asarray(y_true) - _np.asarray(y_pred)) ** 2))
        return mse if squared else mse ** 0.5
    def train_test_split(X, y, test_size=0.25, random_state=42):  # type: ignore
        rng = _np.random.default_rng(random_state)
        idx = _np.arange(len(X))
        rng.shuffle(idx)
        n_test = int(len(X) * test_size)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    GradientBoostingRegressor = None  # type: ignore
from joblib import dump

from .data import build_training_frame

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


class _ZeroBaseline:
    def predict(self, X_any):  # noqa: ANN001
        import numpy as _np
        return _np.zeros(len(X_any))


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


def pick_algo(preference: str):
    if preference == "auto":
        if lgb is not None:
            return "lightgbm"
        if xgb is not None:
            return "xgboost"
        if SKLEARN_AVAILABLE:
            return "gbr"
        return "baseline"
    if preference == "gbr" and not SKLEARN_AVAILABLE:
        if lgb is not None:
            return "lightgbm"
        if xgb is not None:
            return "xgboost"
        return "baseline"
    return preference if preference in {"lightgbm","xgboost","gbr","baseline"} else "baseline"


def train(algo: str) -> Dict[str, Any]:
    X, _, y_margin = build_training_frame()
    if X.empty or y_margin.empty:
        print("No training data found (features or games missing).", file=sys.stderr)
        return {"status":"no_data"}
    X_train, X_test, y_train, y_test = train_test_split(X, y_margin, test_size=0.25, random_state=42)

    if algo == "lightgbm" and lgb is not None:
        if SKLEARN_AVAILABLE and hasattr(lgb, "LGBMRegressor"):
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
        else:
            train_set = lgb.Dataset(X_train, label=y_train)
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.04,
                "num_leaves": 64,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "seed": 42,
            }
            booster = lgb.train(params, train_set, num_boost_round=500)
            model = _BoosterWrapper(booster)
    elif algo == "xgboost" and xgb is not None:
        if SKLEARN_AVAILABLE and hasattr(xgb, "XGBRegressor"):
            model = xgb.XGBRegressor(
                n_estimators=700,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
            )
            model.fit(X_train, y_train)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            params = {
                "objective": "reg:squarederror",
                "eta": 0.03,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
                "seed": 42,
            }
            booster = xgb.train(params, dtrain, num_boost_round=700)
            model = _XGBWrapper(booster)
    elif algo == "gbr" and SKLEARN_AVAILABLE and GradientBoostingRegressor is not None:
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)
    else:
        model = _ZeroBaseline()

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    baseline_mae = float(mean_absolute_error(y_test, np.full_like(y_test, 0)))  # Zero margin baseline
    artifact_dir = OUT / "models" / dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dump(model, artifact_dir / "margin_model.pkl")
    meta = {
        "algo": algo,
        "features": list(X.columns),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "status":"ok"
    }
    with open(artifact_dir / "margin_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="auto", help="Preferred algo: auto|lightgbm|xgboost|gbr")
    args = ap.parse_args()
    algo = pick_algo(args.algo)
    train(algo)

if __name__ == "__main__":
    main()
