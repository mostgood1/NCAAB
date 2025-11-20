"""Prediction variance diagnostics.

Computes distribution statistics for model predictions on the current training frame
(using LightGBM baseline or chosen algorithm) WITHOUT persisting artifacts, meant for
monitoring uniformity (are predictions too clustered?).

Run:
  python -m src.modeling.diagnose_variance --target total --algo auto
  python -m src.modeling.diagnose_variance --target margin --split date
"""
from __future__ import annotations
import argparse, statistics, json, pathlib, datetime as dt
import numpy as np
from .data import build_training_frame
from .train_total import pick_algo as pick_algo_total  # reuse selection logic
from .train_margin import pick_algo as pick_algo_margin

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _split(X, y, dates, split: str, test_size: float):
    if split == "date" and dates is not None and not dates.isna().all():
        order = np.argsort(dates.fillna(dates.max()))
        n_test = int(len(X) * test_size)
        test_idx = order[-n_test:]
        train_idx = order[:-n_test]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    else:
        try:
            from sklearn.model_selection import train_test_split  # type: ignore
            return train_test_split(X, y, test_size=test_size, random_state=42)
        except Exception:
            rng = np.random.default_rng(42)
            idx = np.arange(len(X))
            rng.shuffle(idx)
            n_test = int(len(X) * test_size)
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _train_quick(X_train, y_train, algo: str):
    if algo == "lightgbm" and lgb is not None:
        try:
            if hasattr(lgb, "LGBMRegressor"):
                m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
                m.fit(X_train, y_train)
                return m
        except Exception:
            pass
        ds = lgb.Dataset(X_train, label=y_train)
        params = {"objective":"regression","learning_rate":0.05,"num_leaves":64,"metric":"l2","seed":42}
        booster = lgb.train(params, ds, num_boost_round=200)
        class _W:  # wrapper
            def __init__(self,b): self._b=b
            def predict(self,X): return self._b.predict(X)
        return _W(booster)
    if algo == "xgboost" and xgb is not None:
        if hasattr(xgb, "XGBRegressor"):
            m = xgb.XGBRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
            m.fit(X_train, y_train)
            return m
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            params = {"objective":"reg:squarederror","eta":0.05,"max_depth":6,"seed":42}
            booster = xgb.train(params, dtrain, num_boost_round=250)
            class _W:  # wrapper
                def __init__(self,b): self._b=b
                def predict(self,X):
                    import xgboost as _xgb
                    return self._b.predict(_xgb.DMatrix(X))
            return _W(booster)
    # Fallback mean
    class _Mean:
        def __init__(self,v): self.v=float(v)
        def predict(self,X): return np.full(len(X), self.v)
    return _Mean(np.mean(y_train))


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["total","margin"], default="total")
    ap.add_argument("--algo", default="auto")
    ap.add_argument("--split", default="random", choices=["random","date"], help="Split strategy")
    args = ap.parse_args()

    X, y_total, y_margin, dates = build_training_frame(return_dates=(args.split == "date"))
    if X.empty:
        print(json.dumps({"status":"no_data"}))
        return
    y = y_total if args.target == "total" else y_margin
    split_used = args.split
    from .train_total import pick_algo as _pick_total
    from .train_margin import pick_algo as _pick_margin
    algo = (_pick_total if args.target == "total" else _pick_margin)(args.algo)
    X_train, X_test, y_train, y_test = _split(X, y, dates, split_used, test_size=0.25 if args.target=="margin" else 0.2)
    model = _train_quick(X_train, y_train, algo)
    preds = model.predict(X_test)
    preds = np.asarray(preds)
    variance = float(np.var(preds))
    stdev = float(np.std(preds))
    iqr = float(np.percentile(preds, 75) - np.percentile(preds, 25))
    summary = {
        "target": args.target,
        "algo": algo,
        "split": split_used,
        "test_size": len(X_test),
        "pred_variance": variance,
        "pred_std": stdev,
        "pred_iqr": iqr,
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "status": "ok"
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":  # pragma: no cover
    main()
