from __future__ import annotations
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Optional ML backends
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
except Exception:
    RandomForestRegressor = None  # type: ignore
    GradientBoostingRegressor = None  # type: ignore
    train_test_split = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

LABEL = "actual_total"

EXCLUDE_COLS = set([
    # identifiers & meta
    "game_id", "date", "home_team", "away_team", "conference",
    "_src", "status", "venue", "neutral_site", "display_date",
    # labels & scores
    "home_score", "away_score", "home_score_1h", "away_score_1h", "home_score_2h", "away_score_2h",
    "actual_total_1h", "actual_total_2h", LABEL,
    # odds/targets that can leak
    "market_total", "closing_total", "spread_home", "closing_spread_home",
    "total_1h", "total_2h", "closing_total_1h", "closing_total_2h",
    # pipeline predictions to avoid target leakage
    "pred_total", "pred_total_calibrated", "pred_margin", "pred_total_1h", "pred_total_2h",
    # common derived error columns
    "err_model_total", "err_model_total_1h", "err_model_total_2h",
])

@dataclasses.dataclass
class TrainConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    use_all: bool = True
    recent: Optional[int] = None
    model_name: str = "totals_v1"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _resolve_feature_files(start: Optional[str], end: Optional[str], use_all: bool, recent: Optional[int]) -> list[Path]:
    # Prefer all-season, else per-date selection
    if use_all:
        # Prefer augmented variants when available
        for name in [
            "features_all_augmented.csv",
            "features_hist_augmented.csv",
            "features_history_augmented.csv",
            "features_all.csv",
            "features_hist.csv",
            "features_history.csv",
        ]:
            p = OUT / name
            if p.exists():
                return [p]
    # Date-range or recent
    files = sorted([p for p in OUT.glob("features_*.csv")])
    if recent:
        # filter valid YYYY-MM-DD tokens
        tokens = [(p, (lambda t: t if len(t) == 10 and t[4] == '-' and t[7] == '-' else None)(p.stem.split("_")[-1])) for p in files]
        dates = [t for (_, t) in tokens if t]
        keep = set(dates[-recent:])
        return [p for p in files if p.stem.split("_")[-1] in keep]
    # fallback to current
    for name in ["features_curr_augmented.csv", "features_curr.csv"]:
        p = OUT / name
        if p.exists():
            return [p]
    return []


def _resolve_results_files(start: Optional[str], end: Optional[str], recent: Optional[int]) -> list[Path]:
    daily = OUT / "daily_results"
    files = sorted([p for p in daily.glob("results_*.csv")])
    if recent:
        tokens = [(p, (lambda t: t if len(t) == 10 and t[4] == '-' and t[7] == '-' else None)(p.stem.split("_")[-1])) for p in files]
        dates = [t for (_, t) in tokens if t]
        keep = set(dates[-recent:])
        return [p for p in files if p.stem.split("_")[-1] in keep]
    return files


def _coerce_actual_total(df: pd.DataFrame) -> pd.Series:
    for cand in (LABEL, "final_total", "total_final"):
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    # derive from scores if present
    score_cols_h = [c for c in df.columns if c.lower() in ("home_score", "home_points")]
    score_cols_a = [c for c in df.columns if c.lower() in ("away_score", "away_points")]
    if score_cols_h and score_cols_a:
        return pd.to_numeric(df[score_cols_h[0]], errors="coerce") + pd.to_numeric(df[score_cols_a[0]], errors="coerce")
    return pd.Series([np.nan] * len(df))


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    # numeric-only features, excluding known leakage/labels
    num = df.select_dtypes(include=[np.number]).copy()
    cols = [c for c in num.columns if c not in EXCLUDE_COLS]
    # Ensure basic factor coverage if present
    # pace/tempo, shooting, offense/defense ratings, fouls, rebounding, rest, venue
    return num[cols]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {"count": 0}
    mae = float(mean_absolute_error(y_true, y_pred)) if mean_absolute_error else float(np.nanmean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred))) if mean_squared_error else float(np.sqrt(np.nanmean(np.square(y_true - y_pred))))
    bias = float(np.nanmean(y_pred - y_true))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.isfinite(y_true).all() and np.isfinite(y_pred).all() else None
    return {"count": int(y_true.size), "mae": round(mae, 3), "rmse": round(rmse, 3), "bias": round(bias, 3), "corr": corr}


class TotalsModel:
    def __init__(self, name: str = "totals_v1"):
        self.name = name
        self.mean_model = None
        self.q_models: dict[str, Any] = {}
        self.feature_cols: list[str] = []
        self.feature_means: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        self.feature_cols = list(X.columns)
        # Cache feature means for missing-value imputation at predict time
        try:
            self.feature_means = {c: float(pd.to_numeric(X[c], errors="coerce").mean()) for c in self.feature_cols}
        except Exception:
            self.feature_means = {c: 0.0 for c in self.feature_cols}
        # Mean model
        if lgb is not None:
            params = {
                "objective": "regression",
                "num_leaves": 64,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "metric": "rmse",
                "verbose": -1,
            }
            dtrain = lgb.Dataset(X.values, label=y.values)
            self.mean_model = lgb.train(params, dtrain, num_boost_round=400)
        elif RandomForestRegressor is not None:
            self.mean_model = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)
            self.mean_model.fit(X.values, y.values)
        else:
            raise RuntimeError("No ML backend available for mean model")
        # Quantile models if backend supports
        if lgb is not None:
            for q in (0.1, 0.5, 0.9):
                params_q = {
                    "objective": "quantile",
                    "alpha": q,
                    "num_leaves": 64,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "metric": "quantile",
                    "verbose": -1,
                }
                dq = lgb.Dataset(X.values, label=y.values)
                self.q_models[str(q)] = lgb.train(params_q, dq, num_boost_round=400)
        elif GradientBoostingRegressor is not None:
            for q in (0.1, 0.5, 0.9):
                m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300, max_depth=3, learning_rate=0.05)
                m.fit(X.values, y.values)
                self.q_models[str(q)] = m
        # Metrics on training split (or holdout if we have sklearn)
        try:
            if train_test_split is not None:
                Xtr, Xte, ytr, yte = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
                if hasattr(self.mean_model, "predict"):
                    yhat = self.mean_model.predict(Xte)
                else:
                    yhat = self.mean_model.predict(Xte)
                metrics = _metrics(yte, yhat)
            else:
                yhat = self.mean_model.predict(X.values)
                metrics = _metrics(y.values, yhat)
        except Exception:
            metrics = {"count": int(len(y))}
        return metrics

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.feature_cols:
            raise RuntimeError("Model not fitted")
        # Align columns: reindex to training feature set and fill missing with training means
        try:
            Xv_df = X.reindex(columns=self.feature_cols)
            # Fill missing columns
            for c in self.feature_cols:
                if c not in X.columns:
                    Xv_df[c] = self.feature_means.get(c, 0.0)
            # Fill NaNs within present columns
            for c in self.feature_cols:
                if Xv_df[c].isna().any():
                    Xv_df[c] = Xv_df[c].fillna(self.feature_means.get(c, 0.0))
            Xv = Xv_df[self.feature_cols].values
        except Exception:
            # Fallback: best-effort take intersection
            inter = [c for c in self.feature_cols if c in X.columns]
            Xv = X[inter].values
        yhat = self.mean_model.predict(Xv)
        out: dict[str, np.ndarray] = {"mean": np.asarray(yhat)}
        for k, m in self.q_models.items():
            try:
                out[f"q{k}"] = np.asarray(m.predict(Xv))
            except Exception:
                pass
        return out

    def save(self, path: Path) -> None:
        try:
            import joblib  # type: ignore
        except Exception:
            raise RuntimeError("joblib not available for saving model")
        payload = {
            "name": self.name,
            "feature_cols": self.feature_cols,
            "feature_means": self.feature_means,
            "backend": "lightgbm" if lgb is not None else ("sklearn_rf" if RandomForestRegressor is not None else "unknown"),
        }
        (path.parent).mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg": payload, "mean": self.mean_model, "q": self.q_models}, path)

    @staticmethod
    def load(path: Path) -> "TotalsModel":
        import joblib  # type: ignore
        blob = joblib.load(path)
        m = TotalsModel(name=str(blob.get("cfg", {}).get("name", "totals_v1")))
        m.feature_cols = list(blob.get("cfg", {}).get("feature_cols", []))
        m.feature_means = dict(blob.get("cfg", {}).get("feature_means", {}))
        m.mean_model = blob.get("mean")
        m.q_models = blob.get("q", {})
        return m


def train_full_season(cfg: TrainConfig) -> dict[str, Any]:
    feat_files = _resolve_feature_files(cfg.start, cfg.end, cfg.use_all, cfg.recent)
    res_files = _resolve_results_files(cfg.start, cfg.end, cfg.recent)
    feats_list = [(_safe_read_csv(p).assign(_src=str(p))) for p in feat_files]
    results_list = [(_safe_read_csv(p).assign(_src=str(p))) for p in res_files]
    feats = pd.concat([d for d in feats_list if not d.empty], ignore_index=True) if feats_list else pd.DataFrame()
    results = pd.concat([d for d in results_list if not d.empty], ignore_index=True) if results_list else pd.DataFrame()
    if feats.empty or results.empty:
        return {"error": "Missing features or results", "features_rows": len(feats), "results_rows": len(results)}
    # Normalize keys
    for d in (feats, results):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
    join_keys = [k for k in ("game_id", "date") if k in feats.columns and k in results.columns]
    if not join_keys:
        join_keys = ["game_id"]
    df = feats.merge(results, on=join_keys, how="inner", suffixes=("", "_res"))
    if df.empty:
        return {"error": "Join produced no rows"}
    df[LABEL] = _coerce_actual_total(df)
    df = df[pd.to_numeric(df[LABEL], errors="coerce").notna()].copy()
    X = _select_features(df)
    y = pd.to_numeric(df[LABEL], errors="coerce")
    # Optional sample weights: use quotes_count if present
    weights = None
    if "quotes_count" in df.columns:
        try:
            qc = pd.to_numeric(df["quotes_count"], errors="coerce").fillna(0)
            weights = np.clip(1 + qc, 1.0, 10.0).values
        except Exception:
            weights = None
    model = TotalsModel(cfg.model_name)
    metrics = model.fit(X, y)
    # Save model
    model_path = MODELS / f"{cfg.model_name}.joblib"
    model.save(model_path)
    out = {
        "model": cfg.model_name,
        "features_rows": int(len(feats)),
        "results_rows": int(len(results)),
        "train_rows": int(len(df)),
        "metrics": metrics,
        "feature_cols": model.feature_cols[:40],
        "model_path": str(model_path),
    }
    (OUT / f"train_totals_{cfg.model_name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
