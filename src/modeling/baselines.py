from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss
except Exception:  # pragma: no cover
    Pipeline = object  # type: ignore
    TimeSeriesSplit = object  # type: ignore
    SimpleImputer = object  # type: ignore
    StandardScaler = object  # type: ignore
    GradientBoostingClassifier = object  # type: ignore
    log_loss = None  # type: ignore

from .metrics import classification_metrics


LEAK_PATTERNS = (
    r"score", r"actual", r"_outcome", r"ou_", r"ats_",
    r"closing_",  # optionally allow later
)


def choose_feature_columns(df: pd.DataFrame, allow_market: bool = False) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        # Skip obvious identifiers or text
        lc = str(c).lower()
        if lc in {"game_id", "home_team", "away_team", "venue", "city", "state", "_source_file"}:
            continue
        # target columns will be selected explicitly, so exclude those by pattern
        if any(pat in lc for pat in LEAK_PATTERNS):
            if allow_market and lc.startswith("closing_"):
                pass
            else:
                continue
        # keep numeric only
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _time_series_splits(dates: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Robust time-aware splitting based on chronological order of rows
    if dates.isna().all():
        # fallback to simple chunking
        n = len(dates)
        fold_size = max(1, n // (n_splits + 1))
        splits = []
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_end = min(n, (i + 1) * fold_size)
            if train_end >= n:
                break
            tr = np.arange(0, train_end)
            te = np.arange(train_end, test_end)
            if len(te) == 0:
                break
            splits.append((tr, te))
        return splits
    order = np.argsort(pd.to_datetime(dates, errors="coerce").values)
    n = len(order)
    fold_size = max(1, n // (n_splits + 1))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = min(n, (i + 1) * fold_size)
        if train_end >= n:
            break
        tr = order[:train_end]
        te = order[train_end:test_end]
        if len(te) == 0:
            break
        splits.append((tr, te))
    return splits


def train_eval_classifier(
    df: pd.DataFrame,
    target_col: str,
    allow_market: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float], List[Dict[str, float]]]:
    """
    Train GradientBoostingClassifier with time-aware CV and return fitted pipeline,
    average metrics, and per-fold metrics list.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in DataFrame")

    # Drop rows missing target
    work = df.copy()
    work = work.dropna(subset=[target_col])
    if work.empty:
        raise ValueError(f"No rows with target '{target_col}' available for training")

    # Feature selection
    feats = choose_feature_columns(work, allow_market=allow_market)
    if not feats:
        raise ValueError("No numeric feature columns found after leakage filtering")

    X = work[feats]
    y = work[target_col].astype(int)

    # Build pipeline
    model = GradientBoostingClassifier(random_state=random_state)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    # Time-aware CV
    splits = _time_series_splits(work["date"]) if "date" in work.columns else _time_series_splits(pd.Series([pd.NaT] * len(work)))

    fold_metrics: List[Dict[str, float]] = []
    for tr, te in splits:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)
        try:
            prob = pipe.predict_proba(Xte)[:, 1]
        except Exception:
            # Fallback when predict_proba unsupported
            pred = pipe.predict(Xte)
            prob = (pred.astype(float) + 1e-6)  # crude proxy
        fold_metrics.append(classification_metrics(yte.values, prob))

    # Fit on full dataset for final artifact
    pipe.fit(X, y)

    # Average metrics
    avg: Dict[str, float] = {}
    if fold_metrics:
        keys = sorted({k for fm in fold_metrics for k in fm.keys()})
        for k in keys:
            vals = [fm.get(k, np.nan) for fm in fold_metrics]
            avg[k] = float(np.nanmean(vals))

    return pipe, avg, fold_metrics
