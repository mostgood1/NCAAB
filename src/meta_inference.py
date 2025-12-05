import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    from .meta_schema import load_feature_schema, build_ordered_matrix  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for script-style imports when src isn't treated as a package
    from meta_schema import load_feature_schema, build_ordered_matrix  # type: ignore

OUT = Path(os.getenv("NCAAB_OUTPUTS_DIR", Path(__file__).resolve().parents[1] / "outputs"))

COVER_MODEL = OUT / "meta_cover_lgbm.joblib"
OVER_MODEL = OUT / "meta_over_lgbm.joblib"


def _load_model(path: Path):
    if joblib is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _align_features(df: pd.DataFrame, schema_kind: str) -> Optional[pd.DataFrame]:
    feats = load_feature_schema(schema_kind)
    if not feats:
        return None
    try:
        X = build_ordered_matrix(df, feats)
        # Fill NaNs with zeros for LightGBM predict
        return X.fillna(0)
    except Exception:
        return None


def _predict_safely(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if model is None or X is None or X.empty:
        return None
    try:
        # Try standard predict
        return model.predict(X)
    except Exception:
        # Try LightGBM booster with disable shape check if available
        try:
            booster = getattr(model, "booster_", None)
            if booster is None:
                booster = getattr(model, "booster", None)
            if booster is not None and hasattr(booster, "predict"):
                return booster.predict(X, predict_disable_shape_check=True)  # type: ignore[arg-type]
        except Exception:
            pass
    return None


def enrich_meta_probs(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Add meta probability columns to the given DataFrame using aligned features.
    Returns (df_out, info) where info contains status and sources used.
    This function is defensive: if models or schemas are missing or prediction fails,
    it falls back to existing distribution-based probabilities when present.
    """
    out = df.copy()
    info = {
        "cover": {"source": None, "ok": False},
        "over": {"source": None, "ok": False},
    }

    # COVER (ATS)
    cover_model = _load_model(COVER_MODEL)
    X_cover = _align_features(out, "cover")
    cover_pred = None
    if cover_model is not None and X_cover is not None:
        cover_pred = _predict_safely(cover_model, X_cover)
    if cover_pred is not None:
        out["p_home_cover_meta"] = np.clip(cover_pred, 0.0, 1.0)
        info["cover"]["source"] = "meta_lgbm"
        info["cover"]["ok"] = True
    else:
        # Fallback: use distribution-based cover prob if present
        src = None
        for c in ("p_home_cover_display", "p_home_cover_dist"):
            if c in out.columns:
                out["p_home_cover_meta"] = out[c]
                src = c
                break
        info["cover"]["source"] = src or "missing"
        info["cover"]["ok"] = src is not None

    # TOTALS (OVER)
    over_model = _load_model(OVER_MODEL)
    X_over = _align_features(out, "total")
    over_pred = None
    if over_model is not None and X_over is not None:
        over_pred = _predict_safely(over_model, X_over)
    if over_pred is not None:
        out["p_over_meta"] = np.clip(over_pred, 0.0, 1.0)
        info["over"]["source"] = "meta_lgbm"
        info["over"]["ok"] = True
    else:
        # Fallback: use distribution-based over prob if present
        src = None
        for c in ("p_over_display", "p_over_dist"):
            if c in out.columns:
                out["p_over_meta"] = out[c]
                src = c
                break
        info["over"]["source"] = src or "missing"
        info["over"]["ok"] = src is not None

    return out, info
