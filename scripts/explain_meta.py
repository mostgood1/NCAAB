import json, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

OUT = Path('outputs')
TODAY = datetime.datetime.utcnow().date().isoformat()
ARTIFACT = {
    'date': TODAY,
    'models': {}
}

# Define a placeholder LogisticRegression so joblib can unpickle models
# that were trained with a minimal class under __main__.
class LogisticRegression:  # noqa: N801 - matching pickled class name
    def __init__(self, *args, **kwargs):
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_in_ = None

def load_enriched(date: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_enriched_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

ENRICHED = load_enriched(TODAY)

def load_metrics():
    p = OUT / 'meta_probs_metrics.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

METRICS = load_metrics()

def _load_portable(kind: str):
    p = OUT / f'meta_{kind}_portable.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def explain(model_path: Path, prefix: str):
    clf = None
    portable = None
    if model_path.exists():
        try:
            clf = joblib.load(model_path)
        except Exception as e:
            load_err = str(e)
            # Try portable fallback
            kind = 'cover' if 'cover' in model_path.name else 'over'
            portable = _load_portable(kind)
            if not portable:
                return {'load_error': load_err}
    else:
        kind = 'cover' if 'cover' in model_path.name else 'over'
        portable = _load_portable(kind)
        if not portable:
            return {'missing_model': True}
    if ENRICHED.empty:
        return {'missing_enriched': True}
    feat_cols = [c for c in ENRICHED.columns if c.startswith(prefix) and not c.endswith('_meta') and not c.endswith('_final')]
    if not feat_cols:
        # Fallback to metrics/portable-declared features if discovery failed
        kind = 'cover' if 'cover' in model_path.name else 'over'
        metric_node = METRICS.get(kind) if isinstance(METRICS, dict) else None
        from_metrics = (metric_node.get('features_used') if isinstance(metric_node, dict) else None) or []
        from_metrics = [c for c in from_metrics if c in ENRICHED.columns]
        if from_metrics:
            feat_cols = from_metrics
        else:
            port2 = _load_portable(kind)
            fn = (port2.get('feature_names') if isinstance(port2, dict) else None) or []
            fn = [c for c in fn if c in ENRICHED.columns]
            if fn:
                feat_cols = fn
            else:
                return {'missing_features': True}
    # Filter calibrated-first ordering
    calibrated = [c for c in feat_cols if c.endswith('_cal')]
    non_cal = [c for c in feat_cols if c not in calibrated]
    ordered = calibrated + non_cal
    # Keep only features used in training if recorded inside model (optional custom attribute)
    try:
        if hasattr(clf, 'feature_names_in_'):
            ordered = [c for c in ordered if c in clf.feature_names_in_]
    except Exception:
        pass
    # If no model feature names, fall back to metrics-reported features
    try:
        if (not ordered) or (clf is not None and not hasattr(clf, 'feature_names_in_')):
            kind = 'cover' if 'cover' in model_path.name else 'over'
            metric_node = METRICS.get(kind)
            if isinstance(metric_node, dict):
                fnames = metric_node.get('features_used') or []
                if fnames:
                    ordered = [c for c in fnames if c in ENRICHED.columns]
    except Exception:
        pass
    X = ENRICHED[ordered].astype(float)
    for c in ordered:
        col = X[c]
        if col.isna().any():
            X[c] = col.fillna(col.mean())
    # Logistic contributions: w_i * x_i; probability via sigmoid(intercept + sum)
    if portable is not None:
        coef = np.asarray(portable.get('coef', [[0]]), dtype=float)[0]
        intercept = float(np.asarray(portable.get('intercept', [0]), dtype=float)[0])
        # If portable includes feature names, align ordering
        fn = portable.get('feature_names') or []
        if fn:
            aligned = [c for c in fn if c in X.columns]
            if aligned:
                ordered = aligned
                X = ENRICHED[ordered].astype(float)
                for c in ordered:
                    col = X[c]
                    if col.isna().any():
                        X[c] = col.fillna(col.mean())
    else:
        coef = getattr(clf, 'coef_', None)
        if coef is None:
            return {'missing_coef': True}
        coef = np.asarray(coef, dtype=float)[0]
        intercept = float(np.asarray(getattr(clf, 'intercept_', [0]), dtype=float)[0])
    # Sanitize coefficients/intercept and inputs to avoid NaNs in contributions
    try:
        coef = np.where(np.isfinite(coef), coef, 0.0)
        if not np.isfinite(intercept):
            intercept = 0.0
        X = X.replace([np.inf, -np.inf], np.nan)
        for c in ordered:
            col = X[c]
            if col.isna().any():
                X[c] = col.fillna(col.mean())
    except Exception:
        pass
    # Final safety: ensure shapes compatible
    try:
        if X.shape[1] != len(coef):
            kind = 'cover' if 'cover' in model_path.name else 'over'
            target_k = len(coef)
            candidates: list[str] = []
            # 1) Model-declared feature names
            try:
                fn = getattr(clf, 'feature_names_in_', None)
                if isinstance(fn, (list, tuple, np.ndarray)):
                    candidates = [c for c in fn if c in ENRICHED.columns]
            except Exception:
                pass
            # 2) Metrics feature list
            if not candidates:
                try:
                    metric_node = METRICS.get(kind) if isinstance(METRICS, dict) else None
                    fm = (metric_node.get('features_used') if isinstance(metric_node, dict) else None) or []
                    fm = [c for c in fm if c in ENRICHED.columns]
                    candidates = fm
                except Exception:
                    pass
            # 3) Portable feature list
            if not candidates and portable is not None:
                try:
                    pf = portable.get('feature_names') or []
                    pf = [c for c in pf if c in ENRICHED.columns]
                    candidates = pf
                except Exception:
                    pass
            # 4) Current ordered as last resort
            if not candidates:
                candidates = [c for c in ordered if c in ENRICHED.columns]
            # Trim to match coef length if possible
            if candidates and len(candidates) >= target_k:
                ordered = candidates[:target_k]
                X = ENRICHED[ordered].astype(float)
    except Exception:
        pass
    xv = np.nan_to_num(X.values, nan=float(np.nanmean(X.values)))
    raw_scores = xv @ coef + intercept
    # Compute per-feature contributions (matrix: n_rows x n_features)
    contrib = xv * coef.reshape(1, -1)
    abs_mean_vals = np.nanmean(np.abs(contrib), axis=0)
    signed_mean_vals = np.nanmean(contrib, axis=0)
    abs_mean = dict(zip(ordered, abs_mean_vals))
    signed_mean = dict(zip(ordered, signed_mean_vals))
    top_abs = sorted(abs_mean.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        'n_rows': int(len(X)),
        'features_used': ordered,
        'intercept': intercept,
        'top_abs_contributions': top_abs,
        'mean_abs_contribution': abs_mean,
        'mean_signed_contribution': signed_mean,
    }

ARTIFACT['models']['cover'] = explain(OUT/'meta_cover.joblib', 'p_home_cover')
ARTIFACT['models']['over'] = explain(OUT/'meta_over.joblib', 'p_over')
# If joblib load failed but portable exists, recompute using portable to avoid load_error
for k in ['cover','over']:
    node = ARTIFACT['models'].get(k)
    if isinstance(node, dict) and ('load_error' in node or node.get('missing_model')):
        port = _load_portable(k)
        if port is not None:
            # Re-run explain pathway by simulating loaded portable
            model_path = OUT / f'meta_{k}.joblib'
            # Force the above explain to consume portable via the internal logic
            ARTIFACT['models'][k] = explain(model_path, 'p_home_cover' if k=='cover' else 'p_over')

out_path = OUT / f'meta_explain_{TODAY}.json'
out_path.write_text(json.dumps(ARTIFACT, indent=2))
print('Meta explanation artifact written:', json.dumps(ARTIFACT, indent=2))
