import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import brier_score_loss, log_loss  # type: ignore
except Exception:  # Fallback minimal implementations when scikit-learn unavailable (e.g. ARM build failure)
    class LogisticRegression:  # minimal binary logistic
        def __init__(self, max_iter: int = 800, class_weight=None, penalty: str = 'l2', C: float = 1.0):
            self.max_iter = max_iter
            self.class_weight = class_weight
            self.penalty = penalty
            self.C = C
            self.coef_ = None
            self.intercept_ = None
            self.feature_names_in_ = None
        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            n, m = X.shape
            w = np.zeros(m)
            b = 0.0
            # Class weights
            cw_pos = cw_neg = 1.0
            if isinstance(self.class_weight, dict):
                cw_pos = self.class_weight.get(1,1.0)
                cw_neg = self.class_weight.get(0,1.0)
            lr = 0.05
            for _ in range(self.max_iter):
                z = X @ w + b
                p = 1/(1+np.exp(-z))
                # Gradient
                grad = X.T @ ((p - y) * np.where(y>0, cw_pos, cw_neg)) / n
                gb = np.mean((p - y) * np.where(y>0, cw_pos, cw_neg))
                if self.penalty == 'l2':
                    grad += (1/self.C) * w / n
                w -= lr * grad
                b -= lr * gb
            self.coef_ = np.vstack([w])
            self.intercept_ = np.array([b])
            return self
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            z = X @ self.coef_[0] + self.intercept_[0]
            p = 1/(1+np.exp(-z))
            return np.vstack([1-p, p]).T
    def brier_score_loss(y_true, y_prob):
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        return float(np.mean((y_prob - y_true)**2))
    def log_loss(y_true, y_prob, eps: float = 1e-9):
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1-eps)
        return float(-np.mean(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob)))
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except Exception:
        def roc_auc_score(y_true, y_prob):
            y_true = np.asarray(y_true, dtype=float)
            y_prob = np.asarray(y_prob, dtype=float)
            pos = y_true == 1
            neg = y_true == 0
            if pos.sum() == 0 or neg.sum() == 0:
                return float('nan')
            # rank-based AUC (Mann-Whitney U)
            ranks = y_prob.argsort().argsort()
            u = ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2.0
            return float(u / (pos.sum() * neg.sum()))
import joblib

OUT = Path('outputs')

TARGET_MAP = {
    'cover': 'ats_home_cover',  # daily_results derived column
    'over': 'ou_over',
}

# Legacy minimal feature seeds retained for fallback; dynamic discovery will extend.
LEGACY_FEATURES_COVER = ['p_home_cover','p_home_cover_dist','p_home_cover_ensemble','p_home_cover_cdf']
LEGACY_FEATURES_OVER = ['p_over','p_over_dist','p_over_ensemble','p_over_cdf']

def find_prediction_files(limit: int | None = None):
    files = sorted(OUT.glob('predictions_unified_enriched_*.csv'))
    if limit:
        files = files[-limit:]
    return files

def load_daily_results():
    dfs = []
    for f in sorted((OUT).glob('daily_results/results_*.csv')):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                # Derive targets if missing from textual result columns
                if 'ats_home_cover' not in df.columns:
                    if 'ats_result' in df.columns:
                        ar = df['ats_result'].astype(str).str.lower()
                        df['ats_home_cover'] = np.where(ar == 'home cover', 1, np.where(ar == 'away cover', 0, np.nan))
                if 'ou_over' not in df.columns:
                    if 'ou_result_full' in df.columns:
                        orf = df['ou_result_full'].astype(str).str.lower()
                        df['ou_over'] = np.where(orf == 'over', 1, np.where(orf == 'under', 0, np.nan))
                # Ensure numeric scores for final filtering later
                for sc in ['home_score','away_score']:
                    if sc in df.columns:
                        df[sc] = pd.to_numeric(df[sc], errors='coerce')
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # Normalize target columns
    out['ats_home_cover'] = (out.get('home_cover') if 'home_cover' in out.columns else out.get('ats_home_cover'))
    out['ou_over'] = (out.get('over') if 'over' in out.columns else out.get('ou_over'))
    return out

def build_training_frame(pred_files):
    rows = []
    for pf in pred_files:
        try:
            df = pd.read_csv(pf)
            if df.empty:
                continue
            if 'date' not in df.columns:
                # infer from filename
                date_part = pf.name.replace('predictions_unified_enriched_','').replace('.csv','')
                df['date'] = date_part
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    all_preds = pd.concat(rows, ignore_index=True)
    return all_preds

def _compute_ece(p: pd.Series, y: pd.Series, bins: int = 10) -> float:
    try:
        p = pd.to_numeric(p, errors='coerce').clip(0,1)
        y = pd.to_numeric(y, errors='coerce')
        mask = p.notna() & y.notna()
        p = p[mask]; y = y[mask]
        if p.empty:
            return np.nan
        edges = np.linspace(0,1,bins+1)
        ece = 0.0
        n = len(p)
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            m = (p >= lo) & (p < hi) if i < bins-1 else (p >= lo) & (p <= hi)
            if not m.any():
                continue
            ece += (m.sum()/n) * abs(p[m].mean() - y[m].mean())
        return float(ece)
    except Exception:
        return np.nan

def _discover_features(df: pd.DataFrame, prefix: str, exclude_exact: set[str]) -> list[str]:
    # Collect probability columns matching prefix (e.g. p_home_cover / p_over) including method suffixes.
    cols = [c for c in df.columns if c.startswith(prefix)]
    # Exclude meta/final to avoid leakage, and extremely derived columns that duplicate basis.
    cols = [c for c in cols if c not in exclude_exact]
    return cols

def _prune_correlated(df: pd.DataFrame, cols: list[str], threshold: float = 0.995) -> list[str]:
    if len(cols) < 2:
        return cols
    sub = df[cols].astype(float)
    # Impute NaN with column mean for correlation stability
    for c in cols:
        col = sub[c]
        if col.isna().any():
            sub[c] = col.fillna(col.mean())
    corr = sub.corr().abs()
    to_drop = set()
    # Greedy prune: iterate upper triangle
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i+1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] >= threshold:
                # Drop the one with lower standalone variance (less informative) or later lexicographically for determinism
                v1 = float(sub[c1].var())
                v2 = float(sub[c2].var())
                drop = c1 if (v1 < v2 or (v1 == v2 and c1 > c2)) else c2
                to_drop.add(drop)
    pruned = [c for c in cols if c not in to_drop]
    return pruned

def _sanitize_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        col = pd.to_numeric(X[c], errors='coerce')
        if col.isna().any():
            col = col.fillna(col.mean())
        X[c] = col
    return X.astype(float)

def _safe_metrics(y: pd.Series, probs: np.ndarray) -> tuple[float, float, float, float]:
    yv = pd.to_numeric(y, errors='coerce')
    pv = pd.to_numeric(pd.Series(probs), errors='coerce').clip(0,1)
    m = yv.notna() & pv.notna()
    if not m.any():
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    y_arr = yv[m].astype(float).values
    p_arr = pv[m].astype(float).values
    try:
        from sklearn.metrics import brier_score_loss as _brier, log_loss as _ll, roc_auc_score as _auc
        brier = float(_brier(y_arr, p_arr))
        logloss = float(_ll(y_arr, p_arr, eps=1e-9))
        auc = float(_auc(y_arr, p_arr)) if (np.unique(y_arr).size == 2) else float('nan')
    except Exception:
        brier = float(np.mean((p_arr - y_arr)**2))
        p_clip = np.clip(p_arr, 1e-9, 1-1e-9)
        logloss = float(-np.mean(y_arr*np.log(p_clip) + (1-y_arr)*np.log(1-p_clip)))
        # KS as simple separation proxy
        try:
            auc = float(abs(p_arr[y_arr==1].mean() - p_arr[y_arr==0].mean()))
        except Exception:
            auc = float('nan')
    # KS (again) for consistency
    try:
        ks = float(abs(p_arr[y_arr==1].mean() - p_arr[y_arr==0].mean()))
    except Exception:
        ks = float('nan')
    return (brier, logloss, auc, ks)

def train_meta(all_preds: pd.DataFrame, daily: pd.DataFrame, kind: str):
    target_col = TARGET_MAP[kind]
    if daily.empty or target_col not in daily.columns:
        raise ValueError(f'Missing target {target_col} in daily results.')
    # Merge outcome onto predictions by game_id+date
    keys = [c for c in ['game_id','date'] if c in all_preds.columns and c in daily.columns]
    if not keys:
        raise ValueError('No common keys to merge predictions and outcomes.')
    # Merge outcomes plus score columns for finals filtering
    cols_for_merge = [*keys, target_col]
    for extra in ['home_score','away_score']:
        if extra in daily.columns:
            cols_for_merge.append(extra)
    merged = all_preds.merge(daily[cols_for_merge], on=keys, how='inner')
    # Restrict to finalized games (scores present >0, guard against future slated rows)
    if {'home_score','away_score'}.issubset(merged.columns):
        hs = pd.to_numeric(merged['home_score'], errors='coerce')
        as_ = pd.to_numeric(merged['away_score'], errors='coerce')
        finalized_mask = hs.notna() & as_.notna() & (hs + as_ > 0)
        merged = merged[finalized_mask]
    # Drop rows with missing target
    merged = merged[merged[target_col].notna()]
    # Dynamic feature discovery
    if kind == 'cover':
        exclude = {'p_home_cover_meta','p_home_cover_final'}
        seed_prefix = 'p_home_cover'
        legacy = [c for c in LEGACY_FEATURES_COVER if c in merged.columns]
    else:
        exclude = {'p_over_meta','p_over_final'}
        seed_prefix = 'p_over'
        legacy = [c for c in LEGACY_FEATURES_OVER if c in merged.columns]
    discovered = _discover_features(merged, seed_prefix, exclude_exact=exclude)
    # Prefer calibrated variants first in ordering (those ending with _cal)
    calibrated = [c for c in discovered if c.endswith('_cal')]
    non_cal = [c for c in discovered if c not in calibrated]
    ordered = calibrated + non_cal
    # Guarantee legacy presence for backward compatibility
    for c in legacy:
        if c not in ordered:
            ordered.append(c)
    # Prune highly correlated duplicates
    pruned = _prune_correlated(merged, ordered, threshold=0.995)
    if not pruned:
        raise ValueError('No usable stacking features after pruning.')
    X = _sanitize_X(merged[pruned])
    y_raw = merged[target_col].astype(float)
    # Ensure binary 0/1
    y = (y_raw > 0.5).astype(int)

    if y.nunique() < 2:
        raise ValueError(f'Target {target_col} not binary across merged set.')

    # Logistic regression with balanced class weight for robustness
    clf = LogisticRegression(max_iter=800, class_weight='balanced', penalty='l2', C=1.0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:,1]
    # Guard against non-finite coefs/intercept/probs by falling back to a single best feature
    def _finite(arr) -> bool:
        try:
            a = np.asarray(arr, dtype=float)
            return np.isfinite(a).all()
        except Exception:
            return False
    if (not _finite(getattr(clf, 'coef_', None))) or (not _finite(getattr(clf, 'intercept_', None))) or (not _finite(probs)):
        # Pick a sane fallback feature preference order
        prefs = [c for c in pruned if c.endswith('_cal')] + \
                [c for c in pruned if c.endswith('_dist')] + \
                [c for c in pruned if c.endswith('_display')] + pruned
        fb = None
        for c in prefs:
            if c in X.columns:
                fb = c
                break
        if fb is None:
            raise ValueError('No usable feature for fallback meta model.')
        X_fb = _sanitize_X(merged[[fb]])
        clf = LogisticRegression(max_iter=800, class_weight='balanced', penalty='l2', C=1.0)
        clf.fit(X_fb, y)
        probs = clf.predict_proba(X_fb)[:,1]
        pruned = [fb]
        X = X_fb
    # Extended metrics
    brier, logloss, auc, ks = _safe_metrics(y, probs)
    ece = _compute_ece(pd.Series(probs), y)
    metrics = {
        'samples': int(len(y)),
        'features_used': pruned,
        'n_features': len(pruned),
        'calibrated_features_used': [c for c in pruned if c.endswith('_cal')],
        'legacy_features_present': legacy,
        'brier': float(brier) if brier == brier else float('nan'),
        'log_loss': float(logloss) if logloss == logloss else float('nan'),
        'base_rate': float(y.mean()),
        'auc': auc,
        'ks': ks,
        'ece': float(ece),
    }
    return clf, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=None, help='Limit number of enriched prediction days back from newest')
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--no-cover', action='store_true', help='Skip cover meta model')
    ap.add_argument('--no-over', action='store_true', help='Skip over meta model')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_prediction_files(args.limit_days)
    if not pred_files:
        print('No prediction enriched files found.')
        return
    all_preds = build_training_frame(pred_files)
    daily = load_daily_results()

    artifact = {}

    if not args.no_cover:
        try:
            clf_cover, m_cover = train_meta(all_preds, daily, 'cover')
            joblib.dump(clf_cover, out_dir/'meta_cover.joblib')
            # Also write a portable artifact for environments without matching pickle import paths
            try:
                portable = {
                    'coef': getattr(clf_cover, 'coef_', None).tolist() if getattr(clf_cover, 'coef_', None) is not None else None,
                    'intercept': getattr(clf_cover, 'intercept_', None).tolist() if getattr(clf_cover, 'intercept_', None) is not None else None,
                    'feature_names': list(getattr(clf_cover, 'feature_names_in_', [])) or m_cover.get('features_used', []),
                    'kind': 'cover'
                }
                (out_dir/'meta_cover_portable.json').write_text(json.dumps(portable, indent=2))
                # Write feature sidecar
                (out_dir/'meta_features_cover.json').write_text(json.dumps({'features': m_cover.get('features_used', [])}, indent=2))
            except Exception:
                pass
            artifact['cover'] = m_cover
        except Exception as e:
            artifact['cover_error'] = str(e)
    if not args.no_over:
        try:
            clf_over, m_over = train_meta(all_preds, daily, 'over')
            joblib.dump(clf_over, out_dir/'meta_over.joblib')
            try:
                portable = {
                    'coef': getattr(clf_over, 'coef_', None).tolist() if getattr(clf_over, 'coef_', None) is not None else None,
                    'intercept': getattr(clf_over, 'intercept_', None).tolist() if getattr(clf_over, 'intercept_', None) is not None else None,
                    'feature_names': list(getattr(clf_over, 'feature_names_in_', [])) or m_over.get('features_used', []),
                    'kind': 'over'
                }
                (out_dir/'meta_over_portable.json').write_text(json.dumps(portable, indent=2))
                (out_dir/'meta_features_over.json').write_text(json.dumps({'features': m_over.get('features_used', [])}, indent=2))
            except Exception:
                pass
            artifact['over'] = m_over
        except Exception as e:
            artifact['over_error'] = str(e)

    (out_dir/'meta_probs_metrics.json').write_text(json.dumps(artifact, indent=2))
    print('Meta model training complete:', json.dumps(artifact, indent=2))

if __name__ == '__main__':
    main()
