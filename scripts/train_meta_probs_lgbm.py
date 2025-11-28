import os
import json
import glob
import math
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import joblib
except Exception:
    joblib = None

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

FEATURE_CAND_PREFIXES = [
    'p_home_cover', 'p_away_cover', 'p_over', 'p_under',
    'p_home_cover_dist', 'p_over_dist',
    'p_home_cover_cdf', 'p_over_cdf',
    'p_home_cover_skew', 'p_over_skew',
    'p_home_cover_mix', 'p_over_mix',
    'p_home_cover_kde', 'p_over_kde',
    'p_home_cover_piece', 'p_over_piece',
    'pred_total', 'pred_margin', 'edge_total', 'edge_spread'
]

TARGET_COLS = {
    'cover': ['ats_home_win', 'ats_result', 'cover_home'],
    'over': ['ou_over_win', 'ou_result', 'went_over']
}


def _find_recent_enriched(limit_days: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(OUT, 'predictions_unified_enriched_*.csv')))
    if limit_days <= 0:
        return paths[-180:]
    cutoff = datetime.utcnow() - timedelta(days=limit_days)
    selected = []
    for p in paths:
        base = os.path.basename(p)
        try:
            date_part = base.replace('predictions_unified_enriched_', '').replace('.csv', '')
            dt = datetime.strptime(date_part, '%Y-%m-%d')
        except Exception:
            dt = None
        if dt is None or dt >= cutoff:
            selected.append(p)
    return selected[-180:]


def _derive_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # Cover target: 1 if home covers ATS, else 0
    y_cover = pd.Series(np.nan, index=df.index)
    for c in TARGET_COLS['cover']:
        if c in df.columns:
            s = df[c].astype(str).str.lower()
            if c == 'ats_result':
                # expected values like 'home_cover','away_cover','push'
                y_cover = s.map(lambda v: 1 if 'home' in v and 'cover' in v else (0 if 'away' in v and 'cover' in v else np.nan)).fillna(y_cover)
            elif c in ('ats_home_win', 'cover_home'):
                y_cover = pd.to_numeric(df[c], errors='coerce').fillna(y_cover)
    # Over target: 1 if total went over, else 0
    y_over = pd.Series(np.nan, index=df.index)
    for c in TARGET_COLS['over']:
        if c in df.columns:
            s = df[c].astype(str).str.lower()
            if c == 'ou_result':
                # expected values like 'over','under','push'
                y_over = s.map(lambda v: 1 if 'over' in v else (0 if 'under' in v else np.nan)).fillna(y_over)
            elif c in ('ou_over_win', 'went_over'):
                y_over = pd.to_numeric(df[c], errors='coerce').fillna(y_over)
    # If actual_total and market_total exist, infer over via comparison
    if y_over.isna().all():
        if {'actual_total','market_total'}.issubset(df.columns):
            at = pd.to_numeric(df['actual_total'], errors='coerce')
            mt = pd.to_numeric(df['market_total'], errors='coerce')
            y_over = (at > mt).astype(float)
    # Clip to {0,1}
    y_cover = y_cover.map(lambda x: 1.0 if x == 1 else (0.0 if x == 0 else np.nan))
    y_over = y_over.map(lambda x: 1.0 if x == 1 else (0.0 if x == 0 else np.nan))
    return y_cover, y_over


def _discover_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        cn = str(c).lower()
        for pfx in FEATURE_CAND_PREFIXES:
            if cn.startswith(pfx):
                cols.append(c)
                break
    # de-duplicate
    return sorted(list(set(cols)))


def _prune_correlated(df: pd.DataFrame, cols: list[str], thresh: float = 0.97) -> list[str]:
    if not cols:
        return cols
    try:
        mat = pd.DataFrame(df[cols]).apply(pd.to_numeric, errors='coerce').corr().abs()
        keep = []
        used = set()
        for i, c in enumerate(cols):
            if c in used:
                continue
            keep.append(c)
            used.add(c)
            for j in range(i+1, len(cols)):
                c2 = cols[j]
                if c2 in used:
                    continue
                val = mat.loc[c, c2] if (c in mat.index and c2 in mat.columns) else 0.0
                if isinstance(val, float) and val >= thresh:
                    used.add(c2)
        return keep
    except Exception:
        return cols


def _auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        yt = pd.to_numeric(y_true, errors='coerce')
        yp = pd.to_numeric(y_prob, errors='coerce')
        mask = yt.notna() & yp.notna()
        if mask.sum() < 5:
            return float('nan')
        return float(roc_auc_score(yt[mask], yp[mask]))
    except Exception:
        return float('nan')


def train_lgbm(X: pd.DataFrame, y: pd.Series) -> tuple[object, dict]:
    if lgb is None:
        return None, {'error': 'lightgbm_not_installed'}
    Xn = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    yt = pd.to_numeric(y, errors='coerce')
    mask = yt.isin([0.0, 1.0])
    Xn = Xn[mask]
    yt = yt[mask]
    if len(yt) < 40:
        return None, {'error': 'insufficient_rows', 'rows': int(len(yt))}
    dtrain = lgb.Dataset(Xn, label=yt)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_data_in_leaf': 20,
        'verbose': -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=400)
    # metrics
    pred = model.predict(Xn)
    auc = _auc(yt, pred)
    return model, {'rows': int(len(yt)), 'auc': auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=45)
    ap.add_argument('--out-dir', type=str, default=OUT)
    args = ap.parse_args()

    paths = _find_recent_enriched(args.limit_days)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        print('[warn] No enriched files found for training')
        return 0
    df = pd.concat(frames, ignore_index=True)

    # Filter to finalized rows when available
    if 'finalized' in df.columns:
        fin = df['finalized'].astype(str).str.lower().isin(['1','true','yes'])
        if fin.any():
            df = df[fin].copy()

    # Discover and prune features
    feat_cols = _discover_feature_cols(df)
    feat_cols = _prune_correlated(df, feat_cols, thresh=0.98)
    X = df[feat_cols].copy() if feat_cols else pd.DataFrame(index=df.index)

    # Targets
    y_cover, y_over = _derive_targets(df)

    metrics = {'features_used': feat_cols, 'rows_total': int(len(df))}

    # Train cover
    m_cover, met_cover = train_lgbm(X, y_cover)
    metrics['cover'] = met_cover
    if m_cover is not None and joblib:
        joblib.dump(m_cover, os.path.join(args.out_dir, 'meta_cover_lgbm.joblib'))
    # Train over
    m_over, met_over = train_lgbm(X, y_over)
    metrics['over'] = met_over
    if m_over is not None and joblib:
        joblib.dump(m_over, os.path.join(args.out_dir, 'meta_over_lgbm.joblib'))

    # Write metrics JSON
    try:
        with open(os.path.join(args.out_dir, 'meta_probs_metrics_lgbm.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"[ok] Wrote metrics + models. Features={len(feat_cols)} rows={len(df)}")
    except Exception as e:
        print(f"[warn] Failed to write metrics: {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
