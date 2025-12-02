"""Temporal CV for LGBM quantile feature selection (per target/segment).

Outputs:
- outputs/quantile_cv_results.csv (flat summary)
- outputs/quantile_cv/selected_features.json (mapping for training)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
CV_OUT_DIR = OUT / 'quantile_cv'

# Reuse segmentation and defaults from the trainer if available
try:
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.append(str(ROOT))
    from scripts.train_quantiles_lgbm import (
        _load_backtest, _thresholds_from_meta, _assign_segments,
        FEATURES_DEFAULT, TARGETS, SEG_TOTALS, SEG_MARGINS
    )
except Exception as _e:
    raise SystemExit(f"[cv] cannot import trainer helpers: {_e}")


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _train_quantile_model(df: pd.DataFrame, features: List[str], target_col: str, alpha: float, num_round: int = 200):
    present = [f for f in features if f in df.columns]
    if not present:
        return None
    X = df[present].copy()
    # Fill with column medians then 0
    for c in present:
        med = pd.to_numeric(X[c], errors='coerce').median()
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(med if np.isfinite(med) else 0.0)
    y = pd.to_numeric(df[target_col], errors='coerce')
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    if len(X) < 50:
        return None
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'num_leaves': 64,
        'learning_rate': 0.05,
        'min_data_in_leaf': 50,
        'verbose': -1,
    }
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=num_round)


def _interp_q_from_three(q10: float, q50: float, q90: float, tau: float) -> float:
    q10, q50, q90 = float(q10), float(q50), float(q90)
    if tau <= 0.5:
        m = (q50 - q10) / 0.4
        return q10 + (tau - 0.1) * m
    if tau <= 0.9:
        m = (q90 - q50) / 0.4
        return q50 + (tau - 0.5) * m
    m = (q90 - q50) / 0.4
    return q90 + (tau - 0.9) * m


def _approx_crps(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> float:
    taus = np.linspace(0.05, 0.95, 19)
    losses = []
    for t in taus:
        qt = np.vectorize(_interp_q_from_three)(q10, q50, q90, t)
        e = y - qt
        pin = np.maximum(t * e, (t - 1.0) * e)
        losses.append(np.nanmean(pin))
    return float(2.0 * np.nanmean(losses))


def _build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    # Start from defaults but also consider leaner subsets to guard missingness
    present = [f for f in FEATURES_DEFAULT if f in df.columns]
    base_core = [c for c in ['pred_total','pred_margin'] if c in df.columns]
    market = [c for c in ['market_total','market_spread','market_moneyline_home_prob'] if c in df.columns]
    context = [c for c in ['days_rest_home','days_rest_away','travel_dist_km'] if c in df.columns]
    pace = [c for c in ['pace','home_pace','away_pace'] if c in df.columns]
    priors = [c for c in ['preseason_weight','preseason_only_sparse'] if c in df.columns]
    strength = [c for c in ['home_rating','away_rating','home_off','home_def','away_off','away_def'] if c in df.columns]
    spread_pred = [c for c in ['home_spread_pred','home_win_prob'] if c in df.columns]
    sets = {
        'core': base_core,
        'core_market': base_core + market,
        'core_context': base_core + context,
        'core_market_context': base_core + market + context,
        'core_market_context_pace': base_core + market + context + pace,
        'core_market_context_strength': base_core + market + context + strength,
        'core_plus_all': list(dict.fromkeys(base_core + market + context + pace + priors + strength + spread_pred)),
    }
    # Ensure non-empty and present-only
    sets = {k: [c for c in v if c in df.columns] for k,v in sets.items()}
    sets = {k: v for k,v in sets.items() if v}
    if present and 'default' not in sets:
        sets['default'] = present
    return sets


def _time_folds(dfd: pd.DataFrame, n_folds: int = 5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # Return list of (train_end, val_end) date boundaries using unique ordered dates
    dates = pd.to_datetime(dfd['date'], errors='coerce')
    u = np.array(sorted(set(dates.dropna())))
    if u.size < (n_folds + 1):
        return []
    # Equal bins over unique dates
    edges = np.linspace(0, u.size - 1, n_folds + 1).astype(int)
    bounds = []
    for i in range(1, len(edges)):
        train_end = u[edges[i-1]]
        val_end = u[edges[i]]
        bounds.append((train_end, val_end))
    return bounds


def _segment_keys(target: str) -> List[str]:
    return SEG_TOTALS if target == 'total' else SEG_MARGINS


def _seg_col(target: str) -> str:
    return '_seg_total' if target == 'total' else '_seg_margin'


def evaluate_cv(df: pd.DataFrame, target: str, seg: str, feature_sets: Dict[str, List[str]], n_folds: int = 5) -> List[dict]:
    results = []
    seg_df = df[df[_seg_col(target)] == seg].copy()
    seg_df['date_dt'] = pd.to_datetime(seg_df['date'], errors='coerce')
    seg_df = seg_df.sort_values('date_dt')
    if len(seg_df) < 600:
        return results
    folds = _time_folds(seg_df, n_folds)
    tgt_col = TARGETS[target]['actual_col']
    for name, feats in feature_sets.items():
        crps_list = []; cov_list = []; width_list = []; used_folds = 0; rows_total = 0
        for tr_end, val_end in folds:
            train_df = seg_df[seg_df['date_dt'] <= tr_end]
            val_df = seg_df[(seg_df['date_dt'] > tr_end) & (seg_df['date_dt'] <= val_end)]
            if len(train_df) < 400 or len(val_df) < 100:
                continue
            # Train three quantiles
            try:
                m10 = _train_quantile_model(train_df, feats, tgt_col, 0.1)
                m50 = _train_quantile_model(train_df, feats, tgt_col, 0.5)
                m90 = _train_quantile_model(train_df, feats, tgt_col, 0.9)
            except Exception:
                m10 = m50 = m90 = None
            if not (m10 and m50 and m90):
                continue
            Xv = val_df[[f for f in feats if f in val_df.columns]].copy()
            for c in Xv.columns:
                med = pd.to_numeric(Xv[c], errors='coerce').median()
                Xv[c] = pd.to_numeric(Xv[c], errors='coerce').fillna(med if np.isfinite(med) else 0.0)
            q10 = np.asarray(m10.predict(Xv, predict_disable_shape_check=True), float)
            q50 = np.asarray(m50.predict(Xv, predict_disable_shape_check=True), float)
            q90 = np.asarray(m90.predict(Xv, predict_disable_shape_check=True), float)
            y = pd.to_numeric(val_df[tgt_col], errors='coerce').to_numpy(float)
            # Metrics
            crps = _approx_crps(y, q10, q50, q90)
            cov = float(np.nanmean((y >= q10) & (y <= q90)))
            width = float(np.nanmedian(q90 - q10))
            crps_list.append(crps); cov_list.append(cov); width_list.append(width)
            used_folds += 1
            rows_total += len(val_df)
        if used_folds == 0:
            continue
        results.append({
            'target': target,
            'segment': seg,
            'feature_set': name,
            'features': feats,
            'folds': int(used_folds),
            'rows': int(rows_total),
            'crps_mean': float(np.nanmean(crps_list)),
            'coverage_mean': float(np.nanmean(cov_list)),
            'width_median': float(np.nanmedian(width_list)),
        })
    return results


def select_best(results: List[dict], target_cov: float = 0.8) -> Dict[Tuple[str,str], dict]:
    best: Dict[Tuple[str,str], dict] = {}
    by_key: Dict[Tuple[str,str], List[dict]] = {}
    for r in results:
        key = (r['target'], r['segment'])
        by_key.setdefault(key, []).append(r)
    for key, arr in by_key.items():
        arr = [x for x in arr if x.get('folds',0) > 0]
        if not arr:
            continue
        arr_sorted = sorted(arr, key=lambda x: (x['crps_mean'], abs((x['coverage_mean'] or 0) - target_cov), x['width_median'] or 1e9))
        best[key] = arr_sorted[0]
    return best


def main(folds: int = 5, target_cov: float = 0.8):
    if lgb is None:
        print('[cv] lightgbm not installed; skipping CV.')
        return
    bt = _load_backtest()
    total_thr, margin_thr = _thresholds_from_meta()
    bt = _assign_segments(bt, total_thr, margin_thr)
    # Ensure required columns are numeric
    bt = _ensure_numeric(bt, ['pred_total','pred_margin','actual_total','actual_margin'])
    feat_sets = _build_feature_sets(bt)
    if not feat_sets:
        print('[cv] No viable feature sets; skipping.')
        return
    all_results: List[dict] = []
    for target in ['total','margin']:
        for seg in _segment_keys(target):
            res = evaluate_cv(bt, target, seg, feat_sets, n_folds=folds)
            all_results.extend(res)
    if not all_results:
        print('[cv] No CV results produced (insufficient data).')
        return
    # Persist results
    CV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame(all_results)
    flat = flat.sort_values(['target','segment','crps_mean','coverage_mean'])
    flat.to_csv(OUT / 'quantile_cv_results.csv', index=False)
    flat.to_csv(CV_OUT_DIR / 'quantile_cv_results.csv', index=False)
    # Select best per target/segment
    best = select_best(all_results, target_cov=target_cov)
    selected = {'total': {}, 'margin': {}}
    for (tgt, seg), row in best.items():
        selected[tgt][seg] = row['features']
    payload = {
        'latest_date': str(sorted(pd.Series(bt['date']).dropna().astype(str).unique())[-1]) if 'date' in bt.columns else None,
        'selected': selected,
        'candidates': {k: v for k, v in _build_feature_sets(bt).items()},
        'cv_results_path': 'outputs/quantile_cv_results.csv',
        'target_coverage': target_cov,
        'timestamp': pd.Timestamp.utcnow().isoformat() + 'Z',
    }
    (CV_OUT_DIR / 'selected_features.json').write_text(json.dumps(payload, indent=2))
    print('[cv] Wrote CV results and selected features.')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--target-coverage', type=float, default=0.8)
    args = p.parse_args()
    main(folds=args.folds, target_cov=args.target_coverage)
