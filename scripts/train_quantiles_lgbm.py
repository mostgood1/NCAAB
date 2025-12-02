import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import lightgbm as lgb
except Exception:
    lgb = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
ART = OUT / 'quantile_models'

SEG_TOTALS = ['low','mid','high']
SEG_MARGINS = ['small','med','large']

FEATURES_DEFAULT = [
    # Core model signals
    'pred_total','pred_margin','home_win_prob','home_spread_pred',
    # Team strength
    'home_rating','away_rating','home_off','home_def','away_off','away_def',
    # Tempo/pace and context
    'pace','home_pace','away_pace','days_rest_home','days_rest_away','travel_dist_km',
    # Market anchors (if present)
    'market_total','market_spread','market_moneyline_home_prob',
    # Priors/preseason blending indicators
    'preseason_weight','preseason_only_sparse'
]

TARGETS = {
    'total': {'pred_col': 'pred_total', 'actual_col': 'actual_total'},
    'margin': {'pred_col': 'pred_margin', 'actual_col': 'actual_margin'},
}

PARAMS = {
    0.1: {'objective':'quantile','alpha':0.1,'num_leaves':64,'learning_rate':0.05,'min_data_in_leaf':50},
    0.5: {'objective':'quantile','alpha':0.5,'num_leaves':64,'learning_rate':0.05,'min_data_in_leaf':50},
    0.9: {'objective':'quantile','alpha':0.9,'num_leaves':64,'learning_rate':0.05,'min_data_in_leaf':50},
}


def _load_backtest() -> pd.DataFrame:
    # Prefer enriched backtest if available
    p_en = OUT / 'backtest_reports' / 'backtest_joined_enriched.csv'
    p = p_en if p_en.exists() else (OUT / 'backtest_reports' / 'backtest_joined.csv')
    if not p.exists():
        raise SystemExit('missing backtest_joined.csv')
    return pd.read_csv(p)


def _load_selected_features():
    """Load per-target, per-segment features from CV selection if present."""
    selp = OUT / 'quantile_cv' / 'selected_features.json'
    if not selp.exists():
        return None
    try:
        data = json.loads(selp.read_text(encoding='utf-8'))
        selected = data.get('selected') or {}
        # expected shape: {'total': {'low': [...], 'mid': [...], 'high': [...]}, 'margin': {...}}
        return selected
    except Exception:
        return None


def _thresholds_from_meta():
    mpath = OUT / 'quantile_model_selection.json'
    total_thr = (135.0, 155.0)
    margin_thr = (5.0, 10.0)
    try:
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding='utf-8'))
            sr = m.get('segment_rules') or {}
            tt = sr.get('total_thresholds'); mt = sr.get('margin_thresholds')
            if isinstance(tt,(list,tuple)) and len(tt)==2:
                total_thr = (float(tt[0]), float(tt[1]))
            if isinstance(mt,(list,tuple)) and len(mt)==2:
                margin_thr = (float(mt[0]), float(mt[1]))
    except Exception:
        pass
    return total_thr, margin_thr


def _assign_segments(df: pd.DataFrame, total_thr, margin_thr):
    def lab_total(v: float):
        a,b = total_thr
        if v <= a: return 'low'
        if v <= b: return 'mid'
        return 'high'
    def lab_margin(v: float):
        a,b = margin_thr
        av = abs(v)
        if av <= a: return 'small'
        if av <= b: return 'med'
        return 'large'
    df = df.copy()
    df['_seg_total'] = pd.to_numeric(df['pred_total'], errors='coerce').apply(lambda v: lab_total(float(v)) if np.isfinite(v) else 'mid')
    df['_seg_margin'] = pd.to_numeric(df['pred_margin'], errors='coerce').apply(lambda v: lab_margin(float(v)) if np.isfinite(v) else 'med')
    return df


def _train_quantile(df: pd.DataFrame, features: list, target_col: str, alpha: float):
    # Use only present features and guarantee columns exist
    present = [f for f in features if f in df.columns]
    X = df[present].copy()
    y = pd.to_numeric(df[target_col], errors='coerce')
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    # Fill missing with column medians where possible, else 0
    X = X.apply(lambda col: col.fillna(col.median()) if np.isfinite(col.median()) else col.fillna(0))
    if lgb is None:
        raise SystemExit('lightgbm not installed')
    params = PARAMS[alpha].copy()
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def _save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def main():
    bt = _load_backtest()
    total_thr, margin_thr = _thresholds_from_meta()
    bt = _assign_segments(bt, total_thr, margin_thr)
    # ensure features exist (global fallback)
    feats_global = [c for c in FEATURES_DEFAULT if c in bt.columns]
    if not feats_global:
        # fall back to using predictions as minimal set
        feats_global = [c for c in [TARGETS['total']['pred_col'], TARGETS['margin']['pred_col']] if c in bt.columns]
    if not feats_global:
        raise SystemExit('no usable features found')
    # Try load per-target/segment selected features from CV
    selected = _load_selected_features()
    latest_date = str(sorted(bt['date'].dropna().astype(str).unique())[-1])
    meta = {
        'latest_date': latest_date,
        'features': feats_global,
        'features_by_segment_total': (selected or {}).get('total', {}),
        'features_by_segment_margin': (selected or {}).get('margin', {}),
        'thresholds': {'total': total_thr, 'margin': margin_thr}
    }
    print('Training LGBM quantiles with features (global fallback):', feats_global)
    # train per target and per segment
    for tgt_name, cfg in TARGETS.items():
        pred_col = cfg['pred_col']; actual_col = cfg['actual_col']
        seg_col = '_seg_total' if tgt_name=='total' else '_seg_margin'
        for seg in (SEG_TOTALS if tgt_name=='total' else SEG_MARGINS):
            dseg = bt[bt[seg_col] == seg].copy()
            # require minimum rows
            if len(dseg) < 500:
                continue
            # choose features for this target/segment if selected, else global fallback
            feats_seg = None
            if selected and isinstance(selected, dict):
                feats_seg = (selected.get('total' if tgt_name=='total' else 'margin') or {}).get(seg)
            feats_use = [c for c in (feats_seg or feats_global) if c in dseg.columns]
            if not feats_use:
                feats_use = feats_global
            for alpha in [0.1, 0.5, 0.9]:
                # target is actual, optionally residual learn: we include pred as feature, so direct modeling is fine
                try:
                    print(f'Training {tgt_name} seg={seg} q={alpha} rows={len(dseg)} using {len(feats_use)} features')
                    model = _train_quantile(dseg, feats_use, actual_col, alpha)
                    mpath = ART / tgt_name / seg / f'model_q{int(alpha*100)}.txt'
                    _save_model(model, mpath)
                except Exception as e:
                    print('train error', tgt_name, seg, alpha, e)
    # save meta
    (ART).mkdir(parents=True, exist_ok=True)
    (ART / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('trained LGBM quantiles; artifacts at', str(ART))


if __name__ == '__main__':
    main()
