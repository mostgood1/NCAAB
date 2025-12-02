import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os
ROOT = Path(__file__).resolve().parents[1]


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _window(df: pd.DataFrame, latest_date: str, days: int) -> pd.DataFrame:
    if 'date' not in df.columns:
        return df
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    ref = pd.to_datetime(latest_date, errors='coerce')
    if pd.isna(ref):
        ref = df['date_dt'].max()
    start = ref - pd.Timedelta(days=days)
    return df[(df['date_dt'] >= start) & (df['date_dt'] <= ref)].copy()


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.nanmean(np.maximum(tau * e, (tau - 1.0) * e)))


def candidate_residual(window_df: pd.DataFrame, preds_col: str, target_col: str, target_cov: float):
    eps = (1.0 - target_cov) / 2.0
    ql = float(np.nanquantile(window_df[target_col] - window_df[preds_col], eps))
    qm = float(np.nanquantile(window_df[target_col] - window_df[preds_col], 0.5))
    qh = float(np.nanquantile(window_df[target_col] - window_df[preds_col], 1.0 - eps))
    def produce(preds):
        preds = np.asarray(preds, float)
        q10 = preds + ql
        q50 = preds + qm
        q90 = preds + qh
        return np.vstack([q10, q50, q90]).T
    return {'name':'residual_conformal', 'ql': ql, 'qm': qm, 'qh': qh, 'produce': produce}


def candidate_normal(window_df: pd.DataFrame, preds_col: str, target_col: str, target_cov: float):
    # sigma from residual std (robust fallback to MAD if needed)
    resid = (window_df[target_col] - window_df[preds_col]).astype(float)
    sig = float(np.nanstd(resid))
    if not np.isfinite(sig) or sig <= 1e-6:
        mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))))
        sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else 8.0
    z = float(np.abs(np.nanquantile(np.random.randn(100000), (1.0 - target_cov) / 2.0)))  # approx 1.2816 for 80%
    # Use exact for speed and determinism
    from math import sqrt
    # Invert Normal CDF only for 0.1 and 0.9 directly
    z = 1.2815515655446004 if abs(target_cov - 0.8) < 1e-6 else z
    def produce(preds):
        preds = np.asarray(preds, float)
        return np.vstack([preds - z * sig, preds, preds + z * sig]).T
    return {'name':'normal_sigma', 'sigma': sig, 'z': z, 'produce': produce}


def _load_lgbm_models(root: Path, target: str, segment: str):
    try:
        import lightgbm as lgb
    except Exception:
        return None
    base = root / 'outputs' / 'quantile_models' / target / segment
    models = {}
    for q in [10,50,90]:
        p = base / f'model_q{q}.txt'
        if p.exists():
            try:
                models[q] = lgb.Booster(model_file=str(p))
            except Exception:
                return None
        else:
            return None
    return models


def candidate_lgbm(window_df: pd.DataFrame, preds_col: str, target_col: str, target_cov: float, segment: str, target_name: str, features: list, models):
    # produce q10/q50/q90 via LightGBM models; conformalize residuals in window to hit coverage
    # Synthesize common missing features from available columns to reduce gaps
    synth = window_df.copy()
    # Rest days: default to 3 if missing
    for col in ['days_rest_home','days_rest_away']:
        if col not in synth.columns:
            synth[col] = 3
    # Travel distance: default 0 if missing
    if 'travel_dist_km' not in synth.columns:
        synth['travel_dist_km'] = 0.0
    # Market anchors: try derive
    if 'market_spread' not in synth.columns:
        # prefer closing/home spread if present, else model spread pred
        alt_cols = [c for c in ['closing_spread','home_spread_pred','spread','odds_spread'] if c in synth.columns]
        synth['market_spread'] = synth[alt_cols[0]] if alt_cols else 0.0
    if 'market_total' not in synth.columns:
        alt_cols = [c for c in ['closing_total','total','odds_total'] if c in synth.columns]
        synth['market_total'] = synth[alt_cols[0]] if alt_cols else synth.get('pred_total', pd.Series(0.0, index=synth.index))
    if 'market_moneyline_home_prob' not in synth.columns:
        if 'home_ml_prob' in synth.columns:
            synth['market_moneyline_home_prob'] = synth['home_ml_prob']
        elif 'home_ml_price' in synth.columns:
            def _american_to_prob(price):
                try:
                    price = float(price)
                except Exception:
                    return np.nan
                if price > 0:
                    return 100.0 / (price + 100.0)
                elif price < 0:
                    return abs(price) / (abs(price) + 100.0)
                return np.nan
            synth['market_moneyline_home_prob'] = synth['home_ml_price'].apply(_american_to_prob)
        else:
            synth['market_moneyline_home_prob'] = 0.5
    # Verify features presence and log missing
    missing = [f for f in features if f not in synth.columns]
    if missing:
        print(f'[LGBM {target_name}/{segment}] Missing features in window after synth: {missing}')
    present_feats = [f for f in features if f in synth.columns]
    Xw = synth[present_feats].copy().fillna(0)
    q10_pred = np.asarray(models[10].predict(Xw, predict_disable_shape_check=True), float)
    q50_pred = np.asarray(models[50].predict(Xw, predict_disable_shape_check=True), float)
    q90_pred = np.asarray(models[90].predict(Xw, predict_disable_shape_check=True), float)
    # conformal shift using residuals of central interval
    y = window_df[target_col].to_numpy(float)
    eps = (1.0 - target_cov) / 2.0
    # compute adjustment on residuals to center interval to target coverage
    # error relative to median
    e_med = y - q50_pred
    adj_low = float(np.nanquantile(e_med, eps))
    adj_high = float(np.nanquantile(e_med, 1.0 - eps))
    def produce_row(xrow: pd.DataFrame):
        # Build numeric frame directly to avoid FutureWarning on fillna downcasting
        arr = np.array(xrow, dtype=float)
        xdf = pd.DataFrame(arr, columns=present_feats)
        # Replace NaNs with 0 in numeric dtype (no object downcast occurs)
        xdf = xdf.fillna(0.0)
        q10r = float(models[10].predict(xrow, predict_disable_shape_check=True)[0]) + adj_low
        q50r = float(models[50].predict(xdf, predict_disable_shape_check=True)[0])
        q90r = float(models[90].predict(xdf, predict_disable_shape_check=True)[0]) + adj_high
        return np.array([[q10r, q50r, q90r]], float)
    def produce(preds_unused):
        # ignore preds input, rely on features/models
        return np.vstack([q10_pred + adj_low, q50_pred, q90_pred + adj_high]).T
    return {'name': f'lgbm_quantile_conformal_{target_name}_{segment}', 'produce': produce, 'produce_row': produce_row, 'params': {'segment': segment, 'features': present_feats, 'missing_features': missing}}


def _interp_quantile_from_three(q10: float, q50: float, q90: float, tau: float) -> float:
    # Linear interpolation in tau across (0.1,0.5,0.9); extrapolate with end slopes
    q10, q50, q90 = float(q10), float(q50), float(q90)
    if tau <= 0.1:
        # extrapolate left using slope between 0.1 and 0.5
        m = (q50 - q10) / 0.4
        return q10 + (tau - 0.1) * m
    if tau <= 0.5:
        m = (q50 - q10) / 0.4
        return q10 + (tau - 0.1) * m
    if tau <= 0.9:
        m = (q90 - q50) / 0.4
        return q50 + (tau - 0.5) * m
    # extrapolate right using slope between 0.5 and 0.9
    m = (q90 - q50) / 0.4
    return q90 + (tau - 0.9) * m


def _approx_crps_from_three(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, taus=None) -> float:
    # Approximate CRPS via integral over pinball loss across taus
    # CRPS(F,y) = 2 * ∫_0^1 Pinball_tau(y, q_tau) d tau (approx)
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)
    y = np.asarray(y, float)
    q10 = np.asarray(q10, float); q50 = np.asarray(q50, float); q90 = np.asarray(q90, float)
    losses = []
    for t in taus:
        qt = np.vectorize(_interp_quantile_from_three)(q10, q50, q90, t)
        losses.append(pinball_loss(y, qt, t))
    return float(2.0 * np.nanmean(losses))


def score_candidate(win_df: pd.DataFrame, preds_col: str, target_col: str, cand):
    y = win_df[target_col].to_numpy(float)
    preds = win_df[preds_col].to_numpy(float)
    # If candidate provides feature-based production, build feature matrix
    if 'produce_row' in cand and 'params' in cand:
        feats = cand.get('params', {}).get('features', [])
        X = win_df[feats].copy().fillna(0)
        q10_list = []
        q50_list = []
        q90_list = []
        for i in range(len(X)):
            qs = cand['produce_row']([list(X.iloc[i].values)])
            q10_list.append(qs[0,0]); q50_list.append(qs[0,1]); q90_list.append(qs[0,2])
        qs = np.vstack([np.array(q10_list), np.array(q50_list), np.array(q90_list)]).T
    else:
        qs = cand['produce'](preds)
    q10, q50, q90 = qs[:,0], qs[:,1], qs[:,2]
    # Coverage, width, CRPS (approx from three quantiles)
    cov = float(np.nanmean((y >= q10) & (y <= q90)))
    width = float(np.nanmedian(q90 - q10))
    crps = _approx_crps_from_three(y, q10, q50, q90)
    return {
        'name': cand['name'],
        'coverage': cov,
        'width': width,
        'crps': float(crps),
        'params': {k:v for k,v in cand.items() if k not in {'produce'}}
    }


def select_and_write(latest_date: str, out_dir: Path, preds_hist: pd.DataFrame, sel_total_by_seg: dict, sel_margin_by_seg: dict, seg_rules: dict):
    # Create today's quantiles for both targets
    today = preds_hist[preds_hist['date'].astype(str) == latest_date].copy()
    today['game_id'] = today['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    if any(c not in today.columns for c in ['pred_total','pred_margin']):
        raise ValueError('predictions_history_enriched missing pred_total or pred_margin')
    # Segment assignment helpers
    def bin_total(x: float) -> str:
        a,b = seg_rules['total_thresholds']
        if x <= a: return 'low'
        if x <= b: return 'mid'
        return 'high'
    def bin_margin(x: float) -> str:
        a,b = seg_rules['margin_thresholds']
        ax = abs(x)
        if ax <= a: return 'small'
        if ax <= b: return 'med'
        return 'large'
    today['_seg_total'] = today['pred_total'].apply(lambda v: bin_total(float(v)) if pd.notna(v) else 'mid')
    today['_seg_margin'] = today['pred_margin'].apply(lambda v: bin_margin(float(v)) if pd.notna(v) else 'med')
    # totals by segment
    q10_t = []; q50_t = []; q90_t = []
    arr_t = today['pred_total'].to_numpy(float)
    for i,seg in enumerate(today['_seg_total']):
        sel = sel_total_by_seg.get(seg) or sel_total_by_seg.get('overall')
        if 'produce_row' in sel:
            # build feature row from preds_hist by game_id
            gid = today.iloc[i]['game_id']
            r = preds_hist[preds_hist['game_id'].astype(str) == str(gid)].iloc[0] if not preds_hist.empty else None
            x = []
            feats = sel.get('params',{}).get('features',[])
            for f in feats:
                x.append(r.get(f) if r is not None else np.nan)
            qs = sel['produce_row']([x])
        else:
            qs = sel['produce'](np.array([arr_t[i]]))
        q10_t.append(qs[0,0]); q50_t.append(qs[0,1]); q90_t.append(qs[0,2])
    today['q10_total'] = q10_t; today['q50_total'] = q50_t; today['q90_total'] = q90_t
    # margins by segment
    q10_m = []; q50_m = []; q90_m = []
    arr_m = today['pred_margin'].to_numpy(float)
    for i,seg in enumerate(today['_seg_margin']):
        sel = sel_margin_by_seg.get(seg) or sel_margin_by_seg.get('overall')
        if 'produce_row' in sel:
            gid = today.iloc[i]['game_id']
            r = preds_hist[preds_hist['game_id'].astype(str) == str(gid)].iloc[0] if not preds_hist.empty else None
            x = []
            feats = sel.get('params',{}).get('features',[])
            for f in feats:
                x.append(r.get(f) if r is not None else np.nan)
            qs = sel['produce_row']([x])
        else:
            qs = sel['produce'](np.array([arr_m[i]]))
        q10_m.append(qs[0,0]); q50_m.append(qs[0,1]); q90_m.append(qs[0,2])
    today['q10_margin'] = q10_m; today['q50_margin'] = q50_m; today['q90_margin'] = q90_m

    # Enforce monotonicity per row
    def mono3(a,b,c,row):
        v = sorted([row[a], row[b], row[c]])
        return pd.Series(v, index=[a,b,c])
    today[['q10_total','q50_total','q90_total']] = today.apply(lambda r: mono3('q10_total','q50_total','q90_total', r), axis=1)
    today[['q10_margin','q50_margin','q90_margin']] = today.apply(lambda r: mono3('q10_margin','q50_margin','q90_margin', r), axis=1)

    hist_cols = ['date','game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin']
    qhist_path = out_dir / 'quantiles_history.csv'
    if qhist_path.exists():
        try:
            old = pd.read_csv(qhist_path)
            old['game_id'] = old['game_id'].astype(str)
        except Exception:
            old = pd.DataFrame(columns=hist_cols)
        old = old[old['date'].astype(str) != latest_date]
        new_hist = pd.concat([old, today[hist_cols]], ignore_index=True)
    else:
        new_hist = today[hist_cols]
    new_hist.to_csv(qhist_path, index=False)

    (out_dir / 'quantiles_selected.csv').write_text('')
    today[hist_cols].to_csv(out_dir / 'quantiles_selected.csv', index=False)
    return today[hist_cols]


def main(window_days: int = 28, target_cov: float = 0.8):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'outputs'
    # Prefer enriched backtest for feature availability
    bt_en = out_dir / 'backtest_reports' / 'backtest_joined_enriched.csv'
    bt = _read_csv(bt_en if bt_en.exists() else (out_dir / 'backtest_reports' / 'backtest_joined.csv'))
    preds_hist = _read_csv(out_dir / 'predictions_history_enriched.csv')
    if bt.empty or preds_hist.empty:
        raise SystemExit('Missing backtest_joined or predictions_history_enriched')
    latest_date = str(sorted(preds_hist['date'].dropna().astype(str).unique())[-1])
    # Prep window
    preds_col_total = 'pred_total_calibrated' if 'pred_total_calibrated' in bt.columns else 'pred_total'
    preds_col_margin = 'pred_margin_calibrated' if 'pred_margin_calibrated' in bt.columns else 'pred_margin'
    win = _window(bt, latest_date, window_days)
    # Ensure targets available
    if any(c not in win.columns for c in [preds_col_total, 'actual_total', preds_col_margin, 'actual_margin']):
        raise SystemExit('backtest_joined.csv missing required columns')

    # Build candidates per target
    cands_total = [
        candidate_residual(win, preds_col_total, 'actual_total', target_cov),
        candidate_normal(win, preds_col_total, 'actual_total', target_cov),
    ]
    cands_margin = [
        candidate_residual(win, preds_col_margin, 'actual_margin', target_cov),
        candidate_normal(win, preds_col_margin, 'actual_margin', target_cov),
    ]
    # Attempt to add LightGBM quantile candidates (overall baseline using 'mid' segment models)
    # Load features from quantile_models/meta.json
    qmeta_path = out_dir / 'quantile_models' / 'meta.json'
    qfeatures = []
    qfeatures_by_seg_total = {}
    qfeatures_by_seg_margin = {}
    if qmeta_path.exists():
        try:
            qmeta = json.loads(qmeta_path.read_text(encoding='utf-8'))
            qfeatures = qmeta.get('features') or []
            qfeatures_by_seg_total = qmeta.get('features_by_segment_total') or {}
            qfeatures_by_seg_margin = qmeta.get('features_by_segment_margin') or {}
        except Exception:
            qfeatures = []
    if qfeatures:
        mt = _load_lgbm_models(ROOT, 'total', 'mid')
        mm = _load_lgbm_models(ROOT, 'margin', 'med')
        if mt:
            try:
                feats_mid = qfeatures_by_seg_total.get('mid') or qfeatures
                cands_total.append(candidate_lgbm(win, preds_col_total, 'actual_total', target_cov, 'mid', 'total', feats_mid, mt))
            except Exception:
                pass
        if mm:
            try:
                feats_med = qfeatures_by_seg_margin.get('med') or qfeatures
                cands_margin.append(candidate_lgbm(win, preds_col_margin, 'actual_margin', target_cov, 'med', 'margin', feats_med, mm))
            except Exception:
                pass
    # Score overall
    def _serializable_score(s: dict):
        return {k: v for k, v in s.items() if k not in {'params','produce','produce_row'}}
    scores_total_raw = [score_candidate(win, preds_col_total, 'actual_total', c) for c in cands_total]
    scores_margin_raw = [score_candidate(win, preds_col_margin, 'actual_margin', c) for c in cands_margin]
    scores_total = [_serializable_score(s) for s in scores_total_raw]
    scores_margin = [_serializable_score(s) for s in scores_margin_raw]

    # Select by CRPS then coverage closeness then width
    def pick(scores):
        tgt = target_cov
        scores = sorted(scores, key=lambda s: (s['crps'], abs((s['coverage'] or 0) - tgt), s['width'] or 1e9))
        return scores[0]
    best_t = pick(scores_total)
    best_m = pick(scores_margin)
    # retrieve corresponding candidate objects including non-serializable fields for application
    def _find_cand(cands, name):
        for c in cands:
            if c.get('name') == name:
                return c
        return cands[0]
    sel_total_overall = _find_cand(cands_total, best_t['name'])
    sel_margin_overall = _find_cand(cands_margin, best_m['name'])

    # Segment rules (adaptive thresholds with sensible fallbacks)
    def _round_half(x: float) -> float:
        if not np.isfinite(x):
            return float(x)
        return float(np.round(x * 2.0) / 2.0)

    def _adaptive_thresholds(series: pd.Series, kind: str):
        # kind: 'total' uses raw preds; 'margin' uses absolute preds
        arr = pd.to_numeric(series, errors='coerce').to_numpy(float)
        if kind == 'margin':
            arr = np.abs(arr)
        arr = arr[np.isfinite(arr)]
        total_default = (135.0, 155.0)
        margin_default = (5.0, 10.0)
        if arr.size < 300:  # require enough samples
            return total_default if kind == 'total' else margin_default, {'mode': 'fallback_min_samples', 'n': int(arr.size)}
        q1 = float(np.nanpercentile(arr, 33.0))
        q2 = float(np.nanpercentile(arr, 66.0))
        q1r, q2r = _round_half(q1), _round_half(q2)
        # Ensure separation and reasonable ranges
        if not np.isfinite(q1r) or not np.isfinite(q2r) or (q2r - q1r) < (0.5 if kind == 'total' else 0.5):
            return total_default if kind == 'total' else margin_default, {'mode': 'fallback_bad_spread', 'q1': q1, 'q2': q2}
        # Check segment sizes
        if kind == 'total':
            counts = {
                'low': int(np.sum(arr <= q1r)),
                'mid': int(np.sum((arr > q1r) & (arr <= q2r))),
                'high': int(np.sum(arr > q2r)),
            }
        else:
            counts = {
                'small': int(np.sum(arr <= q1r)),
                'med': int(np.sum((arr > q1r) & (arr <= q2r))),
                'large': int(np.sum(arr > q2r)),
            }
        min_count = 100
        if any(c < min_count for c in counts.values()):
            return total_default if kind == 'total' else margin_default, {'mode': 'fallback_small_segment', 'counts': counts}
        meta = {'mode': 'adaptive', 'q33': q1, 'q66': q2, 'q33_rounded': q1r, 'q66_rounded': q2r, 'counts': counts}
        return (q1r, q2r), meta

    total_thresholds, total_thr_meta = _adaptive_thresholds(win[preds_col_total], 'total')
    margin_thresholds, margin_thr_meta = _adaptive_thresholds(win[preds_col_margin], 'margin')

    # Segment-wise selection
    def seg_label_total(v: float) -> str:
        a,b = total_thresholds
        if v <= a: return 'low'
        if v <= b: return 'mid'
        return 'high'
    def seg_label_margin(v: float) -> str:
        a,b = margin_thresholds
        av = abs(v)
        if av <= a: return 'small'
        if av <= b: return 'med'
        return 'large'

    win['_seg_total'] = win[preds_col_total].apply(lambda v: seg_label_total(float(v)) if pd.notna(v) else 'mid')
    win['_seg_margin'] = win[preds_col_margin].apply(lambda v: seg_label_margin(float(v)) if pd.notna(v) else 'med')

    def pick(scores):
        tgt = target_cov
        scores = sorted(scores, key=lambda s: (s['crps'], abs((s['coverage'] or 0) - tgt), s['width'] or 1e9))
        return scores[0]

    totals_segments = ['low','mid','high']
    margins_segments = ['small','med','large']
    sel_total_by_seg = {'overall': sel_total_overall}
    sel_margin_by_seg = {'overall': sel_margin_overall}
    totals_seg_scores = {}
    margins_seg_scores = {}
    for seg in totals_segments:
        seg_df = win[win['_seg_total'] == seg]
        if len(seg_df) >= 100:
            # add segment LGBM candidate if available
            lmt = _load_lgbm_models(ROOT, 'total', seg)
            seg_cands = list(cands_total)
            if lmt and (qfeatures or qfeatures_by_seg_total):
                try:
                    feats_seg = qfeatures_by_seg_total.get(seg) or qfeatures
                    seg_cands.append(candidate_lgbm(seg_df, preds_col_total, 'actual_total', target_cov, seg, 'total', feats_seg, lmt))
                except Exception:
                    pass
            sc = [score_candidate(seg_df, preds_col_total, 'actual_total', c) for c in seg_cands]
            best = pick(sc)
            # locate candidate by name across seg_cands
            sel_total_by_seg[seg] = next(c for c in seg_cands if c['name'] == best['name'])
            totals_seg_scores[seg] = sc
    for seg in margins_segments:
        seg_df = win[win['_seg_margin'] == seg]
        if len(seg_df) >= 100:
            lmm = _load_lgbm_models(ROOT, 'margin', seg)
            seg_cands = list(cands_margin)
            if lmm and (qfeatures or qfeatures_by_seg_margin):
                try:
                    feats_seg = qfeatures_by_seg_margin.get(seg) or qfeatures
                    seg_cands.append(candidate_lgbm(seg_df, preds_col_margin, 'actual_margin', target_cov, seg, 'margin', feats_seg, lmm))
                except Exception:
                    pass
            sc = [score_candidate(seg_df, preds_col_margin, 'actual_margin', c) for c in seg_cands]
            best = pick(sc)
            sel_margin_by_seg[seg] = next(c for c in seg_cands if c['name'] == best['name'])
            margins_seg_scores[seg] = sc

    # Write outputs
    seg_rules = {'total_thresholds': total_thresholds, 'margin_thresholds': margin_thresholds}
    sel_df = select_and_write(latest_date, out_dir, preds_hist, sel_total_by_seg, sel_margin_by_seg, seg_rules)
    meta = {
        'selected_method_total': best_t['name'],
        'selected_method_margin': best_m['name'],
        'target_coverage': target_cov,
        'window_days': window_days,
        'scores_total': scores_total,
        'scores_margin': scores_margin,
        'crps_total': float(next(s['crps'] for s in scores_total if s['name'] == best_t['name'])),
        'crps_margin': float(next(s['crps'] for s in scores_margin if s['name'] == best_m['name'])),
        'segment_rules': {**seg_rules, 'total_thresholds_meta': total_thr_meta, 'margin_thresholds_meta': margin_thr_meta},
        'segment_scores_total': {
            k: [
                {kk: vv for kk, vv in s.items() if kk not in {'params','produce','produce_row'}}
                for s in v
            ] for k, v in totals_seg_scores.items()
        },
        'segment_scores_margin': {
            k: [
                {kk: vv for kk, vv in s.items() if kk not in {'params','produce','produce_row'}}
                for s in v
            ] for k, v in margins_seg_scores.items()
        },
        'selected_methods_by_segment_total': {k: v['name'] for k,v in sel_total_by_seg.items() if isinstance(v, dict) and 'name' in v},
        'selected_methods_by_segment_margin': {k: v['name'] for k,v in sel_margin_by_seg.items() if isinstance(v, dict) and 'name' in v},
        'median_width_total': float(np.nanmedian(sel_df['q90_total'] - sel_df['q10_total'])) if not sel_df.empty else np.nan,
        'median_width_margin': float(np.nanmedian(sel_df['q90_margin'] - sel_df['q10_margin'])) if not sel_df.empty else np.nan,
        'latest_date': latest_date,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    with open(out_dir / 'quantile_model_selection.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Selected methods:', best_t['name'], best_m['name'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--window-days', type=int, default=28)
    p.add_argument('--target-coverage', type=float, default=0.8)
    args = p.parse_args()
    main(window_days=args.window_days, target_cov=args.target_coverage)
