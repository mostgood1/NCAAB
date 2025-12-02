"""Train LightGBM quantile models for totals and margins.

Inputs (expected):
  - outputs/predictions_history_enriched.csv (features + targets: pred_total/pred_margin optional)
  - outputs/daily_results/results_*.csv (actual_total, actual_margin)

Outputs:
  - outputs/quantiles_history.csv with columns: date, game_id, q10_total, q50_total, q90_total, q10_margin, q50_margin, q90_margin

Note: This is a lightweight placeholder using existing predicted totals/margins
as base features. Extend with richer feature matrices when available.
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

OUTPUTS = Path('outputs')
RESULTS_GLOB = 'daily_results/results_*.csv'

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob(RESULTS_GLOB):
        df = _safe_read(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    if {'date','game_id'}.issubset(res.columns):
        res = res.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    return res

def build_dataset() -> pd.DataFrame:
    preds = _safe_read(OUTPUTS / 'predictions_history_enriched.csv')
    res = load_results()
    if preds.empty or res.empty:
        return pd.DataFrame()
    preds['game_id'] = preds['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    df = res.merge(preds, on=['date','game_id'], how='left')
    # Optional enriched features
    feats = _safe_read(OUTPUTS / 'features_history.csv')
    if not feats.empty and {'date','game_id'}.issubset(feats.columns):
        # Normalize ids
        feats['game_id'] = feats['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
        df = df.merge(feats, on=['date','game_id'], how='left')
    return df

def train_quantile(y: pd.Series, X: pd.DataFrame, alpha: float):
    if lgb is None:
        return None
    # Load model selection if present
    sel = {}
    try:
        with open(OUTPUTS / 'quantile_model_selection.json', 'r', encoding='utf-8') as f:
            sel = json.load(f)
    except Exception:
        sel = {}
    lr = sel.get('params', {}).get('learning_rate', 0.05)
    nl = sel.get('params', {}).get('num_leaves', 31)
    mdl = sel.get('params', {}).get('min_data_in_leaf', 20)
    nround = sel.get('params', {}).get('num_boost_round', 200)
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'learning_rate': lr,
        'num_leaves': nl,
        'min_data_in_leaf': mdl,
        'verbose': -1,
    }
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=nround)
    return model

def main():
    df = build_dataset()
    if df.empty:
        print('[quantiles] Missing inputs; aborting.')
        return
    # Feature matrix: include existing prediction/market and engineered features
    base_cols = [c for c in ['pred_total','pred_margin','market_total','spread_home'] if c in df.columns]
    # Engineered features: any numeric columns starting with home_/away_ from features_history
    eng_cols = [c for c in df.columns if (c.startswith('home_') or c.startswith('away_'))]
    feat_cols = base_cols + eng_cols
    # If engineered features absent, fall back to base only
    if not feat_cols:
        X = pd.DataFrame(index=df.index)
    else:
        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out_rows = []
    # Optional fallback spreads from selection (if no LGBM or no features)
    try:
        with open(OUTPUTS / 'quantile_model_selection.json', 'r', encoding='utf-8') as f:
            sel = json.load(f)
    except Exception:
        sel = {}
    fb_total = float(sel.get('fallback_spread_total', 5.0))
    fb_margin = float(sel.get('fallback_spread_margin', 6.0))

    # Totals quantiles
    if 'actual_total' in df.columns:
        y_tot = df['actual_total']
        if lgb is None or X.shape[1] == 0:
            base = df['pred_total'] if 'pred_total' in df.columns else df.get('market_total', pd.Series(index=df.index, dtype=float))
            q50 = base
            spread = fb_total
            q10 = base - spread
            q90 = base + spread
        else:
            q10_m = train_quantile(y_tot, X, 0.10)
            q50_m = train_quantile(y_tot, X, 0.50)
            q90_m = train_quantile(y_tot, X, 0.90)
            q10 = q10_m.predict(X) if q10_m else np.full(len(X), np.nan)
            q50 = q50_m.predict(X) if q50_m else np.full(len(X), np.nan)
            q90 = q90_m.predict(X) if q90_m else np.full(len(X), np.nan)
    else:
        q10 = q50 = q90 = np.full(len(X), np.nan)
    # Margins quantiles
    if 'actual_margin' in df.columns:
        y_mar = df['actual_margin']
        if lgb is None or X.shape[1] == 0:
            base = df['pred_margin'] if 'pred_margin' in df.columns else df.get('spread_home', pd.Series(index=df.index, dtype=float))
            mq50 = base
            spread = fb_margin
            mq10 = base - spread
            mq90 = base + spread
        else:
            q10m = train_quantile(y_mar, X, 0.10)
            q50m = train_quantile(y_mar, X, 0.50)
            q90m = train_quantile(y_mar, X, 0.90)
            mq10 = q10m.predict(X) if q10m else np.full(len(X), np.nan)
            mq50 = q50m.predict(X) if q50m else np.full(len(X), np.nan)
            mq90 = q90m.predict(X) if q90m else np.full(len(X), np.nan)
    else:
        mq10 = mq50 = mq90 = np.full(len(X), np.nan)
    out = pd.DataFrame({
        'date': df['date'],
        'game_id': df['game_id'],
        'q10_total': q10,
        'q50_total': q50,
        'q90_total': q90,
        'q10_margin': mq10,
        'q50_margin': mq50,
        'q90_margin': mq90,
    })
    # Enforce monotonic quantiles per row (q10 <= q50 <= q90)
    def _mono(a, b, c):
        vals = [a, b, c]
        try:
            arr = np.array(vals, dtype=float)
        except Exception:
            return a, b, c
        if not np.isfinite(arr).any():
            return a, b, c
        arr = np.sort(arr)
        return float(arr[0]), float(arr[1]), float(arr[2])
    out[['q10_total','q50_total','q90_total']] = out[['q10_total','q50_total','q90_total']].apply(
        lambda r: pd.Series(_mono(r['q10_total'], r['q50_total'], r['q90_total'])), axis=1
    )
    out[['q10_margin','q50_margin','q90_margin']] = out[['q10_margin','q50_margin','q90_margin']].apply(
        lambda r: pd.Series(_mono(r['q10_margin'], r['q50_margin'], r['q90_margin'])), axis=1
    )
    out = out.sort_values(['date','game_id']).drop_duplicates(subset=['date','game_id'], keep='last')
    out.to_csv(OUTPUTS / 'quantiles_history.csv', index=False)
    print('[quantiles] Wrote outputs/quantiles_history.csv')

if __name__ == '__main__':
    main()
