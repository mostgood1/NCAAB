"""Meta Ensemble Training Script

Builds LightGBM meta-learners for totals and margins using historical unified
prediction exports + daily_results actuals. Produces booster model artifacts:
  outputs/meta_ensemble_totals.txt
  outputs/meta_ensemble_margin.txt
and a summary JSON with feature importances + calibration stats:
  outputs/meta_ensemble_summary.json

Usage (PowerShell):
  python scripts/train_meta_ensemble.py --days 120 --min-games 500

Requirements: lightgbm must be installed (see requirements.txt).
Falls back gracefully if insufficient data or LightGBM import fails.
"""
from __future__ import annotations
import argparse, json, pathlib, datetime as dt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # allow running just to see message if lib missing
    lgb = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
RES = OUT / 'daily_results'

MIN_FEATURE_ROWS = 50

TOTAL_FEATURE_CANDIDATES = [
    'pred_total_model','pred_total_model_raw','pred_total_calibrated','market_total','closing_total','pred_total_sigma','pred_total_sigma_bootstrap'
]
MARGIN_FEATURE_CANDIDATES = [
    'pred_margin_model','pred_margin_calibrated','spread_home','closing_spread_home','pred_margin_sigma','pred_margin_sigma_bootstrap'
]


def load_unified(date_str: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_results(date_str: str) -> pd.DataFrame:
    p = RES / f'results_{date_str}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_dataset(days: int) -> pd.DataFrame:
    today = dt.date.today()
    frames = []
    for i in range(1, days+1):
        d = (today - dt.timedelta(days=i)).strftime('%Y-%m-%d')
        uni = load_unified(d)
        res = load_results(d)
        if uni.empty or res.empty:
            continue
        if {'game_id','home_score','away_score'}.issubset(res.columns):
            res['game_id'] = res['game_id'].astype(str)
            uni['game_id'] = uni['game_id'].astype(str)
            uni = uni.merge(res[['game_id','home_score','away_score']], on='game_id', how='left', suffixes=('','_r'))
        frames.append(uni)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_xy(df: pd.DataFrame, feature_candidates: list[str], target_expr: str):
    # Filter rows with target available
    if target_expr == 'actual_total':
        target = pd.to_numeric(df.get('home_score'), errors='coerce') + pd.to_numeric(df.get('away_score'), errors='coerce')
    elif target_expr == 'actual_margin':
        target = pd.to_numeric(df.get('home_score'), errors='coerce') - pd.to_numeric(df.get('away_score'), errors='coerce')
    else:
        raise ValueError('Unknown target expression')
    df['_target'] = target
    good = df['_target'].notna()
    sub = df[good].copy()
    feats = {}
    used = []
    for c in feature_candidates:
        if c in sub.columns:
            ser = pd.to_numeric(sub[c], errors='coerce')
            if ser.notna().sum() >= MIN_FEATURE_ROWS:
                # Impute mean for NaNs
                m = ser.mean()
                ser = ser.fillna(m)
                feats[c] = ser.values
                used.append(c)
    if not used:
        return None, None, []
    X = np.vstack([feats[c] for c in used]).T
    y = sub['_target'].values
    return X, y, used


def train_booster(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    if lgb is None:
        return None
    dataset = lgb.Dataset(X, label=y, feature_name=feature_names)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': -1,
        'seed': 42,
    }
    booster = lgb.train(params, dataset, num_boost_round=300, valid_sets=[dataset], verbose_eval=False)
    return booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90, help='Number of past days to include')
    ap.add_argument('--min-games', type=int, default=400, help='Minimum completed games to proceed')
    args = ap.parse_args()

    df = build_dataset(args.days)
    if df.empty:
        print('No historical unified/results data available; aborting meta ensemble training.')
        return
    # Ensure sufficient completed games
    completed_mask = pd.to_numeric(df.get('home_score'), errors='coerce').notna() & pd.to_numeric(df.get('away_score'), errors='coerce').notna()
    completed = int(completed_mask.sum())
    if completed < args.min_games:
        print(f'Insufficient completed games ({completed} < {args.min_games}); aborting.')
        return

    summary = {
        'completed_games': completed,
        'days_window': args.days,
        'totals_model': None,
        'margin_model': None,
    }

    # Totals booster
    X_t, y_t, feats_t = prepare_xy(df, TOTAL_FEATURE_CANDIDATES, 'actual_total')
    if X_t is not None and lgb is not None:
        booster_t = train_booster(X_t, y_t, feats_t)
        if booster_t is not None:
            out_t = OUT / 'meta_ensemble_totals.txt'
            booster_t.save_model(str(out_t))
            importance_t = booster_t.feature_importance(importance_type='gain')
            summary['totals_model'] = {
                'features': feats_t,
                'importance': {f: float(i) for f,i in zip(feats_t, importance_t)},
                'rmse': float(np.sqrt(np.mean((booster_t.predict(X_t) - y_t)**2)))
            }
            print(f'Saved totals meta ensemble to {out_t}')

    # Margin booster
    X_m, y_m, feats_m = prepare_xy(df, MARGIN_FEATURE_CANDIDATES, 'actual_margin')
    if X_m is not None and lgb is not None:
        booster_m = train_booster(X_m, y_m, feats_m)
        if booster_m is not None:
            out_m = OUT / 'meta_ensemble_margin.txt'
            booster_m.save_model(str(out_m))
            importance_m = booster_m.feature_importance(importance_type='gain')
            summary['margin_model'] = {
                'features': feats_m,
                'importance': {f: float(i) for f,i in zip(feats_m, importance_m)},
                'rmse': float(np.sqrt(np.mean((booster_m.predict(X_m) - y_m)**2)))
            }
            print(f'Saved margin meta ensemble to {out_m}')

    # Persist summary
    (OUT / 'meta_ensemble_summary.json').write_text(json.dumps(summary, indent=2))
    print('Meta ensemble training summary written.')

if __name__ == '__main__':
    main()
