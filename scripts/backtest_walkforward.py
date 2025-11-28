from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.modeling.datasets import load_training_data
from src.modeling.metrics import classification_metrics, regression_metrics

try:
    from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
except Exception:
    LGBMRegressor = None  # type: ignore
    LGBMClassifier = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier  # type: ignore
except Exception:
    GradientBoostingRegressor = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore


def _regressor():
    if LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=400, learning_rate=0.05)
    if GradientBoostingRegressor is not None:
        return GradientBoostingRegressor()
    raise RuntimeError('No regressor available')


def _classifier():
    if LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=500, learning_rate=0.05)
    if GradientBoostingClassifier is not None:
        return GradientBoostingClassifier()
    raise RuntimeError('No classifier available')


def walkforward_dates(df: pd.DataFrame, k: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted({d for d in pd.to_datetime(df['date'], errors='coerce') if pd.notna(d)})
    if len(dates) < k + 5:
        k = max(1, len(dates) // 4)
    folds = []
    for i in range(k, len(dates)):
        train_end = dates[i - 1]
        test_date = dates[i]
        folds.append((train_end, test_date))
    return folds[-k:]


def main() -> None:
    ap = argparse.ArgumentParser(description='Walk-forward backtest for totals regression and win/ATS/OU classifiers.')
    ap.add_argument('--outputs-dir', default='outputs')
    ap.add_argument('--date-start', default=None)
    ap.add_argument('--date-end', default=None)
    ap.add_argument('--folds', type=int, default=8)
    args = ap.parse_args()

    df = load_training_data(args.outputs_dir, date_start=args.date_start, date_end=args.date_end)
    if df.empty:
        print('No data.')
        return
    if 'date' not in df.columns:
        print('Date column required for walk-forward.')
        return
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    folds = walkforward_dates(df, args.folds)
    results = []

    for train_end, test_date in folds:
        train = df[df['date'] <= train_end]
        test = df[df['date'] == test_date]
        if train.empty or test.empty:
            continue

        fold_res = {'train_end': train_end.strftime('%Y-%m-%d'), 'test_date': test_date.strftime('%Y-%m-%d')}

        # Totals regression
        if {'home_score','away_score'}.issubset(train.columns):
            train['actual_total'] = train['home_score'] + train['away_score']
            test['actual_total'] = test['home_score'] + test['away_score']
            reg_feats = [c for c in train.columns if c not in {'home_score','away_score','home_win','ats_home_cover','ou_over','actual_total','date'} and pd.api.types.is_numeric_dtype(train[c])]
            t_tr = train.dropna(subset=['actual_total'])
            t_te = test.dropna(subset=['home_score','away_score'])
            if not t_tr.empty and not t_te.empty:
                R = _regressor()
                R.fit(t_tr[reg_feats], t_tr['actual_total'].astype(float))
                preds = R.predict(t_te[reg_feats])
                fold_res['totals'] = regression_metrics(t_te['actual_total'].astype(float).values, preds)
                fold_res['totals']['n_test'] = int(len(t_te))
        # Classifiers
        for key, col in {'win':'home_win','ats':'ats_home_cover','ou':'ou_over'}.items():
            if col in train.columns:
                tr = train.dropna(subset=[col])
                te = test.dropna(subset=[col])
                feats = [c for c in tr.columns if c not in {col,'home_score','away_score','actual_total','home_win','ats_home_cover','ou_over','date'} and pd.api.types.is_numeric_dtype(tr[c])]
                if not tr.empty and not te.empty and feats:
                    C = _classifier()
                    C.fit(tr[feats], tr[col].astype(int))
                    prob = C.predict_proba(te[feats])[:,1]
                    fold_res[key] = classification_metrics(te[col].astype(int).values, prob)
                    fold_res[key]['n_test'] = int(len(te))
        results.append(fold_res)

    out = {
        'folds': results,
        'n_folds': len(results),
        'outputs_dir': args.outputs_dir,
        'date_start': args.date_start,
        'date_end': args.date_end,
    }
    out_path = Path(args.outputs_dir) / f'walkforward_backtest_{len(results)}folds.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Backtest report:', out_path)


if __name__ == '__main__':
    main()
