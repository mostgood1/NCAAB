import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_df(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def fit_residual_quantiles(outputs: Path, window_days: int = 60) -> dict:
    bt = load_df(outputs / 'backtest_reports' / 'backtest_joined.csv')
    if bt.empty:
        hist = load_df(outputs / 'predictions_history_enriched.csv')
        if hist.empty:
            return {"error": "no data"}
        res_t = (hist['market_total'] - hist['pred_total']).dropna()
        res_m = (hist['closing_spread_home'] - hist['pred_margin']).dropna()
    else:
        bt['_date'] = pd.to_datetime(bt['date'], errors='coerce')
        ref = pd.to_datetime(bt['_date'].max())
        win = bt[(bt['_date'] >= ref - pd.Timedelta(days=window_days)) & (bt['_date'] <= ref)]
        res_t = (win['actual_total'] - win['pred_total']).dropna()
        res_m = (win['actual_margin'] - win['pred_margin']).dropna()
    if len(res_t) < 50 or len(res_m) < 50:
        return {"warning": "insufficient residuals", "n_total": int(len(res_t)), "n_margin": int(len(res_m))}
    eps = 0.10
    qt = {
        'q10': float(np.quantile(res_t, eps)),
        'q50': float(np.quantile(res_t, 0.5)),
        'q90': float(np.quantile(res_t, 1-eps)),
    }
    qm = {
        'q10': float(np.quantile(res_m, eps)),
        'q50': float(np.quantile(res_m, 0.5)),
        'q90': float(np.quantile(res_m, 1-eps)),
    }
    return {
        'residual_quantiles': {
            'total': qt,
            'margin': qm,
        },
        'n_total': int(len(res_t)),
        'n_margin': int(len(res_m)),
        'window_days': window_days,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    res = fit_residual_quantiles(out_dir, args.window_days)
    model_path = out_dir / 'quantile_model.json'
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_df(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def fit_residual_quantiles(outputs: Path, window_days: int = 60) -> dict:
    bt = load_df(outputs / 'backtest_reports' / 'backtest_joined.csv')
    if bt.empty:
        # Fallback: use predictions_history_enriched vs market_total/spread_home
        hist = load_df(outputs / 'predictions_history_enriched.csv')
        if hist.empty:
            return {"error": "no data"}
        # Build residuals
        res_t = (hist['market_total'] - hist['pred_total']).dropna()
        res_m = (hist['closing_spread_home'] - hist['pred_margin']).dropna()
    else:
        bt['_date'] = pd.to_datetime(bt['date'], errors='coerce')
        ref = pd.to_datetime(bt['_date'].max())
        win = bt[(bt['_date'] >= ref - pd.Timedelta(days=window_days)) & (bt['_date'] <= ref)]
        import argparse
        from pathlib import Path
        import json
        import pandas as pd
        import numpy as np

        def load_df(p: Path) -> pd.DataFrame:
            try:
                return pd.read_csv(p)
        if lgb is None or X.shape[1] == 0:
                return pd.DataFrame()

        def fit_residual_quantiles(outputs: Path, window_days: int = 60) -> dict:
            bt = load_df(outputs / 'backtest_reports' / 'backtest_joined.csv')
            if bt.empty:
                hist = load_df(outputs / 'predictions_history_enriched.csv')
                if hist.empty:
                    return {"error": "no data"}
                res_t = (hist['market_total'] - hist['pred_total']).dropna()
                res_m = (hist['closing_spread_home'] - hist['pred_margin']).dropna()
            else:
                bt['_date'] = pd.to_datetime(bt['date'], errors='coerce')
                ref = pd.to_datetime(bt['_date'].max())
                win = bt[(bt['_date'] >= ref - pd.Timedelta(days=window_days)) & (bt['_date'] <= ref)]
                res_t = (win['actual_total'] - win['pred_total']).dropna()
                res_m = (win['actual_margin'] - win['pred_margin']).dropna()
            if len(res_t) < 50 or len(res_m) < 50:
                return {"warning": "insufficient residuals", "n_total": int(len(res_t)), "n_margin": int(len(res_m))}
            eps = 0.10
            qt = {
                'q10': float(np.quantile(res_t, eps)),
                'q50': float(np.quantile(res_t, 0.5)),
                'q90': float(np.quantile(res_t, 1-eps)),
            }
            qm = {
                'q10': float(np.quantile(res_m, eps)),
                'q50': float(np.quantile(res_m, 0.5)),
                'q90': float(np.quantile(res_m, 1-eps)),
            }
            return {
                'residual_quantiles': {
                    'total': qt,
                    'margin': qm,
                },
                'n_total': int(len(res_t)),
                'n_margin': int(len(res_m)),
                'window_days': window_days,
            }

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument('--window-days', type=int, default=60)
            ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
            args = ap.parse_args()
            out_dir = Path(args.outputs)
            res = fit_residual_quantiles(out_dir, args.window_days)
            model_path = out_dir / 'quantile_model.json'
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=2)
            print(json.dumps(res, indent=2))

        if __name__ == '__main__':
            main()
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
