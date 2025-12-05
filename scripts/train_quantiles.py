import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def fit_residual_quantiles(outputs: Path, window_days: int = 60) -> dict:
    """
    Compute empirical residual quantiles for totals and margins.
    Priority source: outputs/backtest_reports/backtest_joined.csv with actuals.
    Fallback source: outputs/predictions_history_enriched.csv vs market metrics.
    """
    bt = _safe_read_csv(outputs / 'backtest_reports' / 'backtest_joined.csv')
    res_t = pd.Series(dtype=float)
    res_m = pd.Series(dtype=float)

    if not bt.empty and {'date', 'actual_total', 'pred_total', 'actual_margin', 'pred_margin'}.issubset(bt.columns):
        bt['_date'] = pd.to_datetime(bt['date'], errors='coerce')
        ref = pd.to_datetime(bt['_date'].max())
        win = bt[(bt['_date'] >= ref - pd.Timedelta(days=window_days)) & (bt['_date'] <= ref)]
        res_t = pd.to_numeric(win['actual_total'], errors='coerce') - pd.to_numeric(win['pred_total'], errors='coerce')
        res_m = pd.to_numeric(win['actual_margin'], errors='coerce') - pd.to_numeric(win['pred_margin'], errors='coerce')
        res_t = res_t.replace([np.inf, -np.inf], np.nan).dropna()
        res_m = res_m.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        hist = _safe_read_csv(outputs / 'predictions_history_enriched.csv')
        if hist.empty or not {'market_total', 'pred_total', 'closing_spread_home', 'pred_margin'}.issubset(hist.columns):
            return {"error": "no data"}
        res_t = pd.to_numeric(hist['market_total'], errors='coerce') - pd.to_numeric(hist['pred_total'], errors='coerce')
        res_m = pd.to_numeric(hist['closing_spread_home'], errors='coerce') - pd.to_numeric(hist['pred_margin'], errors='coerce')
        res_t = res_t.replace([np.inf, -np.inf], np.nan).dropna()
        res_m = res_m.replace([np.inf, -np.inf], np.nan).dropna()

    n_t = int(len(res_t))
    n_m = int(len(res_m))
    if n_t < 50 or n_m < 50:
        return {"warning": "insufficient residuals", "n_total": n_t, "n_margin": n_m, "window_days": window_days}

    # Use symmetric tails unless otherwise specified.
    eps = 0.10
    qt = {
        'q10': float(np.quantile(res_t, eps)),
        'q50': float(np.quantile(res_t, 0.5)),
        'q90': float(np.quantile(res_t, 1 - eps)),
    }
    qm = {
        'q10': float(np.quantile(res_m, eps)),
        'q50': float(np.quantile(res_m, 0.5)),
        'q90': float(np.quantile(res_m, 1 - eps)),
    }

    return {
        'residual_quantiles': {
            'total': qt,
            'margin': qm,
        },
        'n_total': n_t,
        'n_margin': n_m,
        'window_days': window_days,
        'source': 'backtest_joined' if not bt.empty else 'predictions_history_enriched',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-days', type=int, default=60)
    ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit_residual_quantiles(out_dir, args.window_days)
    model_path = out_dir / 'quantile_model.json'
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"[quantiles] wrote {model_path}")
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
