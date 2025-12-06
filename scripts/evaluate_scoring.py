import argparse
import json
import math
from pathlib import Path
import datetime as dt

import pandas as pd
import numpy as np


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _today_str_local() -> str:
    # Use local timezone naive date string
    return dt.datetime.now().strftime('%Y-%m-%d')


def _load_sigma(outputs: Path) -> tuple[float | None, float | None, bool, bool]:
    """Return (sigma_total, sigma_margin, total_default_used, margin_default_used)."""
    qm_path = outputs / 'quantile_model.json'
    sigma_t = None
    sigma_m = None
    if qm_path.exists():
        try:
            payload = json.loads(qm_path.read_text(encoding='utf-8'))
            st = payload.get('residual_quantiles', {})
            qt10 = st.get('total', {}).get('q10')
            qt90 = st.get('total', {}).get('q90')
            qm10 = st.get('margin', {}).get('q10')
            qm90 = st.get('margin', {}).get('q90')
            z90 = 1.2815515655446004
            if isinstance(qt10, (int, float)) and isinstance(qt90, (int, float)):
                span_t = float(qt90) - float(qt10)
                if abs(span_t) > 1e-6:
                    sigma_t = span_t / (2.0 * z90)
            if isinstance(qm10, (int, float)) and isinstance(qm90, (int, float)):
                span_m = float(qm90) - float(qm10)
                if abs(span_m) > 1e-6:
                    sigma_m = span_m / (2.0 * z90)
        except Exception:
            sigma_t = None
            sigma_m = None
    t_def_used = False
    m_def_used = False
    if not isinstance(sigma_t, (int, float)) or (sigma_t is not None and sigma_t <= 0):
        sigma_t = 11.0
        t_def_used = True
    if not isinstance(sigma_m, (int, float)) or (sigma_m is not None and sigma_m <= 0):
        sigma_m = 10.5
        m_def_used = True
    return float(sigma_t), float(sigma_m), t_def_used, m_def_used


def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z ** 2))


def _Phi(z: np.ndarray) -> np.ndarray:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + (2.0 / math.sqrt(math.pi)) * np.vectorize(math.erf)(z / math.sqrt(2.0)))


def crps_normal(mean: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for normal N(mean, sigma) at observation x.
    Gneiting & Raftery (2007): CRPS = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ], z=(x-mean)/sigma
    """
    sigma = np.asarray(sigma, dtype=float)
    mean = np.asarray(mean, dtype=float)
    x = np.asarray(x, dtype=float)
    z = (x - mean) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / math.sqrt(math.pi))


def loglik_normal(mean: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    mean = np.asarray(mean, dtype=float)
    x = np.asarray(x, dtype=float)
    z = (x - mean) / sigma
    return -np.log(sigma * math.sqrt(2.0 * math.pi)) - 0.5 * (z ** 2)


def evaluate(outputs: Path, window_days: int = 90, date: str | None = None) -> dict:
    bt = _safe_read_csv(outputs / 'backtest_reports' / 'backtest_joined.csv')
    pt = pd.Series(dtype=float)
    pm = pd.Series(dtype=float)
    at = pd.Series(dtype=float)
    am = pd.Series(dtype=float)
    sel_date = None
    # When an explicit date is provided, prefer per-date evaluation against daily results + predictions
    if date:
        try:
            sel_date = str(date)
            dfr = _safe_read_csv(outputs / 'daily_results' / f'results_{sel_date}.csv')
            dfp = _safe_read_csv(outputs / f'predictions_unified_enriched_{sel_date}.csv')
            # Pending mode: if finals are missing for the selected date but predictions exist,
            # emit a non-error payload so the UI doesn't show a red warning during the day.
            if dfr.empty and not dfp.empty:
                return {
                    "pending": True,
                    "status": "pending",
                    "reason": "daily_results missing for selected date",
                    "source_date": sel_date,
                    "totals_rows": 0,
                    "margins_rows": 0,
                    "window_days": int(window_days),
                }
            if dfr.empty and dfp.empty:
                return {"error": f"no artifacts for {sel_date}"}
            hs = pd.to_numeric(dfr.get('home_score'), errors='coerce')
            as_ = pd.to_numeric(dfr.get('away_score'), errors='coerce')
            at = (hs + as_)
            am = (hs - as_)
            pt = pd.to_numeric(dfp.get('pred_total'), errors='coerce')
            pm = pd.to_numeric(dfp.get('pred_margin'), errors='coerce')
            n = min(len(pt), len(at))
            pt = pt.iloc[:n]; at = at.iloc[:n]; pm = pm.iloc[:n]; am = am.iloc[:n]
        except Exception:
            return {"error": f"failed per-date evaluation for {sel_date}"}
    elif not bt.empty and {'date','pred_total','pred_margin','actual_total','actual_margin'}.issubset(bt.columns):
        bt['_date'] = pd.to_datetime(bt['date'], errors='coerce')
        ref = pd.to_datetime(bt['_date'].max())
        win = bt[(bt['_date'] >= ref - pd.Timedelta(days=window_days)) & (bt['_date'] <= ref)].copy()
        # Coerce columns
        pt = pd.to_numeric(win.get('pred_total'), errors='coerce')
        pm = pd.to_numeric(win.get('pred_margin'), errors='coerce')
        at = pd.to_numeric(win.get('actual_total'), errors='coerce')
        am = pd.to_numeric(win.get('actual_margin'), errors='coerce')
    else:
        # Fallback: evaluate using the most recent available daily_results + matching predictions_enriched
        candidates = sorted((outputs / 'daily_results').glob('results_*.csv'))
        dfp = pd.DataFrame(); dfr = pd.DataFrame()
        for p in reversed(candidates):
            try:
                name = p.name  # results_YYYY-MM-DD.csv
                d = name.replace('results_', '').replace('.csv', '')
                pred_path = outputs / f'predictions_unified_enriched_{d}.csv'
                dfp = _safe_read_csv(pred_path)
                dfr = _safe_read_csv(p)
                if not dfp.empty and not dfr.empty:
                    sel_date = d
                    break
            except Exception:
                continue
        if dfp.empty or dfr.empty:
            return {"error": "no backtest data"}
        # Build actuals from scores
        hs = pd.to_numeric(dfr.get('home_score'), errors='coerce')
        as_ = pd.to_numeric(dfr.get('away_score'), errors='coerce')
        at = (hs + as_)
        am = (hs - as_)
        pt = pd.to_numeric(dfp.get('pred_total'), errors='coerce')
        pm = pd.to_numeric(dfp.get('pred_margin'), errors='coerce')
        # Align lengths
        n = min(len(pt), len(at))
        pt = pt.iloc[:n]
        at = at.iloc[:n]
        pm = pm.iloc[:n]
        am = am.iloc[:n]
    sigma_t, sigma_m, t_def_used, m_def_used = _load_sigma(outputs)
    st = np.full_like(pt, float(sigma_t), dtype=float)
    sm = np.full_like(pm, float(sigma_m), dtype=float)
    # Drop NA prior to numpy conversion
    mask_t = pt.notna() & at.notna()
    mask_m = pm.notna() & am.notna()
    pt_np = pt[mask_t].to_numpy()
    at_np = at[mask_t].to_numpy()
    pm_np = pm[mask_m].to_numpy()
    am_np = am[mask_m].to_numpy()
    st_np = np.full_like(pt_np, float(sigma_t), dtype=float)
    sm_np = np.full_like(pm_np, float(sigma_m), dtype=float)
    crps_t = crps_normal(pt_np, st_np, at_np) if len(pt_np) else np.array([])
    crps_m = crps_normal(pm_np, sm_np, am_np) if len(pm_np) else np.array([])
    ll_t = loglik_normal(pt_np, st_np, at_np) if len(pt_np) else np.array([])
    ll_m = loglik_normal(pm_np, sm_np, am_np) if len(pm_np) else np.array([])
    res = {
        'totals_crps_mean': float(crps_t.mean()) if len(crps_t) else None,
        'margins_crps_mean': float(crps_m.mean()) if len(crps_m) else None,
        'totals_loglik_mean': float(ll_t.mean()) if len(ll_t) else None,
        'margins_loglik_mean': float(ll_m.mean()) if len(ll_m) else None,
        'totals_rows': int(len(pt_np)),
        'margins_rows': int(len(pm_np)),
        'sigma_total_default_used': bool(t_def_used),
        'sigma_margin_default_used': bool(m_def_used),
        'window_days': int(window_days),
        'source_date': sel_date,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
    ap.add_argument('--window-days', type=int, default=90)
    ap.add_argument('--date', type=str, default=None, help='Evaluate a specific date (YYYY-MM-DD)')
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate(out_dir, args.window_days, args.date)
    # Write date-specific scoring artifact for app ingestion
    target_date = args.date if args.date else _today_str_local()
    scoring_path = out_dir / f'scoring_{target_date}.json'
    with open(scoring_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"[scoring] wrote {scoring_path}")
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
