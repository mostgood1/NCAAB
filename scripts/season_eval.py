import json, math
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')
REPORTS = OUT / 'backtest_reports'
REPORTS.mkdir(parents=True, exist_ok=True)

def safe_read_csv(p: Path, **kw):
    try:
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, **kw)
    except Exception:
        return pd.DataFrame()

def calibration_bins(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10):
    try:
        bins = []
        q = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = q[i], q[i+1]
            mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
            if mask.sum() == 0:
                bins.append({"bin": i, "range": [float(lo), float(hi)], "n": 0, "prob_mean": None, "outcome_rate": None})
                continue
            pm = float(prob[mask].mean())
            orate = float(outcome[mask].mean())
            bins.append({"bin": i, "range": [float(lo), float(hi)], "n": int(mask.sum()), "prob_mean": pm, "outcome_rate": orate, "abs_gap": abs(pm - orate)})
        return bins
    except Exception:
        return []

preds = safe_read_csv(OUT / 'predictions_all.csv')
daily_frames = []
for p in (OUT / 'daily_results').glob('results_*.csv'):
    df = safe_read_csv(p)
    if df.empty:
        continue
    df['date'] = p.stem.split('_')[-1]
    daily_frames.append(df)

daily = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()

# Normalize keys/types early
for df_ in (preds, daily):
    if 'date' in df_.columns:
        try:
            df_['date'] = df_['date'].astype(str).str.slice(0, 10)
        except Exception:
            pass
    if 'game_id' in df_.columns:
        try:
            df_['game_id'] = df_['game_id'].astype(str)
        except Exception:
            pass

if 'game_id' in preds.columns:
    preds['game_id'] = preds['game_id'].astype(str)
if 'game_id' in daily.columns:
    daily['game_id'] = daily['game_id'].astype(str)

# Merge on game_id + date; avoid silent column drops by suffixing
base = preds.copy()
if not daily.empty:
    try:
        base = base.merge(daily, on=['game_id','date'], how='left', suffixes=('', '_dr'))
    except Exception:
        pass

# Fallback: per-date enriched predictions when predictions_all is sparse
try:
    # Identify unique dates from daily results; load enriched per-date and union
    enriched_frames = []
    for d in sorted(set(daily.get('date', pd.Series(dtype=str)).dropna().astype(str))):
        p = OUT / f'predictions_unified_enriched_{d}.csv'
        df_e = safe_read_csv(p)
        if not df_e.empty:
            # Ensure keys align
            for k in ('date','game_id'):
                if k in df_e.columns:
                    df_e[k] = df_e[k].astype(str)
            enriched_frames.append(df_e)
    if enriched_frames:
        enriched = pd.concat(enriched_frames, ignore_index=True, sort=False)
        # Prefer enriched rows for dates present; outer merge preserves any extras
        try:
            base = enriched.merge(daily, on=['game_id','date'], how='left', suffixes=('', '_dr'))
        except Exception:
            pass
except Exception:
    pass

summary = {}

# Actuals derivation (coalesce from any suffixed sources)
def coalesce_cols(df: pd.DataFrame, target: str, candidates: list[str]):
    vals = None
    for c in candidates:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors='coerce') if df[c].dtype != 'O' else pd.to_numeric(df[c], errors='coerce')
            vals = series if vals is None else vals.where(vals.notna(), series)
    if vals is not None:
        df[target] = vals

# Derive actual_total/margin if not present
if not {'actual_total','actual_margin'}.issubset(base.columns):
    hs = pd.to_numeric(base.get('home_score') if 'home_score' in base.columns else base.get('home_score_dr'), errors='coerce')
    as_ = pd.to_numeric(base.get('away_score') if 'away_score' in base.columns else base.get('away_score_dr'), errors='coerce')
    base['actual_total'] = hs + as_
    base['actual_margin'] = hs - as_

# Closing/coalesce: totals and spreads from multiple sources
coalesce_cols(base, 'closing_total', ['closing_total','closing_total_dr','market_total_close','market_total_last','market_total'])
coalesce_cols(base, 'closing_spread_home', ['closing_spread_home','closing_spread_home_dr','spread_home_close','spread_home'])

# Regression metrics
pt = pd.to_numeric(base.get('pred_total_calibrated') if 'pred_total_calibrated' in base.columns else base.get('pred_total'), errors='coerce')
pm = pd.to_numeric(base.get('pred_margin_calibrated') if 'pred_margin_calibrated' in base.columns else base.get('pred_margin'), errors='coerce')
at = pd.to_numeric(base.get('actual_total'), errors='coerce')
am = pd.to_numeric(base.get('actual_margin'), errors='coerce')
mask_t = pt.notna() & at.notna()
mask_m = pm.notna() & am.notna()
if mask_t.sum() > 0:
    err_t = pt[mask_t] - at[mask_t]
    summary['regression_total'] = {
        'n': int(mask_t.sum()),
        'mae': float(np.mean(np.abs(err_t))),
        'rmse': float(np.sqrt(np.mean(err_t**2))),
        'bias': float(np.mean(err_t)),
    }
else:
    summary['regression_total'] = {'n': 0}
if mask_m.sum() > 0:
    err_m = pm[mask_m] - am[mask_m]
    summary['regression_margin'] = {
        'n': int(mask_m.sum()),
        'mae': float(np.mean(np.abs(err_m))),
        'rmse': float(np.sqrt(np.mean(err_m**2))),
        'bias': float(np.mean(err_m)),
    }
else:
    summary['regression_margin'] = {'n': 0}

# Probability metrics (over/cover)
prob_summary = {}
# Over outcome using closing_total
if {'closing_total','actual_total'}.issubset(base.columns):
    ct = pd.to_numeric(base['closing_total'], errors='coerce')
    at2 = pd.to_numeric(base['actual_total'], errors='coerce')
    outcome_over = np.where(at2.notna() & ct.notna(), (at2 > ct).astype(int), np.nan)
    outcome_over = pd.Series(outcome_over, index=base.index)
    # choose best prob column available
    over_cols = [c for c in ['p_over_emp','p_over_meta_cal','p_over_display','p_over_dist','p_over_meta'] if c in base.columns]
    best_col = None; best_n = -1
    for c in over_cols:
        p = pd.to_numeric(base[c], errors='coerce')
        mask = p.notna() & outcome_over.notna()
        if mask.sum() > best_n:
            best_col = c; best_n = mask.sum()
    if best_col:
        p = pd.to_numeric(base[best_col], errors='coerce')
        mask = p.notna() & outcome_over.notna()
        y = outcome_over[mask].astype(float)
        pr = p[mask].astype(float)
        brier = float(np.mean((pr.values - y.values) ** 2))
        logloss = float(-np.mean(y.values * np.log(np.clip(pr.values,1e-9,1-1e-9)) + (1-y.values) * np.log(np.clip(1-pr.values,1e-9,1-1e-9))))
        bins = calibration_bins(pr.values, y.values, n_bins=10)
        prob_summary['over'] = {'n': int(mask.sum()), 'brier': brier, 'log_loss': logloss, 'prob_column_used': best_col, 'ece_mean_abs_gap': float(np.nanmean([b.get('abs_gap') for b in bins if b.get('abs_gap') is not None]))}

# Cover outcome using closing_spread_home
if {'closing_spread_home','actual_margin'}.issubset(base.columns):
    cs = pd.to_numeric(base['closing_spread_home'], errors='coerce')
    am2 = pd.to_numeric(base['actual_margin'], errors='coerce')
    # home covers if margin > -spread (spread negative when favored)
    outcome_cover = np.where(am2.notna() & cs.notna(), (am2 > -cs).astype(int), np.nan)
    outcome_cover = pd.Series(outcome_cover, index=base.index)
    cover_cols = [c for c in ['p_home_cover_emp','p_home_cover_meta_cal','p_cover_display','p_home_cover_dist','p_home_cover_meta'] if c in base.columns]
    best_c = None; best_n2 = -1
    for c in cover_cols:
        p = pd.to_numeric(base[c], errors='coerce')
        mask = p.notna() & outcome_cover.notna()
        if mask.sum() > best_n2:
            best_c = c; best_n2 = mask.sum()
    if best_c:
        p = pd.to_numeric(base[best_c], errors='coerce')
        mask = p.notna() & outcome_cover.notna()
        y = outcome_cover[mask].astype(float)
        pr = p[mask].astype(float)
        brier = float(np.mean((pr.values - y.values) ** 2))
        logloss = float(-np.mean(y.values * np.log(np.clip(pr.values,1e-9,1-1e-9)) + (1-y.values) * np.log(np.clip(1-pr.values,1e-9,1-1e-9))))
        bins = calibration_bins(pr.values, y.values, n_bins=10)
        prob_summary['cover'] = {'n': int(mask.sum()), 'brier': brier, 'log_loss': logloss, 'prob_column_used': best_c, 'ece_mean_abs_gap': float(np.nanmean([b.get('abs_gap') for b in bins if b.get('abs_gap') is not None]))}

summary['probabilities'] = prob_summary

# CRPS for totals if sigma present
if {'pred_total_sigma','actual_total'}.issubset(base.columns):
    mu = pd.to_numeric(base.get('pred_total_calibrated') if 'pred_total_calibrated' in base.columns else base.get('pred_total'), errors='coerce')
    sig = pd.to_numeric(base['pred_total_sigma'], errors='coerce')
    act = pd.to_numeric(base['actual_total'], errors='coerce')
    m = mu.notna() & sig.notna() & act.notna()
    if m.sum() > 0:
        try:
            from math import sqrt, pi
            z = (act[m] - mu[m]) / sig[m]
            # Approximate CRPS using normal closed form (like backtest_full)
            from math import erf
            Phi = 0.5 * (1 + erf(z / math.sqrt(2)))
            phi = (1 / math.sqrt(2*math.pi)) * np.exp(-0.5 * z**2)
            crps_vals = sig[m] * (z * (2*Phi - 1) + 2*phi - 1/math.sqrt(math.pi))
            summary['crps_total'] = float(np.mean(crps_vals))
        except Exception:
            summary['crps_total'] = None

# Date range
dates = sorted(set([str(d) for d in base.get('date', pd.Series(dtype=str)).dropna().unique()]))
if dates:
    summary['date_range'] = {'min': dates[0], 'max': dates[-1]}

(REPORTS / 'season_eval_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
print(f"Wrote summary -> {(REPORTS / 'season_eval_summary.json').as_posix()}")
