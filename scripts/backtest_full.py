#!/usr/bin/env python
"""Comprehensive pipeline backtest and evaluation including derivatives.

Joins historical predictions, closing/last odds, and finalized daily results to compute:
- Totals regression metrics (MAE, RMSE, bias, distribution)
- Margin regression metrics
- Classification metrics for cover/over probabilities (Brier, LogLoss, calibration bins, AUC placeholder)
- Edge realization: ROI and PnL vs closing using Kelly fractions (totals, spreads, ML) where available
- Interval coverage diagnostics (if prediction intervals present)
- Half-time derivative performance (1H -> full) when 1H predictions present
- Drift metrics across monthly slices (error mean/variance, probability distribution JS divergence)

Outputs:
  outputs/backtest_reports/backtest_full_summary.json
  outputs/backtest_reports/backtest_cohort.csv (per-game metrics/bets)

Usage:
  python scripts/backtest_full.py --outputs-dir outputs --min-date 2025-11-01 --max-date 2025-11-30 --edge-threshold 0.02 --kelly-floor 0.01

Notes:
- Requires predictions_all.csv and closing_lines.csv accumulation, plus daily_results/*.csv.
- Gracefully skips missing columns; marks metrics as null when inputs absent.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# ---------- Utility ----------

def safe_read_csv(p: Path, **kwargs) -> pd.DataFrame:
    try:
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()

def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    try:
        return float(np.mean((prob - outcome) ** 2))
    except Exception:
        return math.nan

def log_loss(prob: np.ndarray, outcome: np.ndarray, eps=1e-12) -> float:
    try:
        p = np.clip(prob, eps, 1 - eps)
        return float(-np.mean(outcome * np.log(p) + (1 - outcome) * np.log(1 - p)))
    except Exception:
        return math.nan

def calibration_bins(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
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

def interval_coverage(df: pd.DataFrame, point_col: str, lower_col: str, upper_col: str) -> Dict[str, Any]:
    if not {point_col, lower_col, upper_col, 'actual_total'}.issubset(df.columns):
        return {}
    try:
        act = pd.to_numeric(df['actual_total'], errors='coerce')
        lo = pd.to_numeric(df[lower_col], errors='coerce')
        hi = pd.to_numeric(df[upper_col], errors='coerce')
        mask = act.notna() & lo.notna() & hi.notna()
        if mask.sum() == 0:
            return {}
        covered = (act[mask] >= lo[mask]) & (act[mask] <= hi[mask])
        return {"n": int(mask.sum()), "coverage_rate": float(covered.mean())}
    except Exception:
        return {}

def auc_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Compute AUC via rank method (no sklearn dependency)."""
    try:
        y = outcome.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return math.nan
        order = np.argsort(prob)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(prob))
        pos_ranks = ranks[y == 1]
        n_pos = y.sum(); n_neg = len(y) - n_pos
        auc = (pos_ranks.sum() - n_pos*(n_pos-1)/2) / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return math.nan

def normal_crps(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> float:
    """Closed form CRPS for normal forecast vs observation (mean over rows)."""
    try:
        from math import pi, erf
        if len(mu) == 0:
            return math.nan
        z = (x - mu) / sigma
        Phi = 0.5 * (1 + erf(z / math.sqrt(2)))
        phi = (1 / math.sqrt(2*math.pi)) * np.exp(-0.5 * z**2)
        crps_vals = sigma * (z * (2*Phi - 1) + 2*phi - 1/math.sqrt(math.pi))
        return float(np.mean(crps_vals))
    except Exception:
        return math.nan

def halftime_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    needed = {'home_score_1h','away_score_1h','pred_total_1h','pred_margin_1h'}
    if not needed.issubset(df.columns):
        return {}
    try:
        hs = pd.to_numeric(df['home_score_1h'], errors='coerce')
        as_ = pd.to_numeric(df['away_score_1h'], errors='coerce')
        act_tot_1h = hs + as_
        act_marg_1h = hs - as_
        pt1 = pd.to_numeric(df['pred_total_1h'], errors='coerce')
        pm1 = pd.to_numeric(df['pred_margin_1h'], errors='coerce')
        mask_t = act_tot_1h.notna() & pt1.notna()
        mask_m = act_marg_1h.notna() & pm1.notna()
        out: Dict[str, Any] = {}
        if mask_t.sum() > 0:
            err_t = pt1[mask_t] - act_tot_1h[mask_t]
            out['total_1h'] = {'n': int(mask_t.sum()), 'mae': float(np.mean(np.abs(err_t))), 'bias': float(np.mean(err_t))}
        if mask_m.sum() > 0:
            err_m = pm1[mask_m] - act_marg_1h[mask_m]
            out['margin_1h'] = {'n': int(mask_m.sum()), 'mae': float(np.mean(np.abs(err_m))), 'bias': float(np.mean(err_m))}
        if 'pred_total_calibrated' in df.columns:
            pt_full = pd.to_numeric(df['pred_total_calibrated'], errors='coerce')
        else:
            pt_full = pd.to_numeric(df.get('pred_total'), errors='coerce')
        mask_trans = pt1.notna() & pt_full.notna()
        if mask_trans.sum() > 0:
            trans_err = (2*pt1[mask_trans]) - pt_full[mask_trans]
            out['translation_total'] = {'n': int(mask_trans.sum()), 'mae': float(np.mean(np.abs(trans_err))), 'bias': float(np.mean(trans_err))}
        return out
    except Exception:
        return {}

# ---------- Core Backtest ----------

def load_daily_results(outputs_dir: Path, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    dr_dir = outputs_dir / 'daily_results'
    if not dr_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(dr_dir.glob('results_*.csv')):
        date_part = p.stem.split('_')[-1]
        if min_date and date_part < min_date: continue
        if max_date and date_part > max_date: continue
        df = safe_read_csv(p)
        if not df.empty:
            df['date'] = date_part
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if 'game_id' in out.columns:
        out['game_id'] = out['game_id'].astype(str)
    return out

def load_enriched_predictions(outputs_dir: Path, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    """Load per-date enriched unified predictions as a robust fallback.
    Files: outputs/predictions_unified_enriched_<date>.csv
    """
    enr_dir = outputs_dir
    frames: List[pd.DataFrame] = []
    for p in sorted(enr_dir.glob('predictions_unified_enriched_*.csv')):
        date_part = p.stem.split('_')[-1]
        if min_date and date_part < min_date: continue
        if max_date and date_part > max_date: continue
        df = safe_read_csv(p)
        if not df.empty:
            df['date'] = date_part
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if 'game_id' in out.columns:
        out['game_id'] = out['game_id'].astype(str)
    # Normalize date
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return out

def compute_regression_metrics(pred: pd.Series, actual: pd.Series) -> Dict[str, Any]:
    predn = pd.to_numeric(pred, errors='coerce')
    actn = pd.to_numeric(actual, errors='coerce')
    mask = predn.notna() & actn.notna()
    if mask.sum() == 0:
        return {"n": 0}
    err = predn[mask] - actn[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pred_mean": float(predn[mask].mean()),
        "actual_mean": float(actn[mask].mean()),
        "pred_std": float(predn[mask].std(ddof=0)),
        "actual_std": float(actn[mask].std(ddof=0)),
    }

def derive_outcomes(join: pd.DataFrame) -> pd.DataFrame:
    # Actual margin and O/U outcomes using closing_total/spread where present.
    if {'home_score','away_score'}.issubset(join.columns):
        hs = pd.to_numeric(join['home_score'], errors='coerce')
        as_ = pd.to_numeric(join['away_score'], errors='coerce')
        join['actual_total'] = hs + as_
        join['actual_margin'] = hs - as_
    # Over outcome vs closing (1 if actual_total > closing_total)
    if {'actual_total','closing_total'}.issubset(join.columns):
        ct = pd.to_numeric(join['closing_total'], errors='coerce')
        at = pd.to_numeric(join['actual_total'], errors='coerce')
        join['outcome_over'] = np.where(at.notna() & ct.notna(), (at > ct).astype(int), np.nan)
    # Cover outcome for home vs closing_spread_home (home covers if margin > -spread_home)
    if {'actual_margin','closing_spread_home'}.issubset(join.columns):
        sp = pd.to_numeric(join['closing_spread_home'], errors='coerce')
        am = pd.to_numeric(join['actual_margin'], errors='coerce')
        # Convention: closing_spread_home negative when favored; home covers if actual_margin > -spread
        join['outcome_home_cover'] = np.where(am.notna() & sp.notna(), (am > -sp).astype(int), np.nan)
    return join

def bet_selection(df: pd.DataFrame, edge_col: str, kelly_col: str, threshold: float) -> pd.DataFrame:
    if edge_col not in df.columns or kelly_col not in df.columns:
        return pd.DataFrame()
    e = pd.to_numeric(df[edge_col], errors='coerce')
    k = pd.to_numeric(df[kelly_col], errors='coerce')
    mask = e.notna() & k.notna() & (e.abs() >= threshold) & (k > 0)
    return df.loc[mask].copy()

def realized_pnl(row: pd.Series, odds_prob: float, stake: float, outcome_win: int) -> float:
    # Approx EV realized: using fair probability; if outcome_win=1 profit = (1/odds_prob - 1)*stake else -stake
    # If odds_prob missing, return 0.
    try:
        if math.isnan(odds_prob) or stake <= 0:
            return 0.0
        if outcome_win == 1:
            return ((1.0 / odds_prob) - 1.0) * stake
        else:
            return -stake
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs-dir', default='outputs')
    ap.add_argument('--min-date')
    ap.add_argument('--max-date')
    ap.add_argument('--edge-threshold', type=float, default=0.02, help='Absolute edge threshold for selecting bets.')
    ap.add_argument('--kelly-floor', type=float, default=0.01, help='Minimum Kelly fraction to count a bet.')
    # Prefer empirical probabilities first when available
    ap.add_argument('--prob-cols-cover', nargs='*', default=['p_home_cover_emp','p_home_cover_meta_cal','p_home_cover_meta','p_cover_display','p_home_cover_dist'])
    ap.add_argument('--prob-cols-over', nargs='*', default=['p_over_emp','p_over_meta_cal','p_over_meta','p_over_display','p_over_dist'])
    ap.add_argument('--interval-point-col', default='pred_total_calibrated')
    ap.add_argument('--interval-lower-col', default='pred_total_calibrated_low_90')
    ap.add_argument('--interval-upper-col', default='pred_total_calibrated_high_90')
    ap.add_argument('--interval-lower-col-95', default='pred_total_calibrated_low_95')
    ap.add_argument('--interval-upper-col-95', default='pred_total_calibrated_high_95')
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    out_dir_reports = out_dir / 'backtest_reports'
    out_dir_reports.mkdir(parents=True, exist_ok=True)

    preds_all = safe_read_csv(out_dir / 'predictions_all.csv')
    closing = safe_read_csv(out_dir / 'closing_lines.csv')
    last_odds = safe_read_csv(out_dir / 'last_odds.csv')
    daily = load_daily_results(out_dir, args.min_date, args.max_date)

    if 'game_id' in preds_all.columns:
        preds_all['game_id'] = preds_all['game_id'].astype(str)
    if 'game_id' in closing.columns:
        closing['game_id'] = closing['game_id'].astype(str)
    if 'game_id' in last_odds.columns:
        last_odds['game_id'] = last_odds['game_id'].astype(str)
    if 'game_id' in daily.columns:
        daily['game_id'] = daily['game_id'].astype(str)
    # Coerce 'date' to canonical YYYY-MM-DD when present
    for df_ in (preds_all, closing, last_odds, daily):
        if 'date' in df_.columns:
            try:
                df_['date'] = pd.to_datetime(df_['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception:
                df_['date'] = df_['date'].astype(str)

    # If predictions_all sparse, fallback to enriched unified predictions
    preds_enriched = load_enriched_predictions(out_dir, args.min_date, args.max_date)
    base_source = 'predictions_all.csv'
    base = preds_all.copy()
    if base.empty or len(base) < 50:
        if not preds_enriched.empty:
            base = preds_enriched.copy()
            base_source = 'predictions_unified_enriched_<range>.csv'
    # Merge core frame with robust key fallback
    join_cols = ['game_id','date']
    if 'date' not in base.columns:
        # Attempt derive from commence/start_dt if present
        if 'start_dt' in base.columns:
            try:
                base['date'] = pd.to_datetime(base['start_dt'], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception:
                base['date'] = None
        else:
            base['date'] = None
    for df_extra, suff in [(closing,'_closing'), (last_odds,'_last'), (daily,'_daily')]:
        inter = df_extra.copy()
        # Decide merge keys dynamically
        keys = join_cols if set(join_cols).issubset(inter.columns) else (['game_id'] if 'game_id' in inter.columns else None)
        if keys is None and 'commence_time' in inter.columns:
            try:
                inter['date'] = pd.to_datetime(inter['commence_time'], errors='coerce').dt.strftime('%Y-%m-%d')
                keys = join_cols if set(join_cols).issubset(inter.columns) else (['game_id'] if 'game_id' in inter.columns else None)
            except Exception:
                keys = ['game_id'] if 'game_id' in inter.columns else None
        try:
            if keys:
                base = base.merge(inter, on=keys, how='left', suffixes=('', suff))
        except Exception:
            pass

    # Secondary actuals join by game_id only to avoid date mismatches
    if not daily.empty and 'game_id' in daily.columns:
        try:
            dr = daily.copy()
            # Prefer the latest record per game_id within the selected window
            dr = dr.sort_values(['game_id','date']).drop_duplicates(['game_id'], keep='last')
            cols_keep = [c for c in ['game_id','home_score','away_score','actual_total','actual_margin'] if c in dr.columns]
            dr = dr[cols_keep]
            base = base.merge(dr, on=['game_id'], how='left', suffixes=('', '_daily_gid'))
        except Exception:
            pass

    # Coalesce key columns from merged sources back into canonical names
    def coalesce(target: str, candidates: list[str]):
        nonlocal base
        # Initialize target as all-NaN if missing
        if target not in base.columns:
            base[target] = np.nan
        for c in candidates:
            if c in base.columns:
                ser = pd.to_numeric(base[c], errors='coerce') if base[c].dtype.kind in 'biufc' else base[c]
                # Fill where target is NaN and candidate has a value
                try:
                    mask = base[target].isna() & ser.notna()
                except Exception:
                    # Fallback when target dtype not supporting isna
                    mask = pd.Series([True]*len(base)) & ser.notna()
                if mask.any():
                    base.loc[mask, target] = ser[mask]
        return

    # Actual scores
    if 'home_score' not in base.columns:
        coalesce('home_score', ['home_score_daily','home_score'])
    if 'away_score' not in base.columns:
        coalesce('away_score', ['away_score_daily','away_score'])
    # Closing market values
    if 'closing_total' not in base.columns:
        coalesce('closing_total', ['closing_total_closing','closing_total_last','closing_total'])
    if 'closing_spread_home' not in base.columns:
        coalesce('closing_spread_home', ['closing_spread_home_closing','closing_spread_home_last','closing_spread_home'])
    # Actuals from daily by gid
    if 'actual_total' not in base.columns:
        coalesce('actual_total', ['actual_total_daily','actual_total_daily_gid'])
    if 'actual_margin' not in base.columns:
        coalesce('actual_margin', ['actual_margin_daily','actual_margin_daily_gid'])

    base = derive_outcomes(base)

    # Regression metrics
    # Prefer calibrated series when present; otherwise fallback to raw model/display columns.
    # Prefer calibrated predictions; otherwise fallback to display/model/raw
    pred_total_series = None
    for c in ['pred_total_calibrated','pred_total_model_unified','pred_total']:
        if c in base.columns:
            pred_total_series = base[c]
            break
    pred_margin_series = None
    for c in ['pred_margin_calibrated','pred_margin_model_unified','pred_margin']:
        if c in base.columns:
            pred_margin_series = base[c]
            break
    if pred_total_series is None:
        pred_total_series = base.get('pred_total')
    if pred_margin_series is None:
        pred_margin_series = base.get('pred_margin')
    reg_total = compute_regression_metrics(pred_total_series, base.get('actual_total'))
    reg_margin = compute_regression_metrics(pred_margin_series, base.get('actual_margin'))

    # Probability metrics (cover/over)
    prob_metrics = {}
    for kind, cols, outcome_col in [
        ('cover', args.prob_cols_cover, 'outcome_home_cover'),
        ('over', args.prob_cols_over, 'outcome_over'),
    ]:
        outcome = pd.to_numeric(base.get(outcome_col), errors='coerce') if outcome_col in base.columns else pd.Series(dtype=float)
        outcome_mask = outcome.notna()
        best_col = None
        best_n = -1
        for c in cols:
            if c in base.columns:
                probs = pd.to_numeric(base[c], errors='coerce')
                mask = outcome_mask & probs.notna()
                if mask.sum() > best_n:
                    best_col = c
                    best_n = mask.sum()
        if best_col:
            probs = pd.to_numeric(base[best_col], errors='coerce')
            mask = outcome_mask & probs.notna()
            if mask.sum() > 0:
                p_arr = probs[mask].to_numpy()
                o_arr = outcome[mask].to_numpy()
                prob_metrics[kind] = {
                    'prob_column_used': best_col,
                    'n': int(mask.sum()),
                    'brier': brier_score(p_arr, o_arr),
                    'log_loss': log_loss(p_arr, o_arr),
                    'base_rate': float(o_arr.mean()),
                    'calibration_bins': calibration_bins(p_arr, o_arr, n_bins=10),
                }
        else:
            prob_metrics[kind] = {'n': 0}

    # Interval coverage
    interval_cov_90 = interval_coverage(base, args.interval_point_col, args.interval_lower_col, args.interval_upper_col)
    interval_cov_95 = interval_coverage(base, args.interval_point_col, args.interval_lower_col_95, args.interval_upper_col_95)

    # Bets & ROI (totals + spread + moneyline where edges/kelly available)
    cohort_records = []
    def collect_bets(edge_col: str, kelly_col: str, outcome_col: str, label: str, odds_prob_col: str | None = None):
        if edge_col not in base.columns or kelly_col not in base.columns or outcome_col not in base.columns:
            return
        sel = bet_selection(base, edge_col, kelly_col, args.edge_threshold)
        if sel.empty:
            return
        for _, r in sel.iterrows():
            stake_frac = float(pd.to_numeric(r.get(kelly_col), errors='coerce') or 0.0)
            if stake_frac < args.kelly_floor:
                continue
            outcome_win = int(pd.to_numeric(r.get(outcome_col), errors='coerce') == 1) if pd.notna(r.get(outcome_col)) else None
            odds_prob = float(pd.to_numeric(r.get(odds_prob_col), errors='coerce')) if odds_prob_col and odds_prob_col in r else math.nan
            pnl = realized_pnl(r, odds_prob, stake_frac, outcome_win) if outcome_win is not None else 0.0
            cohort_records.append({
                'date': r.get('date'),
                'game_id': r.get('game_id'),
                'bet_type': label,
                'edge': float(pd.to_numeric(r.get(edge_col), errors='coerce') or math.nan),
                'kelly_frac': stake_frac,
                'pnl_units': pnl,
                'won': outcome_win,
                'odds_prob_used': odds_prob,
            })

    collect_bets('edge_total','kelly_fraction_total','outcome_over','total', 'home_ml_prob_market')  # odds prob placeholder
    collect_bets('edge_margin_model','kelly_fraction_total_adj','outcome_home_cover','spread', 'home_ml_prob_market')
    # Moneyline evaluation could use ml_home/away columns; placeholder not implemented fully here.

    bets_df = pd.DataFrame(cohort_records)
    roi_metrics = {}
    if not bets_df.empty:
        for bt in bets_df['bet_type'].unique():
            sub = bets_df[bets_df['bet_type'] == bt]
            n = len(sub)
            pnl = sub['pnl_units'].sum()
            avg_edge = sub['edge'].mean()
            win_rate = sub['won'].mean() if 'won' in sub and sub['won'].notna().any() else math.nan
            avg_kelly = sub['kelly_frac'].mean()
            roi_metrics[bt] = {
                'n_bets': int(n),
                'pnl_units': float(pnl),
                'avg_edge': float(avg_edge) if not math.isnan(avg_edge) else None,
                'win_rate': float(win_rate) if not math.isnan(win_rate) else None,
                'avg_kelly_frac': float(avg_kelly) if not math.isnan(avg_kelly) else None,
                'roi_per_bet': float(pnl / n) if n > 0 else None,
            }

    # Drift analysis monthly slices (basic)
    drift = {}
    if 'date' in base.columns and 'pred_total_calibrated' in base.columns and 'actual_total' in base.columns:
        try:
            pred = pd.to_numeric(base['pred_total_calibrated'], errors='coerce')
            act = pd.to_numeric(base['actual_total'], errors='coerce')
            err = pred - act
            base['month'] = base['date'].str.slice(0,7)
            for m, grp in base.groupby('month'):
                gmask = pd.to_numeric(grp['pred_total_calibrated'], errors='coerce').notna() & pd.to_numeric(grp['actual_total'], errors='coerce').notna()
                if gmask.sum() == 0:
                    continue
                e = (pd.to_numeric(grp['pred_total_calibrated'], errors='coerce') - pd.to_numeric(grp['actual_total'], errors='coerce'))[gmask]
                drift[m] = {
                    'n': int(gmask.sum()),
                    'mae': float(np.mean(np.abs(e))),
                    'bias': float(np.mean(e)),
                }
        except Exception:
            pass

    # AUC + CRPS if probability / sigma columns present
    aucs = {}
    crps_total = None
    if 'pred_total_sigma' in base.columns:
        mu = pd.to_numeric(base.get('pred_total_calibrated') or base.get('pred_total'), errors='coerce')
        sig = pd.to_numeric(base['pred_total_sigma'], errors='coerce')
        act = pd.to_numeric(base.get('actual_total'), errors='coerce')
        mask_crps = mu.notna() & sig.notna() & act.notna()
        if mask_crps.sum() > 0:
            crps_total = normal_crps(mu[mask_crps].to_numpy(), sig[mask_crps].to_numpy(), act[mask_crps].to_numpy())

    # AUC for probabilities using selected columns
    for kind, pm in prob_metrics.items():
        col_used = pm.get('prob_column_used')
        if col_used and pm.get('n',0) > 5:
            probs = pd.to_numeric(base[col_used], errors='coerce')
            outcome_col = 'outcome_home_cover' if kind == 'cover' else 'outcome_over'
            outcome = pd.to_numeric(base.get(outcome_col), errors='coerce')
            mask_auc = probs.notna() & outcome.notna()
            if mask_auc.sum() > 10:
                aucs[kind] = auc_score(probs[mask_auc].to_numpy(), outcome[mask_auc].to_numpy())

    # Halftime derivative metrics
    ht_metrics = halftime_metrics(base)

    summary = {
        'date_range': {'min': args.min_date, 'max': args.max_date},
        'regression': {'total': reg_total, 'margin': reg_margin},
        'probabilities': prob_metrics,
        'interval_coverage_90': interval_cov_90,
        'interval_coverage_95': interval_cov_95,
        'roi': roi_metrics,
        'drift_monthly': drift,
        'edge_threshold_used': args.edge_threshold,
        'kelly_floor_used': args.kelly_floor,
        'bets_total': int(len(bets_df)),
        'auc': aucs,
        'crps_total': crps_total,
        'halftime_derivatives': ht_metrics,
    }

    # Write outputs
    (out_dir_reports / 'backtest_full_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    bets_df.to_csv(out_dir_reports / 'backtest_cohort.csv', index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {(out_dir_reports / 'backtest_full_summary.json').as_posix()}")
    print(f"Wrote cohort -> {(out_dir_reports / 'backtest_cohort.csv').as_posix()} (rows={len(bets_df)})")

if __name__ == '__main__':
    main()
