from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model accuracy vs odds (OU and ATS) for a date or range")
    p.add_argument('--source', choices=['model','sim','sim_blend'], default='model', help='Prediction source to evaluate')
    p.add_argument('--date', help='Single date YYYY-MM-DD')
    p.add_argument('--start-date', help='Start date YYYY-MM-DD (inclusive)')
    p.add_argument('--end-date', help='End date YYYY-MM-DD (inclusive)')
    p.add_argument('--use-closing', action='store_true', help='Prefer closing_total/spread columns when available')
    p.add_argument('--tau-ou', type=float, default=0.0, help='Select OU bets only when |pred_total - line_total| >= tau')
    p.add_argument('--tau-ats', type=float, default=0.0, help='Select ATS bets only when |pred_margin - market_margin| >= tau')
    p.add_argument('--pmin-ou', type=float, default=0.0, help='Select OU bets only when max(p_over, p_under) >= pmin')
    p.add_argument('--pmin-ats', type=float, default=0.0, help='Select ATS bets only when max(p_home_cover, p_away_cover) >= pmin')
    p.add_argument('--edge-min', type=float, default=0.0, help='Select bets only when (p_side - implied_prob(price)) >= edge_min')
    p.add_argument('--assume-price', type=int, default=-110, help='Fallback American odds when book price missing')
    return p.parse_args()


def _normalize_game_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'game_id' in df.columns:
        df = df.copy()
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    return df


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_day(date: str, source: str):
    if source == 'model':
        pr_path = Path(f'outputs/predictions_unified_enriched_{date}.csv')
    elif source == 'sim':
        pr_path = Path(f'outputs/sim_quantiles_{date}.csv')
    else:
        pr_path = Path(f'outputs/sim_blend_{date}.csv')
    res_path = Path(f'outputs/daily_results/results_{date}.csv')
    if not (pr_path.exists() and res_path.exists()):
        return None, None
    pr = _safe_read_csv(pr_path)
    res = _safe_read_csv(res_path)
    if pr.empty or res.empty:
        return None, None
    return _normalize_game_id(pr), _normalize_game_id(res)


def _implied_prob_from_american(odds: int | float | None) -> float | None:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return float(100.0 / (o + 100.0))
    return float((-o) / ((-o) + 100.0))


def _profit_unit_from_american(odds: int | float) -> float:
    o = float(odds)
    if o > 0:
        return float(o / 100.0)
    return float(100.0 / (-o))


def _phi(z: float) -> float:
    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def eval_day(date: str, use_closing: bool) -> dict:
    raise NotImplementedError("Use eval_day2")


def eval_day2(
    date: str,
    source: str,
    use_closing: bool,
    tau_ou: float,
    tau_ats: float,
    pmin_ou: float,
    pmin_ats: float,
    edge_min: float,
    assume_price: int,
) -> dict:
    pr, res = load_day(date, source)
    if pr is None:
        return {'date': date, 'status': 'missing'}

    # Join on game_id if available, else on date+teams
    keys = ['game_id'] if 'game_id' in pr.columns and 'game_id' in res.columns else ['date','home_team','away_team']
    merged = pr.merge(res, on=keys, how='inner', suffixes=('_pred','_res'))

    def get_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce')
        return pd.Series([np.nan] * len(df))

    def get_price(df: pd.DataFrame, candidates: list[str], fallback: int) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                return s.fillna(float(fallback))
        return pd.Series([float(fallback)] * len(df))

    # Choose prediction columns
    if source == 'model':
        p_total = get_series(merged, ['pred_total_market_blend','pred_total_calibrated','pred_total'])
        p_margin = get_series(merged, ['pred_margin_market_blend','pred_margin_calibrated','pred_margin'])
        # Model source doesn't always carry prices.
        over_px = pd.Series([float(assume_price)] * len(merged))
        under_px = pd.Series([float(assume_price)] * len(merged))
        home_px = pd.Series([float(assume_price)] * len(merged))
        away_px = pd.Series([float(assume_price)] * len(merged))
        p_over = pd.Series([np.nan] * len(merged))
        p_home_cover = pd.Series([np.nan] * len(merged))
    elif source == 'sim':
        p_total = get_series(merged, ['mu_total','q50_total'])
        p_margin = get_series(merged, ['mu_margin','q50_margin'])
        over_px = get_price(merged, ['over_price','over_price_res'], assume_price)
        under_px = get_price(merged, ['under_price','under_price_res'], assume_price)
        home_px = get_price(merged, ['home_spread_price','home_spread_price_res'], assume_price)
        away_px = get_price(merged, ['away_spread_price','away_spread_price_res'], assume_price)
        # Prefer the sim-computed probability vs market line if present.
        p_over = get_series(merged, ['p_over_market','p_over'])
        p_home_cover = pd.Series([np.nan] * len(merged))
    else:
        # sim_blend is OU-only for now
        p_total = get_series(merged, ['mu_total_sim','q50_total_sim'])
        p_margin = pd.Series([np.nan] * len(merged))
        over_px = pd.Series([float(assume_price)] * len(merged))
        under_px = pd.Series([float(assume_price)] * len(merged))
        home_px = pd.Series([float(assume_price)] * len(merged))
        away_px = pd.Series([float(assume_price)] * len(merged))
        p_over = get_series(merged, ['p_over_blend','p_over_sim','p_over_model'])
        p_home_cover = pd.Series([np.nan] * len(merged))

    # Market lines: for sim-based recommendation evaluation, prefer the simulator's own
    # market_total/spread_home (the probabilities are defined relative to those).
    if source in ('sim', 'sim_blend') and 'market_total' in merged.columns:
        m_total = get_series(merged, ['market_total'])
    elif use_closing:
        m_total = get_series(merged, ['closing_total','closing_total_res','market_total_res','market_total'])
    else:
        m_total = get_series(merged, ['market_total_res','market_total','closing_total','closing_total_res'])

    if source == 'sim' and 'spread_home' in merged.columns:
        spread_home = get_series(merged, ['spread_home'])
    elif use_closing:
        spread_home = get_series(merged, ['closing_spread_home','closing_spread_home_res','spread_home_res','spread_home'])
    else:
        spread_home = get_series(merged, ['spread_home_res','spread_home','closing_spread_home','closing_spread_home_res'])
    # Actuals may exist on both sides; prefer _res (from results)
    a_total = get_series(merged, ['actual_total_res','actual_total_pred','actual_total'])
    a_margin = get_series(merged, ['actual_margin_res','actual_margin_pred','actual_margin'])

    # OU accuracy / recommendation accuracy
    ou_mask = p_total.notna() & m_total.notna() & a_total.notna()
    ou_edge_pts = (p_total - m_total)
    ou_push = a_total.eq(m_total)
    ou_actual_over = (a_total > m_total)
    ou_pred_over = ou_edge_pts.gt(0)
    # Derive p_over when missing (normal approx)
    if p_over.isna().all() and (source in ('sim','sim_blend')):
        sig_t = get_series(merged, ['sigma_total'])
        z = (m_total - p_total) / sig_t.replace(0.0, np.nan)
        p_over = z.map(lambda v: 1.0 - _phi(float(v)) if pd.notna(v) else np.nan)
    # Accuracy excluding pushes
    ou_valid = ou_mask & (~ou_push)
    ou_acc = float((ou_pred_over[ou_valid] == ou_actual_over[ou_valid]).mean()) if ou_valid.any() else None

    # Selection gating for OU
    p_over2 = pd.to_numeric(p_over, errors='coerce')
    p_over2 = p_over2.where(p_over2.between(0.0, 1.0))
    p_side_ou = np.where(ou_pred_over, p_over2, (1.0 - p_over2))
    ou_price = np.where(ou_pred_over, over_px, under_px)
    ou_imp = pd.Series([_implied_prob_from_american(v) for v in ou_price], index=merged.index, dtype=float)
    ou_edge_prob = pd.Series(p_side_ou, index=merged.index, dtype=float) - ou_imp
    ou_sel = ou_mask
    if tau_ou > 0:
        ou_sel = ou_sel & (ou_edge_pts.abs() >= float(tau_ou))
    if pmin_ou > 0:
        ou_sel = ou_sel & pd.Series(p_side_ou, index=merged.index, dtype=float).ge(float(pmin_ou))
    if edge_min > 0:
        ou_sel = ou_sel & ou_edge_prob.ge(float(edge_min))
    ou_sel = ou_sel & (~ou_push)

    ou_win = (ou_pred_over == ou_actual_over)
    ou_profit = pd.Series([0.0] * len(merged), index=merged.index)
    for i in merged.index[ou_sel]:
        px = float(ou_price[i])
        if bool(ou_win[i]):
            ou_profit[i] = _profit_unit_from_american(px)
        else:
            ou_profit[i] = -1.0

    # ATS accuracy / recommendation accuracy
    ats_mask = p_margin.notna() & spread_home.notna() & a_margin.notna()
    mkt_margin = -spread_home
    ats_edge_pts = (p_margin - mkt_margin)
    ats_push = a_margin.eq(mkt_margin)
    actual_home_cover = a_margin.gt(mkt_margin)
    pred_home_cover = ats_edge_pts.ge(0)

    # Derive p_home_cover if missing and sigma exists
    if p_home_cover.isna().all() and (source == 'sim'):
        sig_m = get_series(merged, ['sigma_margin'])
        z = (mkt_margin - p_margin) / sig_m.replace(0.0, np.nan)
        p_home_cover = z.map(lambda v: 1.0 - _phi(float(v)) if pd.notna(v) else np.nan)

    ats_valid = ats_mask & (~ats_push)
    ats_acc = float((pred_home_cover[ats_valid] == actual_home_cover[ats_valid]).mean()) if ats_valid.any() else None

    # Selection gating for ATS
    p_cov = pd.to_numeric(p_home_cover, errors='coerce')
    p_cov = p_cov.where(p_cov.between(0.0, 1.0))
    p_side_ats = np.where(pred_home_cover, p_cov, (1.0 - p_cov))
    ats_price = np.where(pred_home_cover, home_px, away_px)
    ats_imp = pd.Series([_implied_prob_from_american(v) for v in ats_price], index=merged.index, dtype=float)
    ats_edge_prob = pd.Series(p_side_ats, index=merged.index, dtype=float) - ats_imp
    ats_sel = ats_mask
    if tau_ats > 0:
        ats_sel = ats_sel & (ats_edge_pts.abs() >= float(tau_ats))
    if pmin_ats > 0:
        ats_sel = ats_sel & pd.Series(p_side_ats, index=merged.index, dtype=float).ge(float(pmin_ats))
    if edge_min > 0:
        ats_sel = ats_sel & ats_edge_prob.ge(float(edge_min))
    ats_sel = ats_sel & (~ats_push)

    ats_win = (pred_home_cover == actual_home_cover)
    ats_profit = pd.Series([0.0] * len(merged), index=merged.index)
    for i in merged.index[ats_sel]:
        px = float(ats_price[i])
        if bool(ats_win[i]):
            ats_profit[i] = _profit_unit_from_american(px)
        else:
            ats_profit[i] = -1.0

    return {
        'date': date,
        'status': 'ok',
        'source': source,
        'n_ou_all': int(ou_mask.sum()),
        'n_ou_sel': int(ou_sel.sum()),
        'ou_accuracy_all_ex_push': ou_acc,
        'ou_roi_units': float(ou_profit[ou_sel].sum()) if ou_sel.any() else 0.0,
        'ou_roi_per_bet': float(ou_profit[ou_sel].mean()) if ou_sel.any() else None,
        'ou_avg_edge_pts': float(ou_edge_pts[ou_sel].mean()) if ou_sel.any() else None,
        'ou_avg_edge_prob': float(ou_edge_prob[ou_sel].mean()) if ou_sel.any() else None,
        'n_ats_all': int(ats_mask.sum()),
        'n_ats_sel': int(ats_sel.sum()),
        'ats_accuracy_all_ex_push': ats_acc,
        'ats_roi_units': float(ats_profit[ats_sel].sum()) if ats_sel.any() else 0.0,
        'ats_roi_per_bet': float(ats_profit[ats_sel].mean()) if ats_sel.any() else None,
        'ats_avg_edge_pts': float(ats_edge_pts[ats_sel].mean()) if ats_sel.any() else None,
        'ats_avg_edge_prob': float(ats_edge_prob[ats_sel].mean()) if ats_sel.any() else None,
    }


def main():
    args = parse_args()
    dates = []
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        s = pd.to_datetime(args.start_date)
        e = pd.to_datetime(args.end_date)
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(s, e, freq='D')]
    else:
        # Use all available daily_results files
        root = Path('outputs/daily_results')
        for p in sorted(root.glob('results_*.csv')):
            m = p.stem.split('_', 1)[1]
            dates.append(m)

    results = []
    for d in dates:
        r = eval_day2(
            d,
            args.source,
            args.use_closing,
            args.tau_ou,
            args.tau_ats,
            args.pmin_ou,
            args.pmin_ats,
            args.edge_min,
            args.assume_price,
        )
        results.append(r)

    df = pd.DataFrame(results)
    def _wavg(num_col: str, den_col: str) -> float | None:
        if num_col not in df.columns or den_col not in df.columns:
            return None
        den = pd.to_numeric(df[den_col], errors='coerce').fillna(0.0)
        num = pd.to_numeric(df[num_col], errors='coerce')
        m = den.gt(0) & num.notna()
        if not m.any():
            return None
        return float((num[m] * den[m]).sum() / den[m].sum())

    overall = {
        'dates': int(len(df)),
        'source': args.source,
        'use_closing': bool(args.use_closing),
        'tau_ou': float(args.tau_ou),
        'tau_ats': float(args.tau_ats),
        'pmin_ou': float(args.pmin_ou),
        'pmin_ats': float(args.pmin_ats),
        'edge_min': float(args.edge_min),
        'ou_total_all': int(pd.to_numeric(df.get('n_ou_all', 0), errors='coerce').fillna(0).sum()) if 'n_ou_all' in df.columns else 0,
        'ou_total_sel': int(pd.to_numeric(df.get('n_ou_sel', 0), errors='coerce').fillna(0).sum()) if 'n_ou_sel' in df.columns else 0,
        'ou_accuracy_all_ex_push_wavg': _wavg('ou_accuracy_all_ex_push','n_ou_all'),
        'ou_roi_units_total': float(pd.to_numeric(df.get('ou_roi_units', 0.0), errors='coerce').fillna(0.0).sum()) if 'ou_roi_units' in df.columns else 0.0,
        'ou_roi_per_bet_wavg': _wavg('ou_roi_per_bet','n_ou_sel'),
        'ats_total_all': int(pd.to_numeric(df.get('n_ats_all', 0), errors='coerce').fillna(0).sum()) if 'n_ats_all' in df.columns else 0,
        'ats_total_sel': int(pd.to_numeric(df.get('n_ats_sel', 0), errors='coerce').fillna(0).sum()) if 'n_ats_sel' in df.columns else 0,
        'ats_accuracy_all_ex_push_wavg': _wavg('ats_accuracy_all_ex_push','n_ats_all'),
        'ats_roi_units_total': float(pd.to_numeric(df.get('ats_roi_units', 0.0), errors='coerce').fillna(0.0).sum()) if 'ats_roi_units' in df.columns else 0.0,
        'ats_roi_per_bet_wavg': _wavg('ats_roi_per_bet','n_ats_sel'),
    }
    print(json.dumps({'overall': overall, 'daily': results}, indent=2))


if __name__ == '__main__':
    main()
