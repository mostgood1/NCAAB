from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate thresholded (tuned) OU/ATS picks vs odds over dates")
    p.add_argument('--date', help='Single date YYYY-MM-DD')
    p.add_argument('--start-date', help='Start date YYYY-MM-DD (inclusive)')
    p.add_argument('--end-date', help='End date YYYY-MM-DD (inclusive)')
    p.add_argument('--thresholds', default='outputs/metrics/thresholds_vs_odds.json', help='Path to tuned thresholds JSON')
    p.add_argument('--enforce-guardrails', action='store_true', help='Skip picks when mismatch flags present (full game)')
    return p.parse_args()


def load_day(date: str):
    pr_path = Path(f'outputs/predictions_unified_enriched_{date}.csv')
    res_path = Path(f'outputs/daily_results/results_{date}.csv')
    if not (pr_path.exists() and res_path.exists()):
        return None, None
    try:
        pr = pd.read_csv(pr_path)
        res = pd.read_csv(res_path)
    except Exception:
        return None, None
    for df in (pr, res):
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    return pr, res


def choose_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce')
    return pd.Series([np.nan] * len(df))


def eval_day(date: str, tau_ou: float, tau_ats: float, enforce_guardrails: bool) -> dict:
    pr, res = load_day(date)
    if pr is None:
        return {'date': date, 'status': 'missing'}

    keys = ['game_id'] if 'game_id' in pr.columns and 'game_id' in res.columns else ['date','home_team','away_team']
    merged = pr.merge(res, on=keys, how='inner', suffixes=('_pred','_res'))

    p_total = choose_series(merged, ['pred_total_market_blend','pred_total_calibrated','pred_total'])
    p_margin = choose_series(merged, ['pred_margin_market_blend','pred_margin_calibrated','pred_margin'])

    m_total = choose_series(merged, ['market_total_res','market_total'])
    spread_home = choose_series(merged, ['spread_home_res','spread_home'])

    a_total = choose_series(merged, ['actual_total_res','actual_total_pred','actual_total'])
    a_margin = choose_series(merged, ['actual_margin_res','actual_margin_pred','actual_margin'])

    period = merged.get('period') if 'period' in merged.columns else pd.Series(['full_game'] * len(merged))
    f_total_mm = merged.get('flag_market_total_mismatch') if 'flag_market_total_mismatch' in merged.columns else pd.Series([False]*len(merged))
    f_margin_mm = merged.get('flag_market_margin_mismatch') if 'flag_market_margin_mismatch' in merged.columns else pd.Series([False]*len(merged))

    # OU selection by threshold and guardrails
    ou_mask_all = p_total.notna() & m_total.notna() & a_total.notna()
    ou_edge = (p_total - m_total).abs()
    ou_sel = ou_mask_all & (ou_edge >= tau_ou)
    if enforce_guardrails:
        ou_sel = ou_sel & (~(period.eq('full_game') & (f_total_mm.fillna(False) | f_margin_mm.fillna(False))))

    ou_pred_over = (p_total[ou_sel] - m_total[ou_sel]) > 0
    ou_actual_over = (a_total[ou_sel] - m_total[ou_sel]) > 0
    ou_acc = float((ou_pred_over == ou_actual_over).mean()) if ou_sel.any() else None

    # ATS selection by threshold and guardrails
    ats_mask_all = p_margin.notna() & spread_home.notna() & a_margin.notna()
    ats_edge = (p_margin - (-spread_home)).abs()
    ats_sel = ats_mask_all & (ats_edge >= tau_ats)
    if enforce_guardrails:
        ats_sel = ats_sel & (~(period.eq('full_game') & f_margin_mm.fillna(False)))

    pred_home_cover = p_margin[ats_sel] >= (-spread_home[ats_sel])
    actual_home_cover = a_margin[ats_sel] >= (-spread_home[ats_sel])
    ats_acc = float((pred_home_cover == actual_home_cover).mean()) if ats_sel.any() else None

    return {
        'date': date,
        'n_ou_all': int(ou_mask_all.sum()),
        'n_ats_all': int(ats_mask_all.sum()),
        'n_ou_sel': int(ou_sel.sum()),
        'n_ats_sel': int(ats_sel.sum()),
        'ou_accuracy_sel': ou_acc,
        'ats_accuracy_sel': ats_acc,
    }


def main():
    args = parse_args()
    thr_path = Path(args.thresholds)
    if not thr_path.exists():
        raise SystemExit(f"Thresholds file not found: {thr_path}")
    payload = json.loads(thr_path.read_text())
    tau_ou = float(payload.get('ou', {}).get('tau'))
    tau_ats = float(payload.get('ats', {}).get('tau'))

    dates: list[str]
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        s = pd.to_datetime(args.start_date)
        e = pd.to_datetime(args.end_date)
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(s, e, freq='D')]
    else:
        root = Path('outputs/daily_results')
        dates = [p.stem.split('_', 1)[1] for p in sorted(root.glob('results_*.csv'))]

    rows = []
    for d in dates:
        rows.append(eval_day(d, tau_ou, tau_ats, args.enforce_guardrails))

    df = pd.DataFrame(rows)
    overall = {
        'dates': int(len(df)),
        'ou_total_all': int(df['n_ou_all'].fillna(0).sum()),
        'ats_total_all': int(df['n_ats_all'].fillna(0).sum()),
        'ou_total_sel': int(df['n_ou_sel'].fillna(0).sum()),
        'ats_total_sel': int(df['n_ats_sel'].fillna(0).sum()),
        'ou_accuracy_sel_mean': float(df['ou_accuracy_sel'].dropna().mean()) if df['ou_accuracy_sel'].notna().any() else None,
        'ats_accuracy_sel_mean': float(df['ats_accuracy_sel'].dropna().mean()) if df['ats_accuracy_sel'].notna().any() else None,
        'ou_coverage': (float(df['n_ou_sel'].fillna(0).sum())/float(df['n_ou_all'].fillna(0).sum())) if df['n_ou_all'].fillna(0).sum() else None,
        'ats_coverage': (float(df['n_ats_sel'].fillna(0).sum())/float(df['n_ats_all'].fillna(0).sum())) if df['n_ats_all'].fillna(0).sum() else None,
        'tau_ou': tau_ou,
        'tau_ats': tau_ats,
        'enforce_guardrails': bool(args.enforce_guardrails),
    }
    print(json.dumps({'overall': overall, 'daily': rows}, indent=2))


if __name__ == '__main__':
    main()
