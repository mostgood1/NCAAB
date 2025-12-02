import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _daterange(end: datetime, days: int):
    start = end - timedelta(days=max(days - 1, 0))
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _load_results_for_date(out_dir: Path, date_str: str) -> pd.DataFrame:
    p = out_dir / 'daily_results' / f'results_{date_str}.csv'
    try:
        df = pd.read_csv(p)
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def _load_stake_for_date(out_dir: Path, date_str: str) -> list[tuple[str, pd.DataFrame]]:
    res = []
    for kind in ('base', 'cal'):
        p = out_dir / f'stake_sheet_{date_str}_{kind}.csv'
        if p.exists():
            try:
                df = pd.read_csv(p)
                df['__kind__'] = kind
                if 'game_id' in df.columns:
                    df['game_id'] = df['game_id'].astype(str)
                res.append((kind, df))
            except Exception:
                pass
    return res


def _settle_row(row: pd.Series, results: pd.DataFrame) -> float:
    # Expected columns in stake sheet: game_id, market, selection, stake, price_decimal or american
    gid = str(row.get('game_id')) if 'game_id' in row else None
    stake = float(row.get('stake', 0.0) or 0.0)
    if not gid or stake <= 0 or results.empty:
        return 0.0
    r = results[results['game_id'].astype(str) == gid]
    if r.empty:
        return 0.0
    r = r.iloc[0]
    market = str(row.get('market') or '').lower()
    selection = str(row.get('selection') or '').lower()
    price = row.get('price_decimal')
    if pd.isna(price) and not pd.isna(row.get('price_american')):
        try:
            amer = float(row['price_american'])
            price = (1 + amer/100.0) if amer > 0 else (1 + 100.0/abs(amer))
        except Exception:
            price = np.nan
    try:
        price = float(price)
    except Exception:
        price = np.nan

    win = False
    if market.startswith('totals'):
        total = r.get('total_points')
        target = row.get('target_total')
        if not pd.isna(total) and not pd.isna(target):
            if 'over' in selection:
                win = float(total) > float(target)
            elif 'under' in selection:
                win = float(total) < float(target)
    elif market.startswith('spreads') or market.startswith('ats'):
        margin = r.get('margin')
        target = row.get('target_margin') if not pd.isna(row.get('target_margin')) else row.get('spread_home')
        if not pd.isna(margin) and not pd.isna(target):
            # selection typically 'home' or 'away' with line applied to home
            if 'home' in selection:
                win = float(margin) + float(target) > 0
            elif 'away' in selection:
                win = float(-margin) - float(target) > 0
    # Moneyline could be added when available

    if not win:
        return -stake
    if not np.isfinite(price) or price <= 1.0:
        # default -110 style if missing
        price = 1.9091
    return stake * (price - 1.0)


def main():
    ap = argparse.ArgumentParser(description='Backtest ROI from dated stake sheets vs results')
    ap.add_argument('--start-date', type=str)
    ap.add_argument('--end-date', type=str)
    ap.add_argument('--days', type=int)
    ap.add_argument('--outputs-dir', type=str, default='outputs')
    ap.add_argument('--name', type=str, default='latest')
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    today = datetime.now().date()
    if args.days:
        end = today
        start = today - timedelta(days=max(args.days - 1, 0))
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit('Provide --days or both --start-date and --end-date')
        start = _parse_date(args.start_date).date()
        end = _parse_date(args.end_date).date()

    rows = []
    for d in _daterange(datetime.combine(end, datetime.min.time()), (end - start).days + 1):
        date_str = d
        results = _load_results_for_date(out_dir, date_str)
        stake_sets = _load_stake_for_date(out_dir, date_str)
        if not stake_sets:
            continue
        for kind, sdf in stake_sets:
            pnl = 0.0
            risk = 0.0
            for _, r in sdf.iterrows():
                stake = float(r.get('stake', 0.0) or 0.0)
                risk += max(0.0, stake)
                pnl += _settle_row(r, results)
            rows.append({'date': date_str, 'kind': kind, 'pnl': pnl, 'risk': risk, 'roi': (pnl / risk) if risk > 0 else None})

    if not rows:
        print('[roi] No dated stake sheets found; ensure daily_update archives stake sheets.')
        return
    daily = pd.DataFrame(rows)
    summary = daily.groupby('kind').agg(
        days=('date', 'nunique'),
        pnl=('pnl', 'sum'),
        risk=('risk', 'sum'),
    ).reset_index()
    summary['roi'] = summary.apply(lambda r: (r['pnl']/r['risk']) if r['risk'] > 0 else np.nan, axis=1)

    daily.to_csv(out_dir / f'backtest_roi_daily_{args.name}.csv', index=False)
    summary.to_csv(out_dir / f'backtest_roi_{args.name}.csv', index=False)
    # Also copy to stable filenames
    (out_dir / 'backtest_roi_latest.csv').write_text((summary.to_csv(index=False)), encoding='utf-8')
    print('[roi] Wrote ROI backtest daily and summary.')


if __name__ == '__main__':
    main()
