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
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _load_stake_for_date(out_dir: Path, date_str: str, kinds: list[str]) -> list[tuple[str, pd.DataFrame]]:
    res = []
    for kind in kinds:
        p = out_dir / f'stake_sheet_{date_str}_{kind}.csv'
        if p.exists():
            try:
                df = pd.read_csv(p)
                df['__kind__'] = kind
                if 'game_id' in df.columns:
                    df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                res.append((kind, df))
            except Exception:
                pass
    return res


def _settle_row(row: pd.Series, results: pd.DataFrame) -> tuple[float, bool]:
    # Expected columns in stake sheet (current):
    #   game_id, market, selection, line, price (american), stake
    # Also supports older variants with: target_total/target_margin, price_decimal, price_american.
    gid = str(row.get('game_id')) if 'game_id' in row else None
    if gid is not None:
        gid = str(gid).replace('.0', '').strip()
    stake = float(row.get('stake', 0.0) or 0.0)
    if not gid or stake <= 0 or results.empty:
        return 0.0, False
    r = results[results['game_id'].astype(str) == gid]
    if r.empty:
        return 0.0, False
    r = r.iloc[0]
    market = str(row.get('market') or '').lower()
    selection = str(row.get('selection') or '').lower()

    # Price handling (prefer explicit decimal, else derive from american)
    price = row.get('price_decimal')
    if pd.isna(price):
        # common column names in this repo
        amer_raw = row.get('price_american')
        if pd.isna(amer_raw):
            amer_raw = row.get('price')
        if not pd.isna(amer_raw):
            try:
                amer = float(amer_raw)
                price = (1 + amer / 100.0) if amer > 0 else (1 + 100.0 / abs(amer))
            except Exception:
                price = np.nan
    try:
        price = float(price)
    except Exception:
        price = np.nan

    win = False
    push = False
    if market.startswith('totals'):
        total = r.get('total_points')
        if pd.isna(total):
            total = r.get('actual_total')
        target = row.get('target_total')
        if pd.isna(target):
            target = row.get('line')
        if not pd.isna(total) and not pd.isna(target):
            if 'over' in selection:
                win = float(total) > float(target)
                push = float(total) == float(target)
            elif 'under' in selection:
                win = float(total) < float(target)
                push = float(total) == float(target)
    elif market.startswith('spreads') or market.startswith('ats'):
        margin = r.get('margin')
        if pd.isna(margin):
            margin = r.get('actual_margin')
        target = row.get('target_margin')
        if pd.isna(target):
            target = row.get('spread_home')
        if pd.isna(target):
            target = row.get('line')
        if not pd.isna(margin) and not pd.isna(target):
            # selection typically 'home' or 'away' with line applied to home
            if 'home' in selection:
                s = float(margin) + float(target)
                win = s > 0
                push = s == 0
            elif 'away' in selection:
                s = float(-margin) - float(target)
                win = s > 0
                push = s == 0
    # Moneyline could be added when available

    if push:
        return 0.0, True

    if not win:
        return -stake, True
    if not np.isfinite(price) or price <= 1.0:
        # default -110 style if missing
        price = 1.9091
    return stake * (price - 1.0), True


def main():
    ap = argparse.ArgumentParser(description='Backtest ROI from dated stake sheets vs results')
    ap.add_argument('--start-date', type=str)
    ap.add_argument('--end-date', type=str)
    ap.add_argument('--days', type=int)
    ap.add_argument('--outputs-dir', type=str, default='outputs')
    ap.add_argument('--name', type=str, default='latest')
    ap.add_argument('--kinds', type=str, default='base,cal', help='Comma-separated stake sheet variants to include (default: base,cal)')
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)

    kinds = [k.strip() for k in str(args.kinds).split(',') if k.strip()]
    if not kinds:
        kinds = ['base', 'cal']

    # Optional: map game_id -> true slate date (helps when stake sheets are archived under wall-clock date).
    game_date_map: dict[str, str] = {}
    try:
        ga = out_dir / 'games_all.csv'
        if ga.exists():
            gdf = pd.read_csv(ga)
            if {'game_id', 'date'}.issubset(gdf.columns):
                gdf['game_id'] = gdf['game_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                gdf['date'] = gdf['date'].astype(str).str.strip()
                game_date_map = dict(zip(gdf['game_id'].tolist(), gdf['date'].tolist()))
    except Exception:
        game_date_map = {}

    results_cache: dict[str, pd.DataFrame] = {}

    def _get_results(date_key: str) -> pd.DataFrame:
        if date_key not in results_cache:
            results_cache[date_key] = _load_results_for_date(out_dir, date_key)
        return results_cache[date_key]
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
        stake_sets = _load_stake_for_date(out_dir, date_str, kinds)
        if not stake_sets:
            continue
        for kind, sdf in stake_sets:
            pnl = 0.0
            risk = 0.0
            for _, r in sdf.iterrows():
                stake = float(r.get('stake', 0.0) or 0.0)
                pnl_row, settled = _settle_row(r, results)
                if not settled and game_date_map and 'game_id' in r:
                    gid = str(r.get('game_id')).replace('.0', '').strip()
                    alt_date = game_date_map.get(gid)
                    if alt_date and alt_date != date_str:
                        alt_results = _get_results(alt_date)
                        pnl_row, settled = _settle_row(r, alt_results)
                if settled:
                    risk += max(0.0, stake)
                    pnl += pnl_row
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
