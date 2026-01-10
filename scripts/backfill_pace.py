#!/usr/bin/env python
from __future__ import annotations
import argparse, datetime as dt, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.pace import attach_pace_features, estimate_team_possessions

OUT = ROOT / 'outputs'


def _dates_in_range(start: str | None, end: str | None) -> list[str]:
    if start and end:
        s = dt.date.fromisoformat(start)
        e = dt.date.fromisoformat(end)
        step = 1 if s <= e else -1
        days = []
        cur = s
        while True:
            days.append(cur.strftime('%Y-%m-%d'))
            if cur == e:
                break
            cur = cur + dt.timedelta(days=step)
        return days
    return []


def _load_enriched_for_date(date: str) -> pd.DataFrame:
    for suffix in ['', '_force_fill']:
        p = OUT / f'predictions_unified_enriched_{date}{suffix}.csv'
        if p.exists():
            try:
                df = pd.read_csv(p)
                df['game_id'] = df.get('game_id', '').astype(str)
                return df
            except Exception:
                pass
    return pd.DataFrame()


def _write_enriched_with_pace(date: str, df: pd.DataFrame) -> Path:
    out_path = OUT / f'predictions_unified_enriched_{date}_with_pace.csv'
    out_path.write_text(df.to_csv(index=False), encoding='utf-8')
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Backfill pace/possessions into enriched predictions.')
    ap.add_argument('--date', type=str, help='Target date YYYY-MM-DD')
    ap.add_argument('--start', type=str, help='Start date YYYY-MM-DD')
    ap.add_argument('--end', type=str, help='End date YYYY-MM-DD')
    ap.add_argument('--minutes-col', type=str, default=None, help='Column name with minutes played, defaults to regulation 40 when missing')
    args = ap.parse_args()

    dates: list[str] = []
    if args.date:
        dates = [args.date]
    else:
        dates = _dates_in_range(args.start, args.end)
    if not dates:
        # Default to yesterday
        y = (dt.date.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        dates = [y]

    written = []
    for d in dates:
        df = _load_enriched_for_date(d)
        if df.empty:
            print(f'[warn] No enriched file for {d}; skipping')
            continue
        try:
            # Attach pace features; choose minutes scaling if available
            df2 = attach_pace_features(df, minutes_col=args.minutes_col)
            path = _write_enriched_with_pace(d, df2)
            written.append(str(path))
            print(f'[ok] Wrote pace-enriched artifact for {d}: {path}')
        except Exception as e:
            print(f'[err] Pace attach failed for {d}: {e}')
    if not written:
        print('[warn] No files written')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
