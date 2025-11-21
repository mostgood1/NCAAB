"""Produce per-game closing medians for totals, spreads, and moneylines.

Usage:
  python scripts/closing_join.py --date 2025-11-19

Outputs:
  outputs/games_with_closing_<date>.csv (game_id, closing_total, closing_spread_home,
                                        closing_ml_home, closing_ml_away)
If --date omitted defaults to yesterday (local date) to target a resolved slate.
"""
from __future__ import annotations
import argparse, datetime as dt, re
from pathlib import Path
import pandas as pd

OUT = Path('outputs')

def _safe_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='Date YYYY-MM-DD (default: yesterday)')
    ap.add_argument('--period-filter', default='full_game', help='Period label to treat as closing (default full_game)')
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime('%Y-%m-%d')

    src = _safe_csv(OUT / 'closing_lines.csv')
    if src.empty:
        print('No closing_lines.csv found; abort.')
        return

    # Normalize columns (exclude game_id which may be synthesized later)
    for c in ['market','period']:
        if c in src.columns:
            src[c] = src[c].astype(str)

    # Synthesize game_id when absent using date + away + home canonical slugs.
    if 'game_id' not in src.columns:
        # Attempt to locate team columns flexibly.
        home_col = next((c for c in ['home_team','home_team_name','home','home_name','home_name_full'] if c in src.columns), None)
        away_col = next((c for c in ['away_team','away_team_name','away','away_name','away_name_full'] if c in src.columns), None)
        if home_col and away_col:
            def _slug(v: str) -> str:
                v = (str(v) if v is not None else '').lower()
                v = re.sub(r'[^a-z0-9]+','_', v).strip('_')
                return v
            # Use provided date_str (argument) as authoritative slate date; if commence_time exists per row, try to refine.
            date_series = pd.Series([date_str]*len(src))
            if 'commence_time' in src.columns:
                try:
                    dtc = pd.to_datetime(src['commence_time'], errors='coerce')
                    mask_valid = ~dtc.isna()
                    date_series.loc[mask_valid] = dtc.dt.strftime('%Y-%m-%d')[mask_valid]
                except Exception:
                    pass
            home_slug = src[home_col].astype(str).map(_slug)
            away_slug = src[away_col].astype(str).map(_slug)
            src['game_id'] = [f"{d}:{a}:{h}" for d,a,h in zip(date_series.astype(str), away_slug, home_slug)]
            print(f"Synthesized game_id for {len(src)} rows using {away_col}/{home_col}.")
        else:
            print('closing_lines.csv missing game_id AND team columns; cannot aggregate.')
            return
    else:
        src['game_id'] = src['game_id'].astype(str)

    # Date filtering attempts
    if 'commence_time' in src.columns:
        try:
            dtc = pd.to_datetime(src['commence_time'], errors='coerce')
            src = src[dtc.dt.strftime('%Y-%m-%d') == date_str]
        except Exception:
            pass
    if 'date' in src.columns:
        src = src[src['date'].astype(str) == date_str]

    if src.empty:
        print(f'No closing rows for {date_str}')
        return

    period_norm = src['period'].astype(str).str.lower().str.replace(' ','_') if 'period' in src.columns else pd.Series(['full_game']*len(src))
    src['_period_norm'] = period_norm
    target_periods = {args.period_filter, 'full_game','fg','game','match'}
    mask_period = src['_period_norm'].isin(target_periods)
    base = src[mask_period].copy()

    # Aggregate medians
    def med(series: pd.Series):
        return pd.to_numeric(series, errors='coerce').median()

    rows = []
    for gid, g in base.groupby('game_id'):
        rec = {'game_id': str(gid)}
        # Carry team identifiers when available for downstream fallback joins
        for hcand in ['home_team','home_team_name','home']:
            if hcand in g.columns:
                rec['home_team'] = str(g[hcand].iloc[0])
                break
        for acand in ['away_team','away_team_name','away']:
            if acand in g.columns:
                rec['away_team'] = str(g[acand].iloc[0])
                break
        try:
            # Totals
            tot_rows = g[g['market'].astype(str).str.lower()=='totals'] if 'market' in g.columns else g
            if not tot_rows.empty and 'total' in tot_rows.columns:
                rec['closing_total'] = med(tot_rows['total'])
            # Spreads
            sp_rows = g[g['market'].astype(str).str.lower()=='spreads'] if 'market' in g.columns else g
            for cand in ['home_spread','spread_home']:
                if cand in sp_rows.columns:
                    rec['closing_spread_home'] = med(sp_rows[cand])
                    break
            # Moneylines
            ml_rows = g[g['market'].astype(str).str.lower()=='h2h'] if 'market' in g.columns else g
            for home_c in ['moneyline_home','ml_home']:
                if home_c in ml_rows.columns:
                    rec['closing_ml_home'] = med(ml_rows[home_c])
                    break
            for away_c in ['moneyline_away','ml_away']:
                if away_c in ml_rows.columns:
                    rec['closing_ml_away'] = med(ml_rows[away_c])
                    break
        except Exception:
            pass
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_path = OUT / f'games_with_closing_{date_str}.csv'
    try:
        out_df.to_csv(out_path, index=False)
        print(f'Wrote {len(out_df)} closing medians to {out_path}')
    except Exception as e:
        print(f'Failed to write closing medians: {e}')

if __name__ == '__main__':
    main()
