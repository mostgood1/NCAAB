import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
BT_DIR = OUT / 'backtest_reports'

BT_SRC = BT_DIR / 'backtest_joined.csv'
BT_OUT = BT_DIR / 'backtest_joined_enriched.csv'
PREDH = OUT / 'predictions_history_enriched.csv'
ODDS = OUT / 'games_with_last.csv'
ODDS_NORM = OUT / 'games_with_last_normalized.csv'
TEAM_MAP = ROOT / 'data' / 'team_map.csv'


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _parse_date(s):
    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.NaT


def _compute_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Expect columns like home_last_date, away_last_date; else use rolling by team/date if available
    for side in ['home','away']:
        last_col = f'{side}_last_date'
        if last_col in df.columns:
            last_dt = pd.to_datetime(df[last_col], errors='coerce')
            cur_dt = pd.to_datetime(df['date'], errors='coerce')
            df[f'days_rest_{side}'] = (cur_dt - last_dt).dt.days
    return df


def _compute_rest_from_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not all(c in df.columns for c in ['date','home_team','away_team']):
        return df
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    # Build a long frame of team/date for both home and away appearances
    home = df[['home_team','date_dt']].rename(columns={'home_team':'team'})
    away = df[['away_team','date_dt']].rename(columns={'away_team':'team'})
    long = pd.concat([home, away], ignore_index=True)
    long = long.dropna(subset=['team','date_dt'])
    long.sort_values(['team','date_dt'], inplace=True)
    long['prev_date'] = long.groupby('team')['date_dt'].shift(1)
    # Map previous dates back to the row level for both home/away
    prev_home = long.rename(columns={'team':'home_team','prev_date':'home_prev_date'})[['home_team','date_dt','home_prev_date']]
    prev_away = long.rename(columns={'team':'away_team','prev_date':'away_prev_date'})[['away_team','date_dt','away_prev_date']]
    df = df.merge(prev_home, left_on=['home_team','date_dt'], right_on=['home_team','date_dt'], how='left')
    df = df.merge(prev_away, left_on=['away_team','date_dt'], right_on=['away_team','date_dt'], how='left')
    df['days_rest_home'] = df['days_rest_home'] if 'days_rest_home' in df.columns else (df['date_dt'] - df['home_prev_date']).dt.days
    df['days_rest_away'] = df['days_rest_away'] if 'days_rest_away' in df.columns else (df['date_dt'] - df['away_prev_date']).dt.days
    df.drop(columns=['home_prev_date','away_prev_date'], inplace=True)
    return df


def _load_team_locations() -> pd.DataFrame:
    tm = _read_csv(TEAM_MAP)
    # Expect columns: team, lat, lon (if available); else empty
    cols = {c.lower(): c for c in tm.columns}
    lat_col = cols.get('lat')
    lon_col = cols.get('lon') or cols.get('lng')
    name_col = cols.get('team') or cols.get('name') or cols.get('provider_name')
    if not (lat_col and lon_col and name_col):
        return pd.DataFrame()
    tm2 = tm[[name_col, lat_col, lon_col]].rename(columns={name_col: 'team', lat_col: 'lat', lon_col: 'lon'})
    return tm2


def _haversine_km(lat1, lon1, lat2, lon2):
    # Great-circle distance
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def _compute_travel(df: pd.DataFrame, team_locs: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if team_locs.empty:
        return df
    # Expect columns home_team, away_team; compute distance between home venue and away team base as proxy
    if not all(c in df.columns for c in ['home_team','away_team']):
        return df
    tl = team_locs.copy()
    df = df.merge(tl.rename(columns={'team':'home_team','lat':'home_lat','lon':'home_lon'}), on='home_team', how='left')
    df = df.merge(tl.rename(columns={'team':'away_team','lat':'away_lat','lon':'away_lon'}), on='away_team', how='left')
    df['travel_dist_km'] = _haversine_km(df['home_lat'], df['home_lon'], df['away_lat'], df['away_lon'])
    return df


def _merge_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Prefer normalized odds if available
    odds = _read_csv(ODDS_NORM if ODDS_NORM.exists() else ODDS)
    if odds.empty:
        # ensure market columns exist even if odds missing
        for col in ['market_total','market_spread','market_moneyline_home_prob']:
            if col not in df.columns:
                df[col] = np.nan
        return df
    # Expect columns: date, game_id, total, spread, price_home or home_ml_prob
    for d in (df, odds):
        if 'game_id' in d.columns:
            d['game_id'] = d['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    join_cols = [c for c in ['date','game_id'] if c in df.columns and c in odds.columns]
    if not join_cols:
        return df
    merged = df.merge(odds, on=join_cols, how='left', suffixes=('', '_odds'))
    # Map plausible columns
    def _first_available(row, cols):
        for c in cols:
            if c in merged.columns and pd.notna(row.get(c)):
                return row.get(c)
        return np.nan
    merged['market_total'] = merged.apply(lambda r: _first_available(r, ['total','odds_total','market_total']), axis=1)
    merged['market_spread'] = merged.apply(lambda r: _first_available(r, ['spread','odds_spread','market_spread']), axis=1)
    # Approximate home moneyline prob if American odds present
    def _american_to_prob(price):
        try:
            price = float(price)
        except Exception:
            return np.nan
        if price > 0:
            return 100.0 / (price + 100.0)
        elif price < 0:
            return abs(price) / (abs(price) + 100.0)
        return np.nan
    merged['market_moneyline_home_prob'] = merged.apply(lambda r: _first_available(r, ['home_ml_prob','market_moneyline_home_prob']), axis=1)
    if 'home_ml_price' in merged.columns and merged['market_moneyline_home_prob'].isna().any():
        merged.loc[merged['market_moneyline_home_prob'].isna(), 'market_moneyline_home_prob'] = merged.loc[merged['market_moneyline_home_prob'].isna(), 'home_ml_price'].apply(_american_to_prob)
    return merged


def main():
    bt = _read_csv(BT_SRC)
    if bt.empty:
        raise SystemExit('Missing backtest_joined.csv')
    # Compute rest
    bt = _compute_rest(bt)
    # If explicit last-date columns absent, derive rest from schedule
    if ('days_rest_home' not in bt.columns) or ('days_rest_away' not in bt.columns) or bt['days_rest_home'].isna().all() or bt['days_rest_away'].isna().all():
        bt = _compute_rest_from_schedule(bt)
    # Travel distance (best effort)
    team_locs = _load_team_locations()
    bt = _compute_travel(bt, team_locs)
    # Merge market anchors
    bt = _merge_market(bt)
    # Ensure engineered columns exist and fill empties with safe defaults
    for col, default in [
        ('days_rest_home', np.nan),
        ('days_rest_away', np.nan),
        ('travel_dist_km', np.nan),
        ('market_total', np.nan),
        ('market_spread', np.nan),
        ('market_moneyline_home_prob', np.nan),
    ]:
        if col not in bt.columns:
            bt[col] = default
    # Optional: basic imputations for modeling stability
    # Use medians for numeric engineered columns where available
    for col in ['days_rest_home','days_rest_away','travel_dist_km','market_total','market_spread','market_moneyline_home_prob']:
        if col in bt.columns:
            med = bt[col].median(skipna=True)
            if np.isfinite(med):
                bt[col] = bt[col].fillna(med)
    # Write enriched
    BT_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(BT_OUT, index=False)
    print('Wrote enriched backtest to', str(BT_OUT))


if __name__ == '__main__':
    main()
