import pandas as pd
from pathlib import Path
from datetime import datetime

OUT = Path('outputs')
DATE_STR = datetime.now().strftime('%Y-%m-%d')
MISSING = OUT / f'missing_real_coverage_{DATE_STR}.csv'
ENRICHED = OUT / f'predictions_unified_enriched_{DATE_STR}.csv'
GAMES_CURR = OUT / 'games_curr.csv'

MIN_COLUMNS = ['game_id','home_team','away_team','date','start_time']


def safe_read(p):
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def main():
    miss = safe_read(MISSING)
    if miss.empty:
        print('No missing coverage rows; nothing to append.')
        return
    enr = safe_read(ENRICHED)
    games = safe_read(GAMES_CURR)
    # Ensure id types
    if 'game_id' in miss.columns:
        miss['game_id'] = miss['game_id'].astype(str)
    if 'game_id' in enr.columns:
        enr['game_id'] = enr['game_id'].astype(str)
    existing_ids = set(enr['game_id']) if 'game_id' in enr.columns else set()
    append_rows = []
    for r in miss.itertuples():
        gid = str(getattr(r,'game_id',''))
        if gid in existing_ids:
            continue  # already present; prediction/odds missing but row exists
        home = getattr(r,'home_team','')
        away = getattr(r,'away_team','')
        # Try to find schedule row to borrow start_time/venue
        start_time = None
        if not games.empty and 'game_id' in games.columns:
            gmatch = games[games['game_id'].astype(str) == gid]
            if not gmatch.empty and 'start_time' in gmatch.columns:
                start_time = gmatch['start_time'].iloc[0]
        # Build stub
        stub = {
            'game_id': gid,
            'home_team': home,
            'away_team': away,
            'date': DATE_STR,
            'start_time': start_time,
            'pred_total': None,
            'pred_margin': None,
            'market_total': None,
            'spread_home': None,
            'pred_total_basis': None,
            'pred_margin_basis': None,
            'market_total_basis': None,
            'spread_home_basis': None,
            'coverage_stub': True
        }
        append_rows.append(stub)
    if not append_rows:
        print('No new stub rows to append.')
        return
    stub_df = pd.DataFrame(append_rows)
    merged = pd.concat([enr, stub_df], ignore_index=True, sort=False)
    merged_path = OUT / f'predictions_unified_enriched_{DATE_STR}_appended.csv'
    merged.to_csv(merged_path, index=False)
    print('Appended stub rows:', len(stub_df), '->', merged_path)

if __name__ == '__main__':
    main()
