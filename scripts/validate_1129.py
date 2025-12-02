import pandas as pd
from pathlib import Path
root = Path('c:/Users/mostg/OneDrive/Coding/NCAAB/outputs')
res = {}
# predictions_week
pw = pd.read_csv(root/'predictions_week.csv')
res['week_rows'] = len(pw)
res['week_missing_display_date'] = int(pw['display_date'].isna().sum()) if 'display_date' in pw.columns else -1
# rollover candidates
try:
    utc = pd.to_datetime(pw['start_time_iso'], errors='coerce', utc=True)
    local = pd.to_datetime(pw['start_time_local'], errors='coerce')
    slate = pd.to_datetime(pw['date'], errors='coerce')
    mask = (utc.dt.date == (slate.dt.date + pd.Timedelta(days=1))) & (local.dt.date == slate.dt.date)
    res['week_rollover_candidates'] = int(mask.sum())
except Exception:
    res['week_rollover_candidates'] = -1
# games_with_odds for date
gmo_path = root/'games_with_odds_2025-11-29.csv'
if gmo_path.exists():
    gmo = pd.read_csv(gmo_path)
    res['games_rows'] = len(gmo)
    res['games_missing_display_date'] = int(gmo['display_date'].isna().sum()) if 'display_date' in gmo.columns else -1
else:
    res['games_rows'] = -1
    res['games_missing_display_date'] = -1
venue_local_count = 0
if 'start_time_local_venue' in pw.columns:
    venue_local_count = int(pw['start_time_local_venue'].notna().sum())
res['week_with_venue_local'] = venue_local_count
gmo_venue_local = 0
if gmo_path.exists():
    gmo = pd.read_csv(gmo_path)
    if 'start_time_local_venue' in gmo.columns:
        gmo_venue_local = int(gmo['start_time_local_venue'].notna().sum())
    res['games_with_venue_local'] = gmo_venue_local
print('VALIDATION:', res)
print('\nWeek predictions sample:')
print(pw.head(5)[['game_id','date','display_date','start_time_iso','start_time_local','start_time_local_venue','start_tz_abbr','start_tz_abbr_venue']].to_string(index=False))
if gmo_path.exists():
    print('\nGames+odds 11/29 sample:')
    print(gmo.head(5)[['game_id','date','display_date','venue','start_time_iso','start_time_local','start_time_local_venue','start_tz_abbr','start_tz_abbr_venue']].to_string(index=False))
