import pandas as pd
from pathlib import Path
from datetime import datetime
import json

OUT = Path('outputs')
DATE_STR = datetime.now().strftime('%Y-%m-%d')
GAMES_CURR = OUT / 'games_curr.csv'
MISSING_CSV = OUT / f'missing_real_coverage_{DATE_STR}.csv'
ENRICHED = OUT / f'predictions_unified_enriched_{DATE_STR}.csv'
REPORT = OUT / f'schedule_alignment_probe_{DATE_STR}.json'


def safe_read(p):
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def norm(name: str) -> str:
    if not isinstance(name,str):
        return ''
    return ' '.join(name.lower().replace(' university','').replace(' univ','').replace(' state',' st').replace('&',' and ').split())

def main():
    games = safe_read(GAMES_CURR)
    missing = safe_read(MISSING_CSV)
    enriched = safe_read(ENRICHED)
    report = {
        'date': DATE_STR,
        'games_rows': len(games),
        'missing_rows': len(missing),
        'enriched_rows': len(enriched)
    }
    if missing.empty:
        report['status'] = 'no_missing_rows'
    else:
        # Check which missing game_ids are absent from enriched
        if 'game_id' in missing.columns and 'game_id' in enriched.columns:
            missing['game_id'] = missing['game_id'].astype(str)
            enriched_ids = set(enriched['game_id'].astype(str)) if not enriched.empty else set()
            absent_ids = [gid for gid in missing['game_id'] if gid not in enriched_ids]
            report['absent_game_ids'] = absent_ids
        # Attempt team-pair alignment
        pair_absent = []
        if not games.empty and {'home_team','away_team'}.issubset(games.columns) and {'home_team','away_team'}.issubset(missing.columns):
            games_pairs = {(norm(r.home_team), norm(r.away_team)) for r in games.itertuples()}
            for r in missing.itertuples():
                pair = (norm(r.home_team), norm(r.away_team))
                if pair not in games_pairs:
                    pair_absent.append({'home_team': r.home_team, 'away_team': r.away_team})
            report['absent_team_pairs'] = pair_absent
        # Commence time availability
        if 'commence_time' in games.columns:
            comm_count = games['commence_time'].notna().sum()
            report['games_commence_time_populated'] = int(comm_count)
    with open(REPORT,'w',encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print('Schedule alignment probe report written:', REPORT)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
