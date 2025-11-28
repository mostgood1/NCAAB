import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import json

OUT = Path('outputs')
MANDATORY = [
    'home_off_rating','away_off_rating','home_def_rating','away_def_rating',
    'home_tempo_rating','away_tempo_rating'
]
# Optional columns we try to propagate if available
OPTIONAL = [
    'home_recent_form','away_recent_form','home_pace_adj','away_pace_adj'
]

DEF_BASIS_COLS = {
    'home_off_rating': 'home_off_rating_basis',
    'away_off_rating': 'away_off_rating_basis',
    'home_def_rating': 'home_def_rating_basis',
    'away_def_rating': 'away_def_rating_basis',
    'home_tempo_rating': 'home_tempo_rating_basis',
    'away_tempo_rating': 'away_tempo_rating_basis'
}

def find_missing_coverage(date_str: str) -> pd.DataFrame:
    p = OUT / f'missing_real_coverage_{date_str}.csv'
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty and 'game_id' in df.columns:
                df['game_id'] = df['game_id'].astype(str)
                return df
        except Exception:
            pass
    return pd.DataFrame()

def load_feature_sources() -> pd.DataFrame:
    sources = ['features_curr.csv','features_all.csv','features_week.csv','features_last2.csv']
    frames = []
    for name in sources:
        p = OUT / name
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty or 'game_id' not in df.columns:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Concatenate for aggregation (avoid duplicate column collisions by select common set)
    base_cols = set(frames[0].columns)
    for f in frames[1:]:
        base_cols &= set(f.columns)
    keep = list(base_cols)
    return pd.concat([f[keep] for f in frames], ignore_index=True)

def build_team_avgs(feat_df: pd.DataFrame) -> dict:
    team_stats = {}
    if feat_df.empty:
        return team_stats
    # Expect home_team / away_team columns; if missing we cannot compute team-specific averages
    has_home = 'home_team' in feat_df.columns
    has_away = 'away_team' in feat_df.columns
    if not (has_home and has_away):
        return team_stats
    # Numeric coercion
    num_df = feat_df.copy()
    for c in MANDATORY + OPTIONAL:
        if c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors='coerce')
    # Iterate teams
    all_teams = pd.unique(pd.concat([num_df['home_team'], num_df['away_team']], ignore_index=True).dropna())
    for team in all_teams:
        mask = (num_df['home_team'] == team) | (num_df['away_team'] == team)
        sub = num_df[mask]
        if sub.empty:
            continue
        stat_map = {}
        for col in MANDATORY + OPTIONAL:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors='coerce').dropna()
                if len(vals):
                    stat_map[col] = float(vals.mean())
        if stat_map:
            team_stats[str(team)] = stat_map
    return team_stats

def league_baselines(feat_df: pd.DataFrame) -> dict:
    base = {}
    if feat_df.empty:
        return base
    for col in MANDATORY + OPTIONAL:
        if col in feat_df.columns:
            vals = pd.to_numeric(feat_df[col], errors='coerce').dropna()
            if len(vals):
                base[col] = float(vals.mean())
    return base


def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1 and sys.argv[1].strip():
        date_str = sys.argv[1].strip()
    miss_df = find_missing_coverage(date_str)
    if miss_df.empty:
        print('No missing coverage rows for date:', date_str)
        return
    feat_df = load_feature_sources()
    team_avgs = build_team_avgs(feat_df)
    league_avg = league_baselines(feat_df)
    out_rows = []
    for r in miss_df.itertuples():
        gid = str(getattr(r,'game_id'))
        home = getattr(r,'home_team','')
        away = getattr(r,'away_team','')
        row_map = {'game_id': gid, 'home_team': home, 'away_team': away}
        # Fill mandatory features
        for col in MANDATORY:
            val = None
            basis = None
            # Prefer direct row match if features already exist for gid
            if not feat_df.empty and 'game_id' in feat_df.columns:
                existing_row = feat_df[feat_df['game_id'].astype(str) == gid]
                if not existing_row.empty and col in existing_row.columns:
                    v = pd.to_numeric(existing_row[col], errors='coerce').iloc[0]
                    if pd.notna(v):
                        val = float(v)
                        basis = 'existing'
            if val is None and home and home in team_avgs and col in team_avgs[home]:
                val = team_avgs[home][col]
                basis = 'team_home_avg'
            if val is None and away and away in team_avgs and col in team_avgs[away]:
                val = team_avgs[away][col]
                basis = 'team_away_avg'
            if val is None and col in league_avg:
                val = league_avg[col]
                basis = 'league_avg'
            if val is not None:
                row_map[col] = val
                row_map[DEF_BASIS_COLS[col]] = basis
            else:
                row_map[col] = None
                row_map[DEF_BASIS_COLS[col]] = 'unfilled'
        # Propagate optional if present via team averages (do not force league fill to avoid over-smoothing)
        for col in OPTIONAL:
            val = None
            if home and home in team_avgs and col in team_avgs[home]:
                val = team_avgs[home][col]
            elif away and away in team_avgs and col in team_avgs[away]:
                val = team_avgs[away][col]
            elif col in league_avg:
                # mark but still fill to improve model coverage; basis recorded
                val = league_avg[col]
            row_map[col] = val
        out_rows.append(row_map)
    out_df = pd.DataFrame(out_rows)
    out_path = OUT / f'features_missing_filled_{date_str}.csv'
    out_df.to_csv(out_path, index=False)
    # Summary stats
    filled_counts = {c: int(out_df[c].notna().sum()) for c in MANDATORY if c in out_df.columns}
    with open(OUT / f'features_missing_filled_{date_str}.json','w',encoding='utf-8') as fh:
        json.dump({'date': date_str,'rows': len(out_df),'filled_counts': filled_counts}, fh, indent=2)
    print('Filled features written:', out_path, 'rows:', len(out_df))
    print('Mandatory fill counts:', filled_counts)

if __name__ == '__main__':
    main()
