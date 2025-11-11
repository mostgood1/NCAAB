import pandas as pd, datetime as dt, os, sys
out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
out_dir = os.path.abspath(out_dir)
try:
    today = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
except Exception:
    today = dt.date.today()
report = {}
# Simple normalizer similar to app's normalize_name for this standalone script
def _canon(s: str) -> str:
    s = str(s or "").lower().strip()
    return "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-"))

# Load D1 set
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root, 'data')
d1_path = os.path.join(data_dir, 'd1_conferences.csv')
d1set: set[str] = set()
if os.path.exists(d1_path):
    try:
        d1df = pd.read_csv(d1_path)
        # pick likely team column
        lc = {c.lower().strip(): c for c in d1df.columns}
        team_col = lc.get('team') or lc.get('school') or lc.get('name') or list(d1df.columns)[0]
        d1set = set(d1df[team_col].astype(str).map(_canon))
    except Exception:
        d1set = set()
# Load games
try:
    g = pd.read_csv(os.path.join(out_dir, 'games_curr.csv'))
except Exception as e:
    print({'error':'games_curr.csv load failed', 'e':str(e)}); g = pd.DataFrame()
if not g.empty and 'game_id' in g.columns:
    g['game_id'] = g['game_id'].astype(str)
if 'date' in g.columns:
    try:
        g['date'] = pd.to_datetime(g['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    except Exception:
        g['date'] = g['date'].astype(str)

g_today = g[g.get('date','') == today.isoformat()].copy() if 'date' in g.columns else g.copy()
report['games_today'] = len(g_today)
# Display subset: at least one D1 team
g_disp = pd.DataFrame()
if not g_today.empty and {'home_team','away_team'}.issubset(g_today.columns) and d1set:
    g_today['_home_n'] = g_today['home_team'].astype(str).map(_canon)
    g_today['_away_n'] = g_today['away_team'].astype(str).map(_canon)
    mask_any = g_today['_home_n'].isin(d1set) | g_today['_away_n'].isin(d1set)
    g_disp = g_today[mask_any].copy()
report['display_games_today_d1_any'] = int(len(g_disp)) if not g_disp.empty else 0
# Load games_with_last
try:
    ml = pd.read_csv(os.path.join(out_dir, 'games_with_last.csv'))
except Exception as e:
    ml = pd.DataFrame()
if not ml.empty and 'game_id' in ml.columns:
    ml['game_id'] = ml['game_id'].astype(str)
if not ml.empty:
    if 'date_game' in ml.columns:
        try:
            ml['date_game'] = pd.to_datetime(ml['date_game'], errors='coerce').dt.strftime('%Y-%m-%d')
            ml_today = ml[ml['date_game'] == today.isoformat()].copy()
        except Exception:
            ml_today = ml.copy()
    else:
        cols = ['game_id','date'] if 'date' in g.columns else ['game_id']
        ml_today = ml.merge(g_today[cols], on='game_id', how='inner') if not g_today.empty else pd.DataFrame()
else:
    ml_today = pd.DataFrame()
covered_exact = ml_today[ml_today.get('partial_pair') != True]['game_id'].nunique() if not ml_today.empty and 'partial_pair' in ml_today.columns else ml_today.get('game_id', pd.Series(dtype=str)).nunique()
covered_partial = ml_today[ml_today.get('partial_pair') == True]['game_id'].nunique() if not ml_today.empty and 'partial_pair' in ml_today.columns else 0
report['covered_exact_games_last'] = int(covered_exact)
report['covered_partial_games_last'] = int(covered_partial)
report['covered_any_games_last'] = int(len(set(ml_today.get('game_id', pd.Series(dtype=str))))) if not ml_today.empty else 0
# Current odds join for the date
try:
    mo = pd.read_csv(os.path.join(out_dir, f'games_with_odds_{today.isoformat()}.csv'))
except Exception:
    mo = pd.DataFrame()
if not mo.empty and 'game_id' in mo.columns:
    mo['game_id'] = mo['game_id'].astype(str)
report['covered_any_games_current'] = int(mo['game_id'].nunique()) if not mo.empty else 0
# Coverage for display subset only
if not g_disp.empty:
    gi = set(g_disp['game_id'].astype(str)) if 'game_id' in g_disp.columns else set()
    covered_last_set = set(ml_today.get('game_id', pd.Series(dtype=str)).astype(str)) if not ml_today.empty else set()
    covered_curr_set = set(mo.get('game_id', pd.Series(dtype=str)).astype(str)) if not mo.empty else set()
    report['display_covered_any_games_last'] = int(len(gi & covered_last_set))
    report['display_covered_any_games_current'] = int(len(gi & covered_curr_set))
missing = []
if not g_today.empty:
    gi = set(g_today.get('game_id', pd.Series(dtype=str)).astype(str))
    covered_last = set(ml_today.get('game_id', pd.Series(dtype=str)).astype(str)) if not ml_today.empty else set()
    missing_ids = sorted(list(gi - covered_last))
    # Try to add names if present
    name_cols = [('home_team_name','away_team_name'), ('home_team','away_team'), ('home_name','away_name'), ('home','away')]
    home_col, away_col = None, None
    for h,a in name_cols:
        if h in g_today.columns and a in g_today.columns:
            home_col, away_col = h, a
            break
    subset_cols = ['game_id', 'date'] + ([home_col, away_col] if home_col and away_col else [])
    subset_cols = [c for c in subset_cols if c in g_today.columns]
    miss_df = g_today[g_today['game_id'].astype(str).isin(missing_ids)][subset_cols].copy()
    if home_col and away_col:
        miss_df['matchup'] = miss_df[away_col].astype(str) + ' @ ' + miss_df[home_col].astype(str)
    out_dir_parent = out_dir
    try:
        out_path = os.path.join(out_dir_parent, f'missing_last_odds_{today.isoformat()}.csv')
        miss_df.to_csv(out_path, index=False)
        report['missing_last_odds_csv'] = out_path
        report['missing_last_odds_count'] = int(len(miss_df))
    except Exception as e:
        report['missing_last_odds_error'] = str(e)
print(report)
