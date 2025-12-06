import json, re, os
from pathlib import Path
import pandas as pd
import datetime as dt

# Resolve target date from env or default to today (YYYY-MM-DD)
DATE = os.environ.get('TARGET_DATE') or dt.date.today().strftime('%Y-%m-%d')
OUT = Path('outputs')
res = {}

# Load games
games_path = OUT / 'games_curr.csv'
if games_path.exists():
    gdf = pd.read_csv(games_path)
    if 'date' in gdf.columns:
        gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        g_today = gdf[gdf['date'] == DATE].copy()
    else:
        g_today = gdf
    res['games_today_rows'] = len(g_today)
    res['games_cols'] = list(g_today.columns)
else:
    g_today = pd.DataFrame()
    res['games_today_rows'] = 0

# Load predictions (priority order)
pred_df = pd.DataFrame()
for cand in [OUT / f'predictions_{DATE}.csv', OUT / 'predictions_week.csv', OUT / 'predictions.csv', OUT / 'predictions_all.csv', OUT / 'predictions_last2.csv']:
    if cand.exists():
        try:
            tmp = pd.read_csv(cand)
            if not tmp.empty:
                pred_df = tmp
                res['predictions_source'] = str(cand)
                break
        except Exception:
            pass
if not pred_df.empty and 'date' in pred_df.columns:
    pred_df['date'] = pd.to_datetime(pred_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    pred_today = pred_df[pred_df['date'] == DATE].copy()
else:
    pred_today = pred_df
res['pred_today_rows'] = len(pred_today)

# Load odds join
odds_join = pd.DataFrame()
for cand in [OUT / f'games_with_odds_{DATE}.csv', OUT / 'games_with_last.csv', OUT / 'games_with_closing.csv']:
    if cand.exists():
        try:
            tmp = pd.read_csv(cand)
            if not tmp.empty:
                odds_join = tmp
                res['odds_join_source'] = str(cand)
                break
        except Exception:
            pass

# ID sets
pred_ids = set(pred_today['game_id'].astype(str)) if 'game_id' in pred_today.columns else set()
game_ids = set(g_today['game_id'].astype(str)) if 'game_id' in g_today.columns else set()
odds_ids = set(odds_join['game_id'].astype(str)) if 'game_id' in odds_join.columns else set()
res['coverage_game_vs_pred'] = {'matched': len(game_ids & pred_ids), 'total_games': len(game_ids)}
res['coverage_game_vs_odds'] = {'matched': len(game_ids & odds_ids), 'total_games': len(game_ids)}
res['games_without_odds'] = sorted(list(game_ids - odds_ids))[:25]
res['games_without_preds'] = sorted(list(game_ids - pred_ids))[:25]

# Low prediction totals
low_pred = []
if not pred_today.empty and 'pred_total' in pred_today.columns:
    pt = pd.to_numeric(pred_today['pred_total'], errors='coerce')
    mask = pt.notna() & (pt < 110)
    if mask.any():
        cols = [c for c in ['game_id','home_team','away_team','pred_total','pred_margin'] if c in pred_today.columns]
        low_pred = pred_today.loc[mask, cols].head(12).to_dict(orient='records')
res['low_pred_count'] = len(low_pred)
res['low_pred_sample'] = low_pred

# Attempt to render index route (diagnostics) to detect half line tokens
try:
    import app  # noqa
    from app import app as flask_app
    with flask_app.test_client() as c:
        r = c.get(f'/?date={DATE}&diag=1')
        html = r.get_data(as_text=True)
        # Count occurrences of 1H rows & derived badge 'D'
        res['html_contains_1H_tokens'] = html.count('1H')
        # Simple pattern for half total rows might include 'market_total_1h' printed in diag sections
        res['html_contains_market_total_1h_tokens'] = html.count('market_total_1h')
        # Badge occurrences (class basis-derived)
        res['html_badge_derived_count'] = html.count('basis-derived')
except Exception as e:
    res['index_render_error'] = str(e)

print(json.dumps(res, indent=2))
