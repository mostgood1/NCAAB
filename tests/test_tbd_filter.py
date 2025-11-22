import importlib
import pandas as pd

def test_tbd_games_filtered():
    import app
    importlib.reload(app)
    # Build results and check no row has both teams as TBD
    df, _ = app._build_results_df('2025-11-22')
    if {'home_team','away_team'}.issubset(df.columns):
        bad = df[(df['home_team'].astype(str).str.upper()=='TBD') & (df['away_team'].astype(str).str.upper()=='TBD')]
        assert bad.empty, 'Found TBD vs TBD rows which should be filtered'
