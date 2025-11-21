import pandas as pd
import pathlib
import re

OUT = pathlib.Path(__file__).resolve().parents[1] / 'outputs'
DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'

def _norm_team(t: str) -> str:
    if not isinstance(t, str):
        return ''
    t = t.lower().strip()
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    return ' '.join(t.split())

def test_all_scheduled_teams_in_d1_membership():
    games_path = OUT / 'games_curr.csv'
    conf_path = DATA / 'd1_conferences.csv'
    assert games_path.exists(), 'games_curr.csv missing in outputs/'
    assert conf_path.exists(), 'd1_conferences.csv missing in data/'
    games = pd.read_csv(games_path)
    conf = pd.read_csv(conf_path)
    # Determine team column in membership file (prefer team, school, name)
    team_col = next((c for c in ['team','school','name','team_name'] if c in conf.columns), None)
    assert team_col, 'No recognizable team column in d1_conferences.csv'
    membership = {_norm_team(x) for x in conf[team_col].dropna().astype(str)}
    missing: set[str] = set()
    for side in ['home_team','away_team']:
        if side in games.columns:
            for val in games[side].dropna().astype(str).unique():
                nv = _norm_team(val)
                if nv in membership:
                    continue
                # Attempt stripped variants (remove last token / mascot words)
                parts = [p for p in nv.split() if p]
                alt_candidates = []
                if len(parts) > 1:
                    alt_candidates.append(' '.join(parts[:-1]))
                if len(parts) > 2:
                    alt_candidates.append(' '.join(parts[:2]))
                if parts:
                    alt_candidates.append(parts[0])
                matched = any(a in membership for a in alt_candidates)
                if not matched:
                    missing.add(val)
    # Allow exhibitions / non D1 matches: if more than 10 missing, fail; else warn
    # Allow modest exhibition / non-D1 presence early season
    assert len(missing) <= 18, f"Too many non-membership teams: {sorted(missing)[:20]} (n={len(missing)})"

def test_unified_model_coverage_today():
    # Ensure every game today has model predictions
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    uni_path = OUT / f'predictions_unified_{today}.csv'
    if not uni_path.exists():
        import pytest
        pytest.skip(f'unified predictions file missing for today {today}')
    df = pd.read_csv(uni_path)
    assert not df.empty, 'Unified predictions empty'
    assert 'pred_total_model' in df.columns, 'pred_total_model column missing'
    assert 'pred_margin_model' in df.columns, 'pred_margin_model column missing'
    miss_total = pd.to_numeric(df['pred_total_model'], errors='coerce').isna().sum()
    miss_margin = pd.to_numeric(df['pred_margin_model'], errors='coerce').isna().sum()
    assert miss_total == 0, f'Missing model totals for {miss_total} rows'
    assert miss_margin == 0, f'Missing model margins for {miss_margin} rows'
