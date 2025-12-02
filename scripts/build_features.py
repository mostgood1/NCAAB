"""Build richer team/game features for quantile training.

Features include:
- Rest metrics: days since last game, back-to-back, 3-in-4
- Rolling form (last N): points for/against, net, total proxy (pace)

Inputs:
- outputs/predictions_history_enriched.csv (must include date, game_id, home_team, away_team)
- outputs/daily_results/results_*.csv (actual_total, actual_margin preferred)

Outputs:
- outputs/features_history.csv with per-game engineered features joined by (date, game_id)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path('outputs')


def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_results() -> pd.DataFrame:
    frames = []
    for p in OUTPUTS.glob('daily_results/results_*.csv'):
        df = _safe_read(p)
        if not df.empty and 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _infer_points(actual_total: float, actual_margin: float, is_home: bool) -> tuple[float, float]:
    if not np.isfinite(actual_total) or not np.isfinite(actual_margin):
        return (np.nan, np.nan)
    home_pts = (actual_total + actual_margin) / 2.0
    away_pts = (actual_total - actual_margin) / 2.0
    return (home_pts, away_pts) if is_home else (away_pts, home_pts)


def main():
    preds = _safe_read(OUTPUTS / 'predictions_history_enriched.csv')
    res = _load_results()
    if preds.empty:
        print('[features] Missing predictions_history_enriched; aborting.')
        return
    # Normalize ids
    preds['game_id'] = preds.get('game_id', '').astype(str).str.replace(r'\.0$', '', regex=True)
    if not res.empty:
        res['game_id'] = res['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Merge actuals if available
    df = preds.merge(res[['date','game_id','actual_total','actual_margin']] if {'date','game_id','actual_total','actual_margin'}.issubset(res.columns) else pd.DataFrame(),
                     on=['date','game_id'], how='left')

    # Team columns with fallbacks
    home_col = 'home_team' if 'home_team' in df.columns else ('team_home' if 'team_home' in df.columns else None)
    away_col = 'away_team' if 'away_team' in df.columns else ('team_away' if 'team_away' in df.columns else None)
    if home_col is None or away_col is None:
        print('[features] Missing home/away team columns; aborting.')
        return
    # Build team-game logs
    rows = []
    for r in df[['date','game_id',home_col,away_col,'actual_total','actual_margin']].itertuples(index=False, name=None):
        date, gid, home_team, away_team, atot, amag = r
        # Home entry
        h_pf, h_pa = _infer_points(atot, amag, True)
        rows.append({'date': date, 'team': home_team, 'opp': away_team, 'is_home': 1, 'game_id': gid, 'pf': h_pf, 'pa': h_pa})
        # Away entry
        a_pf, a_pa = _infer_points(atot, amag, False)
        rows.append({'date': date, 'team': away_team, 'opp': home_team, 'is_home': 0, 'game_id': gid, 'pf': a_pf, 'pa': a_pa})
    tlog = pd.DataFrame(rows)
    # Sort and compute per-team rolling features
    tlog['date'] = pd.to_datetime(tlog['date'], errors='coerce')
    tlog = tlog.sort_values(['team','date']).reset_index(drop=True)
    # Compute rest days
    tlog['prev_date'] = tlog.groupby('team', observed=False)['date'].shift(1)
    tlog['rest_days'] = (tlog['date'] - tlog['prev_date']).dt.days
    tlog['b2b'] = (tlog['rest_days'] <= 1).astype('Int64')
    # 3-in-4: count games in last 4 days window (excluding current)
    # Approximate via rolling count over a 4-day window using expanding + diff over shifted mask
    tlog['game_count'] = 1
    # For efficiency, compute rolling count of games within 4 days using backward-looking logic
    # We'll compute counts of prior games within 4 days using merge-asof-like approach
    # Simplified: rolling window size 5 games and then compute time diff mask
    def three_in_four(g: pd.DataFrame) -> pd.Series:
        dates = g['date'].values
        out = []
        for i in range(len(dates)):
            start = dates[i] - np.timedelta64(4, 'D')
            cnt = np.sum((dates[:i] >= start) & (dates[:i] < dates[i]))
            out.append(1 if cnt >= 2 else 0)
        return pd.Series(out, index=g.index, dtype='Int64')
    three4 = tlog.groupby('team', observed=False).apply(three_in_four).reset_index(level=0, drop=True)
    tlog['three_in_four'] = pd.to_numeric(three4, errors='coerce').astype('Int64')

    # Rolling form last N
    N = 5
    def shifted_roll(g: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(g[col], errors='coerce')
        return s.shift(1).rolling(N, min_periods=1).mean()
    pf_roll = tlog.groupby('team', observed=False).apply(lambda g: shifted_roll(g, 'pf')).reset_index(level=0, drop=True)
    pa_roll = tlog.groupby('team', observed=False).apply(lambda g: shifted_roll(g, 'pa')).reset_index(level=0, drop=True)
    tlog['pf_roll'] = pf_roll.values
    tlog['pa_roll'] = pa_roll.values
    tlog['net_roll'] = tlog['pf_roll'] - tlog['pa_roll']
    # Total proxy pace: prior totals average for team
    # If pf/pa are NaN (missing actuals), these will be NaN; allow fallback later
    total_roll = tlog.groupby('team', observed=False).apply(
        lambda g: (pd.to_numeric(g['pf'], errors='coerce') + pd.to_numeric(g['pa'], errors='coerce')).shift(1).rolling(N, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    tlog['total_roll'] = total_roll.values

    # Collapse back to game-level features for both teams
    # Select relevant columns and pivot by role
    tsel = tlog[['date','game_id','team','is_home','rest_days','b2b','three_in_four','pf_roll','pa_roll','net_roll','total_roll']]
    home_feats = tsel[tsel['is_home'] == 1].copy()
    away_feats = tsel[tsel['is_home'] == 0].copy()
    # Prepare rename suffixes
    rename_map = {
        'rest_days':'rest_days', 'b2b':'b2b', 'three_in_four':'three_in_four',
        'pf_roll':'pts_for_l5', 'pa_roll':'pts_against_l5', 'net_roll':'net_l5', 'total_roll':'total_l5'
    }
    home_feats = home_feats.rename(columns={k: f'home_{v}' for k, v in rename_map.items()})
    away_feats = away_feats.rename(columns={k: f'away_{v}' for k, v in rename_map.items()})
    home_feats = home_feats[['date','game_id'] + [c for c in home_feats.columns if c.startswith('home_')]]
    away_feats = away_feats[['date','game_id'] + [c for c in away_feats.columns if c.startswith('away_')]]

    feats = home_feats.merge(away_feats, on=['date','game_id'], how='inner')
    # Fill plausible defaults
    for c in feats.columns:
        if c.endswith('rest_days'):
            feats[c] = feats[c].fillna(3)
        elif c.endswith('b2b') or c.endswith('three_in_four'):
            feats[c] = feats[c].fillna(0).astype('Int64')
        elif c.endswith(('_l5')):
            feats[c] = feats[c].astype(float)
    feats = feats.sort_values(['date','game_id']).drop_duplicates(['date','game_id'], keep='last')
    feats.to_csv(OUTPUTS / 'features_history.csv', index=False)
    print('[features] Wrote outputs/features_history.csv')


if __name__ == '__main__':
    main()
