#!/usr/bin/env python
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
ESPN_SUMMARY_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary'


def fetch_summary(game_id: str | int) -> dict | None:
    try:
        r = requests.get(ESPN_SUMMARY_URL, params={'event': str(game_id)}, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def parse_team_stats(summary: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    try:
        teams = summary.get('boxscore', {}).get('teams') or summary.get('boxscore', {}).get('players')
        # Fallback alt path
        if teams is None:
            teams = summary.get('gameInfo', {}).get('teams', [])
        if not teams:
            return out
        for t in teams:
            # ESPN 'homeAway' exists on 'teams' under boxscore in many summaries
            side = t.get('homeAway') or t.get('position') or ''
            side = 'home' if str(side).lower() == 'home' else ('away' if str(side).lower() == 'away' else None)
            stats = t.get('statistics') or []
            vals = {}
            for s in stats:
                name = s.get('name')
                val = s.get('value')
                try:
                    v = float(val)
                except Exception:
                    continue
                if name == 'fieldGoalsAttempted':
                    vals['fga'] = v
                elif name == 'freeThrowsAttempted':
                    vals['fta'] = v
                elif name == 'offensiveRebounds':
                    vals['or'] = v
                elif name == 'turnovers':
                    vals['to'] = v
            if side:
                out[side] = vals
    except Exception:
        pass
    return out


def enrich_enriched_with_box(date: str, sleep_sec: float = 0.4, in_place: bool = True) -> Path | None:
    p = OUT / f'predictions_unified_enriched_{date}.csv'
    if not p.exists():
        print(f'[err] Missing enriched: {p}')
        return None
    df = pd.read_csv(p)
    if 'game_id' not in df.columns:
        print('[err] enriched lacks game_id')
        return None
    df['game_id'] = df['game_id'].astype(str)
    # Prepare columns
    for c in ['home_fga','home_fta','home_or','home_to','away_fga','away_fta','away_or','away_to']:
        if c not in df.columns:
            df[c] = pd.NA
    gids = sorted(df['game_id'].unique())
    filled = 0
    for gid in gids:
        summary = fetch_summary(gid)
        if not summary:
            time.sleep(sleep_sec)
            continue
        stats = parse_team_stats(summary)
        if not stats:
            time.sleep(sleep_sec)
            continue
        mask = df['game_id'] == gid
        if 'home' in stats:
            df.loc[mask, 'home_fga'] = stats['home'].get('fga')
            df.loc[mask, 'home_fta'] = stats['home'].get('fta')
            df.loc[mask, 'home_or'] = stats['home'].get('or')
            df.loc[mask, 'home_to'] = stats['home'].get('to')
        if 'away' in stats:
            df.loc[mask, 'away_fga'] = stats['away'].get('fga')
            df.loc[mask, 'away_fta'] = stats['away'].get('fta')
            df.loc[mask, 'away_or'] = stats['away'].get('or')
            df.loc[mask, 'away_to'] = stats['away'].get('to')
        filled += int(mask.sum())
        time.sleep(sleep_sec)
    out_path = p if in_place else (OUT / f'predictions_unified_enriched_{date}_with_box.csv')
    df.to_csv(out_path, index=False)
    print(json.dumps({
        'date': date,
        'rows': int(df.shape[0]),
        'games_filled': filled,
        'path': str(out_path)
    }, indent=2))
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Fetch ESPN box scores and enrich daily predictions with FGA/FTA/OR/TOV.')
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--sleep-sec', type=float, default=0.4)
    ap.add_argument('--sidecar', action='store_true', help='Write _with_box.csv sidecar instead of in-place')
    args = ap.parse_args()
    enrich_enriched_with_box(args.date, sleep_sec=args.sleep_sec, in_place=(not args.sidecar))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
