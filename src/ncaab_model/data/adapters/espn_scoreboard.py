from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List
import requests

from ..schemas import Game
from ..cache import cache_path, read_json, write_json


ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={YYYYMMDD}"
)
# Fallback (broader coverage for D1): groups=50 with higher limit via site.web.api
ESPN_WEB_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000&dates={YYYYMMDD}"
)


@dataclass
class FetchResult:
    date: dt.date
    games: List[Game]
    source: str  # "cache" or "network" or "none"


def _fetch_day(date: dt.date, use_cache: bool = True) -> dict | None:
    cache_file = cache_path("espn", f"{date.isoformat()}.json")
    if use_cache and cache_file.exists():
        try:
            return read_json(cache_file)
        except Exception:
            pass
    url = ESPN_URL.format(YYYYMMDD=date.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Heuristic: if too few events, try the broader site.web.api endpoint
        try:
            events = data.get("events", [])
            count = len(events) if isinstance(events, list) else 0
        except Exception:
            count = 0
        # Fallback threshold: if fewer than 20 events (typical mid-season multi-provider slate >30),
        # attempt broader site.web.api endpoint with groups=50 & limit=1000 to capture additional D1 games.
        if count < 20:
            alt_url = ESPN_WEB_URL.format(YYYYMMDD=date.strftime("%Y%m%d"))
            try:
                r2 = requests.get(alt_url, timeout=20)
                r2.raise_for_status()
                data2 = r2.json()
                ev2 = data2.get("events", []) if isinstance(data2, dict) else []
                if isinstance(ev2, list) and len(ev2) > count:
                    data = data2
            except Exception:
                pass
        write_json(cache_file, data)
        return data
    except Exception:
        # Try fallback directly if primary failed
        try:
            alt_url = ESPN_WEB_URL.format(YYYYMMDD=date.strftime("%Y%m%d"))
            r2 = requests.get(alt_url, timeout=20)
            r2.raise_for_status()
            data2 = r2.json()
            write_json(cache_file, data2)
            return data2
        except Exception:
            return None


def _parse_games(date: dt.date, payload: dict) -> List[Game]:
    games: List[Game] = []
    events = payload.get("events", [])
    for ev in events:
        try:
            game_id = str(ev.get("id") or f"{date.isoformat()}-{len(games)}")
            comps = (ev.get("competitions") or [{}])[0]
            neutral_site = comps.get("neutralSite")
            venue_name = None
            try:
                venue = comps.get("venue") or {}
                venue_name = venue.get("fullName") or venue.get("address", {}).get("city")
            except Exception:
                venue_name = None
            competitors = comps.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_team = (
                (home.get("team") or {}).get("displayName")
                or (home.get("team") or {}).get("shortDisplayName")
                or "HOME"
            )
            away_team = (
                (away.get("team") or {}).get("displayName")
                or (away.get("team") or {}).get("shortDisplayName")
                or "AWAY"
            )
            # Scores
            def parse_int(x):
                try:
                    return int(x) if x is not None else None
                except Exception:
                    return None

            home_score = parse_int(home.get("score"))
            away_score = parse_int(away.get("score"))

            # Linescores contain period scoring
            def sum_period(competitor, period_numbers):
                total = 0
                found = False
                for ls in competitor.get("linescores", []):
                    num = ls.get("period") or ls.get("sequence") or ls.get("number")
                    val = parse_int(ls.get("value"))
                    if num in period_numbers and val is not None:
                        total += val
                        found = True
                return total if found else None

            home_1h = sum_period(home, {1})
            away_1h = sum_period(away, {1})
            home_2h = sum_period(home, {2})
            away_2h = sum_period(away, {2})

            # Start/commence time where available
            start_time = None
            try:
                # ESPN sometimes exposes date string under competitions[0]["date"]
                comp_date = comps.get("date")
                if comp_date:
                    start_time = dt.datetime.fromisoformat(comp_date.replace("Z", "+00:00"))
            except Exception:
                start_time = None

            games.append(
                Game(
                    game_id=game_id,
                    season=date.year,
                    date=dt.datetime.combine(date, dt.time(0, 0)),
                    start_time=start_time,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    home_score_1h=home_1h,
                    away_score_1h=away_1h,
                    home_score_2h=home_2h,
                    away_score_2h=away_2h,
                    neutral_site=bool(neutral_site) if neutral_site is not None else None,
                    venue=venue_name,
                )
            )
        except Exception:
            continue
    return games


def iter_games_by_date(start: dt.date, end: dt.date, use_cache: bool = True) -> Iterable[FetchResult]:
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        payload = _fetch_day(cur, use_cache=use_cache)
        if payload is None:
            yield FetchResult(cur, [], source="none")
        else:
            games = _parse_games(cur, payload)
            src = "cache" if cache_path("espn", f"{cur.isoformat()}.json").exists() else "network"
            yield FetchResult(cur, games, src)
        cur += one
