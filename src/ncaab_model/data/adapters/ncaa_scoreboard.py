from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Optional
import requests
import os
from zoneinfo import ZoneInfo

from ..schemas import Game
from ..cache import cache_path, read_json, write_json


SCOREBOARD_URL = (
    "https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{Y}/{M}/{D}/scoreboard.json"
)


@dataclass
class FetchResult:
    date: dt.date
    games: List[Game]
    source: str  # "cache" or "network"


def _fetch_scoreboard(date: dt.date, use_cache: bool = True) -> dict | None:
    cache_file = cache_path("scoreboard", f"{date.isoformat()}.json")
    if use_cache and cache_file.exists():
        try:
            return read_json(cache_file)
        except Exception:
            pass
    url = SCOREBOARD_URL.format(Y=date.year, M=str(date.month).zfill(2), D=str(date.day).zfill(2))
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        write_json(cache_file, data)
        return data
    except Exception:
        return None


def _parse_games(date: dt.date, payload: dict) -> List[Game]:
    games: List[Game] = []
    # The schema may vary; attempt to parse conservatively.
    # Expected keys: 'scoreboard' -> list of events with 'id', 'game', 'teams', 'status'
    items = payload.get("scoreboard") or payload.get("games") or []
    for item in items:
        try:
            game_id = str(item.get("id") or item.get("gameId") or f"{date.isoformat()}-{len(games)}")
            # Teams
            home_team = None
            away_team = None
            # Try multiple possible structures
            teams = item.get("teams") or item.get("participants") or []
            if isinstance(teams, list) and len(teams) >= 2:
                # assume first is away, second is home, but attempt to check a 'isHome' flag
                t0 = teams[0]
                t1 = teams[1]
                if (t0.get("isHome") or t0.get("homeAway") == "home") and not (t1.get("isHome") or t1.get("homeAway") == "home"):
                    home_team = t0.get("name") or t0.get("displayName") or t0.get("shortName")
                    away_team = t1.get("name") or t1.get("displayName") or t1.get("shortName")
                elif (t1.get("isHome") or t1.get("homeAway") == "home"):
                    home_team = t1.get("name") or t1.get("displayName") or t1.get("shortName")
                    away_team = t0.get("name") or t0.get("displayName") or t0.get("shortName")
                else:
                    away_team = t0.get("name") or t0.get("displayName") or t0.get("shortName")
                    home_team = t1.get("name") or t1.get("displayName") or t1.get("shortName")
            # Scores
            home_score = away_score = None
            home_score_1h = away_score_1h = None
            home_score_2h = away_score_2h = None
            score_obj = item.get("status") or item.get("score") or {}
            if isinstance(score_obj, dict):
                # Try totals
                try:
                    home_score = int(score_obj.get("homeScore")) if score_obj.get("homeScore") is not None else None
                    away_score = int(score_obj.get("awayScore")) if score_obj.get("awayScore") is not None else None
                except Exception:
                    pass
                # Try periods array
                periods = score_obj.get("periods") or score_obj.get("linescores") or []
                if isinstance(periods, list) and len(periods) >= 2:
                    # Sum period scores to halves if labeled, else split first half of periods as 1H
                    try:
                        # attempt direct half labels
                        for p in periods:
                            label = str(p.get("label") or p.get("sequence") or "").lower()
                            h = p.get("home") or p.get("homeScore") or p.get("home_points")
                            a = p.get("away") or p.get("awayScore") or p.get("away_points")
                            if label.startswith("1"):
                                home_score_1h = int(h) if h is not None else home_score_1h
                                away_score_1h = int(a) if a is not None else away_score_1h
                            elif label.startswith("2"):
                                home_score_2h = int(h) if h is not None else home_score_2h
                                away_score_2h = int(a) if a is not None else away_score_2h
                    except Exception:
                        pass

            # NCAA feed often lacks explicit tipoff time; derive approximate local schedule time if available from item
            start_time_local = None
            start_tz_abbr = None
            try:
                # Some feeds include 'startTime' or 'game'->'startTime'
                raw_start = item.get('startTime') or (item.get('game') or {}).get('startTime')
                if raw_start:
                    # Attempt ISO parse; treat as UTC if 'Z' present else naive assume Eastern
                    try:
                        st_dt = dt.datetime.fromisoformat(str(raw_start).replace('Z','+00:00'))
                        sched_tz = os.getenv('SCHEDULE_TZ') or 'America/New_York'
                        local_dt = st_dt.astimezone(ZoneInfo(sched_tz))
                        start_time_local = local_dt.strftime('%Y-%m-%d %H:%M')
                        start_tz_abbr = local_dt.tzname()
                    except Exception:
                        pass
            except Exception:
                pass
            games.append(
                Game(
                    game_id=game_id,
                    season=date.year,
                    date=dt.datetime.combine(date, dt.time(0, 0)),
                    start_time_local=start_time_local,
                    start_tz_abbr=start_tz_abbr,
                    home_team=home_team or "HOME",
                    away_team=away_team or "AWAY",
                    home_score=home_score,
                    away_score=away_score,
                    home_score_1h=home_score_1h,
                    away_score_1h=away_score_1h,
                    home_score_2h=home_score_2h,
                    away_score_2h=away_score_2h,
                )
            )
        except Exception:
            continue
    return games


def iter_games_by_date(start: dt.date, end: dt.date, use_cache: bool = True) -> Iterable[FetchResult]:
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        payload = _fetch_scoreboard(cur, use_cache=use_cache)
        if payload is None:
            yield FetchResult(cur, [], source="none")
        else:
            games = _parse_games(cur, payload)
            src = "cache" if cache_path("scoreboard", f"{cur.isoformat()}.json").exists() else "network"
            yield FetchResult(cur, games, src)
        cur += one
