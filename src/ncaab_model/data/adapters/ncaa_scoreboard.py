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
    # Common schema: payload['games'] is list of {'game': {...}} objects.
    items = payload.get("games") or payload.get("scoreboard") or payload.get("events") or []
    for item in items:
        try:
            gobj = item.get("game") if isinstance(item, dict) and isinstance(item.get("game"), dict) else item
            if not isinstance(gobj, dict):
                continue

            game_id = str(
                gobj.get("gameID")
                or gobj.get("gameId")
                or gobj.get("id")
                or item.get("id")
                or f"{date.isoformat()}-{len(games)}"
            )

            # Teams + scores (new NCAA schema: game.home / game.away)
            home_team = None
            away_team = None
            home_score = away_score = None
            home_score_1h = away_score_1h = None
            home_score_2h = away_score_2h = None

            home = gobj.get("home") or {}
            away = gobj.get("away") or {}
            try:
                home_names = home.get("names") or {}
                away_names = away.get("names") or {}
                home_team = home_names.get("short") or home_names.get("full") or home.get("name")
                away_team = away_names.get("short") or away_names.get("full") or away.get("name")
            except Exception:
                home_team = None
                away_team = None

            def _int(x):
                try:
                    return int(x) if x is not None and str(x).strip() != "" else None
                except Exception:
                    return None

            home_score = _int(home.get("score"))
            away_score = _int(away.get("score"))

            # Status/completion (heuristics)
            status = None
            completed: Optional[bool] = None
            try:
                status = (
                    gobj.get("gameState")
                    or gobj.get("finalMessage")
                    or gobj.get("currentPeriod")
                    or gobj.get("contestClock")
                )
                gs = str(gobj.get("gameState") or "").strip().lower()
                fm = str(gobj.get("finalMessage") or "").strip().upper()
                if gs:
                    completed = gs in {"final", "completed", "complete"}
                if completed is None and fm:
                    completed = fm.startswith("FINAL")
            except Exception:
                status = None
                completed = None

            # NCAA feed often lacks explicit tipoff time; derive approximate local schedule time if available from item
            start_time = None
            start_time_local = None
            start_tz_abbr = None
            try:
                # Some feeds include 'startTime' or 'game'->'startTime'
                raw_start = gobj.get('startTime') or item.get('startTime')
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

            # Prefer epoch time when present
            try:
                epoch = gobj.get("startTimeEpoch")
                if epoch is not None and str(epoch).strip() != "":
                    start_time = dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc)
            except Exception:
                start_time = None
            games.append(
                Game(
                    game_id=game_id,
                    season=date.year,
                    date=dt.datetime.combine(date, dt.time(0, 0)),
                    start_time=start_time,
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
                    status=status,
                    completed=completed,
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
