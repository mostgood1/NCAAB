from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Iterable, List
import requests
import os
from zoneinfo import ZoneInfo

from ..schemas import Game
from ..cache import cache_path, read_json, write_json


ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={YYYYMMDD}"
)
# Fallback (broader coverage for D1): groups=50 with higher limit via site.web.api
ESPN_WEB_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000&dates={YYYYMMDD}"
)
ESPN_SCHEDULE_URL = "https://www.espn.com/mens-college-basketball/schedule/_/date/{YYYYMMDD}"
SCHEDULE_BOOTSTRAP_RE = re.compile(r"window\['__espnfitt__'\]=(\{.*?\});</script>", re.S)


@dataclass
class FetchResult:
    date: dt.date
    games: List[Game]
    source: str  # "cache" or "network" or "none"


def _fetch_day(date: dt.date, use_cache: bool = True, cache_only: bool = False) -> dict | None:
    cache_file = cache_path("espn", f"{date.isoformat()}.json")
    if use_cache and cache_file.exists():
        try:
            return read_json(cache_file)
        except Exception:
            pass
    if cache_only:
        return None
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


def _extract_schedule_payload(html: str) -> dict | None:
    if not html:
        return None
    match = SCHEDULE_BOOTSTRAP_RE.search(html)
    if not match:
        return None
    try:
        bootstrap = json.loads(match.group(1))
    except Exception:
        return None
    try:
        content = ((bootstrap.get("page") or {}).get("content") or {})
        events = content.get("events") or {}
        if isinstance(events, dict):
            return {"events": events}
    except Exception:
        return None
    return None


def _fetch_schedule_day(date: dt.date, use_cache: bool = True, cache_only: bool = False) -> dict | None:
    cache_file = cache_path("espn_schedule", f"{date.isoformat()}.json")
    if use_cache and cache_file.exists():
        try:
            return read_json(cache_file)
        except Exception:
            pass
    if cache_only:
        return None
    url = ESPN_SCHEDULE_URL.format(YYYYMMDD=date.strftime("%Y%m%d"))
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = _extract_schedule_payload(resp.text)
        if payload:
            write_json(cache_file, payload)
        return payload
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

            # Game status
            status_name = None
            completed = None
            try:
                st = comps.get("status") or ev.get("status") or {}
                st_type = (st.get("type") or {}) if isinstance(st, dict) else {}
                if isinstance(st_type, dict):
                    status_name = st_type.get("name") or st_type.get("description") or st_type.get("state")
                    c = st_type.get("completed")
                    if isinstance(c, bool):
                        completed = c
            except Exception:
                status_name = None
                completed = None

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
            start_time_local = None
            start_tz_abbr = None
            try:
                # ESPN sometimes exposes date string under competitions[0]["date"] (UTC ISO)
                comp_date = comps.get("date")
                if comp_date:
                    start_time = dt.datetime.fromisoformat(comp_date.replace("Z", "+00:00"))
                    # Localize to schedule timezone for midnight drift handling
                    sched_tz = os.getenv("SCHEDULE_TZ") or "America/New_York"
                    try:
                        local_dt = start_time.astimezone(ZoneInfo(sched_tz))
                        start_time_local = local_dt.strftime("%Y-%m-%d %H:%M")
                        start_tz_abbr = local_dt.tzname()
                    except Exception:
                        start_time_local = None
                        start_tz_abbr = None
            except Exception:
                start_time = None
                start_time_local = None
                start_tz_abbr = None

            games.append(
                Game(
                    game_id=game_id,
                    season=date.year,
                    date=dt.datetime.combine(date, dt.time(0, 0)),
                    start_time=start_time,
                    start_time_local=start_time_local,
                    start_tz_abbr=start_tz_abbr,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    home_score_1h=home_1h,
                    away_score_1h=away_1h,
                    home_score_2h=home_2h,
                    away_score_2h=away_2h,
                    status=status_name,
                    completed=completed,
                    neutral_site=bool(neutral_site) if neutral_site is not None else None,
                    venue=venue_name,
                )
            )
        except Exception:
            continue
    return games


def _parse_schedule_games(date: dt.date, payload: dict) -> List[Game]:
    games: List[Game] = []
    events_by_date = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events_by_date, dict):
        return games
    date_key = date.strftime("%Y%m%d")
    events = events_by_date.get(date_key) or []
    for ev in events:
        try:
            if not isinstance(ev, dict):
                continue
            game_id = str(ev.get("id") or f"{date.isoformat()}-{len(games)}")
            competitors = ev.get("competitors") or ev.get("teams") or []
            home = next((c for c in competitors if bool(c.get("isHome"))), None)
            away = next((c for c in competitors if not bool(c.get("isHome"))), None)
            if (not home or not away) and len(competitors) >= 2:
                home = home or competitors[0]
                away = away or competitors[1]
            if not home or not away:
                continue

            def parse_int(x):
                try:
                    return int(x) if x is not None and str(x).strip() != "" else None
                except Exception:
                    return None

            home_team = (
                home.get("displayName")
                or home.get("name")
                or home.get("shortName")
                or home.get("abbrev")
                or "HOME"
            )
            away_team = (
                away.get("displayName")
                or away.get("name")
                or away.get("shortName")
                or away.get("abbrev")
                or "AWAY"
            )

            start_time = None
            start_time_local = None
            start_tz_abbr = None
            try:
                raw_date = ev.get("date")
                if raw_date:
                    start_time = dt.datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                    sched_tz = os.getenv("SCHEDULE_TZ") or "America/New_York"
                    try:
                        local_dt = start_time.astimezone(ZoneInfo(sched_tz))
                        start_time_local = local_dt.strftime("%Y-%m-%d %H:%M")
                        start_tz_abbr = local_dt.tzname()
                    except Exception:
                        start_time_local = None
                        start_tz_abbr = None
            except Exception:
                start_time = None
                start_time_local = None
                start_tz_abbr = None

            venue_name = None
            try:
                venue = ev.get("venue") or {}
                full_name = venue.get("fullName")
                address = venue.get("address") or {}
                city = address.get("city")
                state = address.get("state")
                if full_name and city and state:
                    suffix = f"{city}, {state}"
                    venue_name = full_name if suffix in str(full_name) else f"{full_name}, {suffix}"
                else:
                    venue_name = full_name or city
            except Exception:
                venue_name = None

            status_name = None
            completed = ev.get("completed")
            try:
                status = ev.get("status") or {}
                if isinstance(status, dict):
                    status_name = status.get("detail") or status.get("state") or status.get("id")
                    if completed is None:
                        state = str(status.get("state") or "").strip().lower()
                        if state:
                            completed = state in {"post", "final", "complete", "completed"}
            except Exception:
                status_name = None

            season = (ev.get("season") or {}).get("year") or date.year
            games.append(
                Game(
                    game_id=game_id,
                    season=int(season),
                    date=dt.datetime.combine(date, dt.time(0, 0)),
                    start_time=start_time,
                    start_time_local=start_time_local,
                    start_tz_abbr=start_tz_abbr,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=parse_int(home.get("score")),
                    away_score=parse_int(away.get("score")),
                    status=status_name,
                    completed=completed,
                    neutral_site=bool(ev.get("neutralSite")) if ev.get("neutralSite") is not None else None,
                    venue=venue_name,
                )
            )
        except Exception:
            continue
    return games


def iter_games_by_date(start: dt.date, end: dt.date, use_cache: bool = True, cache_only: bool = False) -> Iterable[FetchResult]:
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        payload = _fetch_day(cur, use_cache=use_cache, cache_only=cache_only)
        games = _parse_games(cur, payload) if payload is not None else []
        src = "cache" if cache_path("espn", f"{cur.isoformat()}.json").exists() else "network"
        if not games:
            sched_payload = _fetch_schedule_day(cur, use_cache=use_cache, cache_only=cache_only)
            sched_games = _parse_schedule_games(cur, sched_payload) if sched_payload is not None else []
            if sched_games:
                games = sched_games
                src = "cache" if cache_path("espn_schedule", f"{cur.isoformat()}.json").exists() else "network"
        if not games:
            yield FetchResult(cur, [], source="none")
        else:
            yield FetchResult(cur, games, src)
        cur += one
