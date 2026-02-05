from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

from ..cache import cache_path, read_json, write_json


PBP_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/playbyplay?event={EVENT_ID}"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={EVENT_ID}"


@dataclass(frozen=True)
class CumTotals:
    # cumulative minutes elapsed from start (regulation 5..40, OT 45+ when requested)
    end_min: int
    home_score: int
    away_score: int

    @property
    def total_score(self) -> int:
        return int(self.home_score + self.away_score)


def fetch_playbyplay(event_id: str, use_cache: bool = True) -> Optional[dict]:
    """Fetch ESPN play-by-play JSON for an event.

    Notes:
    - Uses data/cache/espn_pbp/<event_id>.json
    - Returns None on persistent failure.
    """
    eid = str(event_id)
    cache_file = cache_path("espn_pbp", f"{eid}.json")
    summary_cache_file = cache_path("espn_summary", f"{eid}.json")

    def _has_plays(d: object) -> bool:
        if not isinstance(d, dict):
            return False
        plays = d.get("plays")
        return isinstance(plays, list) and len(plays) > 0

    if use_cache and cache_file.exists():
        try:
            cached = read_json(cache_file)
            # If the cached payload is empty/malformed (common for the PBP endpoint),
            # fall through to refresh from network (prefer summary fallback).
            if _has_plays(cached):
                try:
                    cached["_fetched_from"] = "cache_pbp"
                except Exception:
                    pass
                return cached
        except Exception:
            pass

    # Many flows in this repo already cache ESPN summary payloads (boxscore) which
    # include a populated `plays` list. Reuse it to avoid extra network calls.
    if use_cache and summary_cache_file.exists():
        try:
            cached_summary = read_json(summary_cache_file)
            if _has_plays(cached_summary):
                # Also populate the espn_pbp cache for consistency.
                try:
                    write_json(cache_file, cached_summary)
                except Exception:
                    pass
                payload = dict(cached_summary)
                payload["_fetched_from"] = "cache_summary"
                return payload
        except Exception:
            pass

    max_attempts = 4
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.espn.com/",
        "Origin": "https://www.espn.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def _fetch_json(url: str) -> Optional[dict]:
        data: dict | None = None
        for attempt in range(max_attempts):
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if getattr(r, "status_code", None) == 429:
                    try:
                        retry_after = int(r.headers.get("Retry-After", "0") or 0)
                    except Exception:
                        retry_after = 0
                    wait_s = float(retry_after) if retry_after > 0 else float(1.5 * (2**attempt))
                    time.sleep(min(wait_s, 30.0))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt < (max_attempts - 1):
                    time.sleep(float(1.5 * (2**attempt)))
                    continue
                return None
        return data if isinstance(data, dict) else None

    # Try the dedicated PBP endpoint first; if it returns {} or lacks plays,
    # fall back to the summary endpoint (which includes a populated plays list).
    fetched_from = "network_pbp"
    data = _fetch_json(PBP_URL.format(EVENT_ID=eid))
    if not _has_plays(data):
        fetched_from = "network_summary"
        data = _fetch_json(SUMMARY_URL.format(EVENT_ID=eid))

    if not isinstance(data, dict):
        return None

    try:
        write_json(cache_file, data)
    except Exception:
        pass

    payload = dict(data)
    payload["_fetched_from"] = fetched_from
    return payload


def _iter_plays(payload: dict) -> Iterable[dict]:
    # ESPN formats vary; most commonly: payload["plays"] is list
    plays = payload.get("plays")
    if isinstance(plays, list):
        for p in plays:
            if isinstance(p, dict):
                yield p
        return

    # Fallback: some payloads have nested drives/competitions (rare for basketball)
    for k in ("items", "events"):
        v = payload.get(k)
        if isinstance(v, list):
            for p in v:
                if isinstance(p, dict):
                    yield p


def _clock_to_remaining_seconds(clock_display: object) -> Optional[int]:
    """Parse 'MM:SS' into remaining seconds in the period."""
    if clock_display is None:
        return None
    s = str(clock_display).strip()
    if not s:
        return None
    # Already seconds?
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    if ":" not in s:
        return None
    try:
        mm, ss = s.split(":", 1)
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def _play_period_num(p: dict) -> Optional[int]:
    try:
        per = p.get("period") or {}
        period_num = per.get("number") or per.get("value") or per.get("period")
        return int(period_num)
    except Exception:
        return None


def _play_elapsed_min(p: dict) -> Optional[float]:
    """Return minutes elapsed from start (regulation + OT).

    ESPN uses 20-minute halves (period 1,2) and 5-minute overtime periods (3,4,...).
    We map elapsed as:
      - period 1: [0,20]
      - period 2: [20,40]
      - OT1 (period 3): [40,45]
      - OT2 (period 4): [45,50]
      - ...
    """
    period_num = _play_period_num(p)
    if period_num is None or period_num < 1:
        return None

    if period_num in (1, 2):
        period_len_min = 20.0
        base_elapsed = 20.0 * float(period_num - 1)
    else:
        # Overtime periods are 5 minutes.
        period_len_min = 5.0
        base_elapsed = 40.0 + 5.0 * float(period_num - 3)

    try:
        clock = p.get("clock") or {}
        disp = clock.get("displayValue") if isinstance(clock, dict) else None
        rem_sec = _clock_to_remaining_seconds(disp)
    except Exception:
        rem_sec = None

    if rem_sec is None:
        # Occasionally ESPN provides "clock" as string
        rem_sec = _clock_to_remaining_seconds(p.get("clock"))

    if rem_sec is None:
        return None

    # elapsed_in_period is clamped to [0, period_len_min]
    elapsed_in_period = float(period_len_min) - (float(rem_sec) / 60.0)
    if not math.isfinite(elapsed_in_period):
        return None
    elapsed_in_period = float(min(max(elapsed_in_period, 0.0), period_len_min))

    elapsed = float(base_elapsed + elapsed_in_period)
    if elapsed < 0.0:
        return None

    # Allow a generous upper bound for multi-OT games.
    if elapsed > 80.5:
        return None
    return float(elapsed)


def infer_ot_periods(payload: dict) -> int:
    """Infer number of OT periods present in an ESPN play-by-play payload."""
    max_period = 0
    try:
        for p in _iter_plays(payload):
            per = _play_period_num(p)
            if per is not None and per > max_period:
                max_period = int(per)
    except Exception:
        max_period = 0

    if max_period <= 2:
        return 0
    return int(max_period - 2)


def extract_cum_totals_5min(payload: dict, endpoints: list[int] | None = None) -> list[CumTotals]:
    """Extract cumulative home/away totals at 5-min endpoints from ESPN PBP.

    Strategy: scan plays in order; whenever a play includes homeScore/awayScore and
    a valid clock/period, update the 'current' score at its elapsed timestamp.

    For each endpoint T, we take the latest score with elapsed <= T.
    """
    if endpoints is None:
        endpoints = [5, 10, 15, 20, 25, 30, 35, 40]

    # Track timeline of (elapsed, home, away)
    timeline: list[tuple[float, int, int]] = [(0.0, 0, 0)]

    for p in _iter_plays(payload):
        elapsed = _play_elapsed_min(p)
        if elapsed is None:
            continue

        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                # Some formats nest under "scoreValue" or "score"; ignore if missing
                continue
            home = int(hs)
            away = int(a_s)
        except Exception:
            continue

        timeline.append((float(elapsed), home, away))

    # Ensure monotonic sort (some payloads aren't strictly ordered)
    timeline.sort(key=lambda t: t[0])

    out: list[CumTotals] = []
    idx = 0
    cur_home, cur_away = 0, 0

    for end_min in endpoints:
        # Move idx forward while elapsed <= end_min
        while idx < len(timeline) and timeline[idx][0] <= float(end_min) + 1e-9:
            _, cur_home, cur_away = timeline[idx]
            idx += 1
        out.append(CumTotals(end_min=int(end_min), home_score=int(cur_home), away_score=int(cur_away)))

    return out
