from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class PbpCacheIndex:
    by_date: dict[str, list[str]]
    scanned_files: int
    used_files: int
    errors: int


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _payload_date_utc(payload: dict[str, Any]) -> Optional[dt.date]:
    """Infer event UTC date from ESPN PBP payload.

    Typical shape: payload['header']['competitions'][0]['date'] == 'YYYY-MM-DDTHH:MMZ'
    """
    try:
        header = payload.get("header") or {}
        comps = header.get("competitions") or []
        if not isinstance(comps, list) or not comps:
            return None
        c0 = comps[0] if isinstance(comps[0], dict) else None
        if not isinstance(c0, dict):
            return None
        s = c0.get("date")
        if s is None:
            return None
        s2 = str(s).strip()
        if not s2:
            return None
        # Handle trailing 'Z'
        if s2.endswith("Z"):
            s2 = s2[:-1] + "+00:00"
        t = dt.datetime.fromisoformat(s2)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc).date()
    except Exception:
        return None


def index_pbp_cache_by_date(
    *,
    cache_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    max_files: int = 0,
) -> PbpCacheIndex:
    """Scan data/cache/espn_pbp and group event_ids by UTC event date."""
    by_date: dict[str, list[str]] = {}
    scanned = 0
    used = 0
    errors = 0

    files = list(cache_dir.glob("*.json"))
    if max_files and max_files > 0:
        files = files[: int(max_files)]

    for p in files:
        scanned += 1
        try:
            eid = p.stem
            payload = _safe_read_json(p)
            if not isinstance(payload, dict):
                continue
            d = _payload_date_utc(payload)
            if d is None:
                continue
            if d < start_date or d > end_date:
                continue
            ds = d.isoformat()
            by_date.setdefault(ds, []).append(str(eid))
            used += 1
        except Exception:
            errors += 1
            continue

    return PbpCacheIndex(by_date=by_date, scanned_files=int(scanned), used_files=int(used), errors=int(errors))


def _iter_plays(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    plays = payload.get("plays")
    if isinstance(plays, list):
        for p in plays:
            if isinstance(p, dict):
                yield p


def _clock_to_remaining_seconds(clock_display: object) -> Optional[int]:
    if clock_display is None:
        return None
    s = str(clock_display).strip()
    if not s:
        return None
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


def _play_period_num(p: dict[str, Any]) -> Optional[int]:
    try:
        per = p.get("period") or {}
        if isinstance(per, dict):
            period_num = per.get("number") or per.get("value") or per.get("period")
        else:
            period_num = None
        return int(period_num)
    except Exception:
        return None


def _play_elapsed_min(p: dict[str, Any]) -> Optional[float]:
    """Return minutes elapsed from start for NCAAB (20-min halves; 5-min OT)."""
    period_num = _play_period_num(p)
    if period_num is None or period_num < 1:
        return None

    if period_num in (1, 2):
        period_len_min = 20.0
        base_elapsed = 20.0 * float(period_num - 1)
    else:
        period_len_min = 5.0
        base_elapsed = 40.0 + 5.0 * float(period_num - 3)

    rem_sec = None
    try:
        clock = p.get("clock") or {}
        if isinstance(clock, dict):
            rem_sec = _clock_to_remaining_seconds(clock.get("displayValue"))
    except Exception:
        rem_sec = None

    if rem_sec is None:
        rem_sec = _clock_to_remaining_seconds(p.get("clock"))
    if rem_sec is None:
        return None

    elapsed_in_period = float(period_len_min) - (float(rem_sec) / 60.0)
    if elapsed_in_period < 0.0:
        elapsed_in_period = 0.0
    if elapsed_in_period > period_len_min:
        elapsed_in_period = float(period_len_min)

    elapsed = float(base_elapsed + elapsed_in_period)
    if elapsed < 0.0 or elapsed > 80.5:
        return None
    return float(elapsed)


def _extract_scores_at_endpoints(payload: dict[str, Any], endpoints: list[int]) -> dict[int, tuple[int, int]]:
    """Return {end_min: (home_score, away_score)} at each endpoint."""
    timeline: list[tuple[float, int, int]] = [(0.0, 0, 0)]
    for p in _iter_plays(payload):
        elapsed = _play_elapsed_min(p)
        if elapsed is None:
            continue
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            home = int(hs)
            away = int(a_s)
        except Exception:
            continue
        timeline.append((float(elapsed), int(home), int(away)))
    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass

    out: dict[int, tuple[int, int]] = {}
    idx = 0
    cur_home = 0
    cur_away = 0
    for end_min in endpoints:
        while idx < len(timeline) and timeline[idx][0] <= float(end_min) + 1e-9:
            _, cur_home, cur_away = timeline[idx]
            idx += 1
        out[int(end_min)] = (int(cur_home), int(cur_away))
    return out


def _extract_shot_proxy_at_endpoints(
    payload: dict[str, Any],
    endpoints: list[int],
    *,
    ft_weight: float = 0.44,
) -> dict[int, float]:
    """Approximate 'possessions so far' as shot_proxy = FGA + ft_weight * FTA."""
    fga = 0
    fta = 0
    timeline: list[tuple[float, int, int]] = [(0.0, 0, 0)]

    for p in _iter_plays(payload):
        elapsed = _play_elapsed_min(p)
        if elapsed is None:
            continue

        try:
            if bool(p.get("shootingPlay")):
                pa = p.get("pointsAttempted")
                if pa is not None:
                    pa_i = int(pa)
                    if pa_i == 1:
                        fta += 1
                    elif pa_i in (2, 3):
                        fga += 1
        except Exception:
            pass

        timeline.append((float(elapsed), int(fga), int(fta)))

    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass

    out: dict[int, float] = {}
    idx = 0
    cur_fga = 0
    cur_fta = 0
    for end_min in endpoints:
        while idx < len(timeline) and timeline[idx][0] <= float(end_min) + 1e-9:
            _, cur_fga, cur_fta = timeline[idx]
            idx += 1
        shot_proxy = float(cur_fga) + float(ft_weight) * float(cur_fta)
        out[int(end_min)] = float(max(0.0, shot_proxy))
    return out


def _home_away_team_ids(payload: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    try:
        header = payload.get("header") or {}
        comps = header.get("competitions") or []
        if not isinstance(comps, list) or not comps:
            return (None, None)
        c0 = comps[0] if isinstance(comps[0], dict) else None
        if not isinstance(c0, dict):
            return (None, None)
        competitors = c0.get("competitors") or []
        if not isinstance(competitors, list):
            return (None, None)
        home_id = None
        away_id = None
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            ha = str(comp.get("homeAway") or "").strip().lower()
            team = comp.get("team") or {}
            tid = None
            if isinstance(team, dict):
                tid = team.get("id")
            tid_s = str(tid).strip() if tid is not None else ""
            if not tid_s:
                continue
            if ha == "home":
                home_id = tid_s
            elif ha == "away":
                away_id = tid_s
        return (home_id, away_id)
    except Exception:
        return (None, None)


def _extract_pbp_counters_at_endpoints(
    payload: dict[str, Any],
    endpoints: list[int],
) -> dict[int, dict[str, int]]:
    """Return {end_min: {home_to, away_to, home_orb, away_orb, home_drb, away_drb}}."""
    home_id, away_id = _home_away_team_ids(payload)

    def side_from_team_id(team_id: Optional[str]) -> Optional[str]:
        if team_id is None:
            return None
        tid = str(team_id).strip()
        if not tid:
            return None
        if home_id and tid == str(home_id):
            return "home"
        if away_id and tid == str(away_id):
            return "away"
        return None

    h_to = a_to = 0
    h_orb = a_orb = 0
    h_drb = a_drb = 0
    timeline: list[tuple[float, int, int, int, int, int, int]] = [(0.0, 0, 0, 0, 0, 0, 0)]

    for p in _iter_plays(payload):
        elapsed = _play_elapsed_min(p)
        if elapsed is None:
            continue

        team = p.get("team") or {}
        team_id = None
        if isinstance(team, dict):
            team_id = team.get("id")
        side = side_from_team_id(str(team_id).strip() if team_id is not None else None)

        t = p.get("type") or {}
        type_text = None
        if isinstance(t, dict):
            type_text = t.get("text")
        type_l = str(type_text or "").strip().lower()
        text_l = str(p.get("text") or "").strip().lower()

        is_to = ("turnover" in type_l) or ("turnover" in text_l)
        is_orb = "offensive rebound" in type_l or "offensive rebound" in text_l
        is_drb = ("defensive rebound" in type_l or "defensive rebound" in text_l) or (
            "dead ball rebound" in type_l or "dead ball rebound" in text_l
        )

        if side == "home":
            if is_to:
                h_to += 1
            if is_orb:
                h_orb += 1
            if is_drb:
                h_drb += 1
        elif side == "away":
            if is_to:
                a_to += 1
            if is_orb:
                a_orb += 1
            if is_drb:
                a_drb += 1

        timeline.append((float(elapsed), int(h_to), int(a_to), int(h_orb), int(a_orb), int(h_drb), int(a_drb)))

    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass

    out: dict[int, dict[str, int]] = {}
    idx = 0
    cur = (0, 0, 0, 0, 0, 0)
    for end_min in endpoints:
        while idx < len(timeline) and timeline[idx][0] <= float(end_min) + 1e-9:
            _, c_ht, c_at, c_horb, c_aorb, c_hdrb, c_adrb = timeline[idx]
            cur = (int(c_ht), int(c_at), int(c_horb), int(c_aorb), int(c_hdrb), int(c_adrb))
            idx += 1
        ht, at, horb, aorb, hdrb, adrb = cur
        out[int(end_min)] = {
            "home_to": int(ht),
            "away_to": int(at),
            "home_orb": int(horb),
            "away_orb": int(aorb),
            "home_drb": int(hdrb),
            "away_drb": int(adrb),
        }
    return out


def _coerce_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return float(v)
    except Exception:
        return None


def _extract_total_line(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a best-effort total line from ESPN payload.

    Returns a dict compatible with build_live_feature_table's `last_line` schema:
      {total, book, event_id_provider, last_update}
    """
    # Prefer pickcenter (common for pregame totals)
    try:
        pc = payload.get("pickcenter")
        if isinstance(pc, list) and pc:
            pc0 = pc[0] if isinstance(pc[0], dict) else None
            if isinstance(pc0, dict):
                total = _coerce_float(pc0.get("overUnder"))
                if total is None:
                    total = _coerce_float(pc0.get("total"))
                provider = pc0.get("provider") or {}
                book = provider.get("name") if isinstance(provider, dict) else None
                provider_id = provider.get("id") if isinstance(provider, dict) else None
                last_update = pc0.get("lastUpdated") or pc0.get("updateTime")
                if total is not None:
                    return {
                        "total": float(total),
                        "book": book,
                        "event_id_provider": provider_id,
                        "last_update": last_update,
                    }
    except Exception:
        pass

    # Fallback to odds list (less common in these cached payloads)
    try:
        odds = payload.get("odds")
        if isinstance(odds, list) and odds:
            for o in odds:
                if not isinstance(o, dict):
                    continue
                total = _coerce_float(o.get("overUnder"))
                if total is None:
                    total = _coerce_float(o.get("total"))
                if total is None:
                    continue
                provider = o.get("provider") or {}
                book = provider.get("name") if isinstance(provider, dict) else o.get("book")
                provider_id = provider.get("id") if isinstance(provider, dict) else None
                last_update = o.get("lastUpdated") or o.get("lastUpdate")
                return {
                    "total": float(total),
                    "book": book,
                    "event_id_provider": provider_id,
                    "last_update": last_update,
                }
    except Exception:
        pass

    return {}


def write_synthetic_live_snapshots_jsonl_from_pbp_cache(
    *,
    date: str,
    event_ids: list[str],
    cache_dir: Path,
    out_jsonl: Path,
    endpoints_min: list[int] | None = None,
    ft_weight: float = 0.44,
) -> dict[str, Any]:
    """Write a minimal Live Lens JSONL from ESPN cached PBP.

    Output contains alternating records:
      - endpoint=live_pbp_stats (poss_est proxy)
      - endpoint=live_state (score + remaining_reg_seconds)

    This is intended for offline dataset backfills when real Render snapshots are missing.
    """
    if endpoints_min is None:
        endpoints_min = [5, 10, 15, 20, 25, 30, 35, 40]

    # Ensure deterministic ordering
    event_ids = sorted({str(x).strip() for x in event_ids if str(x).strip()})

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    base_ts = dt.datetime.fromisoformat(f"{date}T00:00:00+00:00")

    written_lines = 0
    used_games = 0
    skipped_games = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for eid in event_ids:
            p = cache_dir / f"{eid}.json"
            payload = _safe_read_json(p)
            if not isinstance(payload, dict):
                skipped_games += 1
                continue
            # Needs plays to be meaningful
            if not isinstance(payload.get("plays"), list) or len(payload.get("plays") or []) == 0:
                skipped_games += 1
                continue

            scores = _extract_scores_at_endpoints(payload, endpoints_min)
            shot_proxy = _extract_shot_proxy_at_endpoints(payload, endpoints_min, ft_weight=float(ft_weight))
            counters = _extract_pbp_counters_at_endpoints(payload, endpoints_min)
            line = _extract_total_line(payload)
            fetched_from = str(payload.get("_fetched_from") or "cache")

            for end_min in endpoints_min:
                end_i = int(end_min)
                hs, as_ = scores.get(end_i, (0, 0))
                total_points = int(hs) + int(as_)
                rem_reg_seconds = int(max(0, (40 - end_i) * 60))
                period = 1 if end_i <= 20 else 2
                ts = (base_ts + dt.timedelta(minutes=end_i)).isoformat().replace("+00:00", "Z")

                poss_est_total = float(shot_proxy.get(end_i, 0.0))
                half = float(poss_est_total / 2.0) if poss_est_total > 0 else 0.0

                # Emit a `live_lines` record so build_live_feature_table can populate live_line_* columns.
                if line:
                    live_lines_rec = {
                        "ts": ts,
                        "endpoint": "live_lines",
                        "event_id": str(eid),
                        "data": dict(line),
                    }
                    f.write(json.dumps(live_lines_rec, ensure_ascii=False) + "\n")
                    written_lines += 1

                c = counters.get(end_i) or {}

                pbp_stats_rec = {
                    "ts": ts,
                    "endpoint": "live_pbp_stats",
                    "event_id": str(eid),
                    "data": {
                        "stats": {
                            "home": {
                                "poss_est": half,
                                "to": int(c.get("home_to") or 0),
                                "orb": int(c.get("home_orb") or 0),
                                "drb": int(c.get("home_drb") or 0),
                            },
                            "away": {
                                "poss_est": half,
                                "to": int(c.get("away_to") or 0),
                                "orb": int(c.get("away_orb") or 0),
                                "drb": int(c.get("away_drb") or 0),
                            },
                        },
                        "fetched_from": fetched_from,
                    },
                }
                f.write(json.dumps(pbp_stats_rec, ensure_ascii=False) + "\n")
                written_lines += 1

                live_state_rec = {
                    "ts": ts,
                    "endpoint": "live_state",
                    "event_id": str(eid),
                    "data": {
                        "period": int(period),
                        "remaining_reg_seconds": int(rem_reg_seconds),
                        "home_score": int(hs),
                        "away_score": int(as_),
                        "total_points": int(total_points),
                    },
                }
                f.write(json.dumps(live_state_rec, ensure_ascii=False) + "\n")
                written_lines += 1

            used_games += 1

    return {
        "status": "ok",
        "date": str(date),
        "out_jsonl": str(out_jsonl),
        "games": int(used_games),
        "skipped_games": int(skipped_games),
        "lines": int(written_lines),
        "endpoints": list(map(int, endpoints_min)),
    }
