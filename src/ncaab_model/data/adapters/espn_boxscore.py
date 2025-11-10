from __future__ import annotations

import datetime as dt
from typing import Optional, Iterable, List
import requests
import numpy as np

from ..cache import cache_path, read_json, write_json
from ..schemas import BoxScoreRow

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={EVENT_ID}"
)


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _get_stat(stats: list[dict], name: str) -> Optional[float]:
    # ESPN uses ids and displayNames; try both
    for s in stats or []:
        key = (s.get("name") or s.get("id") or s.get("displayName") or "").lower()
        if name in key:
            return _safe_float(s.get("value"))
    return None


def _compute_four_factors(team_stats: dict, opp_stats: dict) -> dict:
    # Extract core counts
    fga = _get_stat(team_stats, "fieldgoalsattempted") or 0.0
    fgm = _get_stat(team_stats, "fieldgoalsmade") or 0.0
    tpa = _get_stat(team_stats, "threepointfieldgoalsattempted") or 0.0
    tpm = _get_stat(team_stats, "threepointfieldgoalsmade") or 0.0
    fta = _get_stat(team_stats, "freethrowattempts") or 0.0
    ftm = _get_stat(team_stats, "freethrowsmade") or 0.0
    orb = _get_stat(team_stats, "offensiverebounds") or 0.0
    to = _get_stat(team_stats, "turnovers") or 0.0

    opp_drb = (_get_stat(opp_stats, "defensiverebounds") or 0.0)

    # Possessions estimate (Dean Oliver commonly 0.475 or 0.44 multiplier for FTA)
    ft_mult = 0.475
    poss = fga - orb + to + ft_mult * fta

    # Four factors
    # Effective FG%: (FGM + 0.5*3PM) / FGA
    efg = (fgm + 0.5 * tpm) / fga if fga > 0 else None
    # Turnover rate: TO / Possessions
    tov_rate = to / poss if poss > 0 else None
    # Offensive rebound rate: ORB / (ORB + Opp DRB)
    orb_rate = orb / (orb + opp_drb) if (orb + opp_drb) > 0 else None
    # Free throw rate: FTM / FGA (or FTA/FGA variant). We'll use FTA/FGA here.
    ftr = fta / fga if fga > 0 else None

    return {
        "poss": poss if poss is not None else None,
        "efg": efg,
        "tov_rate": tov_rate,
        "orb_rate": orb_rate,
        "ftr": ftr,
    }


def fetch_boxscore(event_id: str, use_cache: bool = True) -> Optional[BoxScoreRow]:
    cache_file = cache_path("espn_summary", f"{event_id}.json")
    data = None
    if use_cache and cache_file.exists():
        try:
            data = read_json(cache_file)
        except Exception:
            data = None
    if data is None:
        url = SUMMARY_URL.format(EVENT_ID=event_id)
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            write_json(cache_file, data)
        except Exception:
            return None

    # Parse teams and statistics
    comps = (data.get("boxscore") or {}).get("teams") or []
    # Fallback from summary to gameInfo teams if needed
    if len(comps) < 2:
        # ESPN sometimes nests under "competitors" in summary.competitions
        comps = ((data.get("summary") or {}).get("competitions") or [{}])[0].get("competitors") or []

    if len(comps) < 2:
        return None

    # Identify home/away blocks
    def is_home(block: dict) -> bool:
        return (block.get("homeAway") or block.get("homeAwayTeamId") or "").lower() == "home"

    # Normalize to dicts with statistics list
    def to_team_stats(block: dict) -> tuple[str, dict]:
        team = (block.get("team") or {})
        name = team.get("displayName") or team.get("shortDisplayName") or team.get("name") or ""
        stats = block.get("statistics") or []
        return name, {"statistics": stats}

    # ESPN boxscore.teams ordering varies; capture both and then assign based on homeAway
    b0, b1 = comps[0], comps[1]
    # Determine home/away names from summary competitions if available
    try:
        comp0 = ((data.get("summary") or {}).get("competitions") or [{}])[0]
        competitors = comp0.get("competitors", [])
        home_comp = next((c for c in competitors if (c.get("homeAway") or "").lower() == "home"), None)
        away_comp = next((c for c in competitors if (c.get("homeAway") or "").lower() == "away"), None)
        home_name = (home_comp or {}).get("team", {}).get("displayName")
        away_name = (away_comp or {}).get("team", {}).get("displayName")
        # Parse final scores if present
        def _score_from(c):
            try:
                s = c.get("score")
                return float(s) if s is not None and str(s).strip() != "" else None
            except Exception:
                return None
        home_score = _score_from(home_comp) if home_comp else None
        away_score = _score_from(away_comp) if away_comp else None
    except Exception:
        home_name = away_name = None
        home_score = away_score = None

    # Build stat dicts
    name0, ts0 = to_team_stats(b0)
    name1, ts1 = to_team_stats(b1)

    # Map home/away by best-effort names
    # If summary provided explicit home/away names, use those
    if home_name and away_name:
        if name0 == home_name:
            home_stats, away_stats = ts0, ts1
            home_team, away_team = home_name, away_name
        else:
            home_stats, away_stats = ts1, ts0
            home_team, away_team = home_name, away_name
    else:
        # Fallback to b0 homeAway flag
        if is_home(b0):
            home_stats, away_stats = ts0, ts1
            home_team, away_team = name0, name1
        else:
            home_stats, away_stats = ts1, ts0
            home_team, away_team = name1, name0

    # Compute four factors and possessions
    home_ff = _compute_four_factors(home_stats["statistics"], away_stats["statistics"])
    away_ff = _compute_four_factors(away_stats["statistics"], home_stats["statistics"])

    # Pace as average of team possessions
    poss_vals = [v for v in (home_ff["poss"], away_ff["poss"]) if v is not None]
    pace = float(np.mean(poss_vals)) if poss_vals else None

    # Parse halftime line scores if present
    home_score_1h = away_score_1h = None
    home_score_2h = away_score_2h = None
    try:
        comp_root = ((data.get("summary") or {}).get("competitions") or [{}])[0]
        competitors_ls = comp_root.get("competitors", [])
        # Each competitor may have linescores list: [{"value": <points>}, ...] per period
        def _linescores(block: dict) -> list[float]:
            out: list[float] = []
            for ls in block.get("linescores", []) or []:
                v = ls.get("value")
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out
        home_block = next((c for c in competitors_ls if (c.get("homeAway") or '').lower() == 'home'), None)
        away_block = next((c for c in competitors_ls if (c.get("homeAway") or '').lower() == 'away'), None)
        if home_block and away_block:
            h_ls = _linescores(home_block)
            a_ls = _linescores(away_block)
            # NCAAB regulation: 2 periods (halves). If more entries (OT), treat first two as halves; sum first half periods if split further.
            if h_ls and a_ls:
                # First half = first period value (or sum of first two if ESPN splits by quarters in some tournaments)
                # Heuristic: if there are >2 entries and competition "format" indicates quarters, sum half as first half of game duration.
                # Simplify: halftime = sum of first half of entries; second half = sum remaining regulation entries excluding OT.
                # Detect OT entries: if total of linescores equals final score, we can divide by half count; simplest: assume first len(ls)//2 entries = 1H for 2, else first 1 for 2 periods.
                if len(h_ls) == 2 and len(a_ls) == 2:
                    home_score_1h = h_ls[0]; away_score_1h = a_ls[0]
                    home_score_2h = h_ls[1]; away_score_2h = a_ls[1]
                elif len(h_ls) > 2:
                    # If there are exactly 3 (OT), treat first as 1H, second as 2H, ignore OT
                    if len(h_ls) == 3:
                        home_score_1h = h_ls[0]; away_score_1h = a_ls[0]
                        home_score_2h = h_ls[1]; away_score_2h = a_ls[1]
                    else:
                        # Sum first half of regulation periods as 1H (e.g., 2 quarters), remainder as 2H until possible OT segments
                        # Assume last entries beyond 2* (quarters) indicate OT; limit to first 4 entries if quarters shown.
                        reg = h_ls[:4]; reg_a = a_ls[:4]
                        half_split = len(reg) // 2
                        home_score_1h = sum(reg[:half_split]); away_score_1h = sum(reg_a[:half_split])
                        home_score_2h = sum(reg[half_split:]); away_score_2h = sum(reg_a[half_split:])
    except Exception:
        pass

    # Parse date if present
    try:
        date_str = ((data.get("summary") or {}).get("header") or {}).get("competitions", [{}])[0].get("date")
        date = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else None
    except Exception:
        date = None

    return BoxScoreRow(
        game_id=str(event_id),
        date=date,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        home_possessions=home_ff["poss"],
        away_possessions=away_ff["poss"],
        pace=pace,
        home_efg=home_ff["efg"],
        home_tov_rate=home_ff["tov_rate"],
        home_orb_rate=home_ff["orb_rate"],
        home_ftr=home_ff["ftr"],
        away_efg=away_ff["efg"],
        away_tov_rate=away_ff["tov_rate"],
        away_orb_rate=away_ff["orb_rate"],
        away_ftr=away_ff["ftr"],
        home_score_1h=home_score_1h,
        away_score_1h=away_score_1h,
        home_score_2h=home_score_2h,
        away_score_2h=away_score_2h,
    )


def iter_boxscores(event_ids: Iterable[str], use_cache: bool = True) -> Iterable[BoxScoreRow | None]:
    for eid in event_ids:
        yield fetch_boxscore(str(eid), use_cache=use_cache)
