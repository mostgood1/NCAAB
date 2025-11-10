from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import math
import requests
import pandas as pd

from .merge_odds import normalize_name


ESPN_TEAMS_URL = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"


def _hex_to_rgb(hexstr: str | None) -> tuple[int, int, int] | None:
    if not hexstr:
        return None
    s = str(hexstr).strip().lstrip("#").lower()
    if len(s) not in (3, 6):
        return None
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return None


def _luminance(rgb: tuple[int, int, int]) -> float:
    # Relative luminance per WCAG
    def _chan(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def _auto_text_color(bg_hex: str | None) -> str:
    rgb = _hex_to_rgb(bg_hex)
    if rgb is None:
        return "#ffffff"
    lum = _luminance(rgb)
    # Contrast vs white and black; choose the higher contrast
    # Relative luminance for white=1.0, black=0.0
    contrast_white = (1.0 + 0.05) / (lum + 0.05)
    contrast_black = (lum + 0.05) / (0.0 + 0.05)
    return "#ffffff" if contrast_white >= contrast_black else "#000000"


def fetch_espn_branding(timeout: int = 20) -> pd.DataFrame:
    """Fetch all Division I MBB teams from ESPN with logos and colors.

    Returns a DataFrame with columns: team, logo, primary_color, secondary_color, text_color, espn_id, abbreviation, conference
    """
    resp = requests.get(ESPN_TEAMS_URL, params={"limit": 500}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # ESPN shape: sports[0].leagues[0].teams = [{ team: {...} }, ...]
    teams: List[Dict[str, Any]] = []
    try:
        sports = data.get("sports", [])
        leagues = sports[0].get("leagues", []) if sports else []
        raw_teams = leagues[0].get("teams", []) if leagues else []
        for item in raw_teams:
            t = item.get("team", {})
            if not t:
                continue
            teams.append(t)
    except Exception:
        # Fallback: sometimes the top-level may directly contain teams
        raw_teams = data.get("teams", [])
        for t in raw_teams:
            if t:
                teams.append(t)

    rows: List[Dict[str, Any]] = []
    for t in teams:
        name = t.get("displayName") or t.get("name") or t.get("shortDisplayName")
        if not name:
            continue
        # Logos: prefer the largest default/full item
        logo_url = None
        logos = t.get("logos") or []
        if isinstance(logos, list) and logos:
            # sort by width descending and prefer rel containing "full" or "default"
            def _score(l: Dict[str, Any]) -> tuple[int, int]:
                width = int(l.get("width") or 0)
                rel = l.get("rel") or []
                rel_full = 1 if (isinstance(rel, list) and any(r == "full" for r in rel)) else 0
                rel_def = 1 if (isinstance(rel, list) and any(r == "default" for r in rel)) else 0
                return (rel_full + rel_def, width)

            logos_sorted = sorted(logos, key=_score, reverse=True)
            for l in logos_sorted:
                if l.get("href"):
                    logo_url = l.get("href")
                    break
        # Colors
        primary = t.get("color") or t.get("alternateColor") or None
        secondary = t.get("alternateColor") or None
        # Normalize hex to #RRGGBB if provided without '#'
        def _fmt(h: str | None) -> str | None:
            if not h or not str(h).strip():
                return None
            s = str(h).strip()
            if not s.startswith("#"):
                s = "#" + s
            return s

        primary_hex = _fmt(primary)
        secondary_hex = _fmt(secondary)
        text_hex = _auto_text_color(primary_hex or secondary_hex or "#4a5568")

        # Conference (best-effort): ESPN team often has groups or conference fields
        conf_name = None
        try:
            grp = t.get("groups") or t.get("group")
            if isinstance(grp, list) and grp:
                g0 = grp[0]
                conf_name = g0.get("name") or g0.get("shortName") or g0.get("displayName")
            elif isinstance(grp, dict):
                conf_name = grp.get("name") or grp.get("shortName") or grp.get("displayName")
        except Exception:
            conf_name = None

        rows.append({
            "team": name,
            "logo": logo_url,
            "primary_color": primary_hex,
            "secondary_color": secondary_hex,
            "text_color": text_hex,
            "espn_id": t.get("id"),
            "abbreviation": t.get("abbreviation"),
            "conference": conf_name,
        })

    return pd.DataFrame(rows)


def write_branding_csv(out_path: Path, df: pd.DataFrame) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deduplicate by normalized team name; keep the first
    if not df.empty:
        from .team_normalize import canonical_slug as canonical
        df = df.copy()
        df["_key"] = df["team"].astype(str).map(lambda s: canonical(s))
        df = df.drop_duplicates(subset=["_key"], keep="first")
        df = df.drop(columns=["_key"], errors="ignore")
        # Sort for readability
        df = df.sort_values("team")
    df.to_csv(out_path, index=False)
    return out_path
