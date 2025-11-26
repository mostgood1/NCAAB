import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


OUT = Path("outputs")


def _canon_slug(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Collapse common suffixes
    s = s.replace(" state", " st").replace(" university", "").replace("-", " ")
    return s


def fetch_espn_games(date_str: str) -> List[Dict[str, Any]]:
    # ESPN scoreboard API for men's college basketball
    yyyymmdd = date_str.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={yyyymmdd}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("events", [])
    games: List[Dict[str, Any]] = []
    for ev in events:
        try:
            comps = ev.get("competitions", [{}])[0]
            teams = comps.get("competitors", [])
            home = next((t for t in teams if t.get("homeAway") == "home"), None)
            away = next((t for t in teams if t.get("homeAway") == "away"), None)
            home_name = home.get("team", {}).get("displayName") if home else None
            away_name = away.get("team", {}).get("displayName") if away else None
            gid = ev.get("id")
            games.append({
                "game_id": str(gid) if gid is not None else None,
                "home_team": home_name,
                "away_team": away_name,
                "_home_slug": _canon_slug(str(home_name)),
                "_away_slug": _canon_slug(str(away_name)),
                "_pair_key": "::".join(sorted([_canon_slug(str(home_name)), _canon_slug(str(away_name))])),
            })
        except Exception:
            continue
    return games


def load_frontend_pairs(date_str: str) -> pd.DataFrame:
    # Prefer unified predictions export; fallback to games_curr filtered by date
    candidates: List[Path] = []
    candidates.append(OUT / f"predictions_unified_{date_str}.csv")
    candidates.append(OUT / "games_curr.csv")
    candidates.append(OUT / f"games_{date_str}.csv")
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                df = pd.DataFrame()
                continue
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    # Build canonical pair key
    cols = df.columns.astype(str)
    if {"home_team","away_team"}.issubset(cols):
        df["_home_slug"] = df["home_team"].astype(str).map(_canon_slug)
        df["_away_slug"] = df["away_team"].astype(str).map(_canon_slug)
        df["_pair_key"] = df.apply(lambda r: "::".join(sorted([str(r.get("_home_slug")), str(r.get("_away_slug"))])), axis=1)
    else:
        df["_pair_key"] = None
    return df


def compare_coverage(date_str: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    espn_games = fetch_espn_games(date_str)
    espn_pairs = {g["_pair_key"] for g in espn_games if g.get("_pair_key")}
    frontend_df = load_frontend_pairs(date_str)
    fe_pairs = set(frontend_df["_pair_key"].dropna().astype(str).unique()) if not frontend_df.empty else set()
    missing = sorted(list(espn_pairs - fe_pairs))
    missing_list: List[Dict[str, Any]] = []
    if missing:
        for pk in missing:
            try:
                h, a = pk.split("::")
            except Exception:
                h, a = pk, None
            missing_list.append({"pair_key": pk, "home_slug": h, "away_slug": a})
    return len(espn_pairs), len(fe_pairs), missing_list


def main():
    ap = argparse.ArgumentParser(description="Coverage check vs ESPN schedule")
    ap.add_argument("--date", help="Date YYYY-MM-DD; defaults to today", default=None)
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any ESPN game missing on frontend")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")
    total_espn, total_frontend, missing = compare_coverage(date_str)
    result = {
        "date": date_str,
        "espn_games": total_espn,
        "frontend_pairs": total_frontend,
        "missing_pairs": missing,
        "missing_count": len(missing),
    }
    out_path = OUT / f"coverage_check_{date_str}.json"
    try:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(result, indent=2))
    if args.strict and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
