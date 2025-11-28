import argparse, datetime as dt, json, os, pathlib, sys, typing as t
from dataclasses import dataclass

import requests
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


@dataclass
class EventRow:
    game_id: str
    season: int
    date: str
    start_time: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_score_1h: t.Optional[int] = None
    away_score_1h: t.Optional[int] = None
    home_score_2h: t.Optional[int] = None
    away_score_2h: t.Optional[int] = None
    neutral_site: t.Optional[bool] = None
    venue: t.Optional[str] = None


def _fetch_scoreboard(date_str: str) -> dict:
    params = {"dates": date_str.replace("-", "")}
    r = requests.get(SCOREBOARD_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _parse_events(payload: dict) -> list[EventRow]:
    events = payload.get("events", [])
    out: list[EventRow] = []
    for ev in events:
        try:
            game_id = str(ev.get("id"))
            season = int(ev.get("season", {}).get("year")) if ev.get("season") else None
            competitions = ev.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            start_time = comp.get("date")
            # derive date portion (UTC) -> local naive date string for consistency with existing artifacts
            date_part = None
            if start_time:
                try:
                    # ESPN provides ISO8601 with Z; parse
                    ts = dt.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    date_part = ts.date().isoformat()
                except Exception:
                    pass
            if not date_part:
                # fallback to day.date field
                day = payload.get("day", {}).get("date")
                date_part = day or dt.date.today().isoformat()
            competitors = comp.get("competitors", [])
            home_team = away_team = ""
            home_score = away_score = 0
            for c in competitors:
                team_name = c.get("team", {}).get("displayName") or c.get("team", {}).get("name") or "TBD"
                score_raw = c.get("score")
                score_val = int(score_raw) if str(score_raw).isdigit() else 0
                side = c.get("homeAway")
                if side == "home":
                    home_team = team_name
                    home_score = score_val
                elif side == "away":
                    away_team = team_name
                    away_score = score_val
            neutral = comp.get("neutralSite")
            venue = comp.get("venue", {}).get("fullName")
            out.append(EventRow(
                game_id=game_id,
                season=season or dt.date.today().year,
                date=date_part,
                start_time=start_time or "",
                home_team=home_team or "TBD",
                away_team=away_team or "TBD",
                home_score=home_score,
                away_score=away_score,
                neutral_site=bool(neutral) if neutral is not None else None,
                venue=venue,
            ))
        except Exception:
            continue
    return out


def _load_existing_curr() -> pd.DataFrame:
    path = OUT / "games_curr.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _merge_authoritative(authoritative: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Return authoritative set; attempt to preserve any existing score partitions (1h/2h) if present.

    This function replaces placeholder TBD team names and adds any missing games. If existing contains
    halftime scores they are retained via merge on game_id.
    """
    if existing.empty:
        return authoritative
    # Coerce key dtype
    authoritative["game_id"] = authoritative["game_id"].astype(str)
    existing["game_id"] = existing["game_id"].astype(str)
    keep_cols = [c for c in existing.columns if c.startswith("home_score_") or c.startswith("away_score_")]
    if keep_cols:
        merged = authoritative.merge(existing[["game_id"] + keep_cols], on="game_id", how="left")
        # fill authoritative base score columns only if existing halftime present
        for c in keep_cols:
            if c in merged.columns:
                merged[c] = merged[c]
        return merged
    return authoritative


def ingest(date: str, write: bool = True) -> pathlib.Path | None:
    payload = _fetch_scoreboard(date)
    rows = _parse_events(payload)
    if not rows:
        print(f"No events returned for {date}")
        return None
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Ensure stable column order similar to existing games_curr.csv
    ordered = [
        "game_id","season","date","start_time","home_team","away_team","home_score","away_score",
        "home_score_1h","away_score_1h","home_score_2h","away_score_2h","neutral_site","venue"
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = None
    df = df[ordered]
    existing = _load_existing_curr()
    # If scoreboard appears partial (fewer events than existing), patch placeholders only
    if not existing.empty and len(df) < len(existing):
        # map game_id -> (home_team, away_team)
        id_map = {str(r.game_id): (r.home_team, r.away_team) for r in rows}
        existing['game_id'] = existing['game_id'].astype(str)
        for gid, (h, a) in id_map.items():
            mask = existing['game_id'] == gid
            if mask.any():
                # Only replace TBD placeholders
                if h and h != 'TBD':
                    existing.loc[mask & (existing['home_team'] == 'TBD'), 'home_team'] = h
                if a and a != 'TBD':
                    existing.loc[mask & (existing['away_team'] == 'TBD'), 'away_team'] = a
        merged = existing
    else:
        merged = _merge_authoritative(df, existing)
    out_curr = OUT / "games_curr.csv"
    out_date = OUT / f"games_{date}.csv"
    if write:
        merged.to_csv(out_curr, index=False)
        # Write a date-stamped authoritative snapshot for audit
        merged.to_csv(out_date, index=False)
        print(f"Wrote authoritative schedule: {out_curr} (rows={len(merged)})")
    return out_curr


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Ingest ESPN scoreboard and produce authoritative games_curr.csv")
    ap.add_argument("--date", help="Date YYYY-MM-DD (defaults to today local)")
    ap.add_argument("--no-write", action="store_true", help="Parse only; do not write outputs")
    args = ap.parse_args(argv)
    if args.date:
        date_str = args.date
    else:
        date_str = dt.date.today().isoformat()
    ingest(date_str, write=not args.no_write)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
