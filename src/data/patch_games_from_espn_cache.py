import argparse, datetime as dt, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = DATA / "cache" / "espn"

BADS = {"tbd","t.b.d","tba","t.b.a","to be determined","to-be-determined","to be announced","to-be-announced","unknown","na","n/a","","none","null"}

def is_bad(x: object) -> bool:
    try:
        return str(x).strip().lower() in BADS
    except Exception:
        return True

def load_cache(date_str: str) -> dict | None:
    p = CACHE / f"{date_str}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_team_map(payload: dict) -> dict[str, dict[str, str]]:
    team_map: dict[str, dict[str, str]] = {}
    if not isinstance(payload, dict):
        return team_map
    events = payload.get("events") or []
    for ev in events:
        try:
            gid = str(ev.get("id")) if ev.get("id") is not None else None
            comps = (ev.get("competitions") or [{}])[0]
            competitors = comps.get("competitors") or []
            h = next((c for c in competitors if c.get("homeAway") == "home"), None)
            a = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not (gid and h and a):
                continue
            hname = (h.get("team") or {}).get("displayName") or (h.get("team") or {}).get("shortDisplayName")
            aname = (a.get("team") or {}).get("displayName") or (a.get("team") or {}).get("shortDisplayName")
            if hname or aname:
                team_map[gid] = {"home": hname, "away": aname}
        except Exception:
            continue
    return team_map


def patch_games(date_str: str, write_inplace: bool = True) -> Path | None:
    games_path = OUT / "games_curr.csv"
    if not games_path.exists():
        return None
    try:
        df = pd.read_csv(games_path)
    except Exception:
        return None
    if df.empty:
        return None
    if "date" in df.columns:
        df = df[df["date"].astype(str) == date_str].copy()
    if "game_id" not in df.columns or not {"home_team","away_team"}.issubset(df.columns):
        return None
    cache_payload = load_cache(date_str)
    if not cache_payload:
        return None
    team_map = build_team_map(cache_payload)
    if not team_map:
        return None
    # Work on full file so we can persist across all rows
    try:
        full = pd.read_csv(games_path)
    except Exception:
        return None
    if "game_id" not in full.columns:
        return None
    full["game_id"] = full["game_id"].astype(str)
    replaced = 0
    for idx, row in full.iterrows():
        gid = str(row.get("game_id"))
        if str(row.get("date")) != date_str:
            continue
        ht = row.get("home_team")
        at = row.get("away_team")
        if not is_bad(ht) and not is_bad(at):
            continue
        tm = team_map.get(gid)
        if not tm or not tm.get("home") or not tm.get("away"):
            continue
        if is_bad(ht):
            full.at[idx, "home_team"] = tm["home"]
            replaced += 1
        if is_bad(at):
            full.at[idx, "away_team"] = tm["away"]
            replaced += 1
    if replaced and write_inplace:
        backup = OUT / f"games_curr_backup_{date_str}.csv"
        try:
            # Backup once per run
            if not backup.exists():
                pd.read_csv(games_path).to_csv(backup, index=False)
        except Exception:
            pass
        full.to_csv(games_path, index=False)
    return games_path if replaced else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Patch outputs/games_curr.csv TBD rows with ESPN cached team names")
    ap.add_argument("--date", help="YYYY-MM-DD", default=dt.date.today().isoformat())
    ap.add_argument("--no-inplace", action="store_true", help="Do not write back, only print intended changes")
    args = ap.parse_args(argv)
    path = patch_games(args.date, write_inplace=not args.no_inplace)
    if path:
        print(f"Patched {path} for date {args.date}")
        return 0
    else:
        print("No changes applied (no cache/team map/placeholder rows)")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
