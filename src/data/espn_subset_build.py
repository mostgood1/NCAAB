import argparse, datetime as dt, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = DATA / "cache" / "espn"


def build_subset(date_str: str) -> dict:
    # Load ESPN cached events for the date
    cache_path = CACHE / f"{date_str}.json"
    espn_ids: set[str] = set()
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            events = payload.get("events") or []
            for ev in events:
                gid = ev.get("id")
                if gid is not None:
                    espn_ids.add(str(gid))
        except Exception:
            pass
    # Load our local schedule for the date
    curr_ids: set[str] = set()
    gp = OUT / "games_curr.csv"
    if gp.exists():
        try:
            gdf = pd.read_csv(gp)
            if not gdf.empty and "game_id" in gdf.columns:
                if "date" in gdf.columns:
                    gdf = gdf[gdf["date"].astype(str) == date_str]
                curr_ids = set(map(str, gdf["game_id"].astype(str)))
        except Exception:
            pass
    # Intersection: do not inflate slate beyond our curated set
    subset = sorted(list(espn_ids & curr_ids)) if espn_ids else sorted(list(curr_ids))
    return {"date": date_str, "count": len(subset), "game_ids": subset}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write outputs/schedule_espn_subset_<date>.json from ESPN cache and games_curr.csv")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD")
    args = ap.parse_args(argv)
    obj = build_subset(args.date)
    outp = OUT / f"schedule_espn_subset_{args.date}.json"
    outp.write_text(json.dumps(obj, indent=2))
    print(f"Wrote {outp} with {obj['count']} ids")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
