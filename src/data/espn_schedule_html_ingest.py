import argparse, datetime as dt, json, pathlib, re, sys
from typing import List, Set
import requests

OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"
HTML_SCHEDULE_URL = "https://www.espn.com/mens-college-basketball/schedule"

GAME_ID_RE = re.compile(r"gameId/(\d+)")
TEAM_NAME_RE = re.compile(r"/mens-college-basketball/team/_/id/(\d+)/([a-z0-9-]+)")


def fetch_html(date_str: str) -> str:
    params = {"date": date_str.replace('-', '')}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }
    r = requests.get(HTML_SCHEDULE_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text


def extract_game_ids(html: str) -> Set[str]:
    return set(GAME_ID_RE.findall(html))


def extract_team_slug_map(html: str) -> dict:
    # Map team numeric id to slug for enrichment reference
    out = {}
    for tid, slug in TEAM_NAME_RE.findall(html):
        out[tid] = slug
    return out


def write_outputs(date_str: str, ids: Set[str], team_map: dict) -> None:
    ids_path = OUT / f"schedule_espn_ids_{date_str}.json"
    data = {"date": date_str, "count": len(ids), "game_ids": sorted(ids), "team_slug_map": team_map}
    ids_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {ids_path} (count={len(ids)})")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Extract ESPN schedule gameIds from HTML page")
    ap.add_argument("--date", help="Date YYYY-MM-DD (defaults to today)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    date_str = args.date or dt.date.today().isoformat()
    html = fetch_html(date_str)
    ids = extract_game_ids(html)
    team_map = extract_team_slug_map(html)
    if not args.no_write:
        write_outputs(date_str, ids, team_map)
    else:
        print(json.dumps({"date": date_str, "count": len(ids)}, indent=2))
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
