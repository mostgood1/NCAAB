import argparse
import datetime as dt
import json
import os
from pathlib import Path

import sys


def _to_datestr(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def _to_espn_dates_param(d: dt.date) -> str:
    # ESPN expects YYYYMMDD for the scoreboard dates parameter
    return d.strftime("%Y%m%d")


def fetch_espn_scoreboard(target_date: dt.date, out_dir: Path) -> Path:
    """
    Fetch ESPN men's college basketball scoreboard JSON for target_date
    and write to data/cache/espn/<date>.json. Returns the written path.
    """
    import requests  # lazy import

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_to_datestr(target_date)}.json"

    params_date = _to_espn_dates_param(target_date)
    # Try multiple endpoints in case ESPN routing differs
    candidates = [
        f"https://site.web.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/scoreboard?dates={params_date}&groups=50&limit=500",
        f"https://site.web.api.espn.com/apis/v2/sports/basketball/ncb/scoreboard?dates={params_date}&groups=50&limit=500",
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={params_date}&groups=50&limit=500",
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/ncb/scoreboard?dates={params_date}&groups=50&limit=500",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.espn.com/",
        "Origin": "https://www.espn.com",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

    last_err: Exception | None = None
    payload = None
    for url in candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=25)
            if resp.status_code == 200:
                payload = resp.json()
                break
            last_err = Exception(f"HTTP {resp.status_code} for {url}")
        except Exception as e:
            last_err = e
    if payload is None:
        if last_err:
            raise last_err
        raise RuntimeError("Failed to fetch ESPN scoreboard: no candidates succeeded")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch ESPN scoreboard for a given date")
    parser.add_argument(
        "--date",
        dest="date",
        help="Target date in YYYY-MM-DD; defaults to today (local)",
    )
    parser.add_argument(
        "--out",
        dest="out",
        default=str(Path("data") / "cache" / "espn"),
        help="Output directory for ESPN cache (default: data/cache/espn)",
    )
    args = parser.parse_args(argv)

    if args.date:
        try:
            target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        except Exception:
            print(f"Invalid --date value: {args.date}", file=sys.stderr)
            return 2
    else:
        target_date = dt.datetime.now().date()

    out_dir = Path(args.out)
    try:
        path = fetch_espn_scoreboard(target_date, out_dir)
        print(f"Wrote ESPN scoreboard cache: {path}")
    except Exception as e:
        print(f"ERROR fetching ESPN scoreboard: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
