"""Fetch Live Lens signals from Render and run the local analyzer.

Example:
  python scripts/fetch_and_analyze_live_lens_signals.py --date 2026-02-12
  python scripts/fetch_and_analyze_live_lens_signals.py --date 2026-02-12 --base-url https://ncaab.onrender.com --full-game-only

This downloads:
  /api/download_live_lens_signals?date=YYYY-MM-DD
into:
  outputs/live_lens_signals_<date>_render.jsonl
then prints the same summary as scripts/analyze_live_lens_signals.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def _load_analyzer():
  here = Path(__file__).resolve()
  analyzer_path = here.parent / "analyze_live_lens_signals.py"
  spec = importlib.util.spec_from_file_location("analyze_live_lens_signals", analyzer_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load analyzer module from {analyzer_path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def _fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "ncaab-fetch-analyze"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Slate date YYYY-MM-DD")
    ap.add_argument("--base-url", default="https://ncaab.onrender.com", help="Base URL for the deployed app")
    ap.add_argument("--full-game-only", action="store_true", help="Filter to horizon>=39")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output path. Default: outputs/live_lens_signals_<date>_render.jsonl",
    )
    args = ap.parse_args()

    base = str(args.base_url).rstrip("/")
    url = f"{base}/api/download_live_lens_signals?date={args.date}"

    out_path = Path(args.out) if args.out else (Path("outputs") / f"live_lens_signals_{args.date}_render.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _fetch_bytes(url)
    out_path.write_bytes(raw)

    analyzer = _load_analyzer()
    result = analyzer.analyze_path(out_path, full_game_only=args.full_game_only)
    if result.rows_parsed == 0:
        print(f"Downloaded {len(raw)} bytes to {out_path}, but parsed 0 rows.")
        return 0

    analyzer.print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
