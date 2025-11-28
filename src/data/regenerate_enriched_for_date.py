import argparse, datetime as dt
from pathlib import Path

# Import the Flask app
import importlib, sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def run(date_str: str) -> None:
    # Import app lazily to avoid side effects at module import time
    # Ensure project root on sys.path for 'import app'
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    app_mod = importlib.import_module("app")
    app = getattr(app_mod, "app")
    app.testing = True
    with app.test_client() as c:
        # Include diag=1 to encourage full instrumentation
        resp = c.get(f"/?date={date_str}&diag=1")
        print(f"GET /?date={date_str} -> {resp.status_code}")
        # Best-effort: check for enriched artifact written by the route
        p = OUT / f"predictions_unified_enriched_{date_str}.csv"
        if p.exists():
            print(f"Found enriched: {p}")
        else:
            print("Enriched artifact not found; check server logs.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hit index route to regenerate enriched artifacts for a date")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD")
    args = ap.parse_args(argv)
    run(args.date)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
