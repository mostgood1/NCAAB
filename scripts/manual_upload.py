import os
import sys
import json
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

BASE = os.environ.get("NCAAB_BASE_URL", "https://ncaab.onrender.com")
DATE = os.environ.get("NCAAB_DATE", "2026-01-09")
WS = Path(os.environ.get("NCAAB_WS", "C:/Users/mostg/OneDrive/Coding/NCAAB"))
OUT = WS / "outputs"

def post_csv(url: str, path: Path) -> dict:
    if not path.exists():
        return {"error": f"missing {path}"}
    if requests is None:
        return {"error": "requests not available"}
    with open(path, "rb") as fh:
        # send as raw body or multipart file
        files = {"file": (path.name, fh, "text/csv")}
        try:
            r = requests.post(url, files=files, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e), "url": url}

def get_json(url: str) -> dict:
    if requests is None:
        return {"error": "requests not available"}
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "url": url}

if __name__ == "__main__":
    print(f"Base={BASE} Date={DATE}")
    # Upload artifacts
    results = {}
    results["picks_raw"] = post_csv(f"{BASE}/api/upload_picks_raw", OUT / "picks_raw.csv")
    results["ats_picks"] = post_csv(f"{BASE}/api/upload_ats_picks?date={DATE}", OUT / "picks" / f"ats_picks_{DATE}.csv")
    results["edges"] = post_csv(f"{BASE}/api/upload_align_edges?date={DATE}", OUT / f"align_period_{DATE}_edges.csv")
    results["display"] = post_csv(f"{BASE}/api/upload_predictions_display?date={DATE}", OUT / f"predictions_display_{DATE}.csv")
    results["enriched"] = post_csv(f"{BASE}/api/upload_predictions_enriched?date={DATE}", OUT / f"predictions_unified_enriched_{DATE}.csv")
    print("Uploads:", json.dumps(results, indent=2))
    # Verify
    dbg = get_json(f"{BASE}/api/debug_artifacts?date={DATE}")
    print("Debug:", json.dumps(dbg, indent=2))
    recs = get_json(f"{BASE}/api/recommendations?date={DATE}")
    print("Recommendations:", json.dumps(recs, indent=2))
