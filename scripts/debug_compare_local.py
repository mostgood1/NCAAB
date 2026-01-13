import sys, json
root = r"c:\\Users\\mostg\\OneDrive\\Coding\\NCAAB"
if root not in sys.path:
    sys.path.insert(0, root)
try:
    from app import app  # type: ignore
    app.testing = True
    client = app.test_client()
except Exception as e:
    print(json.dumps({"error": f"failed to import app: {e}"}))
    raise
DATE = "2026-01-11"

try:
    la = client.get(f"/api/recommendations?date={DATE}")
    ld = client.get(f"/api/recommendations_display?date={DATE}")
except Exception as e:
    print(json.dumps({"error": f"failed to fetch local endpoints: {e}"}))
    raise

out = {}
for key, resp in (("local_api", la), ("local_display", ld)):
    try:
        j = resp.get_json(force=True)
    except Exception as e:
        print(json.dumps({"warn": f"json parse failed for {key}: {e}"}))
        j = None
    rows = (j or {}).get("data") or []
    gids = sorted({str(r.get("game_id") or "") for r in rows if str(r.get("game_id") or "")})
    codes = sorted({str(r.get("code") or r.get("rec_code") or "").upper() for r in rows})
    out[key] = {
        "status": resp.status_code,
        "rows": len(rows),
        "codes": codes,
        "sample_gids": gids[:10],
    }

print(json.dumps(out, indent=2))
