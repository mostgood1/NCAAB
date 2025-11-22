#!/usr/bin/env python
"""Push a minimal predictions_<date>.csv to the Render ingestion endpoint to prevent synthetic shells.

Usage (PowerShell):
  python scripts/push_slim_predictions.py --url https://your-render-app.onrender.com --date 2025-11-22

If --date omitted, uses today (America/New_York unless NCAAB_SCHEDULE_TZ set).
Environment variables:
  NCAAB_INGEST_TOKEN  Optional auth token; sent as X-Ingest-Token header if present.
  NCAAB_OUTPUTS_DIR   Override outputs directory (same logic as server if set).
  RENDER_BASE_URL     Fallback base URL if --url not provided.

Logic:
  1. Resolve candidate predictions sources (priority: predictions_<date>.csv, predictions_blend_<date>.csv,
     predictions_week.csv, predictions.csv, predictions_all.csv).
  2. Load first non-empty DataFrame containing pred_total or pred_margin.
  3. Construct minimal frame [game_id, date, pred_total, pred_margin] (exclude columns with all NaN).
  4. Write/overwrite outputs/predictions_<date>.csv locally.
  5. POST file bytes to /api/ingest/predictions on remote.
  6. Print local + remote md5 for parity verification.

Exit codes:
  0 success
  2 no suitable predictions source found
  3 remote ingest failed
"""
from __future__ import annotations
import argparse, os, sys, datetime as dt, json, hashlib
from pathlib import Path
from typing import Optional

try:
    import pandas as pd  # type: ignore
except Exception as e:
    print(f"pandas import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import urllib.request, urllib.error
    _HAS_REQUESTS = False

DEF_TZ = os.getenv("NCAAB_SCHEDULE_TZ", "America/New_York")
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore

DEF_OUT = Path(os.getenv("NCAAB_OUTPUTS_DIR", "")).resolve() if os.getenv("NCAAB_OUTPUTS_DIR") else Path(__file__).resolve().parents[1] / "outputs"
if not DEF_OUT.exists():
    DEF_OUT = Path(__file__).resolve().parents[1] / "outputs"

def today_local_iso() -> str:
    try:
        if ZoneInfo:
            tz = ZoneInfo(DEF_TZ)
            return dt.datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        pass
    return dt.date.today().strftime("%Y-%m-%d")

def pick_source(date_str: str) -> Optional[Path]:
    candidates = [
        DEF_OUT / f"predictions_{date_str}.csv",
        DEF_OUT / f"predictions_blend_{date_str}.csv",
        DEF_OUT / "predictions_week.csv",
        DEF_OUT / "predictions.csv",
        DEF_OUT / "predictions_all.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty and ("pred_total" in df.columns or "pred_margin" in df.columns):
                    return p
            except Exception:
                continue
    return None

def build_slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["game_id", "date", "pred_total", "pred_margin"] if c in df.columns]
    slim = df[cols].copy()
    # Normalize types
    if "game_id" in slim.columns:
        try: slim["game_id"] = slim["game_id"].astype(str)
        except Exception: pass
    if "date" in slim.columns:
        try:
            slim["date"] = pd.to_datetime(slim["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            slim["date"] = slim["date"].astype(str)
    # Drop columns that are entirely NaN
    for col in ["pred_total", "pred_margin"]:
        if col in slim.columns:
            ser = pd.to_numeric(slim[col], errors="coerce")
            if ser.notna().sum() == 0:
                slim.drop(columns=[col], inplace=True)
            else:
                slim[col] = ser
    return slim

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def post_file(url_base: str, target_date: str, csv_bytes: bytes) -> tuple[bool, dict]:
    ingest_url = url_base.rstrip("/") + "/api/ingest/predictions?date=" + target_date
    headers = {}
    tok = os.getenv("NCAAB_INGEST_TOKEN", "").strip()
    if tok:
        headers["X-Ingest-Token"] = tok
    files = {"file": (f"predictions_{target_date}.csv", csv_bytes, "text/csv")}
    try:
        if _HAS_REQUESTS:
            resp = requests.post(ingest_url, files=files, headers=headers, timeout=30)
            ok = resp.status_code == 200
            data = resp.json() if ok else {"status": resp.status_code, "text": resp.text}
            return ok, data
        else:
            # Fallback urllib multipart build (simple boundary)
            boundary = "BOUNDARY123456"
            body_chunks = []
            body_chunks.append(f"--{boundary}\r\n")
            body_chunks.append(f"Content-Disposition: form-data; name=\"file\"; filename=\"predictions_{target_date}.csv\"\r\n")
            body_chunks.append("Content-Type: text/csv\r\n\r\n")
            body_chunks.append(csv_bytes.decode("utf-8", errors="replace"))
            body_chunks.append(f"\r\n--{boundary}--\r\n")
            body = "".join(body_chunks).encode("utf-8")
            req = urllib.request.Request(ingest_url, data=body, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            if tok:
                req.add_header("X-Ingest-Token", tok)
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(text)
                except Exception:
                    data = {"raw": text}
            return True, data
    except Exception as e:
        return False, {"error": str(e)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Render base URL (e.g. https://ncaab.onrender.com)")
    ap.add_argument("--date", help="ISO date (YYYY-MM-DD) override")
    ap.add_argument("--out", help="Outputs directory override")
    args = ap.parse_args()

    base_url = args.url or os.getenv("RENDER_BASE_URL", "").strip()
    if not base_url:
        print("ERROR: --url or RENDER_BASE_URL env required", file=sys.stderr)
        return 2
    date_str = args.date or today_local_iso()

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = DEF_OUT
    if not out_dir.exists():
        print(f"ERROR: outputs dir not found: {out_dir}", file=sys.stderr)
        return 2

    src = pick_source(date_str)
    if not src:
        print(f"ERROR: no predictions source found for {date_str}", file=sys.stderr)
        return 2
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f"ERROR: failed reading source {src}: {e}", file=sys.stderr)
        return 2
    slim = build_slim(df)
    if slim.empty or ("pred_total" not in slim.columns and "pred_margin" not in slim.columns):
        print("ERROR: slim predictions has no usable columns", file=sys.stderr)
        return 2
    target = out_dir / f"predictions_{date_str}.csv"
    try:
        slim.to_csv(target, index=False)
    except Exception as e:
        print(f"ERROR: failed writing slim predictions: {e}", file=sys.stderr)
        return 2
    local_bytes = target.read_bytes()
    local_md5 = md5_bytes(local_bytes)
    ok, data = post_file(base_url, date_str, local_bytes)
    if not ok:
        print(f"REMOTE INGEST FAILED: {data}", file=sys.stderr)
        return 3
    remote_md5 = data.get("md5")
    print(json.dumps({
        "date": date_str,
        "source": str(src),
        "slim_path": str(target),
        "local_md5": local_md5,
        "remote_md5": remote_md5,
        "rows_local": len(slim),
        "columns": list(slim.columns),
        "ingest_response": data,
    }, indent=2))
    if remote_md5 and remote_md5 != local_md5:
        print("WARNING: md5 mismatch between local and remote", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
