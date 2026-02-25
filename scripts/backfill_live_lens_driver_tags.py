from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import time

import requests


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    dt.date.fromisoformat(s2)
    return s2


def _iter_date_range(start_date: str, end_date: str) -> list[str]:
    s = dt.date.fromisoformat(_safe_date(start_date))
    e = dt.date.fromisoformat(_safe_date(end_date))
    if s > e:
        s, e = e, s
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur = cur + dt.timedelta(days=1)
    return out


def _coerce_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            x = float(v)
            return x if math.isfinite(x) else None
        s = str(v).strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return None
        x = float(s)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _is_nan_like(v: Any) -> bool:
    try:
        if v is None:
            return False
        if isinstance(v, float):
            return math.isnan(v) or (not math.isfinite(v))
        s = str(v).strip().lower()
        return s in {"nan", "+nan", "-nan", "none", "null"}
    except Exception:
        return False


def _norm_tags(v: Any) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return None
        # Support either CSV or a single token
        parts = [p.strip() for p in s.split(",") if p.strip()] if "," in s else [s]
        return parts or None
    if isinstance(v, (list, tuple, set)):
        out: list[str] = []
        for x in v:
            sx = str(x or "").strip()
            if sx and sx.lower() not in {"none", "null", "nan"} and sx not in out:
                out.append(sx)
        return out or None
    return None


def _sanitize_driver_and_tags(raw_driver: Any, raw_tags: Any) -> tuple[str | None, list[str] | None, bool]:
    """Sanitize driver + driver_tags fields.

    Goal: remove NaN-like junk (including float('nan')) so downstream analytics/learning
    never treat a literal tag named "nan" as real.
    """

    changed = False

    driver: str | None
    if _is_nan_like(raw_driver):
        driver = None
        if raw_driver is not None:
            changed = True
    else:
        try:
            driver = str(raw_driver or "").strip() or None
        except Exception:
            driver = None

    tags = _norm_tags(raw_tags)

    # If raw tags contain NaN-like elements (e.g., float('nan') in a list), _norm_tags
    # will drop them, but we still need to persist the sanitized list back to disk.
    if isinstance(raw_tags, (list, tuple, set)):
        for x in raw_tags:
            if _is_nan_like(x):
                changed = True
                break
    elif _is_nan_like(raw_tags):
        if raw_tags is not None:
            changed = True

    # Detect string case where we normalized away junk like "nan" / "null".
    if isinstance(raw_tags, str):
        s0 = raw_tags.strip().lower()
        if s0 in {"none", "null", "nan"}:
            changed = True
        elif tags is None and s0:
            # Non-empty string that became empty list after normalization.
            changed = True

    # If driver is a non-null string-like, normalize it (strip), and mark changed if it differed.
    if isinstance(raw_driver, str):
        if (raw_driver.strip() or None) != driver:
            changed = True

    return driver, tags, changed


def _try_parse_json_obj(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if v is None:
        return None
    try:
        s = str(v).strip()
        if not s:
            return None
        if not (s.startswith("{") or s.startswith("[")):
            return None
        return json.loads(s)
    except Exception:
        return None


@dataclass
class BackfillStats:
    total_rows: int = 0
    updated_rows: int = 0
    already_tagged_rows: int = 0
    derived_nonempty_rows: int = 0
    tag_counts: Counter[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.tag_counts is None:
            self.tag_counts = Counter()


def derive_driver_tags(row: dict[str, Any]) -> list[str]:
    """Derive tags from already-logged numeric fields.

    - Pace uses possessions per minute (poss/elapsed), matching reconstruction logic.
    - Efficiency uses points per possession (total_points/poss).

    Thresholds pulled from row['tuning'] when present, else safe defaults.
    """

    pbp = row.get("pbp")
    if not isinstance(pbp, dict):
        pbp = _try_parse_json_obj(pbp)
    if not isinstance(pbp, dict):
        pbp = {}

    tuning = row.get("tuning")
    if not isinstance(tuning, dict):
        tuning = _try_parse_json_obj(tuning)
    if not isinstance(tuning, dict):
        tuning = {}

    elapsed = _coerce_float(row.get("elapsed"))
    total_points = _coerce_float(row.get("total_points"))
    poss = _coerce_float(pbp.get("poss"))

    pace_hi = _coerce_float(tuning.get("pace_hi"))
    pace_lo = _coerce_float(tuning.get("pace_lo"))
    pps_hi = _coerce_float(tuning.get("pps_hi"))
    pps_lo = _coerce_float(tuning.get("pps_lo"))

    # Defaults mirror live_lens_recover.py reconstruction defaults.
    if pace_hi is None:
        pace_hi = 3.25
    if pace_lo is None:
        pace_lo = 2.75
    if pps_hi is None:
        pps_hi = 1.18
    if pps_lo is None:
        pps_lo = 0.95

    tags: list[str] = []

    # Pace = possessions per minute.
    if poss is not None and elapsed is not None and elapsed > 0:
        poss_rate = poss / elapsed
        if pace_hi is not None and poss_rate >= pace_hi:
            tags.append("pace_hi")
        elif pace_lo is not None and poss_rate <= pace_lo:
            tags.append("pace_lo")

    # Efficiency = points per possession.
    if total_points is not None and poss is not None and poss > 0:
        pps = total_points / poss
        if pps_hi is not None and pps >= pps_hi:
            tags.append("eff_hi")
        elif pps_lo is not None and pps <= pps_lo:
            tags.append("eff_lo")

    return tags


def backfill_rows(rows: Iterable[dict[str, Any]], *, overwrite: bool) -> tuple[list[dict[str, Any]], BackfillStats]:
    out: list[dict[str, Any]] = []
    st = BackfillStats()

    for r in rows:
        if not isinstance(r, dict):
            continue
        st.total_rows += 1

        row_changed = False

        raw_driver = r.get("driver")
        raw_tags = r.get("driver_tags")

        driver, tags0, sanitized = _sanitize_driver_and_tags(raw_driver, raw_tags)
        if sanitized:
            r = dict(r)
            r["driver"] = driver
            r["driver_tags"] = tags0
            row_changed = True

        if tags0:
            st.already_tagged_rows += 1

        # Derive tags if missing (or overwrite requested).
        if overwrite or not tags0:
            derived = derive_driver_tags(r)
            if derived:
                st.derived_nonempty_rows += 1
            if derived and (overwrite or not tags0):
                r = dict(r)
                r["driver_tags"] = derived
                if not driver:
                    r["driver"] = derived[0]
                row_changed = True
                for t in derived:
                    st.tag_counts[t] += 1

        if row_changed:
            st.updated_rows += 1

        out.append(r)

    return out, st


def read_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        s = line.strip().lstrip("\ufeff")
        if not s:
            continue
        try:
            j = json.loads(s)
        except Exception:
            continue
        if isinstance(j, dict):
            rows.append(j)
    return rows


def to_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def download_signals(base_url: str, date_s: str, *, timeout: float = 45.0) -> str | None:
    url = f"{base_url.rstrip('/')}/api/download_live_lens_signals"
    resp = requests.get(url, params={"date": date_s}, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def _should_retry_status(code: int | None) -> bool:
    if code is None:
        return False
    return int(code) in {408, 425, 429, 500, 502, 503, 504}


def upload_signals(
    base_url: str,
    date_s: str,
    jsonl_text: str,
    *,
    force: bool,
    timeout: float = 90.0,
    max_attempts: int = 5,
) -> dict[str, Any]:
    """Upload signals JSONL back to the server.

    Prefer a raw-body upload (no multipart) because it's simpler and tends to be
    more reliable for large payloads. The server endpoint supports both.
    """

    url = f"{base_url.rstrip('/')}/api/upload_live_lens_signals"
    params = {"date": date_s}
    if force:
        params["force"] = "1"

    data = (jsonl_text or "").encode("utf-8", errors="replace")
    last_err: str | None = None

    for attempt in range(1, int(max_attempts) + 1):
        try:
            # Raw-body NDJSON upload.
            resp = requests.post(
                url,
                params=params,
                data=data,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=timeout,
            )

            if resp.status_code == 409:
                try:
                    return {"status": "conflict", **resp.json()}
                except Exception:
                    return {"status": "conflict", "text": resp.text}

            if not resp.ok:
                if attempt < int(max_attempts) and _should_retry_status(resp.status_code):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    time.sleep(min(15.0, 1.25**attempt + (attempt * 0.25)))
                    continue
                resp.raise_for_status()

            try:
                return resp.json()
            except Exception:
                return {"status": "ok", "text": resp.text}

        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < int(max_attempts):
                time.sleep(min(20.0, 1.6**attempt))
                continue
            return {"status": "error", "error": last_err}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < int(max_attempts):
                time.sleep(min(15.0, 1.5**attempt))
                continue
            return {"status": "error", "error": last_err}

    return {"status": "error", "error": last_err or "upload_failed"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Backfill Live Lens driver/driver_tags for existing signals JSONL.")
    ap.add_argument("--date", default=None, help="Single date YYYY-MM-DD")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--base-url", default=None, help="Render base URL to download/upload signals (e.g. https://ncaab.onrender.com)")
    ap.add_argument("--out-dir", default="outputs", help="Local outputs directory (default outputs)")
    ap.add_argument("--download", action="store_true", help="Download signals from base-url and write to local outputs before backfilling")
    ap.add_argument("--upload", action="store_true", help="Upload backfilled signals to base-url")
    ap.add_argument("--force", action="store_true", help="Use force=1 when uploading (required to overwrite existing server log)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing driver_tags if present (default only fills missing)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write local file or upload; just report counts")

    args = ap.parse_args(argv)

    if args.date:
        dates = [_safe_date(args.date)]
    else:
        if not (args.start and args.end):
            ap.error("Provide --date or both --start and --end")
        dates = _iter_date_range(_safe_date(args.start), _safe_date(args.end))

    base_url = (str(args.base_url).strip() if args.base_url else "") or None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "total_rows": 0,
        "updated_rows": 0,
        "already_tagged_rows": 0,
        "derived_nonempty_rows": 0,
        "tag_counts": Counter(),
        "per_date": {},
    }

    for d in dates:
        local_path = out_dir / f"live_lens_signals_{d}.jsonl"

        text: str | None = None
        if args.download:
            if not base_url:
                ap.error("--download requires --base-url")
            text = download_signals(base_url, d)
            if text is None:
                print(f"[skip] {d} missing on server (404)")
                continue
            if not args.dry_run:
                local_path.write_text(text, encoding="utf-8", errors="replace")
        else:
            if local_path.exists():
                text = local_path.read_text(encoding="utf-8", errors="ignore")
            elif base_url:
                # Fallback: download if local missing but base-url provided.
                text = download_signals(base_url, d)
                if text is None:
                    print(f"[skip] {d} missing on server (404)")
                    continue
                if not args.dry_run:
                    local_path.write_text(text, encoding="utf-8", errors="replace")
            else:
                print(f"[skip] missing local file and no base-url: {local_path}")
                continue

        rows = read_jsonl(text or "")
        new_rows, st = backfill_rows(rows, overwrite=bool(args.overwrite))

        overall["total_rows"] += st.total_rows
        overall["updated_rows"] += st.updated_rows
        overall["already_tagged_rows"] += st.already_tagged_rows
        overall["derived_nonempty_rows"] += st.derived_nonempty_rows
        overall["tag_counts"].update(st.tag_counts)

        overall["per_date"][d] = {
            "path": str(local_path),
            "total_rows": st.total_rows,
            "updated_rows": st.updated_rows,
            "already_tagged_rows": st.already_tagged_rows,
            "derived_nonempty_rows": st.derived_nonempty_rows,
            "tag_counts": dict(st.tag_counts),
        }

        if st.updated_rows > 0 and not args.dry_run:
            local_path.write_text(to_jsonl(new_rows), encoding="utf-8", errors="replace")

        if args.upload:
            if not base_url:
                ap.error("--upload requires --base-url")
            if not args.force:
                ap.error("--upload requires --force (Render endpoint refuses overwrites without force=1)")
            payload = upload_signals(base_url, d, to_jsonl(new_rows), force=True)
            overall["per_date"][d]["upload"] = payload

        print(
            f"[{d}] rows={st.total_rows} updated={st.updated_rows} already_tagged={st.already_tagged_rows} derived_nonempty={st.derived_nonempty_rows} tags={dict(st.tag_counts)}"
        )

    # Print aggregate summary as JSON on stdout (easy to paste into notes)
    agg = dict(overall)
    agg["tag_counts"] = dict(overall["tag_counts"].most_common())
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
