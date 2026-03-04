from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def summarize_live_lines_moves(
    *,
    date_s: str,
    min_total_pts: float = 1.5,
    min_spread_pts: float = 2.0,
    max_age_s: float = 12 * 3600,
    cutoff_ts_by_event_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Summarize latest material line moves from the append-only live snapshots.

    Returns a mapping keyed by ESPN event_id (as string) with fields:
      - total_prev, total_last, delta_total
      - spread_prev, spread_last, delta_spread_home
      - ts_prev, ts_last, age_s
      - badges: list[{label,title}]

    Notes:
    - Uses the last two *distinct* values per market (ignores repeats).
    - Only returns entries where at least one market move is >= threshold.
    """

    date_s2 = str(date_s or "").strip()
    if not date_s2:
        return {}

    p = _get_snapshot_dir() / f"live_{date_s2}.jsonl"
    if not p.exists():
        return {}

    # Track last and previous distinct values per (event,book) to avoid mixing
    # different books/periods which can create misleading deltas.
    state: Dict[tuple[str, str], Dict[str, Any]] = {}

    book_priority = ["draftkings", "fanduel", "betmgm"]

    def _book_rank(book: str) -> int:
        b = str(book or "").strip().lower()
        try:
            return book_priority.index(b)
        except Exception:
            return 999

    def _ts_epoch(ts_iso: str | None) -> float | None:
        if not ts_iso:
            return None

    cutoff_epoch_by_event_id: Dict[str, float] = {}
    if isinstance(cutoff_ts_by_event_id, dict) and cutoff_ts_by_event_id:
        for eid, ts_str in cutoff_ts_by_event_id.items():
            try:
                ee = str(eid or "").strip()
                if not ee:
                    continue
                import datetime as _dt

                s = str(ts_str or "").strip()
                if not s:
                    continue
                d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                if d.tzinfo is None:
                    d = d.replace(tzinfo=_dt.timezone.utc)
                cutoff_epoch_by_event_id[ee] = float(d.timestamp())
            except Exception:
                continue
        try:
            import datetime as _dt

            s = str(ts_iso).strip()
            if not s:
                return None
            d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=_dt.timezone.utc)
            return float(d.timestamp())
        except Exception:
            return None

    def _coerce_num(v: Any) -> float | None:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                return f if f == f else None
            s = str(v).strip()
            if not s or s.lower() in {"nan", "none", "null", "–", "-"}:
                return None
            f = float(s)
            return f if f == f else None
        except Exception:
            return None

    def _age_s(ts_iso: str | None) -> float | None:
        if not ts_iso:
            return None
        try:
            import datetime as _dt

            s = str(ts_iso).strip()
            if not s:
                return None
            d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=_dt.timezone.utc)
            now = _dt.datetime.now(_dt.timezone.utc)
            return float((now - d.astimezone(_dt.timezone.utc)).total_seconds())
        except Exception:
            return None

    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if str(rec.get("endpoint") or "").strip() != "live_lines":
                    continue
                eid = str(rec.get("event_id") or "").strip()
                if not eid:
                    continue
                ts = str(rec.get("ts") or "").strip() or None

                # If we have a per-game cutoff (scheduled start time), only
                # consider movement up to tipoff so badges reflect *pregame* moves.
                cutoff_epoch = cutoff_epoch_by_event_id.get(eid)
                if cutoff_epoch is not None:
                    te = _ts_epoch(ts)
                    if te is None or float(te) > float(cutoff_epoch):
                        continue

                data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
                if not isinstance(data, dict):
                    data = {}

                # Only track pregame full-game lines for movement badges.
                try:
                    if str(data.get("period") or "").strip().lower() != "full_game":
                        continue
                except Exception:
                    continue

                book = str(data.get("book") or "").strip() or ""

                total = _coerce_num(data.get("total"))
                spread_home = _coerce_num(data.get("spread_home"))

                st = state.setdefault(
                    (eid, book),
                    {
                        "total_prev": None,
                        "total_last": None,
                        "spread_prev": None,
                        "spread_last": None,
                        "ts_prev": None,
                        "ts_last": None,
                    },
                )

                # Update totals when distinct.
                if total is not None:
                    last = st.get("total_last")
                    if last is None:
                        st["total_last"] = total
                        st["ts_last"] = ts
                    elif float(total) != float(last):
                        st["total_prev"] = last
                        st["total_last"] = total
                        st["ts_prev"] = st.get("ts_last")
                        st["ts_last"] = ts

                # Update spreads when distinct.
                if spread_home is not None:
                    last = st.get("spread_last")
                    if last is None:
                        st["spread_last"] = spread_home
                        st["ts_last"] = ts
                    elif float(spread_home) != float(last):
                        st["spread_prev"] = last
                        st["spread_last"] = spread_home
                        st["ts_prev"] = st.get("ts_last")
                        st["ts_last"] = ts
    except Exception:
        return {}

    # Pick a single "best" book per event for UI surfacing.
    by_event: Dict[str, list[tuple[str, Dict[str, Any]]]] = {}
    for (eid, book), st in state.items():
        by_event.setdefault(eid, []).append((book, st))

    out: Dict[str, Dict[str, Any]] = {}
    for eid, items in by_event.items():
        if not items:
            continue

        def _item_sort_key(it: tuple[str, Dict[str, Any]]) -> tuple[int, float]:
            book, st = it
            ts_last = st.get("ts_last")
            te = _ts_epoch(ts_last)
            # Prefer known books, then newest timestamp.
            return (_book_rank(book), -(te if te is not None else -1.0))

        book, st = sorted(items, key=_item_sort_key)[0]
        ts_last = st.get("ts_last")
        age = _age_s(ts_last)
        if age is not None and float(age) > float(max_age_s):
            continue

        total_prev = st.get("total_prev")
        total_last = st.get("total_last")
        spread_prev = st.get("spread_prev")
        spread_last = st.get("spread_last")

        d_total = None
        if total_prev is not None and total_last is not None:
            try:
                d_total = float(total_last) - float(total_prev)
            except Exception:
                d_total = None
        d_spread = None
        if spread_prev is not None and spread_last is not None:
            try:
                d_spread = float(spread_last) - float(spread_prev)
            except Exception:
                d_spread = None

        badges = []
        book_disp = str(book or "").strip() or None
        if d_total is not None and abs(float(d_total)) >= float(min_total_pts):
            side_total = "over" if float(d_total) > 0 else "under"
            badges.append(
                {
                    "label": f"T {float(d_total):+.1f}",
                    "title": (
                        f"Total moved {float(total_prev):.1f}→{float(total_last):.1f} (Δ {float(d_total):+.1f})"
                        + (f" • {book_disp}" if book_disp else "")
                    ),
                    "kind": "total",
                }
            )
            badges.append(
                {
                    "label": f"STEAM TOTAL {side_total.upper()}",
                    "title": f"Material total move toward {side_total.title()} (steam)",
                    "kind": "steam_total",
                    "side": side_total,
                }
            )
        if d_spread is not None and abs(float(d_spread)) >= float(min_spread_pts):
            side_spread = "home" if float(d_spread) < 0 else "away"
            badges.append(
                {
                    "label": f"S {float(d_spread):+.1f}",
                    "title": (
                        f"Home spread moved {float(spread_prev):+.1f}→{float(spread_last):+.1f} (Δ {float(d_spread):+.1f})"
                        + (f" • {book_disp}" if book_disp else "")
                    ),
                    "kind": "spread",
                }
            )
            badges.append(
                {
                    "label": f"STEAM {side_spread.upper()} ATS",
                    "title": f"Material spread move toward {side_spread.title()} (steam)",
                    "kind": "steam_spread",
                    "side": side_spread,
                }
            )

        if not badges:
            continue

        out[eid] = {
            "event_id": eid,
            "book": book_disp,
            "total_prev": total_prev,
            "total_last": total_last,
            "delta_total": d_total,
            "spread_prev": spread_prev,
            "spread_last": spread_last,
            "delta_spread_home": d_spread,
            "ts_prev": st.get("ts_prev"),
            "ts_last": ts_last,
            "age_s": age,
            "badges": badges,
        }

    return out


_LOCK = threading.Lock()
_LAST_SEEN: Dict[str, Dict[str, Any]] = {}
_WARNED: set[str] = set()


def _env_truthy(v: Optional[str]) -> bool:
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _now_iso_z() -> str:
    # Avoid importing datetime on the hot path repeatedly.
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_json_obj_strict(obj: Any) -> Any:
    """Convert non-finite floats (NaN/Inf) into None recursively."""
    try:
        import math

        if obj is None:
            return None
        if isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        # numpy/pandas scalar -> python scalar
        if hasattr(obj, "item"):
            try:
                return _sanitize_json_obj_strict(obj.item())
            except Exception:
                pass
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(k, (str, int, float, bool)) or k is None:
                    kk = k
                else:
                    kk = str(k)
                out[str(kk)] = _sanitize_json_obj_strict(v)
            return out
        if isinstance(obj, (list, tuple, set)):
            return [_sanitize_json_obj_strict(v) for v in obj]
        return obj
    except Exception:
        return None


def _get_outputs_dir() -> Path:
    return Path(os.environ.get("NCAAB_OUTPUTS_DIR") or os.path.join(os.getcwd(), "outputs"))


def _get_snapshot_dir() -> Path:
    # Default under outputs/ so it gets picked up by existing artifact upload flows.
    base = os.environ.get("NCAAB_LIVE_SNAPSHOT_DIR")
    if base and str(base).strip():
        return Path(str(base).strip())
    return _get_outputs_dir() / "live_snapshots"


def _max_bytes() -> int:
    try:
        kb = int(str(os.environ.get("NCAAB_LIVE_SNAPSHOT_MAX_KB") or "20480").strip())
    except Exception:
        kb = 20480
    kb = max(256, min(kb, 512000))
    return kb * 1024


def _min_interval_s() -> float:
    try:
        v = float(str(os.environ.get("NCAAB_LIVE_SNAPSHOT_MIN_INTERVAL_S") or "8").strip())
    except Exception:
        v = 8.0
    return max(0.0, min(v, 120.0))


def _unchanged_window_s() -> float:
    try:
        v = float(str(os.environ.get("NCAAB_LIVE_SNAPSHOT_UNCHANGED_WINDOW_S") or "60").strip())
    except Exception:
        v = 60.0
    return max(0.0, min(v, 600.0))


def enabled() -> bool:
    """Whether live snapshot logging is enabled.

    Behavior:
      - If NCAAB_LIVE_SNAPSHOT_LOG is explicitly set, honor it.
      - If it is unset/blank and we're running on Render, default to enabled.

    Rationale: Render env var sync can drift when services pre-exist; default-on
    keeps cron polling useful while still allowing an explicit opt-out.
    """

    try:
        raw = os.environ.get("NCAAB_LIVE_SNAPSHOT_LOG")
    except Exception:
        raw = None
    if raw is None or not str(raw).strip():
        try:
            return bool(os.environ.get("RENDER_SERVICE_ID") or os.environ.get("RENDER_INSTANCE_ID") or os.environ.get("RENDER"))
        except Exception:
            return False
    return _env_truthy(raw)


def _stable_hash(data: Any) -> str:
    try:
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except Exception:
        raw = json.dumps(_sanitize_json_obj_strict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort file size cap.
    try:
        if path.exists() and int(path.stat().st_size) > _max_bytes():
            k = str(path)
            if k not in _WARNED:
                _WARNED.add(k)
            return
    except Exception:
        pass

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize_json_obj_strict(record), ensure_ascii=False) + "\n")
    except Exception:
        # Never break the API response on logging failure.
        return


def _should_log(key: str, payload_hash: str) -> bool:
    now = time.time()
    min_dt = _min_interval_s()
    unchanged_window = _unchanged_window_s()

    with _LOCK:
        prev = _LAST_SEEN.get(key)
        if isinstance(prev, dict):
            ts0 = float(prev.get("ts") or 0.0)
            h0 = str(prev.get("hash") or "")
            if min_dt > 0 and ts0 and (now - ts0) < min_dt:
                return False
            if unchanged_window > 0 and h0 and h0 == payload_hash and ts0 and (now - ts0) < unchanged_window:
                return False

        _LAST_SEEN[key] = {"ts": now, "hash": payload_hash}
        # Occasional prune.
        if len(_LAST_SEEN) > 8000:
            cutoff = now - 6 * 3600
            for k, v in list(_LAST_SEEN.items())[:2000]:
                try:
                    if float(v.get("ts") or 0.0) < cutoff:
                        _LAST_SEEN.pop(k, None)
                except Exception:
                    _LAST_SEEN.pop(k, None)
        return True


def _iter_event_records(
    endpoint: str,
    date_s: str,
    request_args: Dict[str, Any],
    payload: Dict[str, Any],
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (event_id, record) pairs; event_id may be '' for non-splittable payloads."""

    ts = _now_iso_z()

    # Split common shapes so analysis can be per-event.
    if endpoint in ("live_state", "live_pbp_stats"):
        games = payload.get("games")
        if isinstance(games, dict):
            for eid, g in games.items():
                eid_s = str(eid or "").strip()
                yield eid_s, {
                    "ts": ts,
                    "date": date_s,
                    "endpoint": endpoint,
                    "event_id": eid_s,
                    "request": dict(request_args),
                    "data": g,
                    "meta": {"schema_version": 1},
                }
            return

    if endpoint == "live_lines":
        lines = payload.get("lines")
        if isinstance(lines, dict):
            for eid, ln in lines.items():
                eid_s = str(eid or "").strip()
                yield eid_s, {
                    "ts": ts,
                    "date": date_s,
                    "endpoint": endpoint,
                    "event_id": eid_s,
                    "request": dict(request_args),
                    "data": ln,
                    "meta": {"schema_version": 1},
                }
            return

    yield "", {
        "ts": ts,
        "date": date_s,
        "endpoint": endpoint,
        "event_id": None,
        "request": dict(request_args),
        "data": payload,
        "meta": {"schema_version": 1},
    }


def log_live_api_payload(
    *,
    endpoint: str,
    date_s: str,
    request_args: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    """Append a compact snapshot of a live API payload to outputs/live_snapshots.

    Opt-in via env var:
      - NCAAB_LIVE_SNAPSHOT_LOG=1

    Writes JSONL under:
      - outputs/live_snapshots/live_<date>.jsonl

    This function is best-effort and should never raise.
    """

    try:
        if not enabled():
            return
        endpoint_s = str(endpoint or "").strip()
        if not endpoint_s:
            return
        date_s2 = str(date_s or "").strip()
        if not date_s2:
            return
        if not isinstance(payload, dict):
            return

        # Keep request args small and JSON-friendly.
        # IMPORTANT: redact secrets (e.g., cron_key) since these snapshots can
        # be downloaded for debugging.
        req_keep: Dict[str, Any] = {}
        try:
            redact_keys = {
                "cron_key",
                "key",
                "api_key",
                "apikey",
                "token",
                "access_token",
                "authorization",
                "x-ingest-token",
            }
            for k, v in (request_args or {}).items():
                ks = str(k or "").strip()
                if not ks:
                    continue
                if ks.lower() in redact_keys:
                    continue
                # Guard against huge args.
                vs = str(v) if v is not None else None
                if isinstance(vs, str) and len(vs) > 500:
                    vs = vs[:500] + "…"
                req_keep[ks] = vs
        except Exception:
            req_keep = {}

        out_dir = _get_snapshot_dir()
        out_path = out_dir / f"live_{date_s2}.jsonl"

        for eid, rec in _iter_event_records(endpoint_s, date_s2, req_keep, payload):
            key = f"{endpoint_s}|{date_s2}|{eid}" if eid else f"{endpoint_s}|{date_s2}|_bulk"
            h = _stable_hash(rec.get("data"))
            if not _should_log(key, h):
                continue
            rec["hash"] = h
            _append_jsonl(out_path, rec)
    except Exception:
        return
