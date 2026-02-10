from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


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
    return _env_truthy(os.environ.get("NCAAB_LIVE_SNAPSHOT_LOG"))


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
        req_keep: Dict[str, Any] = {}
        try:
            for k, v in (request_args or {}).items():
                ks = str(k or "").strip()
                if not ks:
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
