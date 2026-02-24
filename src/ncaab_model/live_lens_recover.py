from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


@dataclass(frozen=True)
class LiveLensSignalsReconstructConfig:
    date: str
    out_dir: Path = Path("outputs")
    out_path: Path | None = None
    full_game_only: bool = False

    base_url: str | None = None
    download_projections: bool = True
    overwrite_projections: bool = False


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    dt.date.fromisoformat(s2)
    return s2


def projections_path(date: str, out_dir: Path) -> Path:
    d = _safe_date(date)
    return Path(out_dir) / f"live_lens_projections_{d}.jsonl"


def _parse_ts(iso: str | None) -> dt.datetime:
    s = str(iso or "").strip()
    if not s:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    # Handle both ...Z and offsets.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        x = dt.datetime.fromisoformat(s)
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return x
    except Exception:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def _read_jsonl(p: Path, max_lines: int = 500_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= int(max_lines):
                break
            s = (line or "").strip()
            if not s:
                continue
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    rows.append(j)
            except Exception:
                continue
    return rows


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def _to_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _driver_tags_from_parts(parts: Iterable[str]) -> list[str]:
    tags: list[str] = []

    def add(t: str) -> None:
        if t and t not in tags:
            tags.append(t)

    for p in parts:
        tl = str(p or "").strip().lower()
        if not tl:
            continue
        if tl.startswith("edge "):
            add("EDGE")
        elif tl.startswith("d "):
            add("SIM-GAP")
        elif tl.startswith("pace "):
            add("PACE")
        elif tl.startswith("ppp "):
            add("PPP")
        elif tl.startswith("shooting "):
            add("SHOOTING")
        elif tl.startswith("ft rate "):
            add("FT")
        elif tl.startswith("pbp+"):
            add("PBP")
        elif "late-over" in tl:
            add("LATE-OVER")
        elif "early-over" in tl:
            add("EARLY-OVER")
    return tags


def _reconstruct_one_total_signal(row: dict[str, Any]) -> dict[str, Any] | None:
    game_id = str(row.get("game_id") or "").strip()
    if not game_id:
        return None

    lens = row.get("lens")
    lens_s = str(lens or "").strip().lower()
    if lens_s in {"fg", "full", "fullgame", "full_game"}:
        lens_s = "full_game"
    elif lens_s in {"1h", "first_half", "half1"}:
        lens_s = "1h"
    elif lens_s in {"2h", "second_half", "half2"}:
        lens_s = "2h"

    horizon = _to_num(row.get("horizon"))
    elapsed = _to_num(row.get("elapsed"))
    remaining = _to_num(row.get("remaining"))
    if elapsed is None and horizon is not None and remaining is not None:
        elapsed = horizon - remaining
    if elapsed is None or horizon is None:
        return None

    act = _to_num(row.get("total_points"))
    line = _to_num(row.get("live_line"))
    proj = _to_num(row.get("proj_blend"))
    if proj is None:
        proj = _to_num(row.get("proj_final"))

    if line is None or proj is None:
        return None

    edge = proj - line
    if edge > 0.01:
        side = "over"
    elif edge < -0.01:
        side = "under"
    else:
        return None

    # Delta between sim expected total at this time and actual so far.
    delta = None
    sim_exp_total = _to_num(row.get("sim_exp_total"))
    if sim_exp_total is not None and act is not None:
        delta = sim_exp_total - act

    pbp = row.get("pbp") if isinstance(row.get("pbp"), dict) else {}
    poss = _to_num(pbp.get("poss"))
    fg_pct = _to_num(pbp.get("fgPct"))
    tp_pct = _to_num(pbp.get("tpPct"))
    ft_pct = _to_num(pbp.get("ftPct"))
    ftr = _to_num(pbp.get("ftr"))

    # UI defaults (we do not have per-day tuning in the projections log).
    pbp_n_scale = 70.0
    pace_hi = 3.25
    pace_lo = 2.75
    pps_hi = 1.18
    pps_lo = 0.95

    poss_rate = (poss / elapsed) if (poss is not None and elapsed > 0) else None
    ppp = (act / poss) if (act is not None and poss is not None and poss > 0) else None

    w_pbp = 1.0
    if poss is not None:
        w_pbp = min(1.0, max(0.0, poss / pbp_n_scale))

    driver_parts: list[str] = []
    driver_parts.append(f"Edge {edge:+.1f}")
    if delta is not None:
        driver_parts.append(f"d {delta:+.1f}")
    if poss_rate is not None:
        pace_tag = f"pace {poss_rate:.2f}"
        if poss_rate >= pace_hi:
            pace_tag += " hi"
        elif poss_rate <= pace_lo:
            pace_tag += " lo"
        driver_parts.append(pace_tag)
    if ppp is not None:
        eff_tag = f"ppp {ppp:.2f}"
        if ppp >= pps_hi:
            eff_tag += " hi"
        elif ppp <= pps_lo:
            eff_tag += " lo"
        driver_parts.append(eff_tag)

    hot = ((fg_pct is not None and fg_pct >= 0.60) or (tp_pct is not None and tp_pct >= 0.52) or (ft_pct is not None and ft_pct >= 0.92))
    cold = ((fg_pct is not None and fg_pct <= 0.38) or (tp_pct is not None and tp_pct <= 0.22))
    if hot:
        driver_parts.append("shooting hot")
    elif cold:
        driver_parts.append("shooting cold")

    if ftr is not None:
        if ftr >= 0.55:
            driver_parts.append("FT rate high")
        elif ftr <= 0.22:
            driver_parts.append("FT rate low")

    adj = 0.0
    if delta is not None:
        if side == "over":
            adj += 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
        else:
            adj += 1.0 if delta < 0 else (-1.0 if delta > 0 else 0.0)

    pace_adj = 0.0
    if poss_rate is not None:
        if side == "over":
            if poss_rate >= pace_hi:
                pace_adj += 1.0
            elif poss_rate <= pace_lo:
                pace_adj -= 1.0
        else:
            if poss_rate <= pace_lo:
                pace_adj += 1.0
            elif poss_rate >= pace_hi:
                pace_adj -= 1.0

    eff_adj = 0.0
    if ppp is not None:
        if side == "over":
            if ppp <= pps_lo:
                eff_adj += 1.0
            elif ppp >= pps_hi:
                eff_adj -= 1.0
        else:
            if ppp >= pps_hi:
                eff_adj += 1.0
            elif ppp <= pps_lo:
                eff_adj -= 1.0

    if side == "over":
        if hot:
            adj -= 2.0
        if cold:
            adj += 1.0
    else:
        if hot:
            adj += 1.0
        if cold:
            adj -= 2.0

    if ftr is not None:
        if side == "over" and ftr >= 0.55:
            adj += 0.5
        if side == "under" and ftr <= 0.22:
            adj += 0.5

    adj += w_pbp * (pace_adj + eff_adj)

    strength = abs(edge) + adj

    # Mirror UI's base thresholds (ignoring market price gating because projections do not contain prices).
    if horizon <= 20.5:
        min_elapsed = 4.0
        thr = 5.0
        thr_watch = max(3.0, thr - 1.0)
    else:
        min_elapsed = 6.0
        thr = 7.0
        thr_watch = max(4.0, thr - 2.0)

    if elapsed < min_elapsed:
        return None

    # Absolute odds window gate we can replicate.
    # (In the UI this suppresses FG markets in the final minutes.)
    if lens_s == "full_game" and remaining is not None and remaining < 4:
        return None

    is_bet = strength >= thr
    is_watch = (not is_bet) and (strength >= thr_watch)

    is_candidate = False
    candidate_kind = None
    if (not is_bet) and (not is_watch) and side == "under":
        cand_min = max(0.0, thr_watch - 1.0)
        if strength >= cand_min and strength < thr_watch:
            is_candidate = True
            candidate_kind = "under_below_watch"

    if not (is_bet or is_watch or is_candidate):
        return None

    line_q = round(line * 2) / 2
    driver = " | ".join([p for p in driver_parts if str(p or "").strip()])
    tags = _driver_tags_from_parts(driver_parts)

    out: dict[str, Any] = {
        "schema_version": 2,
        "logic_version": "live_lens_reconstruct_v1",
        "ts": str(row.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")),
        "date": row.get("date"),
        "game_id": game_id,
        "lens": lens_s or None,
        "kind": "total",
        "horizon": horizon,
        "elapsed": elapsed,
        "remaining": remaining,
        "total_points": act,
        "live_line": line_q,
        "side": side,
        "edge": edge,
        "strength": strength,
        "thr": thr,
        "thr_watch": thr_watch,
        "driver": (driver or None),
        "driver_tags": (tags if tags else None),
        "is_bet": bool(is_bet),
        "is_watch": bool(is_watch),
        "is_candidate": bool(is_candidate),
        "candidate_kind": candidate_kind,
        "reconstructed_from": "projections",
    }

    return out


def ensure_projections_downloaded(cfg: LiveLensSignalsReconstructConfig) -> dict[str, Any] | None:
    if not cfg.base_url or not cfg.download_projections:
        return None

    url = cfg.base_url.rstrip("/") + "/api/download_live_lens_projections"
    params = {"date": _safe_date(cfg.date)}

    dst = projections_path(cfg.date, cfg.out_dir)
    if dst.exists() and not cfg.overwrite_projections:
        return {"status": "skipped", "path": str(dst), "reason": "exists"}

    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        return {"status": "error", "url": str(r.url), "http_status": int(r.status_code), "text": (r.text[:500] if r.text else "")}

    _atomic_write_bytes(dst, r.content)
    return {"status": "ok", "path": str(dst), "bytes": int(len(r.content))}


def _try_download_original_signals_jsonl(cfg: LiveLensSignalsReconstructConfig, date_s: str) -> dict[str, Any] | None:
    """Best-effort download of the original Live Lens signals JSONL.

    This is used as a fallback when projections don't contain any market line, so
    signals cannot be reconstructed.
    """

    if not cfg.base_url:
        return None

    url = cfg.base_url.rstrip("/") + "/api/download_live_lens_signals"
    params = {"date": date_s}

    try:
        r = requests.get(url, params=params, timeout=60)
    except Exception as e:
        return {"status": "error", "url": str(url), "error": str(e)}

    if r.status_code == 404:
        return {"status": "missing", "url": str(r.url), "http_status": 404}

    if r.status_code != 200:
        return {
            "status": "error",
            "url": str(r.url),
            "http_status": int(r.status_code),
            "text": (r.text[:500] if r.text else ""),
        }

    content = r.content or b""
    if len(content) <= 2:
        return {"status": "empty", "url": str(r.url), "http_status": 200, "bytes": int(len(content))}

    # Sanity-check: ensure at least one JSON row parses.
    parsed = 0
    try:
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                parsed += 1
            except Exception:
                continue
            if parsed >= 1:
                break
    except Exception:
        parsed = 0

    return {
        "status": "ok",
        "url": str(r.url),
        "http_status": 200,
        "bytes": int(len(content)),
        "parsed_rows_min": int(parsed),
        "content": content,
    }


def reconstruct_signals_from_projections(cfg: LiveLensSignalsReconstructConfig) -> dict[str, Any]:
    date_s = _safe_date(cfg.date)
    out_dir = Path(cfg.out_dir)

    dl = ensure_projections_downloaded(cfg)

    proj_p = projections_path(date_s, out_dir)
    if not proj_p.exists():
        return {
            "status": "missing",
            "date": date_s,
            "message": f"No projections file at {proj_p}",
            "download": dl,
        }

    rows = _read_jsonl(proj_p)
    if not rows:
        return {
            "status": "missing",
            "date": date_s,
            "message": f"No projection rows parsed from {proj_p}",
            "download": dl,
        }

    # If the projections log doesn't contain any market line, we cannot reconstruct
    # Over/Under signals (edge requires line).
    live_line_n = 0
    lens_counts: dict[str, int] = {}
    for r in rows:
        if _to_num(r.get("live_line")) is not None:
            live_line_n += 1
        lk = str(r.get("lens") or "").strip() or "(none)"
        lens_counts[lk] = lens_counts.get(lk, 0) + 1
    if live_line_n == 0:
        fallback = _try_download_original_signals_jsonl(cfg, date_s)
        if isinstance(fallback, dict) and fallback.get("status") == "ok" and fallback.get("parsed_rows_min", 0) >= 1:
            out_path = cfg.out_path
            if out_path is None:
                out_path = out_dir / f"live_lens_signals_recovered_{date_s}.jsonl"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(Path(out_path), fallback.get("content", b""))
            fallback.pop("content", None)
            return {
                "status": "downloaded_original",
                "date": date_s,
                "projections_path": str(proj_p),
                "out_path": str(out_path),
                "rows_in": int(len(rows)),
                "rows_with_live_line": 0,
                "lens_counts": lens_counts,
                "download": dl,
                "fallback_download": fallback,
                "message": "Projections contain no live_line; downloaded original signals instead.",
            }

        if isinstance(fallback, dict) and "content" in fallback:
            fallback.pop("content", None)

        return {
            "status": "blocked",
            "date": date_s,
            "projections_path": str(proj_p),
            "rows_in": int(len(rows)),
            "rows_with_live_line": 0,
            "lens_counts": lens_counts,
            "download": dl,
            "fallback_download": fallback,
            "message": "Cannot reconstruct totals signals because projections contain no live_line (market total).",
            "hint": "Recovery requires either the original signals JSONL, or a historical odds source to supply totals lines.",
        }

    rows.sort(key=lambda r: _parse_ts(r.get("ts")))

    out_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        if cfg.full_game_only:
            lens = str(r.get("lens") or "").strip().lower()
            if lens not in {"full_game", "fg", "full", "fullgame"}:
                continue

        sig = _reconstruct_one_total_signal(r)
        if not sig:
            continue

        # Deduplicate: keep only one record per (game,lens,minute,line,side,class)
        gid = str(sig.get("game_id") or "")
        lens_k = str(sig.get("lens") or "")
        horizon = _to_num(sig.get("horizon"))
        elapsed = _to_num(sig.get("elapsed"))
        bucket = int(math.floor(elapsed)) if elapsed is not None else -1
        line_q = _to_num(sig.get("live_line"))
        side = str(sig.get("side") or "")
        cls = "B" if sig.get("is_bet") else ("W" if sig.get("is_watch") else "C")
        key = f"{gid}|{lens_k}|{horizon}|{bucket}|{line_q}|{side}|{cls}"
        if key in seen:
            continue
        seen.add(key)

        sig["date"] = date_s
        out_rows.append(sig)

    out_path = cfg.out_path
    if out_path is None:
        out_path = out_dir / f"live_lens_signals_reconstructed_{date_s}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for j in out_rows:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "date": date_s,
        "projections_path": str(proj_p),
        "out_path": str(out_path),
        "rows_in": int(len(rows)),
        "rows_out": int(len(out_rows)),
        "download": dl,
        "notes": [
            "Reconstruction is approximate: projections do not include market prices or all tuning/calibration fields.",
            "Only totals signals are reconstructed (ATS/ML require spread/price data not present in projections logs).",
        ],
    }
