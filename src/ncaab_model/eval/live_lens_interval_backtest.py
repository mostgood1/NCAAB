from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .live_snapshot_features import market_blend_weight
from .pbp_synthetic_snapshots import _payload_date_utc, _play_elapsed_min, _iter_plays


@dataclass(frozen=True)
class BacktestConfig:
    step_sec: int = 30
    min_elapsed_min: float = 6.0
    max_elapsed_min: float = 39.5
    horizon_min: float = 40.0
    ft_weight: float = 0.44
    use_tuning_clamps: bool = True


@dataclass(frozen=True)
class LiveLensClamps:
    pace_lo: float
    pace_hi: float
    pps_lo: float
    pps_hi: float


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _coerce_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return float(v)
    except Exception:
        return None


def _coerce_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_pregame_total_line(payload: dict[str, Any]) -> Optional[float]:
    # Prefer pickcenter
    try:
        pc = payload.get("pickcenter")
        if isinstance(pc, list) and pc:
            pc0 = pc[0] if isinstance(pc[0], dict) else None
            if isinstance(pc0, dict):
                total = _coerce_float(pc0.get("overUnder"))
                if total is None:
                    total = _coerce_float(pc0.get("total"))
                return total
    except Exception:
        pass

    try:
        odds = payload.get("odds")
        if isinstance(odds, list) and odds:
            for o in odds:
                if not isinstance(o, dict):
                    continue
                total = _coerce_float(o.get("overUnder"))
                if total is None:
                    total = _coerce_float(o.get("total"))
                if total is not None:
                    return total
    except Exception:
        pass

    return None


def _final_total_from_plays(payload: dict[str, Any]) -> Optional[int]:
    last_total: Optional[int] = None
    for p in _iter_plays(payload):
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            t = int(hs) + int(a_s)
            last_total = int(t)
        except Exception:
            continue
    return last_total


def _is_shooting_play(p: dict[str, Any]) -> bool:
    try:
        if bool(p.get("shootingPlay")):
            return True
    except Exception:
        pass
    try:
        txt = str(p.get("text") or "").strip().lower()
        if not txt:
            return False
        return ("makes" in txt) or ("misses" in txt) or ("free throw" in txt)
    except Exception:
        return False


def _update_shot_counts(p: dict[str, Any], fga: int, fta: int) -> tuple[int, int]:
    if not _is_shooting_play(p):
        return fga, fta
    try:
        pa = p.get("pointsAttempted")
        if pa is None:
            # Heuristic from text
            txt = str(p.get("text") or "").lower()
            if "free throw" in txt:
                return fga, fta + 1
            if "3-pt" in txt or "three" in txt:
                return fga + 1, fta
            if "2-pt" in txt or "layup" in txt or "jumper" in txt or "dunk" in txt:
                return fga + 1, fta
            return fga, fta
        pa_i = int(pa)
        if pa_i == 1:
            fta += 1
        elif pa_i in (2, 3):
            fga += 1
    except Exception:
        pass
    return int(fga), int(fta)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(v))))


def _period_from_elapsed(elapsed_min: float) -> int:
    return 1 if float(elapsed_min) <= 20.0 else 2


def iter_interval_rows_for_game(
    *,
    event_id: str,
    payload: dict[str, Any],
    config: BacktestConfig,
    clamps: Optional[LiveLensClamps] = None,
) -> Iterable[dict[str, Any]]:
    eid = str(event_id)
    line_total = _extract_pregame_total_line(payload)
    actual_total = _final_total_from_plays(payload)
    if actual_total is None:
        return []

    # Build a timeline keyed by elapsed seconds.
    timeline: list[tuple[int, int, int, int]] = []  # (elapsed_sec, total_pts, fga, fta)
    fga = 0
    fta = 0
    for p in _iter_plays(payload):
        elapsed_min = _play_elapsed_min(p)
        if elapsed_min is None:
            continue
        if elapsed_min < 0:
            continue
        if elapsed_min > float(config.horizon_min) + 0.01:
            continue
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            total_pts = int(hs) + int(a_s)
        except Exception:
            continue

        fga, fta = _update_shot_counts(p, fga, fta)
        elapsed_sec = int(round(float(elapsed_min) * 60.0))
        if elapsed_sec < 0:
            elapsed_sec = 0
        if elapsed_sec > int(round(float(config.horizon_min) * 60.0)):
            elapsed_sec = int(round(float(config.horizon_min) * 60.0))
        timeline.append((int(elapsed_sec), int(total_pts), int(fga), int(fta)))

    if not timeline:
        return []

    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass

    min_sec = int(round(float(config.min_elapsed_min) * 60.0))
    max_sec = int(round(float(config.max_elapsed_min) * 60.0))
    horizon_sec = int(round(float(config.horizon_min) * 60.0))
    step = max(1, int(config.step_sec))

    idx = 0
    cur_total = 0
    cur_fga = 0
    cur_fta = 0

    for s in range(min_sec, min(max_sec, horizon_sec) + 1, step):
        while idx < len(timeline) and timeline[idx][0] <= s:
            _, cur_total, cur_fga, cur_fta = timeline[idx]
            idx += 1

        elapsed_min = float(s) / 60.0
        remaining_min = max(0.0, float(config.horizon_min) - float(elapsed_min))
        period = _period_from_elapsed(elapsed_min)

        total_points = int(cur_total)
        shot_proxy = float(cur_fga) + float(config.ft_weight) * float(cur_fta)

        proj_pace = None
        if elapsed_min > 0.01:
            proj_pace = (float(total_points) / float(elapsed_min)) * float(config.horizon_min)

        proj_clamped = None
        poss_rate = None
        ppp = None
        if shot_proxy > 0.01 and elapsed_min > 0.01:
            poss_rate = float(shot_proxy) / float(elapsed_min)
            ppp = float(total_points) / float(shot_proxy)

        if proj_pace is not None:
            proj_clamped = float(proj_pace)
            if config.use_tuning_clamps and clamps is not None and poss_rate is not None and ppp is not None:
                pr = _clamp(float(poss_rate), clamps.pace_lo, clamps.pace_hi)
                eff = _clamp(float(ppp), clamps.pps_lo, clamps.pps_hi)
                proj_clamped = float(pr) * float(config.horizon_min) * float(eff)

        w = market_blend_weight(elapsed_min, horizon_min=float(config.horizon_min))
        proj_blend = None
        if line_total is not None and proj_clamped is not None:
            proj_blend = (1.0 - float(w)) * float(proj_clamped) + float(w) * float(line_total)
        else:
            proj_blend = proj_clamped

        err_pace = (float(proj_pace) - float(actual_total)) if (proj_pace is not None) else None
        err_clamped = (float(proj_clamped) - float(actual_total)) if (proj_clamped is not None) else None
        err_blend = (float(proj_blend) - float(actual_total)) if (proj_blend is not None) else None

        yield {
            "event_id": eid,
            "elapsed_sec": int(s),
            "elapsed_min": float(elapsed_min),
            "remaining_min": float(remaining_min),
            "period": int(period),
            "total_points": int(total_points),
            "shot_proxy": float(shot_proxy),
            "poss_rate": float(poss_rate) if poss_rate is not None else None,
            "ppp": float(ppp) if ppp is not None else None,
            "line_total": float(line_total) if line_total is not None else None,
            "actual_total": int(actual_total),
            "proj_pace": float(proj_pace) if proj_pace is not None else None,
            "proj_clamped": float(proj_clamped) if proj_clamped is not None else None,
            "proj_blend": float(proj_blend) if proj_blend is not None else None,
            "err_pace": float(err_pace) if err_pace is not None else None,
            "err_clamped": float(err_clamped) if err_clamped is not None else None,
            "err_blend": float(err_blend) if err_blend is not None else None,
        }


def _infer_clamps_from_tuning() -> LiveLensClamps:
    try:
        from ..live_lens_tuning import DEFAULT_TUNING

        t = DEFAULT_TUNING
        return LiveLensClamps(pace_lo=float(t.pace_lo), pace_hi=float(t.pace_hi), pps_lo=float(t.pps_lo), pps_hi=float(t.pps_hi))
    except Exception:
        return LiveLensClamps(pace_lo=2.75, pace_hi=3.25, pps_lo=0.95, pps_hi=1.18)


def _date_from_payload_or_fallback(payload: dict[str, Any]) -> Optional[str]:
    d = _payload_date_utc(payload)
    return d.isoformat() if isinstance(d, dt.date) else None


def run_interval_backtest(
    *,
    cache_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: Path,
    step_sec: int = 30,
    min_elapsed_min: float = 6.0,
    max_elapsed_min: float = 39.5,
    max_files: int = 0,
    use_tuning_clamps: bool = True,
) -> dict[str, Any]:
    from .pbp_synthetic_snapshots import index_pbp_cache_by_date

    cache_dir = Path(cache_dir)
    idx = index_pbp_cache_by_date(cache_dir=cache_dir, start_date=start_date, end_date=end_date, max_files=max_files)

    clamps = _infer_clamps_from_tuning() if use_tuning_clamps else None

    rows: list[dict[str, Any]] = []
    games = 0
    skipped = 0

    for ds, eids in sorted(idx.by_date.items()):
        for eid in eids:
            p = cache_dir / f"{eid}.json"
            payload = _safe_read_json(p)
            if not isinstance(payload, dict):
                skipped += 1
                continue
            games += 1
            cfg = BacktestConfig(step_sec=int(step_sec), min_elapsed_min=float(min_elapsed_min), max_elapsed_min=float(max_elapsed_min), use_tuning_clamps=bool(use_tuning_clamps))
            for r in iter_interval_rows_for_game(event_id=str(eid), payload=payload, config=cfg, clamps=clamps):
                r["date"] = ds
                rows.append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = summarize_interval_backtest(df)
    return {
        "status": "ok",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "cache_dir": str(cache_dir),
        "out_csv": str(out_csv),
        "rows": int(len(df)),
        "games": int(games),
        "skipped": int(skipped),
        "summary": summary,
    }


def _metric_block(err: pd.Series) -> dict[str, float | int | None]:
    try:
        e = pd.to_numeric(err, errors="coerce").dropna()
        if e.empty:
            return {"n": 0, "mae": None, "rmse": None, "bias": None}
        mae = float(np.mean(np.abs(e.values)))
        rmse = float(np.sqrt(np.mean((e.values) ** 2)))
        bias = float(np.mean(e.values))
        return {"n": int(len(e)), "mae": mae, "rmse": rmse, "bias": bias}
    except Exception:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}


def summarize_interval_backtest(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"overall": {}, "by_elapsed_bucket": []}

    out: dict[str, Any] = {}
    out["overall"] = {
        "pace": _metric_block(df.get("err_pace")),
        "clamped": _metric_block(df.get("err_clamped")),
        "blend": _metric_block(df.get("err_blend")),
    }

    # Bucket by elapsed minutes into 2-min bins for readability.
    try:
        em = pd.to_numeric(df.get("elapsed_min"), errors="coerce")
        bucket = (np.floor(em / 2.0) * 2.0).astype("Int64")
        df2 = df.copy()
        df2["elapsed_bucket_min"] = bucket
        rows: list[dict[str, Any]] = []
        for b, g in df2.dropna(subset=["elapsed_bucket_min"]).groupby("elapsed_bucket_min"):
            rows.append(
                {
                    "elapsed_bucket_min": int(b),
                    "pace": _metric_block(g.get("err_pace")),
                    "clamped": _metric_block(g.get("err_clamped")),
                    "blend": _metric_block(g.get("err_blend")),
                }
            )
        rows.sort(key=lambda r: r["elapsed_bucket_min"])
        out["by_elapsed_bucket"] = rows
    except Exception:
        out["by_elapsed_bucket"] = []

    return out
