from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import settings


@dataclass(frozen=True)
class LiveLensTuning:
    pace_hi: float
    pace_lo: float
    pps_hi: float
    pps_lo: float
    pbp_n_scale: float = 70.0
    shot_proxy_ft_weight: float = 0.44

    def to_dict(self) -> dict[str, Any]:
        return {
            "pace_hi": float(self.pace_hi),
            "pace_lo": float(self.pace_lo),
            "pps_hi": float(self.pps_hi),
            "pps_lo": float(self.pps_lo),
            "pbp_n_scale": float(self.pbp_n_scale),
            "shot_proxy_ft_weight": float(self.shot_proxy_ft_weight),
        }


DEFAULT_TUNING = LiveLensTuning(
    pace_hi=3.25,
    pace_lo=2.75,
    pps_hi=1.18,
    pps_lo=0.95,
    pbp_n_scale=70.0,
    shot_proxy_ft_weight=0.44,
)


def _iter_cache_files(days: int, max_files: int) -> list[Path]:
    cache_dir = settings.data_dir / "cache" / "espn_pbp"
    if not cache_dir.exists():
        return []

    cutoff = None
    if days > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(days))

    files: list[Path] = []
    for p in cache_dir.glob("*.json"):
        try:
            if not p.is_file():
                continue
            if cutoff is not None:
                mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc)
                if mtime < cutoff:
                    continue
            files.append(p)
        except Exception:
            continue

    # newest first, cap
    try:
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return files[: max(0, int(max_files))]


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _compute_timeline(payload: dict[str, Any]) -> list[tuple[float, int | None, int, int]]:
    """Return a timeline of (elapsed_min, total_points, fga, fta) records."""

    try:
        from .data.adapters import espn_playbyplay as pbp_mod
    except Exception:
        return []

    # cumulative counts
    fga = 0
    fta = 0
    total_points: int | None = None
    timeline: list[tuple[float, int | None, int, int]] = []

    for p in pbp_mod._iter_plays(payload):  # type: ignore[attr-defined]
        try:
            elapsed = pbp_mod._play_elapsed_min(p)  # type: ignore[attr-defined]
        except Exception:
            elapsed = None
        if elapsed is None:
            continue

        # Update score if present
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is not None and a_s is not None:
                total_points = int(hs) + int(a_s)
        except Exception:
            pass

        # Update attempts
        try:
            if bool(p.get("shootingPlay")):
                pa = p.get("pointsAttempted")
                if pa is not None:
                    pa_i = int(pa)
                    if pa_i == 1:
                        fta += 1
                    elif pa_i in (2, 3):
                        fga += 1
        except Exception:
            pass

        timeline.append((float(elapsed), total_points, int(fga), int(fta)))

    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass
    return timeline


def _sample_metrics(
    timeline: list[tuple[float, int | None, int, int]],
    *,
    shot_proxy_ft_weight: float,
    min_elapsed: int,
    max_elapsed: int,
    step_min: int,
    min_shot_proxy: float,
) -> tuple[list[float], list[float]]:
    """Return (shot_rate_per_min_samples, pps_samples)."""
    if not timeline:
        return [], []

    # Track latest record as we move forward
    idx = 0
    cur_total: int | None = None
    cur_fga = 0
    cur_fta = 0

    rates: list[float] = []
    pps_list: list[float] = []

    endpoints = list(range(int(min_elapsed), int(max_elapsed) + 1, int(step_min)))
    for m in endpoints:
        m_f = float(m)
        while idx < len(timeline) and timeline[idx][0] <= m_f + 1e-9:
            _, cur_total, cur_fga, cur_fta = timeline[idx]
            idx += 1

        if cur_total is None:
            continue
        if m_f <= 0.01:
            continue
        shot_proxy = float(cur_fga) + float(shot_proxy_ft_weight) * float(cur_fta)
        if not np.isfinite(shot_proxy) or shot_proxy < float(min_shot_proxy):
            continue

        shot_rate = shot_proxy / m_f
        pps = float(cur_total) / shot_proxy
        if np.isfinite(shot_rate):
            rates.append(float(shot_rate))
        if np.isfinite(pps):
            pps_list.append(float(pps))

    return rates, pps_list


def compute_live_lens_tuning(
    *,
    days: int = 21,
    max_files: int = 500,
    min_elapsed: int = 6,
    max_elapsed: int = 30,
    step_min: int = 2,
    min_shot_proxy: float = 12.0,
    shot_proxy_ft_weight: float = 0.44,
) -> tuple[LiveLensTuning, dict[str, Any]]:
    """Compute thresholds from cached ESPN PBP.

    The goal is not to be "perfect"—just to keep thresholds aligned to the
    current season's typical pace/efficiency so Watch/Bet triggers are stable.
    """

    files = _iter_cache_files(days=int(days), max_files=int(max_files))
    if not files:
        return DEFAULT_TUNING, {"source": "default", "files": 0, "samples": 0}

    all_rates: list[float] = []
    all_pps: list[float] = []
    used_files = 0

    for p in files:
        payload = _safe_read_json(p)
        if not isinstance(payload, dict):
            continue
        timeline = _compute_timeline(payload)
        if not timeline:
            continue
        rates, pps_list = _sample_metrics(
            timeline,
            shot_proxy_ft_weight=float(shot_proxy_ft_weight),
            min_elapsed=int(min_elapsed),
            max_elapsed=int(max_elapsed),
            step_min=int(step_min),
            min_shot_proxy=float(min_shot_proxy),
        )
        if rates:
            all_rates.extend(rates)
        if pps_list:
            all_pps.extend(pps_list)
        used_files += 1

    if len(all_rates) < 80 or len(all_pps) < 80:
        # Not enough signal; keep the defaults.
        return DEFAULT_TUNING, {
            "source": "default_insufficient_samples",
            "files": int(used_files),
            "samples": int(min(len(all_rates), len(all_pps))),
        }

    med_rate = float(np.median(np.asarray(all_rates, dtype=float)))
    med_pps = float(np.median(np.asarray(all_pps, dtype=float)))

    # Convert medians into loose thresholds.
    pace_hi = float(med_rate + 0.35)
    pace_lo = float(med_rate - 0.35)
    pps_hi = float(med_pps + 0.15)
    pps_lo = float(med_pps - 0.15)

    # Clamp into sane bounds.
    pace_hi = float(min(max(pace_hi, 2.9), 3.9))
    pace_lo = float(min(max(pace_lo, 2.2), 3.4))
    pps_hi = float(min(max(pps_hi, 1.05), 1.35))
    pps_lo = float(min(max(pps_lo, 0.80), 1.10))

    tuning = LiveLensTuning(
        pace_hi=pace_hi,
        pace_lo=pace_lo,
        pps_hi=pps_hi,
        pps_lo=pps_lo,
        pbp_n_scale=DEFAULT_TUNING.pbp_n_scale,
        shot_proxy_ft_weight=float(shot_proxy_ft_weight),
    )

    meta = {
        "source": "cache_espn_pbp",
        "days": int(days),
        "files": int(used_files),
        "samples": int(min(len(all_rates), len(all_pps))),
        "medians": {"shot_rate_per_min": float(med_rate), "pps": float(med_pps)},
    }
    return tuning, meta


def write_live_lens_tuning(out_path: Path, tuning: LiveLensTuning, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "tuning": tuning.to_dict(),
        "meta": meta or {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
