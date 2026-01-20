from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ncaab_model.data.adapters.espn_playbyplay import fetch_playbyplay, extract_cum_totals_5min


@dataclass(frozen=True)
class TuneSegments5MinConfig:
    out_dir: Path
    start: str
    end: str
    use_cache: bool = True
    sleep_seconds: float = 0.15
    max_games: int = 0
    shrink_to_uniform: float = 0.10  # 0=no shrink, 1=all uniform
    out_path: Path | None = None


def _date_range(start_iso: str, end_iso: str) -> Iterable[dt.date]:
    s = dt.date.fromisoformat(start_iso)
    e = dt.date.fromisoformat(end_iso)
    cur = s
    one = dt.timedelta(days=1)
    while cur <= e:
        yield cur
        cur += one


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.clip(v, 0.0, None)
    s = float(v.sum())
    if s <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return v / s


def tune_segment_weights(cfg: TuneSegments5MinConfig) -> dict:
    # Accumulate segment shares in each half.
    # Half1 segments correspond to endpoints 5,10,15,20.
    # Half2 segments correspond to 25,30,35,40 (relative to halftime).
    sum_share_h1 = np.zeros(4, dtype=float)
    sum_share_h2 = np.zeros(4, dtype=float)
    n_h1 = 0
    n_h2 = 0
    n_games = 0
    n_missing = 0

    for d in _date_range(cfg.start, cfg.end):
        date_iso = d.isoformat()

        # Use any locally available sim_segments_<date>.csv to enumerate game_ids, otherwise
        # fall back to cached ESPN scoreboard (if present) is out-of-scope here.
        seg_path = Path(cfg.out_dir) / f"sim_segments_{date_iso}.csv"
        game_ids: list[str] = []
        if seg_path.exists():
            try:
                import pandas as pd

                seg = pd.read_csv(seg_path)
                if not seg.empty and "game_id" in seg.columns:
                    game_ids = sorted({str(x) for x in seg["game_id"].astype(str).tolist() if str(x).strip()})
            except Exception:
                game_ids = []

        for gid in game_ids:
            if cfg.max_games and int(cfg.max_games) > 0 and n_games >= int(cfg.max_games):
                break

            payload = fetch_playbyplay(gid, use_cache=cfg.use_cache)
            fetched_from = payload.get("_fetched_from") if isinstance(payload, dict) else None
            did_network = isinstance(fetched_from, str) and fetched_from.startswith("network")
            plays = payload.get("plays") if isinstance(payload, dict) else None
            if payload is None or not isinstance(plays, list) or len(plays) == 0:
                n_missing += 1
                if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                    time.sleep(float(cfg.sleep_seconds))
                continue

            cum = extract_cum_totals_5min(payload)
            m = {int(x.end_min): int(x.total_score) for x in cum}
            if not all(k in m for k in (5, 10, 15, 20, 25, 30, 35, 40)):
                n_missing += 1
                if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                    time.sleep(float(cfg.sleep_seconds))
                continue

            # Segment points
            s1 = np.array([m[5] - 0, m[10] - m[5], m[15] - m[10], m[20] - m[15]], dtype=float)
            s2 = np.array([m[25] - m[20], m[30] - m[25], m[35] - m[30], m[40] - m[35]], dtype=float)

            t1 = float(m[20])
            t2 = float(m[40] - m[20])

            if t1 > 0:
                sum_share_h1 += (s1 / t1)
                n_h1 += 1
            if t2 > 0:
                sum_share_h2 += (s2 / t2)
                n_h2 += 1

            n_games += 1

            if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                time.sleep(float(cfg.sleep_seconds))

        if cfg.max_games and int(cfg.max_games) > 0 and n_games >= int(cfg.max_games):
            break

    # Convert to mean shares
    w1 = (sum_share_h1 / float(max(n_h1, 1))).astype(float)
    w2 = (sum_share_h2 / float(max(n_h2, 1))).astype(float)

    w1 = _normalize(w1)
    w2 = _normalize(w2)

    # Optional shrinkage toward uniform
    shrink = float(np.clip(float(cfg.shrink_to_uniform), 0.0, 1.0))
    uni = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    w1 = _normalize((1.0 - shrink) * w1 + shrink * uni)
    w2 = _normalize((1.0 - shrink) * w2 + shrink * uni)

    out = {
        "start": cfg.start,
        "end": cfg.end,
        "method": "empirical_points_share_from_pbp",
        "games_used": int(n_games),
        "half1_games": int(n_h1),
        "half2_games": int(n_h2),
        "pbp_missing_or_incomplete": int(n_missing),
        "shrink_to_uniform": shrink,
        "half1": [float(x) for x in w1.tolist()],
        "half2": [float(x) for x in w2.tolist()],
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    out_path = cfg.out_path
    if out_path is None:
        out_path = Path(cfg.out_dir) / "segment_weights.json"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    out["out_path"] = str(out_path)
    return out
