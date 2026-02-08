from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ncaab_model.data.adapters.espn_playbyplay import fetch_playbyplay, extract_cum_totals_5min, infer_ot_periods
from ncaab_model.data.interval_actuals_5min import _pick_game_ids


@dataclass(frozen=True)
class BuildIntervalActuals2MinConfig:
    out_dir: Path
    date: str
    endpoints: tuple[int, ...] = tuple(range(2, 41, 2))
    include_ot_endpoints: bool = False
    max_ot_periods: int = 4
    use_cache: bool = True
    sleep_seconds: float = 0.15
    max_games: int = 0
    out_prefix: str = "interval_actuals_2min_"


def build_interval_actuals_2min_for_date(cfg: BuildIntervalActuals2MinConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    date = str(cfg.date)
    base_endpoints = [int(x) for x in cfg.endpoints]

    game_ids = _pick_game_ids(out_dir, date)
    if cfg.max_games and int(cfg.max_games) > 0:
        game_ids = game_ids[: int(cfg.max_games)]

    rows: list[dict] = []
    for gid in game_ids:
        payload = fetch_playbyplay(gid, use_cache=bool(cfg.use_cache))
        fetched_from = payload.get("_fetched_from") if isinstance(payload, dict) else None
        did_network = isinstance(fetched_from, str) and fetched_from.startswith("network")

        plays = payload.get("plays") if isinstance(payload, dict) else None
        if payload is None or not isinstance(plays, list) or len(plays) == 0:
            if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                time.sleep(float(cfg.sleep_seconds))
            continue

        # Regulation endpoints are at 2-minute grid. If OT endpoints are enabled,
        # we append 5-minute OT checkpoints (45/50/55/60) for games that went to OT.
        ot_periods = infer_ot_periods(payload) if bool(cfg.include_ot_endpoints) else 0
        endpoints = list(base_endpoints)
        if ot_periods > 0:
            extra = [40 + 5 * i for i in range(1, min(int(cfg.max_ot_periods), int(ot_periods)) + 1)]
            endpoints = sorted(set(endpoints + extra))

        cum = extract_cum_totals_5min(payload, endpoints=endpoints)
        for c in cum:
            end_min = int(c.end_min)
            rows.append(
                {
                    "date": date,
                    "game_id": str(gid),
                    "end_min": end_min,
                    "actual_home_score_end": int(c.home_score),
                    "actual_away_score_end": int(c.away_score),
                    "actual_total_score_end": int(c.total_score),
                    "is_ot_game": 1 if ot_periods > 0 else 0,
                    "is_ot_endpoint": 1 if end_min > 40 else 0,
                    "ot_periods": int(ot_periods),
                    "fetched_from": str(fetched_from) if fetched_from is not None else "",
                }
            )

        if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
            time.sleep(float(cfg.sleep_seconds))

    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
        except Exception:
            pass
        try:
            df = df.sort_values(["game_id", "end_min"], kind="mergesort")
        except Exception:
            pass

        try:
            df = df.drop_duplicates(subset=["game_id", "end_min"], keep="last")
        except Exception:
            pass

        try:
            for c in ["actual_home_score_end", "actual_away_score_end", "actual_total_score_end"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    df[c] = df.groupby("game_id")[c].cummax()
        except Exception:
            pass

    out_path = out_dir / f"{cfg.out_prefix}{date}.csv"
    try:
        df.replace([np.inf, -np.inf], np.nan).to_csv(out_path, index=False, na_rep="")
    except Exception:
        df.to_csv(out_path, index=False)

    return out_path
