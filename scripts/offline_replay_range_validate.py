import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.simulation.game_sim import simulate_game_row


_CLOCK_RE = re.compile(r"^(?P<m>\d+):(?P<s>\d{2})$")


def _safe_float(v: object) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        x = float(v)
        if not np.isfinite(x):
            return None
        return float(x)
    except Exception:
        return None


def _clock_to_sec(clock_display: object) -> int | None:
    if clock_display is None:
        return None
    s = str(clock_display).strip()
    if not s:
        return None
    m = _CLOCK_RE.match(s)
    if m:
        return int(m.group("m")) * 60 + int(m.group("s"))
    try:
        v = float(s)
        if 0 <= v <= 60:
            return int(round(v))
    except Exception:
        return None
    return None


def _infer_input_path(date: str) -> tuple[Path, str] | None:
    p_align = Path("outputs") / f"align_period_{date}.csv"
    if p_align.exists():
        return p_align, "align_period"
    p_simq = Path("outputs") / f"sim_quantiles_{date}.csv"
    if p_simq.exists():
        return p_simq, "sim_quantiles"
    return None


def _ensure_mean_cols(row: pd.Series) -> pd.Series:
    def _first_finite(cands: list[str]) -> float | None:
        for c in cands:
            if c in row and pd.notna(row[c]):
                v = _safe_float(row[c])
                if v is not None:
                    return float(v)
        return None

    if ("pred_total_blend" not in row) or pd.isna(row.get("pred_total_blend")):
        v = _first_finite([
            "mean_total_after_overrides_calib",
            "mean_total_selected",
            "mu_total",
            "pred_total",
            "pred_total_full",
        ])
        if v is not None:
            row["pred_total_blend"] = float(v)

    if ("pred_margin_blend" not in row) or pd.isna(row.get("pred_margin_blend")):
        v = _first_finite([
            "mean_margin_after_overrides_calib",
            "mean_margin_selected",
            "mu_margin",
            "pred_margin",
            "pred_margin_full",
        ])
        if v is not None:
            row["pred_margin_blend"] = float(v)

    return row


def _pick_base_row_from_df(df: pd.DataFrame, input_kind: str, game_id: int) -> pd.Series:
    if "game_id" not in df.columns:
        raise ValueError("Input missing game_id column")

    gid_col = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df_g = df[gid_col == int(game_id)]
    if input_kind == "align_period":
        if "period" in df_g.columns:
            df_g = df_g[df_g["period"].astype(str) == "full_game"]
    if df_g.empty:
        raise ValueError(f"No rows for game_id={game_id}")

    row = df_g.iloc[0].copy()
    return _ensure_mean_cols(row)


def _load_summary_info(summary_path: Path) -> tuple[int, int, bool]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comps = summary["header"]["competitions"][0]["competitors"]
    home_comp = next(c for c in comps if c.get("homeAway") == "home")
    away_comp = next(c for c in comps if c.get("homeAway") == "away")
    home = int(float(home_comp.get("score")))
    away = int(float(away_comp.get("score")))

    status_type = (
        (((summary.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type")
        or {}
    )
    if "completed" in status_type:
        completed = bool(status_type.get("completed"))
    else:
        # Some cache variants only have state/name.
        state = str(status_type.get("state") or "").strip().lower()
        name = str(status_type.get("name") or "").strip().lower()
        completed = state in {"post", "final"} or "final" in name

    return home, away, completed


def _extract_snapshots_from_pbp(pbp_path: Path) -> list[dict]:
    pbp = json.loads(pbp_path.read_text(encoding="utf-8"))
    plays = pbp.get("plays") or []

    out: list[dict] = []
    for p in plays:
        try:
            per = int(((p.get("period") or {}).get("number")) or 0)
            if per not in (1, 2):
                continue
            sec_half = _clock_to_sec(((p.get("clock") or {}).get("displayValue")))
            if sec_half is None:
                continue
            hs = p.get("homeScore")
            aS = p.get("awayScore")
            if hs is None or aS is None:
                continue

            rem_reg = (1200 + int(sec_half)) if per == 1 else int(sec_half)
            out.append(
                {
                    "period": int(per),
                    "sec_half": int(sec_half),
                    "rem_reg": int(rem_reg),
                    "home": int(hs),
                    "away": int(aS),
                }
            )
        except Exception:
            continue

    if not out:
        raise ValueError(f"No usable snapshots parsed from {pbp_path}")
    return out


def _nearest_snapshot(snapshots: list[dict], target_rem_reg: int) -> dict:
    return min(snapshots, key=lambda s: abs(int(s["rem_reg"]) - int(target_rem_reg)))


def _seed_for_game(global_seed: int, game_id: int) -> int:
    return int((int(global_seed) * 1000003 + int(game_id) * 9176) % 2147483647)


@dataclass
class SkipCounts:
    missing_pbp: int = 0
    missing_summary: int = 0
    not_completed: int = 0
    no_snapshots: int = 0
    bad_row: int = 0
    sim_error: int = 0

    def as_dict(self) -> dict:
        return {
            "missing_pbp": int(self.missing_pbp),
            "missing_summary": int(self.missing_summary),
            "not_completed": int(self.not_completed),
            "no_snapshots": int(self.no_snapshots),
            "bad_row": int(self.bad_row),
            "sim_error": int(self.sim_error),
        }


def _summarize_bucket(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "mae_total": float("nan"),
            "bias_total": float("nan"),
            "mae_margin": float("nan"),
            "bias_margin": float("nan"),
            "mae_remaining": float("nan"),
            "bias_remaining": float("nan"),
        }

    def _mae(vals: list[float]) -> float:
        a = np.asarray(vals, dtype=float)
        return float(np.mean(np.abs(a))) if a.size else float("nan")

    def _bias(vals: list[float]) -> float:
        a = np.asarray(vals, dtype=float)
        return float(np.mean(a)) if a.size else float("nan")

    e_t = [float(r["err_total"]) for r in rows]
    e_m = [float(r["err_margin"]) for r in rows]
    e_r = [float(r["err_remaining"]) for r in rows]

    return {
        "n": int(len(rows)),
        "mae_total": _mae(e_t),
        "bias_total": _bias(e_t),
        "mae_margin": _mae(e_m),
        "bias_margin": _bias(e_m),
        "mae_remaining": _mae(e_r),
        "bias_remaining": _bias(e_r),
    }


def evaluate_date(
    date: str,
    targets: list[int],
    samples: int,
    seed: int,
    max_games: int,
) -> tuple[list[dict], SkipCounts] | None:
    inferred = _infer_input_path(date)
    if inferred is None:
        return None
    input_path, input_kind = inferred

    df = pd.read_csv(input_path)
    if "game_id" not in df.columns:
        return None

    gid_col = pd.to_numeric(df["game_id"], errors="coerce")
    game_ids = [int(x) for x in pd.unique(gid_col.dropna()).tolist()]
    game_ids = sorted(game_ids)
    if int(max_games) and int(max_games) > 0:
        game_ids = game_ids[: int(max_games)]

    skipped = SkipCounts()
    out_rows: list[dict] = []

    for gid in game_ids:
        pbp_path = Path("data/cache/espn_pbp") / f"{gid}.json"
        summary_path = Path("data/cache/espn_summary") / f"{gid}.json"
        if not pbp_path.exists():
            skipped.missing_pbp += 1
            continue
        if not summary_path.exists():
            skipped.missing_summary += 1
            continue

        try:
            row0 = _pick_base_row_from_df(df, input_kind=input_kind, game_id=gid)
        except Exception:
            skipped.bad_row += 1
            continue

        try:
            actual_home, actual_away, completed = _load_summary_info(summary_path)
        except Exception:
            skipped.missing_summary += 1
            continue

        if not bool(completed):
            skipped.not_completed += 1
            continue

        try:
            actual_total = int(actual_home + actual_away)
            actual_margin = int(actual_home - actual_away)
            snapshots = _extract_snapshots_from_pbp(pbp_path)
        except Exception:
            skipped.no_snapshots += 1
            continue

        for tgt in targets:
            s = _nearest_snapshot(snapshots, tgt)
            r = row0.copy()
            r["remaining_reg_seconds"] = float(s["rem_reg"])
            r["period"] = int(s["period"])
            r["home_score"] = int(s["home"])
            r["away_score"] = int(s["away"])

            try:
                out = simulate_game_row(
                    r,
                    engine="events",
                    samples=int(samples),
                    rng=np.random.default_rng(_seed_for_game(int(seed), gid) + int(tgt) * 37),
                )
            except Exception:
                skipped.sim_error += 1
                continue

            live_mu_t = out.get("live_mu_total")
            live_mu_m = out.get("live_mu_margin")
            live_mu_rem = out.get("live_mu_total_remaining")
            if live_mu_t is None or live_mu_m is None or live_mu_rem is None:
                skipped.sim_error += 1
                continue

            live_mu_t = float(live_mu_t)
            live_mu_m = float(live_mu_m)
            live_mu_rem = float(live_mu_rem)

            now_total = int(s["home"] + s["away"])
            actual_remaining = actual_total - now_total

            out_rows.append(
                {
                    "date": str(date),
                    "game_id": int(gid),
                    "target_rem_reg": int(tgt),
                    "snap_rem_reg": int(s["rem_reg"]),
                    "period": int(s["period"]),
                    "home_score": int(s["home"]),
                    "away_score": int(s["away"]),
                    "actual_total": int(actual_total),
                    "actual_margin": int(actual_margin),
                    "live_mu_total": float(live_mu_t),
                    "live_mu_margin": float(live_mu_m),
                    "live_mu_total_remaining": float(live_mu_rem),
                    "actual_remaining": int(actual_remaining),
                    "err_total": float(live_mu_t - actual_total),
                    "err_margin": float(live_mu_m - actual_margin),
                    "err_remaining": float(live_mu_rem - float(actual_remaining)),
                }
            )

    return out_rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay validation across a date range (cached ESPN PBP + summary).")
    ap.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-games", type=int, default=0, help="Optional limit per day (0 = no limit)")
    ap.add_argument(
        "--targets",
        type=str,
        default="1200,900,600,300,120,60,0",
        help="Comma-separated remaining regulation seconds to probe",
    )
    ap.add_argument(
        "--time-adj",
        type=str,
        default="both",
        choices=["both", "on", "off"],
        help="Whether to evaluate with NCAAB_LIVE_TIME_RATE_ADJUST enabled/disabled.",
    )
    ap.add_argument(
        "--write-summary",
        type=str,
        default=None,
        help="Optional output CSV path for summary (default outputs/offline_replay_summary_<start>_<end>.csv)",
    )
    args = ap.parse_args()

    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    if end < start:
        raise SystemExit("end-date must be >= start-date")

    targets = [int(x.strip()) for x in str(args.targets).split(",") if x.strip()]

    modes: list[tuple[str, str]]
    if args.time_adj == "both":
        modes = [("on", "1"), ("off", "0")]
    elif args.time_adj == "on":
        modes = [("on", "1")]
    else:
        modes = [("off", "0")]

    all_summary_rows: list[dict] = []

    for d in pd.date_range(start=start, end=end, freq="D"):
        date_str = d.strftime("%Y-%m-%d")

        for mode_name, env_val in modes:
            os.environ["NCAAB_LIVE_TIME_RATE_ADJUST"] = str(env_val)
            evaluated = evaluate_date(
                date=date_str,
                targets=targets,
                samples=int(args.samples),
                seed=int(args.seed),
                max_games=int(args.max_games),
            )
            if evaluated is None:
                continue
            out_rows, skipped = evaluated
            df = pd.DataFrame(out_rows)
            for tgt in targets:
                bucket_rows = []
                if not df.empty:
                    bucket_rows = df[df["target_rem_reg"] == int(tgt)].to_dict(orient="records")
                stats = _summarize_bucket(bucket_rows)
                all_summary_rows.append(
                    {
                        "date": date_str,
                        "time_adj": mode_name,
                        "target_rem_reg": int(tgt),
                        **stats,
                        **{f"sk_{k}": v for k, v in skipped.as_dict().items()},
                    }
                )

    out_path = Path(args.write_summary) if args.write_summary else Path("outputs") / f"offline_replay_summary_{args.start_date}_{args.end_date}.csv"
    pd.DataFrame(all_summary_rows).to_csv(out_path, index=False)
    print(f"Wrote summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
