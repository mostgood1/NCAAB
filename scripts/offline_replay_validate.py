import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.simulation.game_sim import simulate_game_row


_CLOCK_RE = re.compile(r"^(?P<m>\d+):(?P<s>\d{2})$")


def _clock_to_sec(clock_display: object) -> int | None:
    if clock_display is None:
        return None
    s = str(clock_display).strip()
    if not s:
        return None
    m = _CLOCK_RE.match(s)
    if m:
        return int(m.group("m")) * 60 + int(m.group("s"))

    # Sometimes ESPN uses "0.0" / "0.1" style
    try:
        v = float(s)
        if 0 <= v <= 60:
            return int(round(v))
    except Exception:
        return None
    return None


def _fmt_clock(sec: int) -> str:
    mm, ss = divmod(int(sec), 60)
    return f"{mm:02d}:{ss:02d}"


def _load_actual_final_from_summary(summary_path: Path) -> tuple[int, int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comps = summary["header"]["competitions"][0]["competitors"]
    home_comp = next(c for c in comps if c.get("homeAway") == "home")
    away_comp = next(c for c in comps if c.get("homeAway") == "away")
    home = int(float(home_comp.get("score")))
    away = int(float(away_comp.get("score")))
    return home, away


def _load_summary_info(summary_path: Path) -> tuple[int, int, bool, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        comps = summary["header"]["competitions"][0]["competitors"]
        home_comp = next(c for c in comps if c.get("homeAway") == "home")
        away_comp = next(c for c in comps if c.get("homeAway") == "away")
        home = int(float(home_comp.get("score")))
        away = int(float(away_comp.get("score")))
    except Exception as e:
        raise ValueError(f"Could not parse final scores from {summary_path}") from e

    status_type = (
        (((summary.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type")
        or {}
    )
    completed = bool(status_type.get("completed")) if "completed" in status_type else False
    status_name = str(
        status_type.get("name")
        or status_type.get("state")
        or status_type.get("description")
        or status_type.get("detail")
        or ""
    )
    return home, away, completed, status_name


def _infer_input_path(date: str, input_path: Path | None) -> tuple[Path, str]:
    if input_path is not None:
        return input_path, _infer_input_kind(input_path)

    p_align = Path("outputs") / f"align_period_{date}.csv"
    if p_align.exists():
        return p_align, "align_period"

    p_simq = Path("outputs") / f"sim_quantiles_{date}.csv"
    if p_simq.exists():
        return p_simq, "sim_quantiles"

    raise SystemExit(
        f"Could not find input for date={date}. Tried: {p_align} and {p_simq}. "
        "Pass --input to specify a CSV explicitly."
    )


def _infer_input_kind(p: Path) -> str:
    name = p.name.lower()
    if name.startswith("align_period_"):
        return "align_period"
    if name.startswith("sim_quantiles_"):
        return "sim_quantiles"
    return "unknown"


def _pick_base_row_from_df(df: pd.DataFrame, input_kind: str, game_id: int) -> pd.Series:
    if "game_id" not in df.columns:
        raise ValueError("Input missing game_id column")

    gid_col = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df_g = df[gid_col == int(game_id)]
    if input_kind == "align_period":
        if "period" not in df.columns:
            raise ValueError("align_period input missing period column")
        df_g = df_g[df_g["period"].astype(str) == "full_game"]
        if df_g.empty:
            raise ValueError(f"No full_game rows for game_id={game_id}")

        sort_cols = [c for c in ["market", "book"] if c in df_g.columns]
        if sort_cols:
            df_g = df_g.sort_values(sort_cols, kind="stable")
        row = df_g.iloc[0].copy()
    else:
        if df_g.empty:
            raise ValueError(f"No rows for game_id={game_id}")
        row = df_g.iloc[0].copy()

    return _ensure_mean_cols(row)


def _pick_base_row(input_path: Path, input_kind: str, game_id: int) -> pd.Series:
    df = pd.read_csv(input_path)
    try:
        return _pick_base_row_from_df(df, input_kind=input_kind, game_id=game_id)
    except Exception as e:
        raise SystemExit(f"Could not pick base row for game_id={game_id} from {input_path}: {e}") from e


def _ensure_mean_cols(row: pd.Series) -> pd.Series:
    """Make sure simulate_game_row can resolve total/margin means from this row.

    Many artifacts (e.g., sim_quantiles_*.csv) store the selected mean in
    columns like mean_total_selected/mean_margin_selected but do not have
    pred_total_blend/pred_margin_blend columns. We map those through so the
    simulator uses the intended pregame mean rather than falling back to
    feature-derived means.
    """

    def _first_finite(cands: list[str]) -> float | None:
        for c in cands:
            if c in row and pd.notna(row[c]):
                try:
                    v = float(row[c])
                except Exception:
                    continue
                if np.isfinite(v):
                    return float(v)
        return None

    if ("pred_total_blend" not in row) or pd.isna(row.get("pred_total_blend")):
        v = _first_finite(
            [
                "mean_total_after_overrides_calib",
                "mean_total_selected",
                "mu_total",
                "mean_total_first_candidate",
            ]
        )
        if v is not None:
            row["pred_total_blend"] = float(v)

    if ("pred_margin_blend" not in row) or pd.isna(row.get("pred_margin_blend")):
        v = _first_finite(
            [
                "mean_margin_after_overrides_calib",
                "mean_margin_selected",
                "mu_margin",
                "mean_margin_first_candidate",
            ]
        )
        if v is not None:
            row["pred_margin_blend"] = float(v)

    return row


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
    # Stable-ish per-game seed without bringing in hashlib.
    return int((int(global_seed) * 1000003 + int(game_id) * 9176) % 2147483647)


def _summarize_errs(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"{label}: no rows")
        return

    def _mae(vs: list[float]) -> float:
        a = np.asarray(vs, dtype=float)
        return float(np.mean(np.abs(a))) if a.size else float("nan")

    def _bias(vs: list[float]) -> float:
        a = np.asarray(vs, dtype=float)
        return float(np.mean(a)) if a.size else float("nan")

    e_t = [float(r["err_total"]) for r in rows if r.get("err_total") is not None]
    e_m = [float(r["err_margin"]) for r in rows if r.get("err_margin") is not None]
    e_r = [float(r["err_remaining"]) for r in rows if r.get("err_remaining") is not None]
    print(
        f"{label}: n={len(rows)} | "
        f"MAE(total)={_mae(e_t):.2f} bias(total)={_bias(e_t):+.2f} | "
        f"MAE(margin)={_mae(e_m):.2f} bias(margin)={_bias(e_m):+.2f} | "
        f"MAE(rem)={_mae(e_r):.2f} bias(rem)={_bias(e_r):+.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline replay validation using cached ESPN PBP + summary.")
    ap.add_argument("--game-id", type=int, required=False)
    ap.add_argument(
        "--date",
        type=str,
        required=True,
        help="YYYY-MM-DD used to locate outputs/align_period_<date>.csv or outputs/sim_quantiles_<date>.csv",
    )
    ap.add_argument("--input", type=str, default=None, help="Optional explicit CSV path (align_period_*.csv or sim_quantiles_*.csv)")
    ap.add_argument("--batch", action="store_true", help="Evaluate all games in the input file (skips games without cached ESPN PBP+summary)")
    ap.add_argument("--max-games", type=int, default=0, help="Optional limit in batch mode (0 = no limit)")
    ap.add_argument(
        "--debug-errors",
        type=int,
        default=3,
        help="In batch mode, print up to N example simulation errors (0 = none)",
    )
    ap.add_argument(
        "--write-csv",
        type=str,
        default=None,
        help="Optional output path for detailed per-snapshot rows (default outputs/offline_replay_batch_<date>.csv in batch mode)",
    )
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--targets",
        type=str,
        default="1200,900,600,300,120,60,0",
        help="Comma-separated remaining regulation seconds to probe",
    )
    args = ap.parse_args()

    input_path, input_kind = _infer_input_path(args.date, Path(args.input) if args.input else None)
    targets = [int(x.strip()) for x in str(args.targets).split(",") if x.strip()]

    if args.batch:
        df = pd.read_csv(input_path)
        if "game_id" not in df.columns:
            raise SystemExit(f"Input missing game_id column: {input_path}")
        if input_kind == "align_period" and "period" in df.columns:
            df = df[df["period"].astype(str) == "full_game"]

        gid_col = pd.to_numeric(df["game_id"], errors="coerce")
        game_ids = [int(x) for x in pd.unique(gid_col.dropna()).tolist()]
        game_ids = sorted(game_ids)
        if int(args.max_games) and int(args.max_games) > 0:
            game_ids = game_ids[: int(args.max_games)]

        out_rows: list[dict] = []
        by_target: dict[int, list[dict]] = {t: [] for t in targets}
        skipped = {
            "missing_pbp": 0,
            "missing_summary": 0,
            "not_completed": 0,
            "no_snapshots": 0,
            "bad_row": 0,
            "sim_error": 0,
        }
        debug_left = int(args.debug_errors)
        debug_examples: list[str] = []

        for gid in game_ids:
            pbp_path = Path("data/cache/espn_pbp") / f"{gid}.json"
            summary_path = Path("data/cache/espn_summary") / f"{gid}.json"
            if not pbp_path.exists():
                skipped["missing_pbp"] += 1
                continue
            if not summary_path.exists():
                skipped["missing_summary"] += 1
                continue

            try:
                row0 = _pick_base_row_from_df(df, input_kind=input_kind, game_id=gid)
            except Exception:
                skipped["bad_row"] += 1
                continue

            try:
                actual_home, actual_away, completed, status_name = _load_summary_info(summary_path)
            except Exception:
                skipped["missing_summary"] += 1
                continue

            if not bool(completed):
                skipped["not_completed"] += 1
                continue

            try:
                actual_total = int(actual_home + actual_away)
                actual_margin = int(actual_home - actual_away)
                snapshots = _extract_snapshots_from_pbp(pbp_path)
            except Exception:
                skipped["no_snapshots"] += 1
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
                        samples=int(args.samples),
                        rng=np.random.default_rng(_seed_for_game(int(args.seed), gid) + int(tgt) * 37),
                    )
                except Exception as e:
                    skipped["sim_error"] += 1
                    if debug_left > 0:
                        debug_left -= 1
                        debug_examples.append(f"game_id={gid} tgt={tgt} err={type(e).__name__}: {e}")
                    continue

                live_mu_t = out.get("live_mu_total")
                live_mu_m = out.get("live_mu_margin")
                live_mu_rem = out.get("live_mu_total_remaining")
                if live_mu_t is None or live_mu_m is None or live_mu_rem is None:
                    skipped["sim_error"] += 1
                    continue

                live_mu_t = float(live_mu_t)
                live_mu_m = float(live_mu_m)
                live_mu_rem = float(live_mu_rem)

                now_total = int(s["home"] + s["away"])
                actual_remaining = actual_total - now_total

                row_out = {
                    "date": str(args.date),
                    "game_id": int(gid),
                    "target_rem_reg": int(tgt),
                    "snap_rem_reg": int(s["rem_reg"]),
                    "period": int(s["period"]),
                    "sec_half": int(s["sec_half"]),
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
                out_rows.append(row_out)
                by_target[int(tgt)].append(row_out)

        print(f"Batch input: {input_path} (kind={input_kind})")
        print(f"Games considered: {len(game_ids)} | rows: {len(out_rows)} | skipped: {skipped}")
        if debug_examples:
            print("Example sim errors:")
            for msg in debug_examples:
                print(f"  - {msg}")
        for tgt in targets:
            _summarize_errs(by_target[int(tgt)], label=f"tgt_rem_reg={int(tgt)}")

        write_path = None
        if args.write_csv:
            write_path = Path(args.write_csv)
        elif out_rows:
            write_path = Path("outputs") / f"offline_replay_batch_{args.date}.csv"

        if write_path is not None:
            pd.DataFrame(out_rows).to_csv(write_path, index=False)
            print(f"Wrote: {write_path}")

        return 0

    # Single-game mode
    if args.game_id is None:
        raise SystemExit("--game-id is required unless --batch is set")

    game_id = int(args.game_id)
    pbp_path = Path("data/cache/espn_pbp") / f"{game_id}.json"
    summary_path = Path("data/cache/espn_summary") / f"{game_id}.json"

    row0 = _pick_base_row(input_path, input_kind, game_id)
    actual_home, actual_away = _load_actual_final_from_summary(summary_path)
    actual_total = int(actual_home + actual_away)
    actual_margin = int(actual_home - actual_away)

    snapshots = _extract_snapshots_from_pbp(pbp_path)

    rng = np.random.default_rng(_seed_for_game(int(args.seed), game_id))
    pregame_out = simulate_game_row(row0, engine="events", samples=int(args.samples), rng=rng)

    print(f"Input: {input_path} (kind={input_kind})")
    print(f"Game {game_id} actual final: home={actual_home} away={actual_away} total={actual_total} margin={actual_margin}")
    print(
        "Pregame: "
        f"mu_total={float(pregame_out.get('mu_total')):.2f} "
        f"mu_margin={float(pregame_out.get('mu_margin')):.2f} "
        f"(engine={pregame_out.get('sim_engine')}, mean_source_used={pregame_out.get('mean_source_used')})"
    )
    print("\nrem_reg | P clock | score | live_mu_total (err) | live_mu_margin (err) | mu_remaining vs actual_remaining")

    for tgt in targets:
        s = _nearest_snapshot(snapshots, tgt)

        r = row0.copy()
        r["remaining_reg_seconds"] = float(s["rem_reg"])
        r["period"] = int(s["period"])
        r["home_score"] = int(s["home"])
        r["away_score"] = int(s["away"])

        out = simulate_game_row(
            r,
            engine="events",
            samples=int(args.samples),
            rng=np.random.default_rng(_seed_for_game(int(args.seed), game_id)),
        )

        live_mu_t = float(out.get("live_mu_total"))
        live_mu_m = float(out.get("live_mu_margin"))
        live_mu_rem = float(out.get("live_mu_total_remaining"))

        now_total = int(s["home"] + s["away"])
        actual_remaining = actual_total - now_total
        rem_err = live_mu_rem - float(actual_remaining)

        clock = _fmt_clock(int(s["sec_half"]))
        print(
            f"{int(s['rem_reg']):5d} | {int(s['period'])} {clock} | {int(s['home']):3d}-{int(s['away']):3d} | "
            f"{live_mu_t:7.2f} ({live_mu_t - actual_total:+6.2f}) | "
            f"{live_mu_m:7.2f} ({live_mu_m - actual_margin:+6.2f}) | "
            f"{live_mu_rem:7.2f} vs {actual_remaining:3d} ({rem_err:+6.2f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
