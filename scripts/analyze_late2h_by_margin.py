import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


BUCKETS_DEFAULT = [(0, 3, "<=3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, None, "11+")]


@dataclass(frozen=True)
class Bucket:
    lo: int
    hi: int | None
    label: str

    def contains(self, x: int) -> bool:
        if self.hi is None:
            return x >= self.lo
        return self.lo <= x <= self.hi


def _parse_buckets(spec: str | None) -> list[Bucket]:
    if not spec:
        return [Bucket(lo=a, hi=b, label=lab) for a, b, lab in BUCKETS_DEFAULT]

    buckets: list[Bucket] = []
    # Format: "<=3,4-6,7-10,11+" (order matters)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if p.startswith("<="):
            hi = int(p[2:])
            buckets.append(Bucket(lo=0, hi=hi, label=p))
        elif p.endswith("+"):
            lo = int(p[:-1])
            buckets.append(Bucket(lo=lo, hi=None, label=p))
        elif "-" in p:
            lo_s, hi_s = p.split("-", 1)
            buckets.append(Bucket(lo=int(lo_s), hi=int(hi_s), label=p))
        else:
            # single value treated as exact
            v = int(p)
            buckets.append(Bucket(lo=v, hi=v, label=p))
    return buckets


def _find_scores_file(outputs_dir: str, date: str) -> str | None:
    path = os.path.join(outputs_dir, f"scores_raw_{date}_espn.csv")
    return path if os.path.exists(path) else None


def _load_scores(outputs_dir: str, dates: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for d in sorted(set(dates)):
        p = _find_scores_file(outputs_dir, d)
        if not p:
            continue
        df = pd.read_csv(p)
        # Ensure game_id numeric-ish for merge robustness
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        df["date"] = df["date"].astype(str)
        df["abs_margin_final"] = (df["home_score"].astype(float) - df["away_score"].astype(float)).abs().astype(int)
        rows.append(df[["date", "game_id", "abs_margin_final"]])
    if not rows:
        return pd.DataFrame(columns=["date", "game_id", "abs_margin_final"])
    return pd.concat(rows, ignore_index=True)


def _bucketize(margins: pd.Series, buckets: list[Bucket]) -> pd.Series:
    def _lab(v: float) -> str | None:
        if not np.isfinite(v):
            return None
        iv = int(v)
        for b in buckets:
            if b.contains(iv):
                return b.label
        return None

    return margins.apply(_lab)


def _compute_increments(
    bt: pd.DataFrame,
    start_min: int,
    end_min: int,
    step: int,
    quantile_col: str,
) -> pd.DataFrame:
    # Build wide table per game for fast differences.
    bt_small = bt[["date", "game_id", "end_min", "actual_total", quantile_col]].copy()
    bt_small = bt_small.pivot_table(
        index=["date", "game_id"],
        columns="end_min",
        values=["actual_total", quantile_col],
        aggfunc="first",
    )

    def _col(kind: str, m: int) -> tuple[str, int]:
        return (kind, m)

    mins = list(range(start_min, end_min + 1, step))
    # Ensure start and end exist.
    needed = set(mins)
    present = set(bt_small.columns.get_level_values(1).unique())
    missing = sorted(needed - present)
    if missing:
        raise ValueError(f"Backtest missing endpoints: {missing}")

    rows: list[dict] = []

    # Aggregate 30->40 as single window too.
    a0 = bt_small[_col("actual_total", start_min)]
    a1 = bt_small[_col("actual_total", end_min)]
    p0 = bt_small[_col(quantile_col, start_min)]
    p1 = bt_small[_col(quantile_col, end_min)]
    rows.append(
        {
            "seg_start": start_min,
            "seg_end": end_min,
            "actual_inc": (a1 - a0),
            "pred_inc": (p1 - p0),
        }
    )

    # Per step increments.
    for s in range(start_min, end_min, step):
        e = s + step
        a_s = bt_small[_col("actual_total", s)]
        a_e = bt_small[_col("actual_total", e)]
        p_s = bt_small[_col(quantile_col, s)]
        p_e = bt_small[_col(quantile_col, e)]
        rows.append(
            {
                "seg_start": s,
                "seg_end": e,
                "actual_inc": (a_e - a_s),
                "pred_inc": (p_e - p_s),
            }
        )

    out = []
    for r in rows:
        df = pd.DataFrame(
            {
                "date": bt_small.index.get_level_values(0),
                "game_id": bt_small.index.get_level_values(1),
                "seg_start": r["seg_start"],
                "seg_end": r["seg_end"],
                "actual_inc": r["actual_inc"].astype(float).values,
                "pred_inc": r["pred_inc"].astype(float).values,
            }
        )
        df["err"] = df["pred_inc"] - df["actual_inc"]
        out.append(df)

    return pd.concat(out, ignore_index=True)


def _summarize_by_bucket(inc: pd.DataFrame) -> pd.DataFrame:
    g = inc.groupby(["bucket", "seg_start", "seg_end"], dropna=False)
    s = g.agg(
        n_games=("game_id", "nunique"),
        mean_actual_inc=("actual_inc", "mean"),
        mean_pred_inc=("pred_inc", "mean"),
        mean_err=("err", "mean"),
        mae=("err", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze late-2H (2-min grid) increment bias by final-margin bucket.")
    ap.add_argument("--backtest", required=True, help="Path to segments_2min backtest CSV")
    ap.add_argument("--outputs-dir", default="outputs", help="Directory containing scores_raw_<date>_espn.csv")
    ap.add_argument("--out-dir", default="outputs/backtests", help="Where to write summary CSVs")
    ap.add_argument("--tag", default=None, help="Tag used in output filenames")
    ap.add_argument("--start-min", type=int, default=30)
    ap.add_argument("--end-min", type=int, default=40)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--quantile-col", default="pred_q50", help="Which prediction column to analyze")
    ap.add_argument("--buckets", default=None, help='Override buckets, e.g. "<=3,4-6,7-10,11+"')
    ap.add_argument("--exclude-ot", action="store_true", help="Drop OT games (recommended)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bt = pd.read_csv(args.backtest)
    bt["date"] = bt["date"].astype(str)
    bt["game_id"] = pd.to_numeric(bt["game_id"], errors="coerce").astype("Int64")
    bt["end_min"] = pd.to_numeric(bt["end_min"], errors="coerce").astype(int)

    if args.exclude_ot and "is_ot_game" in bt.columns:
        bt = bt.loc[bt["is_ot_game"].astype(int) == 0].copy()

    # Load scores/margins and merge.
    scores = _load_scores(args.outputs_dir, bt["date"].unique().tolist())
    merged = bt.merge(scores, on=["date", "game_id"], how="left")

    buckets = _parse_buckets(args.buckets)
    merged["bucket"] = _bucketize(merged["abs_margin_final"], buckets)

    # Compute increments per game for all segments.
    inc = _compute_increments(
        merged.dropna(subset=[args.quantile_col]),
        start_min=args.start_min,
        end_min=args.end_min,
        step=args.step,
        quantile_col=args.quantile_col,
    )

    inc = inc.merge(merged[["date", "game_id", "bucket"]].drop_duplicates(), on=["date", "game_id"], how="left")
    inc = inc.dropna(subset=["bucket"]).copy()

    summary = _summarize_by_bucket(inc)

    tag = args.tag
    if not tag:
        base = os.path.basename(args.backtest)
        tag = os.path.splitext(base)[0]

    seg_out = os.path.join(args.out_dir, f"late2H_segment_increments_by_margin_{tag}.csv")
    summary.to_csv(seg_out, index=False)

    # Final2 table: last segment only.
    final2 = summary.loc[(summary["seg_start"] == args.end_min - args.step) & (summary["seg_end"] == args.end_min)].copy()
    final2_out = os.path.join(args.out_dir, f"final2_increment_by_margin_{tag}.csv")
    final2.to_csv(final2_out, index=False)

    # Aggregate 30->40 table: seg_start=start_min, seg_end=end_min.
    agg = summary.loc[(summary["seg_start"] == args.start_min) & (summary["seg_end"] == args.end_min)].copy()
    agg_out = os.path.join(args.out_dir, f"late2H_aggregate_{args.start_min}to{args.end_min}_by_margin_{tag}.csv")
    agg.to_csv(agg_out, index=False)

    # Print quick scores for selection.
    # Unweighted and count-weighted L1 of mean_err across buckets for the aggregate window.
    if not agg.empty:
        unweighted = float(np.mean(np.abs(agg["mean_err"].astype(float).values)))
        weighted = float(
            np.average(np.abs(agg["mean_err"].astype(float).values), weights=agg["n_games"].astype(float).values)
        )
        print(f"Wrote: {seg_out}")
        print(f"Wrote: {final2_out}")
        print(f"Wrote: {agg_out}")
        print(f"Aggregate {args.start_min}->{args.end_min} mean_err L1: unweighted={unweighted:.4f} weighted={weighted:.4f}")
    else:
        print(f"Wrote: {seg_out}")
        print(f"Wrote: {final2_out}")
        print(f"Wrote: {agg_out}")
        print("Aggregate window table is empty (no games after filters).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
