from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ncaab_model.eval.live_snapshot_features import market_blend_weight


@dataclass(frozen=True)
class SweepRow:
    start_min: float | None
    max_w: float
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None


def _metric_block(err: pd.Series) -> dict[str, float | int | None]:
    e = pd.to_numeric(err, errors="coerce").dropna()
    if e.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    mae = float(np.mean(np.abs(e.values)))
    rmse = float(np.sqrt(np.mean((e.values) ** 2)))
    bias = float(np.mean(e.values))
    return {"n": int(len(e)), "mae": mae, "rmse": rmse, "bias": bias}


def _bucket_remaining(rem: float) -> str:
    edges = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    for i in range(1, len(edges)):
        if rem <= edges[i]:
            return f"{edges[i-1]}-{edges[i]}"
    return "40+"


def _coerce_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _compute_err_blend(df: pd.DataFrame, *, horizon_min: float, start_min: float | None, max_w: float) -> pd.Series:
    elapsed = _coerce_float_series(df, "elapsed_min")
    proj_model = _coerce_float_series(df, "proj_model_total")
    line = _coerce_float_series(df, "live_line_total")
    actual = _coerce_float_series(df, "actual_total")

    w = elapsed.apply(lambda e: market_blend_weight(e, horizon_min=horizon_min, start_min=start_min, max_w=max_w))
    proj_blend = (1.0 - w) * proj_model + w * line

    ok = proj_model.notna() & line.notna() & actual.notna() & elapsed.notna()
    err = pd.Series(np.nan, index=df.index, dtype="float64")
    err.loc[ok] = (proj_blend.loc[ok] - actual.loc[ok]).astype("float64")
    return err


def run_sweep(
    *,
    in_csv: Path,
    out_csv: Path,
    out_json: Path,
    horizon_min: float = 40.0,
    start_grid: list[float | None] | None = None,
    max_w_grid: list[float] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(in_csv, low_memory=True)

    need = {"elapsed_min", "remaining_min", "proj_model_total", "live_line_total", "actual_total"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if start_grid is None:
        start_grid = [None, 2.0, 3.0, 4.0, 5.0, 6.0]
    if max_w_grid is None:
        max_w_grid = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]

    rows: list[dict[str, Any]] = []

    rem = _coerce_float_series(df, "remaining_min")
    df = df.copy()
    df["bucket_remaining"] = rem.apply(lambda x: _bucket_remaining(float(x)) if pd.notna(x) else "?")

    for start in start_grid:
        for max_w in max_w_grid:
            err = _compute_err_blend(df, horizon_min=float(horizon_min), start_min=start, max_w=float(max_w))
            met = _metric_block(err)
            row = {
                "start_min": (float(start) if start is not None else None),
                "max_w": float(max_w),
                "n": int(met["n"]),
                "mae": met["mae"],
                "rmse": met["rmse"],
                "bias": met["bias"],
            }

            # bucketed MAE (same buckets as evaluate-live-snapshots)
            try:
                maes = []
                for b, g in df.groupby("bucket_remaining"):
                    e2 = pd.to_numeric(err.loc[g.index], errors="coerce").dropna().abs()
                    maes.append({"bucket_remaining": str(b), "n": int(len(e2)), "mae": (float(e2.mean()) if not e2.empty else None)})
                # keep stable order
                def _key(x: dict[str, Any]) -> float:
                    s = str(x.get("bucket_remaining") or "")
                    if "-" in s:
                        try:
                            return float(s.split("-")[0])
                        except Exception:
                            return 999.0
                    return 999.0

                maes.sort(key=_key)
                row["by_bucket"] = maes
            except Exception:
                row["by_bucket"] = []

            rows.append(row)

    out_df = pd.DataFrame(rows)

    # Pick best by overall MAE (tie-break RMSE then |bias|)
    best = None
    try:
        tmp = out_df.copy()
        tmp["abs_bias"] = pd.to_numeric(tmp.get("bias"), errors="coerce").abs()
        tmp["mae"] = pd.to_numeric(tmp.get("mae"), errors="coerce")
        tmp["rmse"] = pd.to_numeric(tmp.get("rmse"), errors="coerce")
        tmp = tmp.dropna(subset=["mae"])
        if not tmp.empty:
            tmp = tmp.sort_values(["mae", "rmse", "abs_bias"], ascending=[True, True, True])
            best = tmp.iloc[0].to_dict()
            # remove big field
            best.pop("by_bucket", None)
            best.pop("abs_bias", None)
    except Exception:
        best = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "status": "ok",
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "horizon_min": float(horizon_min),
        "start_grid": [float(x) if x is not None else None for x in start_grid],
        "max_w_grid": [float(x) for x in max_w_grid],
        "best_overall": best,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep market blend parameters over an evaluate-live-snapshots CSV")
    ap.add_argument("--in-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--horizon-min", type=float, default=40.0)
    args = ap.parse_args()

    run_sweep(in_csv=args.in_csv, out_csv=args.out_csv, out_json=args.out_json, horizon_min=float(args.horizon_min))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
