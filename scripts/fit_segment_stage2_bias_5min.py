from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on sys.path for local imports (src.*)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_stage1_affine(calib_path: Path) -> tuple[dict[int, float], dict[int, float]]:
    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    a_map_raw = payload.get("a_by_end_min") if isinstance(payload, dict) else None
    b_map_raw = payload.get("b_by_end_min") if isinstance(payload, dict) else None
    if not isinstance(a_map_raw, dict) or not isinstance(b_map_raw, dict):
        return {}, {}

    a_map: dict[int, float] = {}
    b_map: dict[int, float] = {}

    for k, v in a_map_raw.items():
        try:
            kk = int(float(k))
            vv = float(v)
        except Exception:
            continue
        if np.isfinite(vv):
            a_map[kk] = vv

    for k, v in b_map_raw.items():
        try:
            kk = int(float(k))
            vv = float(v)
        except Exception:
            continue
        if np.isfinite(vv):
            b_map[kk] = vv

    return a_map, b_map


def fit_stage2_bias(
    backtest_csv: Path,
    stage1_calibration_json: Path,
    out_path: Path,
    start: str | None,
    end: str | None,
    end_mins: list[int],
    pred_already_stage1: bool,
    pred_already_stage2: bool,
    existing_stage2_bias_by_end_min: dict[int, float] | None,
    pred_col: str = "pred_q50",
    actual_col: str = "actual_total",
    date_col: str = "date",
    end_min_col: str = "end_min",
    min_rows_per_end_min: int = 150,
    stat: str = "mean",
) -> dict:
    if not backtest_csv.exists():
        raise FileNotFoundError(f"Missing backtest CSV: {backtest_csv}")
    if not stage1_calibration_json.exists():
        raise FileNotFoundError(f"Missing stage1 calibration JSON: {stage1_calibration_json}")

    df = pd.read_csv(backtest_csv)
    if df.empty:
        raise ValueError(f"Empty backtest CSV: {backtest_csv}")

    for col in (date_col, end_min_col, actual_col, pred_col):
        if col not in df.columns:
            raise ValueError(f"Backtest CSV missing required column: {col}")

    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    if start:
        df = df[df[date_col] >= str(start)]
    if end:
        df = df[df[date_col] <= str(end)]

    df[end_min_col] = pd.to_numeric(df[end_min_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    df = df.dropna(subset=[end_min_col, pred_col, actual_col])
    df[end_min_col] = df[end_min_col].astype(int)

    # Only keep target endpoints
    end_mins = [int(x) for x in end_mins]
    df = df[df[end_min_col].isin(end_mins)]
    if df.empty:
        raise ValueError(f"No rows left after filtering to end_mins={end_mins}")

    if pred_already_stage1:
        # Predictions already reflect stage1 calibration.
        df["pred_stage1"] = df[pred_col]

        if pred_already_stage2:
            # Predictions also reflect an existing stage2 correction.
            # Undo it to recover the stage1-only prediction, then fit the residual bias
            # that should be subtracted as stage2.
            if not existing_stage2_bias_by_end_min:
                raise ValueError(
                    "--pred-already-stage2 requires an existing stage2 bias map (bias_by_end_min) to undo"
                )
            df["_stage2_existing"] = df[end_min_col].map(existing_stage2_bias_by_end_min).fillna(0.0)
            df["pred_stage1_only"] = df["pred_stage1"] + df["_stage2_existing"]
            df["resid"] = df["pred_stage1_only"] - df[actual_col]
        else:
            # Fit residual directly from stage1-calibrated predictions.
            df["resid"] = df["pred_stage1"] - df[actual_col]
    else:
        a_map, b_map = _load_stage1_affine(stage1_calibration_json)
        if not a_map:
            raise ValueError(f"Stage1 calibration has no a_by_end_min: {stage1_calibration_json}")

        # Apply stage 1 calibration to raw predictions, then compute residual bias.
        df["_a"] = df[end_min_col].map(a_map)
        df["_b"] = df[end_min_col].map(b_map).fillna(0.0)
        df = df.dropna(subset=["_a"])

        df["pred_stage1"] = df["_a"] * df[pred_col] + df["_b"]
        df["resid"] = df["pred_stage1"] - df[actual_col]

    bias_by_end_min: dict[str, float] = {}
    rows_by_end_min: dict[str, int] = {}
    rows_used = 0

    for end_min, g in df.groupby(end_min_col):
        n = int(len(g))
        key = str(int(end_min))
        rows_by_end_min[key] = n
        if n < int(min_rows_per_end_min):
            continue
        vals = pd.to_numeric(g["resid"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        if str(stat).lower() == "median":
            m = float(np.median(vals))
        else:
            m = float(np.mean(vals))
        if not np.isfinite(m):
            continue
        bias_by_end_min[key] = m
        rows_used += n

    payload = {
        "kind": "stage2_residual_bias_by_end_min",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "source_backtest_csv": str(backtest_csv),
        "stage1_calibration_json": str(stage1_calibration_json),
        "start": start,
        "end": end,
        "pred_col": pred_col,
        "actual_col": actual_col,
        "end_mins": [int(x) for x in end_mins],
        "min_rows_per_end_min": int(min_rows_per_end_min),
        "rows_used": int(rows_used),
        "rows_by_end_min": rows_by_end_min,
        "bias_by_end_min": bias_by_end_min,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "out_path": str(out_path),
        "rows_used": int(rows_used),
        "bias_by_end_min": bias_by_end_min,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit late-game (stage2) residual bias for 5-min cumulative endpoints")
    ap.add_argument("--backtest-csv", required=True, help="Backtest CSV (e.g., outputs/backtests/segments_5min_*.csv)")
    ap.add_argument(
        "--stage1-calibration",
        default=str(Path("outputs") / "segment_calibration_5min.json"),
        help="Stage 1 affine calibration JSON (default: outputs/segment_calibration_5min.json)",
    )
    ap.add_argument(
        "--out",
        default=str(Path("outputs") / "segment_calibration_stage2_5min.json"),
        help="Output JSON path (default: outputs/segment_calibration_stage2_5min.json)",
    )
    ap.add_argument(
        "--pred-already-stage1",
        action="store_true",
        help=(
            "Treat pred-col values in the backtest CSV as already stage1-calibrated. "
            "In this mode, residual is computed as pred-col - actual (no extra stage1 application)."
        ),
    )
    ap.add_argument(
        "--pred-already-stage2",
        action="store_true",
        help=(
            "Treat pred-col values as already stage2-corrected as well (i.e., stage1 + stage2 already applied). "
            "In this mode, the fitter will undo the existing stage2 bias map (from the current --out file) before "
            "computing residuals, so that the newly-fit stage2 represents the residual bias to subtract from stage1-only predictions."
        ),
    )
    ap.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Merge fitted bias_by_end_min into the existing stage2 JSON (if present), "
            "overriding only the fitted endpoints and preserving any others."
        ),
    )
    ap.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    ap.add_argument(
        "--window-days",
        type=int,
        default=0,
        help=(
            "If >0, fit on a rolling window ending at --end (inclusive). "
            "For example, --window-days 21 uses [end-20, end]."
        ),
    )
    ap.add_argument("--end-mins", default="35,40", help="Comma-separated endpoint minutes to fit (default: 35,40)")
    ap.add_argument("--min-rows-per-end-min", type=int, default=150, help="Minimum rows per endpoint to include")
    ap.add_argument("--min-endpoints", type=int, default=2, help="Minimum number of endpoints (end_min) that must be fit")
    ap.add_argument("--min-rows-used", type=int, default=400, help="Minimum total rows used across fitted endpoints")
    ap.add_argument("--pred-col", default="pred_q50", help="Prediction column (default pred_q50)")
    ap.add_argument("--stat", default="mean", choices=["mean", "median"], help="Residual center to use (default: mean)")
    args = ap.parse_args()

    end_mins = [int(x.strip()) for x in str(args.end_mins).split(",") if x.strip()]

    start = str(args.start) if args.start else None
    end = str(args.end) if args.end else None

    if int(args.window_days) > 0:
        if start is not None:
            raise SystemExit("Do not combine --start with --window-days; use --end + --window-days")
        if end is None:
            raise SystemExit("--window-days requires --end")
        try:
            end_dt = dt.date.fromisoformat(end)
        except Exception:
            raise SystemExit(f"Invalid --end date: {end}")
        start_dt = end_dt - dt.timedelta(days=int(args.window_days) - 1)
        start = start_dt.isoformat()

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    existing_stage2_bias: dict[int, float] | None = None
    if bool(args.pred_already_stage2):
        if not bool(args.pred_already_stage1):
            raise SystemExit("--pred-already-stage2 requires --pred-already-stage1")
        if out_path.exists():
            try:
                existing_obj = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing_obj = {}
            raw = existing_obj.get("bias_by_end_min") if isinstance(existing_obj, dict) else None
            if isinstance(raw, dict) and raw:
                parsed: dict[int, float] = {}
                for k, v in raw.items():
                    try:
                        kk = int(float(k))
                        vv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(vv):
                        parsed[kk] = vv
                existing_stage2_bias = parsed

    fit_res = fit_stage2_bias(
        backtest_csv=Path(args.backtest_csv),
        stage1_calibration_json=Path(args.stage1_calibration),
        out_path=tmp_path,
        start=start,
        end=end,
        end_mins=end_mins,
        pred_already_stage1=bool(args.pred_already_stage1),
        pred_already_stage2=bool(args.pred_already_stage2),
        existing_stage2_bias_by_end_min=existing_stage2_bias,
        pred_col=str(args.pred_col),
        min_rows_per_end_min=int(args.min_rows_per_end_min),
        stat=str(args.stat),
    )

    # Decide whether to promote the freshly-fit calibration.
    try:
        payload = json.loads(tmp_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    bias_map = payload.get("bias_by_end_min") if isinstance(payload, dict) else None
    endpoints_fit = len(bias_map) if isinstance(bias_map, dict) else 0
    rows_used = int(payload.get("rows_used") or 0) if isinstance(payload, dict) else 0

    should_promote = endpoints_fit >= int(args.min_endpoints) and rows_used >= int(args.min_rows_used)

    result = {
        "status": "updated" if should_promote else "skipped",
        "backtest_csv": str(Path(args.backtest_csv)),
        "out_path": str(out_path),
        "tmp_path": str(tmp_path),
        "start": start,
        "end": end,
        "window_days": int(args.window_days),
        "stat": str(args.stat),
        "pred_col": str(args.pred_col),
        "min_rows_per_end_min": int(args.min_rows_per_end_min),
        "min_endpoints": int(args.min_endpoints),
        "min_rows_used": int(args.min_rows_used),
        "rows_used": rows_used,
        "endpoints_fit": endpoints_fit,
        "bias_by_end_min": bias_map if isinstance(bias_map, dict) else {},
    }

    if should_promote:
        if args.merge_existing and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

            try:
                fresh = json.loads(tmp_path.read_text(encoding="utf-8"))
            except Exception:
                fresh = {}

            if isinstance(existing, dict) and isinstance(fresh, dict):
                ex_bias = existing.get("bias_by_end_min")
                fr_bias = fresh.get("bias_by_end_min")
                if isinstance(ex_bias, dict) and isinstance(fr_bias, dict) and fr_bias:
                    merged = dict(ex_bias)
                    merged.update(fr_bias)
                    fresh["bias_by_end_min"] = merged
                tmp_path.write_text(json.dumps(fresh, indent=2, sort_keys=True), encoding="utf-8")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)
    else:
        # Keep existing calibration (if any).
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
