from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Ensure repo root on sys.path for local imports (src.*)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fit affine calibration for 5-min cumulative endpoints from a backtest CSV, "
            "but only overwrite the existing calibration if the fit is sufficiently supported."
        )
    )
    ap.add_argument("--backtest-csv", required=True, help="Backtest CSV with columns date,end_min,actual_total,pred_q50")
    ap.add_argument("--out", default=str(Path("outputs") / "segment_calibration_5min.json"), help="Output JSON path")
    ap.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    ap.add_argument("--pred-col", default="pred_q50", help="Prediction column to calibrate")
    ap.add_argument("--min-a", type=float, default=0.0, help="Minimum slope per end_min")
    ap.add_argument("--max-a", type=float, default=1.30, help="Maximum slope per end_min")
    ap.add_argument("--min-rows-per-end-min", type=int, default=250, help="Minimum rows per end_min to fit")
    ap.add_argument("--min-endpoints", type=int, default=4, help="Minimum number of endpoints (end_min) that must be fit")
    ap.add_argument("--min-rows-used", type=int, default=800, help="Minimum total rows used across fitted endpoints")
    args = ap.parse_args()

    from src.backtests.segments_5min_calibration_fit import FitSegments5MinCalibrationConfig, fit_segments_5min_calibration

    backtest_csv = Path(args.backtest_csv)
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    result = {
        "status": "error",
        "backtest_csv": str(backtest_csv),
        "out_path": str(out_path),
        "tmp_path": str(tmp_path),
        "start": args.start,
        "end": args.end,
        "pred_col": args.pred_col,
        "thresholds": {
            "min_endpoints": int(args.min_endpoints),
            "min_rows_used": int(args.min_rows_used),
            "min_rows_per_end_min": int(args.min_rows_per_end_min),
        },
    }

    cfg = FitSegments5MinCalibrationConfig(
        backtest_csv=backtest_csv,
        out_path=tmp_path,
        start=args.start,
        end=args.end,
        pred_col=args.pred_col,
        min_a=float(args.min_a),
        max_a=float(args.max_a),
        min_rows_per_end_min=int(args.min_rows_per_end_min),
    )

    fit_res = fit_segments_5min_calibration(cfg)

    endpoints_fit = len(fit_res.get("a_by_end_min") or {})
    rows_used = int(fit_res.get("rows_used") or 0)

    result.update({
        "rows_used": rows_used,
        "endpoints_fit": endpoints_fit,
    })

    should_promote = endpoints_fit >= int(args.min_endpoints) and rows_used >= int(args.min_rows_used)

    if should_promote:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)
        result["status"] = "updated"
    else:
        # Keep existing calibration (if any).
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        result["status"] = "skipped"
        result["reason"] = "insufficient_support"

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
