from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date(s: str | None) -> str | None:
	if not s:
		return None
	try:
		return dt.date.fromisoformat(str(s).strip()).isoformat()
	except Exception:
		return None


def main() -> int:
	ap = argparse.ArgumentParser(
		description=(
			"Fit stage2 residual bias for 5-min cumulative endpoints from a stage2-disabled backtest CSV. "
			"This computes mean(pred - actual) by end_min and writes outputs/segment_calibration_stage2_5min.json."
		)
	)
	ap.add_argument("--backtest-csv", required=True, help="Backtest CSV (e.g., outputs/backtests/segments_5min_*.csv)")
	ap.add_argument(
		"--out",
		default=str(Path("outputs") / "segment_calibration_stage2_5min.json"),
		help="Output JSON path (default: outputs/segment_calibration_stage2_5min.json)",
	)
	ap.add_argument("--pred-col", default="pred_q50", help="Prediction column (default: pred_q50)")
	ap.add_argument("--actual-col", default="actual_total", help="Actual column (default: actual_total)")
	ap.add_argument("--date-col", default="date", help="Date column (default: date)")
	ap.add_argument("--end-min-col", default="end_min", help="Endpoint column (default: end_min)")
	ap.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
	ap.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
	ap.add_argument(
		"--window-days",
		type=int,
		default=0,
		help="If >0, fit on a rolling window ending at --end (inclusive): [end-(window_days-1), end]",
	)
	ap.add_argument("--end-mins", default="5,10,15,20,25,30,35,40", help="Comma-separated endpoints to fit")
	ap.add_argument("--min-rows-per-end-min", type=int, default=150, help="Minimum rows per endpoint")
	ap.add_argument("--min-endpoints", type=int, default=2, help="Minimum number of endpoints that must be fit")
	ap.add_argument("--min-rows-used", type=int, default=400, help="Minimum total rows used across fitted endpoints")
	ap.add_argument(
		"--stat",
		default="mean",
		choices=["mean", "median"],
		help="Center statistic for residual (default: mean)",
	)
	ap.add_argument(
		"--merge-existing",
		action="store_true",
		help="Merge fitted endpoints into existing stage2 JSON (if present), preserving other keys",
	)
	ap.add_argument(
		"--zero-end-mins",
		default="20,40",
		help="Comma-separated endpoints to force to 0.0 in the output (default: 20,40)",
	)

	args = ap.parse_args()

	bt_path = Path(args.backtest_csv)
	if not bt_path.exists():
		print(json.dumps({"error": "missing_backtest_csv", "path": str(bt_path)}))
		return 2

	out_path = Path(args.out)

	df = pd.read_csv(bt_path)
	if df.empty:
		print(json.dumps({"error": "empty_backtest_csv", "path": str(bt_path)}))
		return 2

	for c in (args.date_col, args.end_min_col, args.actual_col, args.pred_col):
		if c not in df.columns:
			print(json.dumps({"error": "missing_column", "column": c, "path": str(bt_path)}))
			return 2

	start = _parse_date(args.start)
	end = _parse_date(args.end)

	if int(args.window_days) > 0:
		if args.start:
			print(json.dumps({"error": "do_not_combine_start_and_window_days"}))
			return 2
		if not end:
			print(json.dumps({"error": "window_days_requires_end"}))
			return 2
		end_dt = dt.date.fromisoformat(end)
		start_dt = end_dt - dt.timedelta(days=int(args.window_days) - 1)
		start = start_dt.isoformat()

	df = df.copy()
	df[args.date_col] = df[args.date_col].astype(str)
	if start:
		df = df[df[args.date_col] >= str(start)]
	if end:
		df = df[df[args.date_col] <= str(end)]

	df[args.end_min_col] = pd.to_numeric(df[args.end_min_col], errors="coerce")
	df[args.pred_col] = pd.to_numeric(df[args.pred_col], errors="coerce")
	df[args.actual_col] = pd.to_numeric(df[args.actual_col], errors="coerce")
	df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.end_min_col, args.pred_col, args.actual_col])
	if df.empty:
		print(json.dumps({"error": "no_rows_after_filter"}))
		return 2

	df[args.end_min_col] = df[args.end_min_col].astype(int)

	end_mins = [int(x.strip()) for x in str(args.end_mins).split(",") if x.strip()]
	df = df[df[args.end_min_col].isin(end_mins)]
	if df.empty:
		print(json.dumps({"error": "no_rows_for_end_mins", "end_mins": end_mins}))
		return 2

	# Residual is pred - actual; simulator subtracts this bias.
	df["resid"] = df[args.pred_col] - df[args.actual_col]

	bias_by_end_min: dict[str, float] = {}
	rows_by_end_min: dict[str, int] = {}
	rows_used = 0

	for end_min, g in df.groupby(args.end_min_col):
		n = int(len(g))
		key = str(int(end_min))
		rows_by_end_min[key] = n
		if n < int(args.min_rows_per_end_min):
			continue
		vals = pd.to_numeric(g["resid"], errors="coerce").dropna().to_numpy(dtype=float)
		if vals.size == 0:
			continue
		m = float(np.median(vals)) if str(args.stat).lower() == "median" else float(np.mean(vals))
		if not np.isfinite(m):
			continue
		bias_by_end_min[key] = m
		rows_used += n

	n_endpoints = len(bias_by_end_min)
	if n_endpoints < int(args.min_endpoints) or rows_used < int(args.min_rows_used):
		print(
			json.dumps(
				{
					"error": "insufficient_data",
					"endpoints_fit": n_endpoints,
					"rows_used": int(rows_used),
					"min_endpoints": int(args.min_endpoints),
					"min_rows_used": int(args.min_rows_used),
					"rows_by_end_min": rows_by_end_min,
				},
				indent=2,
			)
		)
		return 2

	if args.merge_existing and out_path.exists():
		try:
			existing = json.loads(out_path.read_text(encoding="utf-8"))
			if isinstance(existing, dict):
				existing_map = existing.get("bias_by_end_min")
				if isinstance(existing_map, dict):
					for k, v in existing_map.items():
						if k not in bias_by_end_min:
							try:
								bias_by_end_min[str(int(float(k)))] = float(v)
							except Exception:
								continue
		except Exception:
			pass

	# Force anchors to zero so stage2 doesn't fight the segment re-anchoring.
	zero_end_mins = [x.strip() for x in str(args.zero_end_mins).split(",") if x.strip()]
	for k in zero_end_mins:
		try:
			kk = str(int(float(k)))
		except Exception:
			continue
		bias_by_end_min[kk] = 0.0

	payload = {
		"kind": "stage2_residual_bias_by_end_min",
		"generated_at": dt.datetime.utcnow().isoformat() + "Z",
		"source_backtest_csv": str(bt_path),
		"start": start,
		"end": end,
		"pred_col": str(args.pred_col),
		"actual_col": str(args.actual_col),
		"end_mins": [int(x) for x in end_mins],
		"min_rows_per_end_min": int(args.min_rows_per_end_min),
		"rows_used": int(rows_used),
		"rows_by_end_min": rows_by_end_min,
		"bias_by_end_min": {str(int(float(k))): float(v) for k, v in bias_by_end_min.items()},
		"zero_end_mins": [str(int(float(x))) for x in zero_end_mins if x],
	}

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

	print(
		json.dumps(
			{
				"out_path": str(out_path),
				"rows_used": int(rows_used),
				"endpoints_fit": int(n_endpoints),
				"bias_by_end_min": payload["bias_by_end_min"],
			},
			indent=2,
		)
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
