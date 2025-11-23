"""Parity comparison script.

Compares local committed prediction artifacts against a remote copy (GitHub raw
or HTTP host) to ensure deterministic parity under commit-mode.

Exit codes:
 0 -> Full parity (no missing rows, all numeric diffs within tolerance)
 2 -> Missing rows locally or remotely
 3 -> Numeric diffs exceed tolerance
 4 -> Remote fetch failure / unreadable

Artifacts (when --write-artifacts):
  outputs/parity/parity_<date>.json  -> summary metrics
  outputs/parity/diff_rows_<date>.csv -> rows with meaningful differences

Usage examples:
  python scripts/compare_parity.py --date 2025-11-22
  python scripts/compare_parity.py --date 2025-11-22 --base-url https://ncaab.onrender.com
  python scripts/compare_parity.py --date 2025-11-22 --remote-source github --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

try:
	import pandas as pd  # type: ignore
except ImportError:  # Minimal fallback if pandas missing (unlikely)
	print("[fatal] pandas is required for parity comparison", file=sys.stderr)
	sys.exit(4)

try:  # Prefer requests; fallback to urllib
	import requests  # type: ignore
except ImportError:
	requests = None
	import urllib.request  # type: ignore


@dataclass
class ParityMetrics:
	date: str
	local_rows: int
	remote_rows: int
	intersection_rows: int
	missing_local: int
	missing_remote: int
	numeric_columns_compared: List[str]
	tolerance: float
	max_abs_diff: Dict[str, float]
	mean_abs_diff: Dict[str, float]
	diff_rows: int
	status: str
	exit_code: int


NUMERIC_COL_CANDIDATES = [
	"pred_total",
	"pred_margin",
	"pred_total_blend",
	"pred_margin_blend",
	"pred_total_seg",
	"pred_margin_seg",
	"blend_weight",
	"seg_n_rows",
]

REMOTE_GITHUB_RAW_TMPL = "https://raw.githubusercontent.com/mostgood1/NCAAB/main/outputs/predictions_{date}.csv"


def fetch_remote(date: str, args: argparse.Namespace) -> pd.DataFrame:
	"""Attempt to fetch remote predictions using strategy priority."""
	errors = []
	# Strategy 1: explicit remote file path override
	if args.remote_file and os.path.exists(args.remote_file):
		try:
			return pd.read_csv(args.remote_file)
		except Exception as e:  # pragma: no cover
			errors.append(f"remote_file read failed: {e}")

	# Strategy 2: github raw (commit-mode canonical source)
	if args.remote_source in ("github", "auto"):
		url = REMOTE_GITHUB_RAW_TMPL.format(date=date)
		df = _try_fetch_url(url, errors)
		if df is not None:
			return df

	# Strategy 3: supplied base URL (attempt a few path variants)
	if args.base_url:
		base = args.base_url.rstrip('/')
		for variant in [
			f"{base}/predictions_{date}.csv",
			f"{base}/outputs/predictions_{date}.csv",
		]:
			df = _try_fetch_url(variant, errors)
			if df is not None:
				return df

	raise RuntimeError("Remote fetch failed; tried: " + "; ".join(errors))


def _try_fetch_url(url: str, errors: List[str]) -> pd.DataFrame | None:
	try:
		if requests is not None:
			r = requests.get(url, timeout=10)
			if r.status_code == 200 and r.text.strip():
				# Use io.StringIO (pandas.compat.StringIO removed in newer pandas)
				import io
				return pd.read_csv(io.StringIO(r.text))
			errors.append(f"{url} status={r.status_code}")
		else:  # urllib fallback
			with urllib.request.urlopen(url) as resp:  # type: ignore
				text = resp.read().decode("utf-8")
				if text.strip():
					import io
					return pd.read_csv(io.StringIO(text))
			errors.append(f"{url} empty body")
	except Exception as e:  # pragma: no cover
		errors.append(f"{url} err={e}")
	return None


def compute_metrics(local_df: pd.DataFrame, remote_df: pd.DataFrame, date: str, tolerance: float) -> ParityMetrics:
	key_cols = [c for c in ["game_id", "date"] if c in local_df.columns and c in remote_df.columns]
	if not key_cols:
		raise ValueError("No common key columns (expected game_id and/or date)")

	# Ensure string for join stability
	for c in key_cols:
		local_df[c] = local_df[c].astype(str)
		remote_df[c] = remote_df[c].astype(str)

	local_rows = len(local_df)
	remote_rows = len(remote_df)

	merged = local_df.merge(remote_df, on=key_cols, how="outer", indicator=True, suffixes=("_local", "_remote"))
	missing_local = int((merged["_merge"] == "right_only").sum())
	missing_remote = int((merged["_merge"] == "left_only").sum())
	intersection_rows = int((merged["_merge"] == "both").sum())

	numeric_cols = [c for c in NUMERIC_COL_CANDIDATES if f"{c}_local" in merged.columns and f"{c}_remote" in merged.columns]
	max_abs_diff: Dict[str, float] = {}
	mean_abs_diff: Dict[str, float] = {}

	diff_mask_any = False
	diff_rows_mask = merged["_merge"] == "both"
	for col in numeric_cols:
		lcol = f"{col}_local"
		rcol = f"{col}_remote"
		diffs = []
		for lv, rv in zip(merged[lcol], merged[rcol]):
			try:
				if pd.isna(lv) or pd.isna(rv):
					continue
				diff = abs(float(lv) - float(rv))
				diffs.append(diff)
			except Exception:
				continue
		if diffs:
			max_abs_diff[col] = max(diffs)
			mean_abs_diff[col] = sum(diffs) / len(diffs)
			if max_abs_diff[col] > tolerance:
				diff_mask_any = True
				# Mark rows where diff > tolerance
				row_mask = (abs(merged[lcol] - merged[rcol]) > tolerance) & (merged["_merge"] == "both")
				diff_rows_mask = diff_rows_mask & row_mask | diff_rows_mask
		else:
			max_abs_diff[col] = math.nan
			mean_abs_diff[col] = math.nan

	diff_rows = int(diff_rows_mask.sum()) if diff_mask_any else 0

	status: str
	exit_code: int
	if missing_local or missing_remote:
		status = "missing_rows"
		exit_code = 2
	elif diff_mask_any:
		status = "numeric_diffs"
		exit_code = 3
	else:
		status = "parity_ok"
		exit_code = 0

	return ParityMetrics(
		date=date,
		local_rows=local_rows,
		remote_rows=remote_rows,
		intersection_rows=intersection_rows,
		missing_local=missing_local,
		missing_remote=missing_remote,
		numeric_columns_compared=numeric_cols,
		tolerance=tolerance,
		max_abs_diff=max_abs_diff,
		mean_abs_diff=mean_abs_diff,
		diff_rows=diff_rows,
		status=status,
		exit_code=exit_code,
	)


def write_artifacts(metrics: ParityMetrics, merged: pd.DataFrame, date: str, out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	summary_path = os.path.join(out_dir, f"parity_{date}.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(asdict(metrics), f, indent=2)

	if metrics.exit_code in (2, 3):
		# Only write diff CSV if issues present
		diff_csv_path = os.path.join(out_dir, f"diff_rows_{date}.csv")
		merged.to_csv(diff_csv_path, index=False)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Compare local vs remote predictions for parity")
	p.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
	p.add_argument("--local-file", help="Override local predictions file path")
	p.add_argument("--remote-file", help="Explicit remote file path (already on disk)")
	p.add_argument("--base-url", help="Base URL of deployed site for direct fetch attempts")
	p.add_argument("--remote-source", choices=["auto", "github", "http"], default="auto", help="Remote fetch strategy preference")
	p.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance for parity")
	p.add_argument("--write-artifacts", action="store_true", help="Write parity JSON + diff rows CSV")
	p.add_argument("--out-dir", default="outputs/parity", help="Directory for parity artifacts")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	date = args.date
	local_path = args.local_file or os.path.join("outputs", f"predictions_{date}.csv")
	if not os.path.exists(local_path):
		print(f"[fatal] Local predictions file not found: {local_path}", file=sys.stderr)
		sys.exit(2)
	try:
		local_df = pd.read_csv(local_path)
	except Exception as e:
		print(f"[fatal] Failed to read local file: {e}", file=sys.stderr)
		sys.exit(2)

	try:
		remote_df = fetch_remote(date, args)
	except Exception as e:
		print(f"[fatal] Remote fetch failed: {e}", file=sys.stderr)
		sys.exit(4)

	try:
		metrics = compute_metrics(local_df, remote_df, date, args.tolerance)
	except Exception as e:
		print(f"[fatal] Parity metric computation failed: {e}", file=sys.stderr)
		sys.exit(3)

	print(json.dumps(asdict(metrics), indent=2))

	if args.write_artifacts:
		merged = local_df.merge(remote_df, on=[c for c in ["game_id", "date"] if c in local_df.columns and c in remote_df.columns], how="outer", indicator=True, suffixes=("_local", "_remote"))
		write_artifacts(metrics, merged, date, args.out_dir)

	sys.exit(metrics.exit_code)


if __name__ == "__main__":  # pragma: no cover
	main()

