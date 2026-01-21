from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_stage2(seg: pd.DataFrame, bias_by_end_min: dict[float, float]) -> pd.DataFrame:
    if seg.empty:
        return seg

    if "end_min" not in seg.columns:
        return seg

    end_min = pd.to_numeric(seg["end_min"], errors="coerce")
    seg = seg.copy()
    seg["end_min"] = end_min

    bias = seg["end_min"].map(bias_by_end_min).fillna(0.0)

    # Stage2 is defined as residual bias to subtract from predictions.
    cols = [
        c
        for c in (
            "mu_total_score_end",
            "q10_total_score_end",
            "q50_total_score_end",
            "q90_total_score_end",
        )
        if c in seg.columns
    ]

    for col in cols:
        seg[col] = pd.to_numeric(seg[col], errors="coerce")
        seg[col] = seg[col] - bias

    # Keep things sane
    for col in cols:
        seg[col] = seg[col].replace([np.inf, -np.inf], np.nan)

    return seg


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply stage2 residual bias to sim_segments CSVs")
    ap.add_argument("--stage2-json", default="outputs/segment_calibration_stage2_5min.json")
    ap.add_argument("--in-dir", default="outputs")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD dates")
    ap.add_argument("--in-prefix", default="sim_segments_stage1only_")
    ap.add_argument("--out-prefix", default="sim_segments_stage2_")
    args = ap.parse_args()

    stage2 = _read_json(Path(args.stage2_json))
    raw_map = stage2.get("bias_by_end_min") if isinstance(stage2, dict) else None
    if not isinstance(raw_map, dict) or not raw_map:
        raise SystemExit(f"No bias_by_end_min found in {args.stage2_json}")

    bias_by_end_min: dict[float, float] = {}
    for k, v in raw_map.items():
        try:
            kk = float(k)
            vv = float(v)
        except Exception:
            continue
        if np.isfinite(kk) and np.isfinite(vv):
            bias_by_end_min[kk] = vv

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in args.dates:
        inp = in_dir / f"{args.in_prefix}{d}.csv"
        outp = out_dir / f"{args.out_prefix}{d}.csv"
        if not inp.exists():
            raise SystemExit(f"Missing input: {inp}")

        seg = pd.read_csv(inp)
        seg2 = apply_stage2(seg, bias_by_end_min)
        seg2.to_csv(outp, index=False)

    print({"dates": args.dates, "out_prefix": args.out_prefix, "out_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
