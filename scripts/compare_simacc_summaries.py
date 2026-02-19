"""Compare sim-accuracy summary JSON files.

Usage (PowerShell):
  .venv\Scripts\python.exe scripts\compare_simacc_summaries.py \
    outputs\backtests\simacc_marg0_a025_2026-02-11_2026-02-16_summary.json \
    outputs\backtests\simacc_marg1_a025_2026-02-11_2026-02-16_summary.json \
    outputs\backtests\simacc_marg1_a050_2026-02-11_2026-02-16_summary.json \
    outputs\backtests\simacc_marg1_a100_2026-02-11_2026-02-16_summary.json

Prints deltas vs the first (baseline) file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _acc(obj: dict[str, Any], key: str) -> float | None:
    try:
        v = (obj.get(key) or {}).get("acc")
        return None if v is None else float(v)
    except Exception:
        return None


def _mean_scoring(obj: dict[str, Any], key: str) -> float | None:
    try:
        scoring = obj.get("scoring") or {}
        v = (scoring.get(key) or {}).get("mean")
        return None if v is None else float(v)
    except Exception:
        return None


def _fmt(x: float | None) -> str:
    return "NA" if x is None else f"{x:.6f}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Provide 2+ summary JSON paths; first is baseline.")
        return 2

    paths = argv[1:]
    objs = [(p, _load(p)) for p in paths]
    base_path, base = objs[0]

    metrics = ["winners", "ats", "totals"]
    scoring_keys = ["crps_margin_final", "crps_total_final"]

    print(f"Baseline: {base_path}")
    print("Baseline ACC:")
    for m in metrics:
        print(f"  {m:7s}: {_fmt(_acc(base, m))}")
    for k in scoring_keys:
        v = _mean_scoring(base, k)
        if v is not None:
            print(f"  {k:18s}: {_fmt(v)}")

    print("\nDeltas vs baseline (candidate - baseline):")
    for cand_path, cand in objs[1:]:
        print(f"\nCandidate: {cand_path}")
        for m in metrics:
            c = _acc(cand, m)
            b = _acc(base, m)
            d = None if (c is None or b is None) else (c - b)
            print(f"  {m:7s}: delta={_fmt(d)} cand={_fmt(c)} base={_fmt(b)}")

        for k in scoring_keys:
            c = _mean_scoring(cand, k)
            b = _mean_scoring(base, k)
            if c is None or b is None:
                continue
            print(f"  {k:18s}: delta={_fmt(c - b)} cand={_fmt(c)} base={_fmt(b)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
