"""Summarize A/B sim-quantile backtest JSON reports.

This is a small convenience utility for comparing multiple
`ab-sim-quantiles` outputs (e.g., alpha sweeps).

Example:
  python scripts/summarize_ab_sim_quantiles.py outputs/backtests/_ab_simq10_roll_a*_ab_*.json

By default, it prints a table sorted by `explicit.crps_total` delta (more
negative = better vs native).
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _get(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


@dataclass(frozen=True)
class Row:
    file: str
    engine: str
    start: str
    end: str
    alpha: float | None
    tq_source: str | None
    tq_total: str | None
    tq_margin: str | None
    d_crps_total: float | None
    d_mae_total: float | None
    d_rmse_total: float | None
    target_total_rate: float | None


def _iter_paths(args_paths: list[str], args_glob: str | None) -> list[Path]:
    paths: list[Path] = []

    for p in args_paths:
        # argparse can pass glob patterns literally; expand here so users can
        # use wildcards without quoting differences across shells.
        expanded = glob.glob(p)
        if expanded:
            paths.extend(Path(x) for x in expanded)
        else:
            paths.append(Path(p))

    if args_glob:
        paths.extend(Path(x) for x in glob.glob(args_glob))

    # De-dup + stable sort
    uniq = sorted({p.resolve() for p in paths if p is not None})
    return [p for p in uniq if p.exists() and p.is_file()]


def _load_row(p: Path) -> Row | None:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    sim_env = obj.get("sim_env") if isinstance(obj.get("sim_env"), dict) else {}
    alpha = _safe_float(sim_env.get("NCAAB_SIM_TARGET_QUANTILES_ALPHA"))
    tq_source = sim_env.get("NCAAB_SIM_TARGET_QUANTILES_SOURCE")
    tq_total = sim_env.get("NCAAB_SIM_TARGET_QUANTILES_TOTAL")
    tq_margin = sim_env.get("NCAAB_SIM_TARGET_QUANTILES_MARGIN")

    engine = str(obj.get("engine") or "")
    r = obj.get("range") if isinstance(obj.get("range"), dict) else {}
    start = str(r.get("start") or "")
    end = str(r.get("end") or "")

    d_exp = _get(obj, "deltas_vs_native", "explicit")
    if not isinstance(d_exp, dict):
        d_exp = {}

    d_crps_total = _safe_float(_get(d_exp, "crps_total", "crps"))
    d_mae_total = _safe_float(_get(d_exp, "totals", "mae"))
    d_rmse_total = _safe_float(_get(d_exp, "totals", "rmse"))
    target_total_rate = _safe_float(_get(d_exp, "targeting", "total", "rate"))

    return Row(
        file=p.name,
        engine=engine,
        start=start,
        end=end,
        alpha=alpha,
        tq_source=str(tq_source) if tq_source is not None else None,
        tq_total=str(tq_total) if tq_total is not None else None,
        tq_margin=str(tq_margin) if tq_margin is not None else None,
        d_crps_total=d_crps_total,
        d_mae_total=d_mae_total,
        d_rmse_total=d_rmse_total,
        target_total_rate=target_total_rate,
    )


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return ""


def _print_table(rows: Iterable[Row]) -> None:
    rows_l = list(rows)
    # Sort by best (most negative) CRPS delta first; then MAE/RMSE.
    rows_l.sort(
        key=lambda r: (
            float("inf") if r.d_crps_total is None else r.d_crps_total,
            float("inf") if r.d_mae_total is None else r.d_mae_total,
            float("inf") if r.d_rmse_total is None else r.d_rmse_total,
            float("inf") if r.alpha is None else r.alpha,
            r.file,
        )
    )

    headers = [
        "file",
        "engine",
        "range",
        "alpha",
        "src",
        "tot",
        "mar",
        "d_crps_total",
        "d_mae_total",
        "d_rmse_total",
        "tgt_rate",
    ]

    body: list[list[str]] = []
    for r in rows_l:
        body.append(
            [
                r.file,
                r.engine,
                f"{r.start}..{r.end}" if r.start or r.end else "",
                _fmt(r.alpha, 2),
                r.tq_source or "",
                r.tq_total or "",
                r.tq_margin or "",
                _fmt(r.d_crps_total, 4),
                _fmt(r.d_mae_total, 4),
                _fmt(r.d_rmse_total, 4),
                _fmt(r.target_total_rate, 3),
            ]
        )

    # Compute widths
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(_line(headers))
    print(_line(["-" * w for w in widths]))
    for row in body:
        print(_line(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize A/B sim-quantile backtest JSON reports")
    ap.add_argument("paths", nargs="*", help="One or more JSON report paths (globs allowed)")
    ap.add_argument("--glob", dest="glob", default=None, help="Optional glob pattern for reports")
    args = ap.parse_args()

    paths = _iter_paths(list(args.paths), args.glob)
    if not paths:
        # Reasonable default for common usage.
        paths = _iter_paths([], "outputs/backtests/_ab_simq*.json")

    rows: list[Row] = []
    for p in paths:
        r = _load_row(p)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No valid JSON reports found.")
        return 2

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
