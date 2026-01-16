import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _write_empty_stake_sheet(path: Path):
    # Keep this aligned with the current CLI outputs (at least the common columns).
    header = [
        "date",
        "game_id",
        "event_id",
        "book",
        "market",
        "period",
        "selection",
        "line",
        "price",
        "edge",
        "kelly",
        "fractional",
        "stake",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n", encoding="utf-8")


def _run_bankroll_optimize(args: list[str]) -> int:
    # Use the active venv interpreter.
    cmd = [sys.executable, "-m", "ncaab_model.cli", "bankroll-optimize", *args]
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate dated stake sheets for a date range (base/calcap/isocap, etc). "
            "This is intended for fair ROI backtests where distributional variants must merge per-day quantiles."
        )
    )
    ap.add_argument("--start-date", required=True, type=str)
    ap.add_argument("--end-date", required=True, type=str)
    ap.add_argument("--outputs-dir", type=str, default="outputs")
    ap.add_argument(
        "--kinds",
        type=str,
        default="base,calcap,isocap",
        help="Comma-separated: base, calcap, isocap (and/or cal, iso)",
    )

    # Bankroll sizing knobs (defaults mirror the fair-comparison settings).
    ap.add_argument("--bankroll", type=float, default=200.0)
    ap.add_argument("--kelly-fraction", type=float, default=0.5)
    ap.add_argument("--include-markets", type=str, default="totals,spreads")
    ap.add_argument("--min-edge-total", type=float, default=0.5)
    ap.add_argument("--min-edge-margin", type=float, default=0.5)
    ap.add_argument("--min-kelly", type=float, default=0.01)
    ap.add_argument("--max-pct-per-bet", type=float, default=0.03)
    ap.add_argument("--max-daily-risk-pct", type=float, default=0.10)

    ap.add_argument(
        "--calibration-artifact",
        type=str,
        default=None,
        help="Optional path to z-recenter calibration JSON (defaults to outputs/models_dist/calibration_totals.json if present).",
    )
    ap.add_argument(
        "--isotonic-params-path",
        type=str,
        default=None,
        help="Optional path to isotonic params JSON (defaults to outputs/calibration_params.json if present).",
    )

    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    kinds = [k.strip() for k in str(args.kinds).split(",") if k.strip()]
    if not kinds:
        raise SystemExit("No kinds specified")

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")

    cal_art = Path(args.calibration_artifact) if args.calibration_artifact else (out_dir / "models_dist" / "calibration_totals.json")
    iso_par = Path(args.isotonic_params_path) if args.isotonic_params_path else (out_dir / "calibration_params.json")

    for date_str in _daterange(start, end):
        merged = out_dir / f"align_period_{date_str}_edges.csv"
        quant = out_dir / f"sim_quantiles_{date_str}.csv"

        if not merged.exists():
            print(f"[gen] {date_str}: missing {merged.name}; skipping")
            continue

        for kind in kinds:
            out_path = out_dir / f"stake_sheet_{date_str}_{kind}.csv"
            # Ensure we don't keep stale content when the CLI exits early without writing.
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            common = [
                "--merged-csv",
                str(merged),
                "--out",
                str(out_path),
                "--bankroll",
                str(args.bankroll),
                "--kelly-fraction",
                str(args.kelly_fraction),
                "--include-markets",
                str(args.include_markets),
                "--min-edge-total",
                str(args.min_edge_total),
                "--min-edge-margin",
                str(args.min_edge_margin),
                "--min-kelly",
                str(args.min_kelly),
                "--max-pct-per-bet",
                str(args.max_pct_per_bet),
                "--max-daily-risk-pct",
                str(args.max_daily_risk_pct),
            ]

            extra: list[str] = []
            kind_l = kind.strip().lower()
            if kind_l in {"cal", "iso", "calcap", "isocap"}:
                extra.append("--use-distributional")
                extra.append("--calibrate-probabilities")
                if quant.exists():
                    extra.extend(["--quantiles-csv", str(quant)])
                if cal_art.exists():
                    extra.extend(["--calibration-artifact", str(cal_art)])

            if kind_l in {"iso", "isocap"}:
                extra.append("--isotonic-prob-calibration")
                if iso_par.exists():
                    extra.extend(["--isotonic-params-path", str(iso_par)])

            rc = _run_bankroll_optimize(common + extra)
            if not out_path.exists():
                # CLI exits 0 when empty, but doesn't write a file; write a header-only CSV for backtests.
                _write_empty_stake_sheet(out_path)
                print(f"[gen] {date_str} {kind}: empty -> wrote header-only")
            else:
                print(f"[gen] {date_str} {kind}: ok (rc={rc})")


if __name__ == "__main__":
    main()
