import argparse
import json
from pathlib import Path
from datetime import date as _date

import numpy as np
import pandas as pd


def _iter_dates(start: str, end: str) -> list[str]:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return []
    if e < s:
        s, e = e, s
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(s, e, freq="D")]


def _summarize_quantiles(path: Path) -> dict:
    df = pd.read_csv(path)
    n = int(len(df))
    if n <= 0:
        return {"games": 0}

    out: dict[str, object] = {"games": n}

    # Spread presence (raw)
    spread_present = None
    if "spread_home" in df.columns:
        sh = pd.to_numeric(df["spread_home"], errors="coerce")
        spread_present = (sh.notna() & np.isfinite(sh))
        out["spread_present"] = int(spread_present.sum())
        out["spread_present_share"] = float(spread_present.mean())

    # Proxy source (preferred)
    if "abs_margin_proxy_source" in df.columns:
        src = df["abs_margin_proxy_source"].astype(str).str.lower().str.strip()
        spread_used = (src == "spread")
        expected_used = (src == "expected")
        out["proxy_source_spread"] = int(spread_used.sum())
        out["proxy_source_expected"] = int(expected_used.sum())
        out["proxy_source_other"] = int((~(spread_used | expected_used)).sum())
        out["proxy_source_spread_share"] = float(spread_used.mean())

        if spread_present is not None:
            # Useful for checking if proxy-source is aligned with raw spread presence.
            disagree = (spread_present != spread_used)
            out["spread_present_vs_proxy_source_disagree"] = int(disagree.sum())
    else:
        # Fallback inference (historical files)
        if spread_present is not None:
            out["proxy_source_spread"] = int(spread_present.sum())
            out["proxy_source_expected"] = int((~spread_present).sum())
            out["proxy_source_spread_share"] = float(spread_present.mean())

    # Proxy numeric summary
    if "abs_margin_proxy" in df.columns:
        v = pd.to_numeric(df["abs_margin_proxy"], errors="coerce")
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            out["abs_margin_proxy_min"] = float(v.min())
            out["abs_margin_proxy_p50"] = float(v.median())
            out["abs_margin_proxy_p90"] = float(v.quantile(0.90))
            out["abs_margin_proxy_max"] = float(v.max())

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose abs-margin proxy usage (spread vs expected-margin)")
    ap.add_argument("--date", type=str, default=None, help="Single date YYYY-MM-DD (default: today)")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs directory (default: outputs)")
    ap.add_argument(
        "--quantiles-prefix",
        type=str,
        default="sim_quantiles_",
        help="Prefix for quantiles CSV files (default: sim_quantiles_)",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON only (default: pretty JSON)")
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)

    if args.start and args.end:
        dates = _iter_dates(args.start, args.end)
    else:
        d = args.date or _date.today().strftime("%Y-%m-%d")
        dates = [d]

    per_date: dict[str, dict] = {}
    totals = {
        "dates": 0,
        "games": 0,
        "proxy_source_spread": 0,
        "proxy_source_expected": 0,
        "missing_files": 0,
    }

    for d in dates:
        qpath = out_dir / f"{args.quantiles_prefix}{d}.csv"
        if not qpath.exists():
            per_date[d] = {"error": f"missing: {qpath}"}
            totals["missing_files"] += 1
            continue

        summ = _summarize_quantiles(qpath)
        per_date[d] = summ

        totals["dates"] += 1
        totals["games"] += int(summ.get("games") or 0)
        totals["proxy_source_spread"] += int(summ.get("proxy_source_spread") or 0)
        totals["proxy_source_expected"] += int(summ.get("proxy_source_expected") or 0)

    if totals["games"] > 0:
        totals["proxy_source_spread_share"] = float(totals["proxy_source_spread"]) / float(totals["games"])

    payload = {"quantiles_prefix": args.quantiles_prefix, "outputs_dir": str(out_dir), "totals": totals, "by_date": per_date}

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
