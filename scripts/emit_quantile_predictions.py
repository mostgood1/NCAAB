import argparse
import datetime as dt
import os
from typing import Optional

import numpy as np
import pandas as pd


Z10 = -1.2815515655446004
Z90 = +1.2815515655446004


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit q10/q50/q90 quantile sidecar from mean + sigma heuristics.")
    p.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD; defaults to today")
    p.add_argument("--input", type=str, default=None, help="Path to predictions_unified_enriched_<date>.csv; defaults under outputs/")
    p.add_argument("--output", type=str, default=None, help="Output sidecar path; defaults to outputs/quantiles_<date>.csv")
    p.add_argument("--sigma-total", type=float, default=None, help="Override sigma for totals if not present in CSV")
    p.add_argument("--sigma-margin", type=float, default=None, help="Override sigma for margins if not present in CSV")
    return p.parse_args()


def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    date = args.date or today_str()

    if args.input:
        inp = args.input
    else:
        inp = os.path.join("outputs", f"predictions_unified_enriched_{date}.csv")

    if args.output:
        outp = args.output
    else:
        outp = os.path.join("outputs", f"quantiles_{date}.csv")

    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input predictions not found: {inp}")

    df = pd.read_csv(inp)
    if "game_id" not in df.columns:
        raise ValueError("Input must contain 'game_id'")
    if "pred_total" not in df.columns or "pred_margin" not in df.columns:
        raise ValueError("Input must contain 'pred_total' and 'pred_margin'")

    # Sigma discovery with sensible defaults
    sigma_total_cols = ["sigma_total", "pred_total_sigma", "total_sigma"]
    sigma_margin_cols = ["sigma_margin", "pred_margin_sigma", "margin_sigma"]

    def col_or_default(cols, default):
        for c in cols:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").fillna(default).to_numpy()
        return np.full(len(df), default, dtype=float)

    sigma_total = col_or_default(sigma_total_cols, args.sigma_total if args.sigma_total is not None else 12.0)
    sigma_margin = col_or_default(sigma_margin_cols, args.sigma_margin if args.sigma_margin is not None else 7.0)

    mu_total = pd.to_numeric(df["pred_total"], errors="coerce").to_numpy()
    mu_margin = pd.to_numeric(df["pred_margin"], errors="coerce").to_numpy()

    # Compute quantiles assuming Gaussian as bootstrap; replace with trained quantile models later
    q10_total = mu_total + Z10 * sigma_total
    q50_total = mu_total
    q90_total = mu_total + Z90 * sigma_total

    q10_margin = mu_margin + Z10 * sigma_margin
    q50_margin = mu_margin
    q90_margin = mu_margin + Z90 * sigma_margin

    sidecar = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "pred_total_q10": q10_total,
            "pred_total_q50": q50_total,
            "pred_total_q90": q90_total,
            "pred_margin_q10": q10_margin,
            "pred_margin_q50": q50_margin,
            "pred_margin_q90": q90_margin,
        }
    )

    os.makedirs(os.path.dirname(outp), exist_ok=True)
    sidecar.to_csv(outp, index=False)
    print(f"Wrote {outp} with {len(sidecar)} rows.")


if __name__ == "__main__":
    main()
