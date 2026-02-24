import argparse

import numpy as np
import pandas as pd


def _prep_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, int]:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mask = df[cols].notna().all(axis=1)
    return df.loc[mask, cols].to_numpy(dtype=float), int(mask.sum())


def _eval_shift_scale(
    actual: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    d: float,
    s: float,
) -> tuple[float, float, float, float, float, float]:
    q50a = q50 + d
    q10a = q50a + s * (q10 - q50)
    q90a = q50a + s * (q90 - q50)

    cov10 = float(np.mean(actual <= q10a))
    cov50 = float(np.mean(actual <= q50a))
    cov90 = float(np.mean(actual <= q90a))
    bias = float(np.mean(q50a - actual))
    width = float(np.mean(q90a - q10a))

    score = abs(cov10 - 0.10) + abs(cov50 - 0.50) + abs(cov90 - 0.90) + abs(bias) / 10.0
    return score, cov10, cov50, cov90, bias, width


def sweep(
    df: pd.DataFrame,
    *,
    label: str,
    actual_col: str,
    q10_col: str,
    q50_col: str,
    q90_col: str,
    d_grid: np.ndarray,
    s_grid: np.ndarray,
    top_k: int,
) -> None:
    arr, n = _prep_numeric(df, [actual_col, q10_col, q50_col, q90_col])
    actual = arr[:, 0]
    q10 = arr[:, 1]
    q50 = arr[:, 2]
    q90 = arr[:, 3]

    baseline = _eval_shift_scale(actual, q10, q50, q90, d=0.0, s=1.0)

    results: list[tuple[float, float, float, float, float, float, float, float]] = []
    for d in d_grid:
        for s in s_grid:
            score, cov10, cov50, cov90, bias, width = _eval_shift_scale(
                actual, q10, q50, q90, d=float(d), s=float(s)
            )
            results.append((score, float(d), float(s), cov10, cov50, cov90, bias, width))
    results.sort(key=lambda t: t[0])

    print(f"--- {label} n={n}")
    print(
        "baseline  d=+0.00 s=1.00  "
        f"score={baseline[0]:.3f}  cov10={baseline[1]:.3f} cov50={baseline[2]:.3f} cov90={baseline[3]:.3f}  "
        f"bias={baseline[4]:+.2f}  width={baseline[5]:.2f}"
    )
    for score, d, s, cov10, cov50, cov90, bias, width in results[:top_k]:
        print(
            f"best      d={d:+.2f} s={s:.2f}  "
            f"score={score:.3f}  cov10={cov10:.3f} cov50={cov50:.3f} cov90={cov90:.3f}  "
            f"bias={bias:+.2f}  width={width:.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grid-search post-hoc shift (d) and width scale (s) for q10/q50/q90 calibration."
    )
    ap.add_argument(
        "--csv",
        default="outputs/backtests/sim_accuracy_pregame_quantcal_2026-02-01_2026-02-22.csv",
        help="Per-game backtest CSV containing q10/q50/q90 and actual columns.",
    )
    ap.add_argument("--top", type=int, default=8, help="How many best candidates to print per target.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    d_grid_tot = np.arange(-10.0, 10.0 + 1e-9, 0.5)
    s_grid = np.arange(0.6, 1.8 + 1e-9, 0.1)
    d_grid_marg = np.arange(-4.0, 4.0 + 1e-9, 0.25)

    sweep(
        df,
        label="TOTAL full-game",
        actual_col="actual_total",
        q10_col="q10_total",
        q50_col="q50_total",
        q90_col="q90_total",
        d_grid=d_grid_tot,
        s_grid=s_grid,
        top_k=args.top,
    )
    sweep(
        df,
        label="TOTAL 1H",
        actual_col="actual_total_1h",
        q10_col="q10_total_1h",
        q50_col="q50_total_1h",
        q90_col="q90_total_1h",
        d_grid=d_grid_tot,
        s_grid=s_grid,
        top_k=args.top,
    )
    sweep(
        df,
        label="MARGIN full-game",
        actual_col="actual_margin",
        q10_col="q10_margin",
        q50_col="q50_margin",
        q90_col="q90_margin",
        d_grid=d_grid_marg,
        s_grid=s_grid,
        top_k=args.top,
    )
    sweep(
        df,
        label="MARGIN 1H",
        actual_col="actual_margin_1h",
        q10_col="q10_margin_1h",
        q50_col="q50_margin_1h",
        q90_col="q90_margin_1h",
        d_grid=d_grid_marg,
        s_grid=s_grid,
        top_k=args.top,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
