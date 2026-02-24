import argparse

import numpy as np
import pandas as pd


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sweep(act: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, d_grid, s_grid):
    # Post-hoc affine adjustment around q50:
    # q50' = q50 + d
    # q10' = q50 + d + s*(q10-q50)
    # q90' = q50 + d + s*(q90-q50)

    best = []
    for d in d_grid:
        q50a = q50 + d
        for s in s_grid:
            q10a = q50a + s * (q10 - q50)
            q90a = q50a + s * (q90 - q50)

            cov10 = float(np.mean(act <= q10a))
            cov50 = float(np.mean(act <= q50a))
            cov90 = float(np.mean(act <= q90a))
            bias = float(np.mean(q50a - act))
            mae = float(np.mean(np.abs(q50a - act)))
            width = float(np.mean(q90a - q10a))

            # Score: focus on coverage targets; light penalty on mean bias
            score = (
                (cov10 - 0.10) ** 2
                + (cov50 - 0.50) ** 2
                + (cov90 - 0.90) ** 2
                + (bias / 10.0) ** 2
            )
            best.append((score, d, s, cov10, cov50, cov90, bias, mae, width))

    best.sort(key=lambda t: t[0])
    return best


def run(csv_path: str, kind: str, d_min: float, d_max: float, d_step: float, s_min: float, s_max: float, s_step: float, top: int):
    df = pd.read_csv(csv_path)

    if kind == "totals":
        a_col, q10_col, q50_col, q90_col = "actual_total", "q10_total", "q50_total", "q90_total"
    elif kind == "totals_1h":
        a_col, q10_col, q50_col, q90_col = "actual_total_1h", "q10_total_1h", "q50_total_1h", "q90_total_1h"
    elif kind == "margin":
        a_col, q10_col, q50_col, q90_col = "actual_margin", "q10_margin", "q50_margin", "q90_margin"
    elif kind == "margin_1h":
        a_col, q10_col, q50_col, q90_col = "actual_margin_1h", "q10_margin_1h", "q50_margin_1h", "q90_margin_1h"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    for c in [a_col, q10_col, q50_col, q90_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")

    act = _coerce_numeric(df[a_col])
    q10 = _coerce_numeric(df[q10_col])
    q50 = _coerce_numeric(df[q50_col])
    q90 = _coerce_numeric(df[q90_col])

    mask = act.notna() & q10.notna() & q50.notna() & q90.notna()
    act = act[mask].to_numpy(dtype=float)
    q10 = q10[mask].to_numpy(dtype=float)
    q50 = q50[mask].to_numpy(dtype=float)
    q90 = q90[mask].to_numpy(dtype=float)

    n = int(mask.sum())
    if n == 0:
        raise ValueError("No non-null rows for selected kind")

    def baseline_metrics():
        cov10 = float(np.mean(act <= q10))
        cov50 = float(np.mean(act <= q50))
        cov90 = float(np.mean(act <= q90))
        bias = float(np.mean(q50 - act))
        mae = float(np.mean(np.abs(q50 - act)))
        width = float(np.mean(q90 - q10))
        return cov10, cov50, cov90, bias, mae, width

    b_cov10, b_cov50, b_cov90, b_bias, b_mae, b_width = baseline_metrics()

    d_grid = np.arange(d_min, d_max + 1e-9, d_step)
    s_grid = np.arange(s_min, s_max + 1e-9, s_step)

    best = _sweep(act, q10, q50, q90, d_grid, s_grid)

    print(f"kind={kind} n={n}")
    print(
        "baseline: "
        f"cov10={b_cov10:.3f} cov50={b_cov50:.3f} cov90={b_cov90:.3f} "
        f"bias={b_bias:+.2f} mae={b_mae:.2f} width={b_width:.2f}"
    )
    print(f"grid: d=[{d_min},{d_max}] step={d_step} ({len(d_grid)} vals), s=[{s_min},{s_max}] step={s_step} ({len(s_grid)} vals)")

    print("\nTop candidates:")
    for score, d, s, cov10, cov50, cov90, bias, mae, width in best[:top]:
        print(
            f"score={score:.5f} d={d:+.2f} s={s:.2f} "
            f"cov10={cov10:.3f} cov50={cov50:.3f} cov90={cov90:.3f} "
            f"bias={bias:+.2f} mae={mae:.2f} width={width:.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="outputs/backtests/sim_accuracy_pregame_quantcal_2026-02-01_2026-02-22.csv",
        help="Per-game backtest CSV file",
    )
    ap.add_argument("--kind", choices=["totals", "totals_1h", "margin", "margin_1h"], default="totals")
    ap.add_argument("--d-min", type=float, default=-10.0)
    ap.add_argument("--d-max", type=float, default=10.0)
    ap.add_argument("--d-step", type=float, default=0.25)
    ap.add_argument("--s-min", type=float, default=0.60)
    ap.add_argument("--s-max", type=float, default=1.80)
    ap.add_argument("--s-step", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    run(
        csv_path=args.csv,
        kind=args.kind,
        d_min=args.d_min,
        d_max=args.d_max,
        d_step=args.d_step,
        s_min=args.s_min,
        s_max=args.s_max,
        s_step=args.s_step,
        top=args.top,
    )


if __name__ == "__main__":
    main()
