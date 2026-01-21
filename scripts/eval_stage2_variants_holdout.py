from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    base_csv = Path("outputs/backtests/segments_5min_holdout_controlled_A_stage1only_2026-01-11_to_2026-01-19.csv")
    if not base_csv.exists():
        raise SystemExit(f"Missing base holdout CSV: {base_csv}")

    df = pd.read_csv(base_csv)
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce").astype("Int64")
    df["actual_total"] = pd.to_numeric(df["actual_total"], errors="coerce")
    df["pred_q50"] = pd.to_numeric(df["pred_q50"], errors="coerce")
    df = df.dropna(subset=["end_min", "actual_total", "pred_q50"])
    df["end_min"] = df["end_min"].astype(int)

    err0 = df["pred_q50"] - df["actual_total"]
    base = {
        "variant": "BASE_stage1only",
        "mae_all": float(np.mean(np.abs(err0))),
        "bias_all": float(np.mean(err0)),
        "mae_35": float(np.mean(np.abs(err0[df["end_min"] == 35]))),
        "bias_35": float(np.mean(err0[df["end_min"] == 35])),
        "mae_40": float(np.mean(np.abs(err0[df["end_min"] == 40]))),
        "bias_40": float(np.mean(err0[df["end_min"] == 40])),
    }

    variants: list[dict] = []
    for p in sorted(Path("outputs").glob("segment_calibration_stage2_5min_w*_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        bm = payload.get("bias_by_end_min") or {}
        bias35 = float(bm.get("35", 0.0))
        bias40 = float(bm.get("40", 0.0))
        bias_map = {35: bias35, 40: bias40}

        pred2 = df["pred_q50"] - df["end_min"].map(bias_map).fillna(0.0)
        err2 = pred2 - df["actual_total"]

        out = {
            "variant": p.name.replace("segment_calibration_stage2_5min_", "").replace(".json", ""),
            "bias35_fit": bias35,
            "bias40_fit": bias40,
            "mae_all": float(np.mean(np.abs(err2))),
            "bias_all": float(np.mean(err2)),
        }

        for em in (35, 40):
            sub = err2[df["end_min"] == em]
            out[f"mae_{em}"] = float(np.mean(np.abs(sub)))
            out[f"bias_{em}"] = float(np.mean(sub))

        variants.append(out)

    res = pd.DataFrame(variants)
    res = res.sort_values(["mae_all", "mae_40", "mae_35"]).reset_index(drop=True)

    print("BASE:", base)
    print("\nTop candidates by mae_all (lower better):")
    cols = [
        "variant",
        "bias35_fit",
        "bias40_fit",
        "mae_all",
        "bias_all",
        "mae_35",
        "bias_35",
        "mae_40",
        "bias_40",
    ]
    print(res[cols].head(12).to_string(index=False))

    base_mae_all = base["mae_all"]
    base_bias_all = base["bias_all"]
    res2 = res.copy()
    res2["d_mae_all"] = res2["mae_all"] - base_mae_all
    res2["d_bias_all"] = res2["bias_all"] - base_bias_all

    print("\nDeltas vs BASE (negative = improvement):")
    print(res2[["variant", "d_mae_all", "d_bias_all", "mae_35", "bias_35", "mae_40", "bias_40"]].head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
