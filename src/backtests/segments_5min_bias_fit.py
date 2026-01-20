from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FitSegments5MinBiasConfig:
    backtest_csv: Path
    out_path: Path
    start: str | None = None
    end: str | None = None


def fit_segments_5min_bias(cfg: FitSegments5MinBiasConfig) -> dict:
    df = pd.read_csv(cfg.backtest_csv)
    if df.empty:
        raise ValueError(f"Empty backtest CSV: {cfg.backtest_csv}")

    for col in ("date", "end_min", "actual_total", "pred_q50"):
        if col not in df.columns:
            raise ValueError(f"Backtest CSV missing required column: {col}")

    if cfg.start:
        df = df[df["date"] >= cfg.start]
    if cfg.end:
        df = df[df["date"] <= cfg.end]

    df = df.dropna(subset=["end_min", "actual_total", "pred_q50"])
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    df = df.dropna(subset=["end_min"])

    df["err_q50"] = pd.to_numeric(df["pred_q50"], errors="coerce") - pd.to_numeric(df["actual_total"], errors="coerce")
    df = df.dropna(subset=["err_q50"])

    bias = df.groupby("end_min")["err_q50"].mean().to_dict()

    out = {
        "start": cfg.start,
        "end": cfg.end,
        "source_backtest_csv": str(cfg.backtest_csv),
        "rows_used": int(len(df)),
        "bias_by_end_min": {str(int(float(k))): float(v) for k, v in bias.items()},
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    out["out_path"] = str(cfg.out_path)
    return out
