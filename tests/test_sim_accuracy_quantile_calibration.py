from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtests.sim_accuracy_backtest import SimAccuracyBacktestConfig, run_sim_accuracy_backtest


def test_sim_accuracy_backtest_quantile_calibration(tmp_path: Path):
    out_dir = tmp_path
    (out_dir / "daily_results").mkdir(parents=True, exist_ok=True)

    date = "2026-01-01"

    # Minimal finalized results input
    results = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_team": "A",
                "away_team": "B",
                "home_score": 55,
                "away_score": 45,
                "market_total": 101.0,
                "spread_home": -1.5,
            },
            {
                "game_id": "g2",
                "home_team": "C",
                "away_team": "D",
                "home_score": 60,
                "away_score": 60,
                "market_total": 118.0,
                "spread_home": 2.5,
            },
            {
                "game_id": "g3",
                "home_team": "E",
                "away_team": "F",
                "home_score": 35,
                "away_score": 45,
                "market_total": 82.0,
                "spread_home": 0.5,
            },
        ]
    )
    results.to_csv(out_dir / "daily_results" / f"results_{date}.csv", index=False)

    # Minimal sim quantiles input
    sim = pd.DataFrame(
        [
            {
                "date": date,
                "game_id": "g1",
                "home_team": "A",
                "away_team": "B",
                "q10_total": 90.0,
                "q50_total": 100.0,
                "q90_total": 110.0,
                "q10_margin": -10.0,
                "q50_margin": 1.0,
                "q90_margin": 12.0,
            },
            {
                "date": date,
                "game_id": "g2",
                "home_team": "C",
                "away_team": "D",
                "q10_total": 110.0,
                "q50_total": 115.0,
                "q90_total": 125.0,
                "q10_margin": -8.0,
                "q50_margin": 0.0,
                "q90_margin": 9.0,
            },
            {
                "date": date,
                "game_id": "g3",
                "home_team": "E",
                "away_team": "F",
                "q10_total": 70.0,
                "q50_total": 85.0,
                "q90_total": 95.0,
                "q10_margin": -20.0,
                "q50_margin": -5.0,
                "q90_margin": 8.0,
            },
        ]
    )
    sim.to_csv(out_dir / f"sim_quantiles_{date}.csv", index=False)

    cfg = SimAccuracyBacktestConfig(out_dir=out_dir, start=date, end=date, out_prefix="simacc_test")
    res = run_sim_accuracy_backtest(cfg)

    summary_path = Path(res["wrote"]["summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "quantiles" in summary

    total_cov = summary["quantiles"]["total_final"]["coverage"]
    assert total_cov["q10"]["n"] == 3
    assert total_cov["q50"]["n"] == 3
    assert total_cov["q90"]["n"] == 3

    # actual totals are: 100, 120, 80
    # q10 totals are:    90, 110, 70  -> coverage = 0/3
    # q50 totals are:   100, 115, 85  -> coverage = 2/3
    # q90 totals are:   110, 125, 95  -> coverage = 3/3
    assert total_cov["q10"]["obs"] == 0.0
    assert abs(total_cov["q50"]["obs"] - (2.0 / 3.0)) < 1e-9
    assert total_cov["q90"]["obs"] == 1.0

    sharp = summary["quantiles"]["total_final"]["sharpness"]["p80_width_q10_q90"]
    assert sharp["n"] == 3
    assert sharp["mean"] == (20.0 + 15.0 + 25.0) / 3.0
