import numpy as np
import pandas as pd

from src.simulation.sim_backtest import _build_labels


def test_build_labels_pushes_become_nan():
    df = pd.DataFrame(
        {
            "actual_total": [150, 150, 150],
            "market_total": [149.5, 150.0, 150.5],
            "actual_margin": [5, 0, -3],
            "spread_home": [-4.5, 0.0, 2.5],
            "home_score_1h": [40, 30, 35],
            "away_score_1h": [35, 30, 38],
            "market_total_1h": [74.5, 60.0, 73.5],
            "spread_home_1h": [-2.5, 0.0, 1.0],
        }
    )

    out = _build_labels(df)

    # Over: middle row is push
    np.testing.assert_allclose(
        out["y_over"].to_numpy(dtype=float),
        np.array([1.0, np.nan, 0.0], dtype=float),
        equal_nan=True,
    )

    # Home cover: margin + spread_home -> [0.5, 0, -0.5] => [1, nan(push), 0]
    np.testing.assert_allclose(
        out["y_cover_home"].to_numpy(dtype=float),
        np.array([1.0, np.nan, 0.0], dtype=float),
        equal_nan=True,
    )

    # Home win: margin==0 is neither win nor loss (treated as NaN)
    np.testing.assert_allclose(
        out["y_home_win"].to_numpy(dtype=float),
        np.array([1.0, np.nan, 0.0], dtype=float),
        equal_nan=True,
    )
