import math

import numpy as np
import pandas as pd

from src.simulation.game_sim import simulate_game_row


def test_event_sim_engine_produces_finite_outputs():
    row = pd.Series(
        {
            # Feature-driven mean inputs
            "pace_game_est": 70.0,
            "home_ppp_mu": 1.05,
            "away_ppp_mu": 1.00,
            "home_ppp_allowed_mu": 1.00,
            "away_ppp_allowed_mu": 1.02,
            # Market lines for probability outputs
            "market_total": 140.5,
            "spread_home": -3.5,
            "market_total_1h": 67.5,
            "spread_home_1h": -2.0,
            # Half split
            "half_frac": 0.5,
        }
    )

    rng = np.random.default_rng(123)
    out = simulate_game_row(
        row,
        samples=600,
        mean_source="features",
        engine="events",
        use_pace=False,
        rng=rng,
    )

    assert out["sim_ok"] is True
    assert out["sim_method"] == "events"

    for k in [
        "mu_total",
        "mu_margin",
        "q10_total",
        "q50_total",
        "q90_total",
        "q10_total_1h",
        "q50_total_1h",
        "q90_total_1h",
        "p_over_market",
        "p_cover_home",
        "p_home_win",
    ]:
        v = out.get(k)
        assert v is None or (isinstance(v, (int, float)) and math.isfinite(float(v)))

    # Probabilities should be in [0, 1]
    for k in ["p_over_market", "p_over_market_1h", "p_cover_home", "p_cover_home_1h", "p_home_win", "p_home_win_1h"]:
        v = out.get(k)
        if v is not None:
            assert 0.0 <= float(v) <= 1.0
