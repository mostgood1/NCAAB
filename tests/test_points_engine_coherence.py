import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from src.simulation.game_sim import simulate_game_row


def test_points_engine_guardrails_and_1h_full_coherence():
    # Construct a row where an explicit 1H mean is implausibly high
    # relative to the full-game mean. The simulation should ignore it
    # and fall back to proportional scaling.
    row = pd.Series(
        {
            "pred_total": 106.0,
            "pred_margin": 0.0,
            # Implausible 1H mean (would imply ~31 2H points at median)
            "pred_total_1h": 75.0,
            "pred_margin_1h": 0.0,
            # Force typical half split so the fallback is near 53
            "half_frac": 0.50,
            # Add market lines just to keep probability outputs defined
            "market_total": 140.5,
            "spread_home": -3.5,
            "market_total_1h": 67.5,
            "spread_home_1h": -2.0,
        }
    )

    rng = np.random.default_rng(123)
    out = simulate_game_row(
        row,
        samples=2500,
        mean_source="auto",
        engine="normal",
        use_pace=False,
        rng=rng,
    )

    assert out["sim_ok"] is True
    assert out["sim_method"] == "points"

    # Guardrail should prevent 1H median from hugging the (implausible) 75
    # when full-game is ~106 and half_frac is 0.5.
    assert float(out["q50_total_1h"]) < 65.0

    # Segments should include an endpoint at 20 minutes with a plausible q50.
    segs = out.get("_segments_rows")
    assert isinstance(segs, list) and len(segs) > 0
    seg20 = next((s for s in segs if int(s.get("end_min")) == 20), None)
    assert seg20 is not None
    assert float(seg20["q50_total_score_end"]) < 65.0
