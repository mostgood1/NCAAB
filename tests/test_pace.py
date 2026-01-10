import math
from src.features.pace import estimate_possessions, pace_per_minutes


def test_estimate_possessions_basic():
    # Simple sanity: FGA=60, FTA=20, OR=10, TOV=12
    poss = estimate_possessions(60, 20, 10, 12)
    assert math.isclose(poss, 60 + 0.475*20 - 10 + 12, rel_tol=1e-6)


def test_pace_per_minutes_scaling():
    # 70 possessions in 45 minutes -> pace per 40
    p = pace_per_minutes(70, 45)
    assert math.isclose(p, 70 * (40/45), rel_tol=1e-6)
