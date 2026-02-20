import math
from pathlib import Path

from app import app as flask_app
from app import OUT


def test_tracked_bets_place_settle_rollup():
    # Use a far-future date so we don't collide with real runs.
    date_s = "2099-01-01"
    out_path = Path(OUT) / f"tracked_bets_{date_s}.jsonl"

    # Ensure clean slate for this test.
    if out_path.exists():
        out_path.unlink()

    try:
        with flask_app.test_client() as c:
            # Place -110 bet with $110 stake -> $100 profit on win
            rv = c.post(
                "/api/tracked_bets/place",
                json={
                    "date": date_s,
                    "bet_id": "pytest-bet-1",
                    "game_id": "test-game-1",
                    "market": "total",
                    "selection": "over",
                    "line": 150.5,
                    "price_american": -110,
                    "stake": 110,
                    "book": "pytest",
                    "source": "test",
                },
            )
            assert rv.status_code == 200
            data = rv.get_json()
            assert data["ok"] is True
            assert data["bet_id"] == "pytest-bet-1"

            # List should show it pending
            rv = c.get(f"/api/tracked_bets?date={date_s}")
            assert rv.status_code == 200
            data = rv.get_json()
            assert data["ok"] is True
            assert data["date"] == date_s
            assert isinstance(data.get("bets"), list)
            assert len(data["bets"]) == 1
            bet = data["bets"][0]
            assert bet["bet_id"] == "pytest-bet-1"
            assert bet.get("status") == "pending"
            assert bet.get("result") in (None, "")

            # Settle as win; profit should be computed as 100
            rv = c.post(
                "/api/tracked_bets/settle",
                json={
                    "date": date_s,
                    "bet_id": "pytest-bet-1",
                    "result": "win",
                },
            )
            assert rv.status_code == 200
            data = rv.get_json()
            assert data["ok"] is True
            assert data["result"] == "win"
            assert math.isclose(float(data["profit"]), 100.0, rel_tol=0, abs_tol=1e-9)

            # List again should show settled result + profit
            rv = c.get(f"/api/tracked_bets?date={date_s}")
            assert rv.status_code == 200
            data = rv.get_json()
            bet = data["bets"][0]
            assert bet.get("status") == "settled"
            assert bet.get("result") == "win"
            assert math.isclose(float(bet.get("profit")), 100.0, rel_tol=0, abs_tol=1e-9)

            # Summary should roll up totals
            rv = c.get(f"/api/tracked_bets/summary?start_date={date_s}&end_date={date_s}")
            assert rv.status_code == 200
            data = rv.get_json()
            assert data["ok"] is True
            total = data["total"]
            assert math.isclose(float(total["stake"]), 110.0, rel_tol=0, abs_tol=1e-9)
            assert math.isclose(float(total["profit"]), 100.0, rel_tol=0, abs_tol=1e-9)
            assert math.isclose(float(total["roi"]), 100.0 / 110.0, rel_tol=0, abs_tol=1e-9)
    finally:
        if out_path.exists():
            out_path.unlink()
