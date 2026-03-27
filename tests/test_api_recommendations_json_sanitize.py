import importlib

import pandas as pd


def test_api_recommendations_sanitizes_nat_rows(monkeypatch, tmp_path):
    app_module = importlib.import_module("app")
    date_str = "2026-03-27"

    monkeypatch.setattr(app_module, "OUT", tmp_path)
    monkeypatch.setattr(app_module, "_today_request_local_str", lambda: date_str)
    monkeypatch.setattr(app_module, "_today_local_str", lambda: date_str)
    monkeypatch.setattr(
        app_module,
        "_load_picks",
        lambda: pd.DataFrame(
            [
                {
                    "date": date_str,
                    "game_id": "401856570",
                    "home_team": "Duke Blue Devils",
                    "away_team": "St. John's Red Storm",
                    "market": "totals",
                    "period": "full_game",
                    "bet": "Under",
                    "line": 140.5,
                    "price": -110,
                    "edge": 4.2,
                    "book": "test",
                    "rec_type": "Total",
                    "rec_code": "OU",
                    "pred_total": 145.1,
                    "start_time_iso": pd.NaT,
                }
            ]
        ),
    )

    app_module.app.testing = True
    with app_module.app.test_client() as client:
        resp = client.get(f"/api/recommendations?date={date_str}")

    assert resp.status_code == 200
    payload = resp.get_json() or {}
    assert payload.get("status") == "ok"
    rows = payload.get("data") or []
    assert len(rows) == 1
    assert rows[0].get("start_time_iso") is None
    assert rows[0].get("home_team") == "Duke Blue Devils"