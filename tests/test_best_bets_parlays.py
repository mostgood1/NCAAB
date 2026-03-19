import datetime as dt
import importlib
from pathlib import Path


def _pick(
    game_id: str,
    rec_code: str,
    score: float,
    p_win: float,
    away_team: str,
    home_team: str,
    selection: str,
    selection_team: str | None,
    line: float | None,
    price: float,
) -> dict:
    return {
        "date": "2026-03-19",
        "game_id": game_id,
        "rec_code": rec_code,
        "selection": selection,
        "selection_team": selection_team,
        "away_team": away_team,
        "home_team": home_team,
        "line": line,
        "price": price,
        "p_win": p_win,
        "score": score,
        "edge": 5.0,
        "start_time_local": "2026-03-19 18:00",
        "reasons": [f"p_win={p_win:.3f}"],
    }


def test_build_best_bets_parlays_ranks_unique_games(monkeypatch):
    module = importlib.import_module("src.eval.parlays")

    picks_by_market = {
        "ATS": [
            _pick("1001", "ATS", 95.0, 0.72, "Road One", "Home One", "Home", "Home One", -4.5, -110.0),
            _pick("1002", "ATS", 90.0, 0.70, "Road Two", "Home Two", "Away", "Road Two", -2.5, -110.0),
        ],
        "OU": [
            _pick("1001", "OU", 96.0, 0.75, "Road One", "Home One", "Under", None, 145.5, -110.0),
            _pick("1003", "OU", 94.0, 0.74, "Road Three", "Home Three", "Over", None, 138.5, -110.0),
        ],
        "ML": [
            _pick("1004", "ML", 88.0, 0.68, "Road Four", "Home Four", "Home Four", "Home Four", None, -125.0),
        ],
    }

    def fake_build_high_likelihood(cfg):
        market = tuple(cfg.include_markets)[0]
        return {"status": "ok", "date": cfg.date, "picks": list(picks_by_market.get(market, []))}

    monkeypatch.setattr(module, "build_high_likelihood", fake_build_high_likelihood)
    monkeypatch.setattr(module, "_current_local_naive", lambda: dt.datetime(2026, 3, 19, 10, 0, 0))

    payload = module.build_best_bets_and_parlays(
        module.BestBetsParlayConfig(
            out_dir=Path("outputs"),
            date="2026-03-19",
            best_bets=4,
            candidate_pool=4,
            parlay_size=2,
            max_parlays=2,
            future_only=False,
        )
    )

    assert payload["status"] == "ok"
    assert payload["best_bet"]["game_id"] == "1001"
    assert payload["best_bet"]["rec_code"] == "OU"
    assert [row["game_id"] for row in payload["best_bets"]] == ["1001", "1003", "1002", "1004"]
    assert len(payload["parlays"]) == 2
    for parlay in payload["parlays"]:
        game_ids = [leg["game_id"] for leg in parlay["legs"]]
        assert len(game_ids) == len(set(game_ids))


def test_api_best_bets_parlays_defaults_to_request_local_day(monkeypatch):
    app_module = importlib.import_module("app")
    parlays_module = importlib.import_module("src.eval.parlays")
    date_today = "2026-03-19"

    monkeypatch.setattr(app_module, "_today_request_local_str", lambda: date_today)
    monkeypatch.setattr(app_module, "_today_local_str", lambda: date_today)
    monkeypatch.setattr(
        parlays_module,
        "build_best_bets_and_parlays",
        lambda cfg: {
            "status": "ok",
            "date": cfg.date,
            "best_bet": None,
            "best_bets": [],
            "parlays": [],
            "generated_utc": "2026-03-19T15:00:00Z",
            "as_of_local": "2026-03-19T10:00:00",
            "source_summary": [],
        },
    )

    app_module.app.testing = True
    with app_module.app.test_client() as client:
        resp = client.get("/api/best-bets-parlays")

    assert resp.status_code == 200
    payload = resp.get_json() or {}
    assert payload.get("date") == date_today


def test_best_bets_parlays_page_renders(monkeypatch):
    app_module = importlib.import_module("app")
    parlays_module = importlib.import_module("src.eval.parlays")

    monkeypatch.setattr(
        parlays_module,
        "build_best_bets_and_parlays",
        lambda cfg: {
            "status": "ok",
            "date": cfg.date,
            "generated_utc": "2026-03-19T15:00:00Z",
            "as_of_local": "2026-03-19T10:00:00",
            "message": None,
            "source_summary": [{"market": "ATS", "status": "ok", "count": 1}],
            "best_bet": {
                "rec_code": "ATS",
                "display_pick": "Sample State -4.5 - Visitor @ Sample State",
                "matchup": "Visitor @ Sample State",
                "price": -110.0,
                "p_win": 0.71,
                "score": 88.5,
                "line": -4.5,
                "reasons": ["p_win=0.710"],
            },
            "best_bets": [
                {
                    "rec_code": "ATS",
                    "display_pick": "Sample State -4.5 - Visitor @ Sample State",
                    "matchup": "Visitor @ Sample State",
                    "price": -110.0,
                    "p_win": 0.71,
                    "score": 88.5,
                    "start_time_local": "2026-03-19 18:00",
                    "reasons": ["p_win=0.710"],
                }
            ],
            "parlays": [
                {
                    "combined_p_win": 0.51,
                    "approx_american_odds": 950,
                    "expected_units_per_1_risk": 4.2,
                    "min_leg_score": 84.0,
                    "legs": [
                        {
                            "rec_code": "ATS",
                            "display_pick": "Sample State -4.5 - Visitor @ Sample State",
                            "matchup": "Visitor @ Sample State",
                            "price": -110.0,
                            "p_win": 0.71,
                            "score": 88.5,
                            "start_time_local": "2026-03-19 18:00",
                        }
                    ],
                }
            ],
        },
    )

    app_module.app.testing = True
    with app_module.app.test_client() as client:
        resp = client.get("/best-bets-parlays?date=2026-03-19")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Best Bets & Parlays" in html
    assert "Sample State" in html