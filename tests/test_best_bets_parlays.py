import datetime as dt
import importlib
from pathlib import Path

import pandas as pd


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
    **extra,
) -> dict:
    row = {
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
    row.update(extra)
    return row


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
    best_bet_game_ids = [row["game_id"] for row in payload["best_bets"]]
    assert best_bet_game_ids[0] == "1001"
    assert set(best_bet_game_ids) == {"1001", "1002", "1003", "1004"}
    assert len(payload["parlays"]) == 2
    for parlay in payload["parlays"]:
        game_ids = [leg["game_id"] for leg in parlay["legs"]]
        assert len(game_ids) == len(set(game_ids))


def test_build_best_bets_parlays_uses_basketball_rationale_and_priority(monkeypatch, tmp_path: Path):
    module = importlib.import_module("src.eval.parlays")
    date_str = "2026-03-19"

    pd.DataFrame(
        [
            {
                "game_id": "1001",
                "home_team": "Home One",
                "away_team": "Road One",
                "home_off_rating": 113.0,
                "away_off_rating": 101.0,
                "home_def_rating": 95.0,
                "away_def_rating": 103.0,
                "pace_game_est": 69.0,
                "home_ppp_mu": 1.13,
                "away_ppp_mu": 0.99,
                "home_ppp_allowed_mu": 0.95,
                "away_ppp_allowed_mu": 1.06,
                "rest_home": 4.0,
                "rest_away": 1.0,
            },
            {
                "game_id": "1002",
                "home_team": "Home Two",
                "away_team": "Road Two",
                "home_off_rating": 108.0,
                "away_off_rating": 107.0,
                "home_def_rating": 101.0,
                "away_def_rating": 100.0,
                "pace_game_est": 70.0,
                "home_ppp_mu": 1.07,
                "away_ppp_mu": 1.06,
                "home_ppp_allowed_mu": 1.02,
                "away_ppp_allowed_mu": 1.01,
                "rest_home": 2.0,
                "rest_away": 2.0,
            },
        ]
    ).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

    picks_by_market = {
        "ATS": [
            _pick(
                "1001",
                "ATS",
                95.0,
                0.91,
                "Road One",
                "Home One",
                "Home",
                "Home One",
                -4.5,
                -110.0,
                pred_margin=8.0,
                edge=3.5,
                reasons=["p_win=0.910", "src=sim_quantiles"],
            )
        ],
        "OU": [
            _pick(
                "1002",
                "OU",
                99.0,
                0.95,
                "Road Two",
                "Home Two",
                "Over",
                None,
                145.5,
                -110.0,
                pred_total=146.0,
                edge=9.0,
                reasons=["p_win=0.950", "src=sim_quantiles"],
            )
        ],
        "ML": [],
    }

    def fake_build_high_likelihood(cfg):
        market = tuple(cfg.include_markets)[0]
        return {"status": "ok", "date": cfg.date, "picks": list(picks_by_market.get(market, []))}

    monkeypatch.setattr(module, "build_high_likelihood", fake_build_high_likelihood)
    monkeypatch.setattr(module, "_current_local_naive", lambda: dt.datetime(2026, 3, 19, 10, 0, 0))

    payload = module.build_best_bets_and_parlays(
        module.BestBetsParlayConfig(
            out_dir=tmp_path,
            date=date_str,
            best_bets=2,
            candidate_pool=2,
            parlay_size=2,
            max_parlays=1,
            future_only=False,
        )
    )

    assert payload["status"] == "ok"
    assert payload["best_bet"]["game_id"] == "1001"
    assert "efficiency matchup" in str(payload["best_bet"].get("basketball_summary") or "").lower()
    assert any("efficiency matchup" in str(reason).lower() for reason in (payload["best_bet"].get("reasons") or []))
    assert float(payload["best_bets"][0].get("basketball_priority_score") or 0.0) > float(payload["best_bets"][1].get("basketball_priority_score") or 0.0)
    assert float(payload["best_bets"][0].get("recommendation_priority_score") or 0.0) > float(payload["best_bets"][1].get("recommendation_priority_score") or 0.0)


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
                "basketball_summary": "Sample State owns the cleaner efficiency matchup and carries the stronger rest profile.",
                "basketball_priority_score": 92.0,
                "sim_support_score": 74.0,
                "value_support_score": 58.0,
                "recommendation_priority_score": 84.0,
                "model_reasons": ["p_win=0.710"],
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
                    "basketball_summary": "Sample State owns the cleaner efficiency matchup and carries the stronger rest profile.",
                    "basketball_priority_score": 92.0,
                    "sim_support_score": 74.0,
                    "value_support_score": 58.0,
                    "recommendation_priority_score": 84.0,
                    "model_reasons": ["p_win=0.710"],
                    "start_time_local": "2026-03-19 18:00",
                    "reasons": ["p_win=0.710"],
                }
            ],
            "parlays": [
                {
                    "combined_p_win": 0.51,
                    "approx_american_odds": 950,
                    "expected_units_per_1_risk": 4.2,
                    "avg_recommendation_priority": 84.0,
                    "min_leg_score": 84.0,
                    "legs": [
                        {
                            "rec_code": "ATS",
                            "display_pick": "Sample State -4.5 - Visitor @ Sample State",
                            "matchup": "Visitor @ Sample State",
                            "price": -110.0,
                            "p_win": 0.71,
                            "score": 88.5,
                            "basketball_summary": "Sample State owns the cleaner efficiency matchup and carries the stronger rest profile.",
                            "basketball_priority_score": 92.0,
                            "recommendation_priority_score": 84.0,
                            "model_reasons": ["p_win=0.710"],
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
    assert "cleaner efficiency matchup" in html.lower()
    assert "Overall 84" in html