import pandas as pd
import pytest

from ncaab_model.data.odds_closing import compute_closing_lines


def test_compute_closing_lines_adds_movement_and_timing_columns():
    commence = "2026-01-01T20:00:00Z"

    rows = []
    for market in ("totals", "spreads", "h2h"):
        base = {
            "event_id": "E1",
            "book": "BookA",
            "market": market,
            "period": "full_game",
            "commence_time": commence,
            "home_team_name": "Home U",
            "away_team_name": "Away U",
        }

        if market == "totals":
            rows.extend(
                [
                    {
                        **base,
                        "fetched_at": "2026-01-01T10:00:00Z",
                        "last_update": "2026-01-01T09:55:00Z",
                        "total": 140.5,
                        "over_price": -110,
                        "under_price": -110,
                    },
                    {
                        **base,
                        "fetched_at": "2026-01-01T19:50:00Z",
                        "last_update": "2026-01-01T19:45:00Z",
                        "total": 142.0,
                        "over_price": -105,
                        "under_price": -115,
                    },
                ]
            )
        elif market == "spreads":
            rows.extend(
                [
                    {
                        **base,
                        "fetched_at": "2026-01-01T10:00:00Z",
                        "last_update": "2026-01-01T09:50:00Z",
                        "home_spread": -2.5,
                        "home_spread_price": -110,
                        "away_spread": 2.5,
                        "away_spread_price": -110,
                    },
                    {
                        **base,
                        "fetched_at": "2026-01-01T19:55:00Z",
                        "last_update": "2026-01-01T19:40:00Z",
                        "home_spread": -4.5,
                        "home_spread_price": -112,
                        "away_spread": 4.5,
                        "away_spread_price": -108,
                    },
                ]
            )
        else:  # h2h
            rows.extend(
                [
                    {
                        **base,
                        "fetched_at": "2026-01-01T10:00:00Z",
                        "last_update": "2026-01-01T09:52:00Z",
                        "moneyline_home": -150,
                        "moneyline_away": 130,
                    },
                    {
                        **base,
                        "fetched_at": "2026-01-01T19:58:00Z",
                        "last_update": "2026-01-01T19:42:00Z",
                        "moneyline_home": -200,
                        "moneyline_away": 170,
                    },
                ]
            )

    df = pd.DataFrame(rows)
    out = compute_closing_lines(df, window_minutes=90)

    assert set(["ts_open", "ts_close", "close_prio"]).issubset(out.columns)
    assert set(["mins_open_to_close", "mins_open_to_tip", "mins_close_to_tip"]).issubset(out.columns)

    # Totals movement + timing
    tot = out[out["market"] == "totals"].iloc[0]
    assert tot["open_total"] == pytest.approx(140.5)
    assert tot["close_total"] == pytest.approx(142.0)
    assert tot["delta_total"] == pytest.approx(1.5)
    assert bool(tot["steam_total_flag"]) is True
    assert int(tot["close_prio"]) == 3
    assert pd.to_datetime(tot["ts_open"], utc=True) == pd.Timestamp("2026-01-01T09:55:00Z")
    assert pd.to_datetime(tot["ts_close"], utc=True) == pd.Timestamp("2026-01-01T19:45:00Z")
    assert float(tot["mins_close_to_tip"]) == pytest.approx(15.0)

    # Spread movement
    spr = out[out["market"] == "spreads"].iloc[0]
    assert spr["open_home_spread"] == pytest.approx(-2.5)
    assert spr["close_home_spread"] == pytest.approx(-4.5)
    assert spr["delta_home_spread"] == pytest.approx(-2.0)
    assert bool(spr["steam_spread_flag"]) is True

    # Moneyline movement
    ml = out[out["market"] == "h2h"].iloc[0]
    assert ml["open_moneyline_home"] == pytest.approx(-150)
    assert ml["close_moneyline_home"] == pytest.approx(-200)
    assert ml["delta_moneyline_home"] == pytest.approx(-50)
    assert bool(ml["steam_ml_home_flag"]) is True
