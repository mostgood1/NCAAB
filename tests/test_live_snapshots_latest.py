import json

from ncaab_model.live_snapshots import latest_live_lines_by_event_id


def test_latest_live_lines_by_event_id_filters_period_and_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("NCAAB_LIVE_SNAPSHOT_DIR", str(tmp_path))

    p = tmp_path / "live_2026-03-04.jsonl"
    rows = [
        {
            "endpoint": "live_state",
            "event_id": "123",
            "ts": "2026-03-04T09:00:00Z",
            "data": {"foo": 1},
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T10:00:00Z",
            "data": {
                "period": "full_game",
                "book": "draftkings",
                "total": 150.5,
                "spread_home": -3.5,
                "moneyline_home": -150,
                "moneyline_away": 130,
                "over_price": -110,
                "under_price": -110,
                "spread_home_price": -110,
                "spread_away_price": -110,
            },
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T12:00:00Z",
            "data": {"period": "first_half", "total": 160.0},
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T11:00:00Z",
            "data": {
                "period": "full_game",
                "book": "draftkings",
                "total": 151.0,
                "spread_home": -4.0,
                "moneyline_home": -155,
                "moneyline_away": 135,
            },
        },
        {
            "endpoint": "live_lines",
            "event_id": "999",
            "ts": None,
            "data": {"period": "full_game", "total": 140.0},
        },
        {
            "endpoint": "live_lines",
            "event_id": "999",
            "ts": None,
            "data": {"period": "full_game", "total": 141.0},
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = latest_live_lines_by_event_id(date_s="2026-03-04", period="full_game")

    assert set(out.keys()) == {"123", "999"}

    # Period filter: ignore first_half even though it is later.
    assert out["123"]["event_id"] == "123"
    assert out["123"]["ts"] == "2026-03-04T11:00:00Z"
    assert out["123"]["total"] == 151.0

    # Timestamp missing: fall back to file order.
    assert out["999"]["total"] == 141.0


def test_latest_live_lines_by_event_id_supports_cutoff_and_min(tmp_path, monkeypatch):
    monkeypatch.setenv("NCAAB_LIVE_SNAPSHOT_DIR", str(tmp_path))

    p = tmp_path / "live_2026-03-04.jsonl"
    rows = [
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T09:00:00Z",
            "data": {"period": "full_game", "total": 150.0},
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T10:00:00Z",
            "data": {"period": "full_game", "total": 151.0},
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T11:00:00Z",
            "data": {"period": "full_game", "total": 152.0},
        },
        {
            "endpoint": "live_lines",
            "event_id": "123",
            "ts": "2026-03-04T12:00:00Z",
            "data": {"period": "full_game", "total": 153.0},
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Cutoff: pick the newest quote at or before cutoff.
    out = latest_live_lines_by_event_id(
        date_s="2026-03-04",
        period="full_game",
        cutoff_ts_by_event_id={"123": "2026-03-04T10:30:00Z"},
    )
    assert out["123"]["ts"] == "2026-03-04T10:00:00Z"
    assert out["123"]["total"] == 151.0

    # Min: pick the newest quote strictly after min.
    out2 = latest_live_lines_by_event_id(
        date_s="2026-03-04",
        period="full_game",
        min_ts_by_event_id={"123": "2026-03-04T10:00:00Z"},
    )
    assert out2["123"]["ts"] == "2026-03-04T12:00:00Z"
    assert out2["123"]["total"] == 153.0

    # Combination window.
    out3 = latest_live_lines_by_event_id(
        date_s="2026-03-04",
        period="full_game",
        min_ts_by_event_id={"123": "2026-03-04T09:30:00Z"},
        cutoff_ts_by_event_id={"123": "2026-03-04T11:00:00Z"},
    )
    assert out3["123"]["ts"] == "2026-03-04T11:00:00Z"
    assert out3["123"]["total"] == 152.0
