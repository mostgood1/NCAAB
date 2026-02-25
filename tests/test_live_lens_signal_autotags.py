import json
import importlib
from pathlib import Path

import pytest


app_module = importlib.import_module("app")
app = getattr(app_module, "app")
app.testing = True


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


def _read_last_jsonl(path: Path) -> dict:
    data = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    assert data, "expected at least one jsonl line"
    return json.loads(data[-1])


def test_live_lens_signal_autotags_backfill_driver_tags(client):
    out_dir = Path(getattr(app_module, "OUT"))
    date = "1900-01-03"
    sig_p = out_dir / f"live_lens_signals_{date}.jsonl"

    payload = {
        "date": date,
        "game_id": "g1",
        "kind": "total",
        "lens": "fg",
        "side": "over",
        "live_line": 140.5,
        "is_bet": True,
        "elapsed": 10,
        "total_points": 40,
        "pbp": {"poss": 35},
        "tuning": {"pace_hi": 3.25, "pace_lo": 2.75, "pps_hi": 1.18, "pps_lo": 0.95},
        "driver": None,
        "driver_tags": None,
        "ts": "1900-01-03T01:00:00Z",
    }

    try:
        resp = client.post("/api/live_lens_signal", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("status") == "ok"

        assert sig_p.exists()
        last = _read_last_jsonl(sig_p)
        assert last.get("game_id") == "g1"
        assert last.get("driver") == "pace_hi"
        assert last.get("driver_tags") == ["pace_hi"]
    finally:
        try:
            if sig_p.exists():
                sig_p.unlink()
        except Exception:
            pass


def test_live_lens_signal_autotags_respect_client_tags(client):
    out_dir = Path(getattr(app_module, "OUT"))
    date = "1900-01-04"
    sig_p = out_dir / f"live_lens_signals_{date}.jsonl"

    payload = {
        "date": date,
        "game_id": "g2",
        "kind": "total",
        "lens": "fg",
        "side": "over",
        "live_line": 140.5,
        "is_bet": True,
        "elapsed": 10,
        "total_points": 10,
        "pbp": {"poss": 10},
        "tuning": {"pace_hi": 3.25, "pace_lo": 2.75, "pps_hi": 1.18, "pps_lo": 0.95},
        "driver": "client_driver",
        "driver_tags": ["client_tag"],
        "ts": "1900-01-04T01:00:00Z",
    }

    try:
        resp = client.post("/api/live_lens_signal", json=payload)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get("status") == "ok"

        last = _read_last_jsonl(sig_p)
        assert last.get("driver") == "client_driver"
        assert last.get("driver_tags") == ["client_tag"]
    finally:
        try:
            if sig_p.exists():
                sig_p.unlink()
        except Exception:
            pass
