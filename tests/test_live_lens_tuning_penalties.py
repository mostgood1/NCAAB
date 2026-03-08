import importlib
import json
from pathlib import Path

import pytest


app_module = importlib.import_module("app")
app = getattr(app_module, "app")
app.testing = True


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


def test_live_lens_tuning_passes_penalty_map(client):
    out_dir = getattr(app_module, "OUT", Path("outputs"))
    p = Path(out_dir) / "live_lens_tuning.json"

    had_existing = p.exists()
    backup_text = None
    if had_existing:
        try:
            backup_text = p.read_text(encoding="utf-8")
        except Exception:
            backup_text = p.read_text(encoding="utf-8", errors="ignore")

    try:
        payload = {
            "meta": {"generated_at": "test"},
            "tuning": {
                "pace_hi": 9.99,
                "driver_tag_strength_penalties": {
                    "pace_hi": 0.5,
                    "eff_mid": 1.25,
                    "BAD": "not-a-number",
                },
                "driver_tagset_strength_penalties": {
                    "kind_total|lens_fg|eff_mid|pace_hi": 2.0,
                },
                "unknown_object_should_be_ignored": {"a": 1.0},
            },
        }

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")

        resp = client.get("/api/live_lens_tuning?ttl=0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert data.get("status") == "ok"

        tune = data.get("tuning")
        assert isinstance(tune, dict)
        assert tune.get("pace_hi") == pytest.approx(9.99)

        pen = tune.get("driver_tag_strength_penalties")
        assert isinstance(pen, dict)
        assert pen.get("pace_hi") == pytest.approx(0.5)
        assert pen.get("eff_mid") == pytest.approx(1.25)
        assert "BAD" not in pen

        pen2 = tune.get("driver_tagset_strength_penalties")
        assert isinstance(pen2, dict)
        assert pen2.get("kind_total|lens_fg|eff_mid|pace_hi") == pytest.approx(2.0)

        assert "unknown_object_should_be_ignored" not in tune
    finally:
        if had_existing:
            assert backup_text is not None
            p.write_text(backup_text, encoding="utf-8")
        else:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
