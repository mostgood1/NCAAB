import json
import importlib
from pathlib import Path

import pandas as pd
import pytest


app_module = importlib.import_module("app")
app = getattr(app_module, "app")
app.testing = True


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_live_lens_side_accuracy_includes_tag_breakdowns(client):
    out_dir = Path(getattr(app_module, "OUT"))
    date = "1900-01-03"

    sig_p = out_dir / f"live_lens_signals_{date}.jsonl"
    res_p = out_dir / "daily_results" / f"results_{date}.csv"

    signals = [
        {
            "game_id": "g1",
            "kind": "total",
            "lens": "fg",
            "side": "over",
            "live_line": 148.0,
            "is_bet": True,
            "edge": 0.03,
            "elapsed": 18,
            "remaining": 22,
            "driver": "pace_hi",
            "driver_tags": ["pace_hi"],
            "ts": "1900-01-03T01:00:00Z",
        },
        {
            "game_id": "g2",
            "kind": "total",
            "lens": "fg",
            "side": "under",
            "live_line": 125.0,
            "is_bet": True,
            "edge": 0.04,
            "elapsed": 26,
            "remaining": 14,
            "driver": "eff_lo",
            "driver_tags": ["eff_lo"],
            "ts": "1900-01-03T02:00:00Z",
        },
    ]

    results = pd.DataFrame(
        [
            {"game_id": "g1", "completed": True, "actual_total": 150, "actual_margin": 5},
            {"game_id": "g2", "completed": True, "actual_total": 120, "actual_margin": -4},
        ]
    )

    _write_jsonl(sig_p, signals)
    res_p.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(res_p, index=False)

    try:
        resp = client.get(f"/api/live_lens_side_accuracy?date={date}&full_game_only=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["date"] == date

        summary = data.get("summary")
        assert isinstance(summary, dict)

        # Newly-added breakdowns
        assert "by_driver_tag" in summary
        assert "by_driver_tag_full" in summary
        assert "by_driver_tag_canonical" in summary
        assert "by_driver_tag_type" in summary

        assert isinstance(summary["by_driver_tag_full"], list)
        assert isinstance(summary["by_driver_tag_canonical"], list)
        assert isinstance(summary["by_driver_tag_type"], list)

        # Canonical tags should include our inputs
        canonical_tags = {r.get("tag") for r in summary["by_driver_tag_canonical"]}
        assert "pace_hi" in canonical_tags
        assert "eff_lo" in canonical_tags

        # Tag types should include at least pace+eff
        tag_types = {r.get("type") for r in summary["by_driver_tag_type"]}
        assert "pace" in tag_types
        assert "eff" in tag_types
    finally:
        try:
            if sig_p.exists():
                sig_p.unlink()
        except Exception:
            pass
        try:
            if res_p.exists():
                res_p.unlink()
        except Exception:
            pass
