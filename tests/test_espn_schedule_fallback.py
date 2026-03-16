import datetime as dt
import json

from ncaab_model.data.adapters import espn_scoreboard


def _schedule_html(date_key: str = "20260319") -> str:
    bootstrap = {
        "page": {
            "content": {
                "events": {
                    date_key: [
                        {
                            "id": "401856479",
                            "date": "2026-03-19T16:15Z",
                            "note": "NCAA Men's Basketball Championship - Midwest Region - 1st Round",
                            "completed": False,
                            "neutralSite": True,
                            "season": {"year": 2026, "type": 3, "slug": "post-season"},
                            "status": {"id": "1", "state": "pre", "detail": "Thu, March 19th at 12:15 PM EDT"},
                            "venue": {
                                "id": "999",
                                "fullName": "Bon Secours Wellness Arena",
                                "address": {"city": "Greenville", "state": "SC"},
                            },
                            "competitors": [
                                {
                                    "id": "194",
                                    "displayName": "Ohio State Buckeyes",
                                    "shortName": "Ohio State",
                                    "abbrev": "OSU",
                                    "isHome": True,
                                    "score": 0,
                                },
                                {
                                    "id": "2628",
                                    "displayName": "TCU Horned Frogs",
                                    "shortName": "TCU",
                                    "abbrev": "TCU",
                                    "isHome": False,
                                    "score": 0,
                                },
                            ],
                        }
                    ]
                }
            }
        }
    }
    return f"<html><body><script>window['__espnfitt__']={json.dumps(bootstrap)};</script></body></html>"


class _DummyResponse:
    def __init__(self, *, status_code: int = 200, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON payload configured")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_extract_schedule_payload_from_bootstrap_html():
    payload = espn_scoreboard._extract_schedule_payload(_schedule_html())
    assert payload is not None
    assert "20260319" in payload["events"]
    assert payload["events"]["20260319"][0]["id"] == "401856479"


def test_iter_games_by_date_falls_back_to_schedule_html(monkeypatch, tmp_path):
    target_date = dt.date(2026, 3, 19)
    scoreboard_payload = {"events": []}
    schedule_html = _schedule_html(target_date.strftime("%Y%m%d"))

    def fake_cache_path(*parts: str):
        return tmp_path.joinpath(*parts)

    def fake_get(url, *args, **kwargs):
        if "/schedule/_/date/" in url:
            return _DummyResponse(text=schedule_html)
        return _DummyResponse(json_data=scoreboard_payload)

    monkeypatch.setattr(espn_scoreboard, "cache_path", fake_cache_path)
    monkeypatch.setattr(espn_scoreboard.requests, "get", fake_get)

    results = list(espn_scoreboard.iter_games_by_date(target_date, target_date, use_cache=False, cache_only=False))
    assert len(results) == 1
    assert len(results[0].games) == 1

    game = results[0].games[0]
    assert game.game_id == "401856479"
    assert game.home_team == "Ohio State Buckeyes"
    assert game.away_team == "TCU Horned Frogs"
    assert game.neutral_site is True
    assert game.completed is False
    assert game.start_time is not None
    assert "Greenville, SC" in (game.venue or "")
    assert game.tournament_label == "NCAA Tournament"
    assert game.tournament_note == "NCAA Men's Basketball Championship - Midwest Region - 1st Round"


def test_parse_games_extracts_tournament_fields_from_scoreboard_notes():
    payload = {
        "events": [
            {
                "id": "401856480",
                "notes": [{"headline": "NCAA Men's Basketball Championship - West Region - 1st Round"}],
                "season": {"year": 2026, "type": 3, "slug": "post-season"},
                "competitions": [
                    {
                        "date": "2026-03-20T19:25Z",
                        "neutralSite": True,
                        "type": {"abbreviation": "TRNMNT"},
                        "venue": {"fullName": "Rupp Arena"},
                        "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"displayName": "Gonzaga Bulldogs"},
                                "score": "0",
                            },
                            {
                                "homeAway": "away",
                                "team": {"displayName": "Drake Bulldogs"},
                                "score": "0",
                            },
                        ],
                    }
                ],
            }
        ]
    }

    games = espn_scoreboard._parse_games(dt.date(2026, 3, 20), payload)
    assert len(games) == 1
    game = games[0]
    assert game.tournament_label == "NCAA Tournament"
    assert game.tournament_note == "NCAA Men's Basketball Championship - West Region - 1st Round"