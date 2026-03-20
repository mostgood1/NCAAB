import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_path(path: Path, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, path)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def test_generate_display_from_edges_collapses_duplicate_games(tmp_path: Path):
	mod = _load_module_from_path(
		ROOT / "scripts" / "generate_display_from_edges.py",
		"test_generate_display_from_edges_mod",
	)
	date_str = "2026-03-07"

	pd.DataFrame(
		[
			{
				"game_id": "1001",
				"date": date_str,
				"home_team": "Home A",
				"away_team": "Away A",
				"pred_total": 151.5,
				"pred_margin": 4.0,
				"display_date": date_str,
				"start_time": "2026-03-07T18:00:00Z",
			},
			{
				"game_id": "1002",
				"date": date_str,
				"home_team": "Home B",
				"away_team": "Away B",
				"pred_total": 144.0,
				"pred_margin": -2.5,
				"display_date": date_str,
				"start_time": "2026-03-07T20:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "1001",
				"date": date_str,
				"period": "full_game",
				"home_team": "Home A",
				"away_team": "Away A",
				"total": 150.0,
				"pred_total": 151.0,
				"pred_margin": 4.0,
				"start_time": "2026-03-07T18:00:00Z",
			},
			{
				"game_id": "1001",
				"date": date_str,
				"period": "full_game",
				"home_team": "Home A",
				"away_team": "Away A",
				"total": 151.0,
				"pred_total": 152.0,
				"pred_margin": 4.5,
				"start_time": "2026-03-07T18:00:00Z",
			},
			{
				"game_id": "1002",
				"date": date_str,
				"period": "full_game",
				"home_team": "Home B",
				"away_team": "Away B",
				"total": 144.5,
				"pred_total": 144.0,
				"pred_margin": -2.5,
				"start_time": "2026-03-07T20:00:00Z",
			},
		]
	).to_csv(tmp_path / f"align_period_{date_str}_edges.csv", index=False)

	out_df = mod.build_display_frame(tmp_path, date_str)
	assert len(out_df) == 2
	assert out_df["game_id"].nunique() == 2

	row = out_df.set_index("game_id").loc["1001"]
	assert row["home_team"] == "Home A"
	assert row["away_team"] == "Away A"
	assert float(row["market_total"]) == pytest.approx(150.5)
	assert float(row["pred_total"]) == pytest.approx(151.5)


def test_api_display_predictions_defaults_to_request_local_day(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_today = "2026-03-06"
	date_next = "2026-03-07"

	pd.DataFrame(
		[
			{
				"game_id": "2001",
				"date": date_today,
				"display_date": date_today,
				"home_team": "Today Home",
				"away_team": "Today Away",
				"pred_total": 140.5,
				"pred_margin": 1.5,
				"market_total": 139.5,
				"start_time": "2026-03-06T19:00:00Z",
			}
		]
	).to_csv(tmp_path / f"predictions_display_{date_today}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "2002",
				"date": date_next,
				"display_date": date_next,
				"home_team": "Next Home",
				"away_team": "Next Away",
				"pred_total": 155.0,
				"pred_margin": 3.0,
				"market_total": 154.0,
				"start_time": "2026-03-07T19:00:00Z",
			}
		]
	).to_csv(tmp_path / f"predictions_display_{date_next}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	monkeypatch.setattr(app_module, "_today_request_local_str", lambda: date_today)
	monkeypatch.setattr(app_module, "_today_local_str", lambda: date_today)
	monkeypatch.setattr(
		app_module,
		"_persist_display",
		lambda df, date_str: (tmp_path / f"predictions_display_{date_str}.csv", "digest"),
	)

	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get("/api/display_predictions")

	assert resp.status_code == 200
	payload = resp.get_json() or {}
	assert payload.get("date") == date_today
	rows = payload.get("rows") or []
	assert len(rows) == 1
	assert rows[0]["game_id"] == "2001"
	assert float(rows[0]["pred_total"]) == pytest.approx(140.5)


def test_api_high_likelihood_defaults_to_request_local_day(monkeypatch):
	app_module = importlib.import_module("app")
	hl_module = importlib.import_module("src.eval.high_likelihood")
	date_today = "2026-03-06"

	monkeypatch.setattr(app_module, "_today_request_local_str", lambda: date_today)
	monkeypatch.setattr(app_module, "_today_local_str", lambda: date_today)
	monkeypatch.setattr(
		hl_module,
		"build_high_likelihood",
		lambda cfg: {"status": "ok", "date": cfg.date, "picks": []},
	)
	monkeypatch.setattr(hl_module, "reconcile_picks", lambda *args, **kwargs: {"status": "ok", "wins": 0, "losses": 0, "pushes": 0, "units": 0.0, "rows": 0})
	monkeypatch.setattr(hl_module, "recent_results_dates", lambda *args, **kwargs: [])

	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get("/api/high_likelihood")

	assert resp.status_code == 200
	payload = resp.get_json() or {}
	assert payload.get("date") == date_today


def test_api_recommendations_dedupes_per_book_rows(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-07"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "3001",
				"home_team": "Home A",
				"away_team": "Away A",
				"market": "totals",
				"period": "full_game",
				"bet": "over",
				"line": 150.5,
				"price": -110,
				"edge": 3.0,
				"book": "book_a",
				"rec_type": "Total",
				"rec_code": "OU",
				"pred_total": 153.5,
			},
			{
				"date": date_str,
				"game_id": "3001",
				"home_team": "Home A",
				"away_team": "Away A",
				"market": "totals",
				"period": "full_game",
				"bet": "over",
				"line": 150.5,
				"price": -108,
				"edge": 4.5,
				"book": "book_b",
				"rec_type": "Total",
				"rec_code": "OU",
				"pred_total": 155.0,
			},
			{
				"date": date_str,
				"game_id": "3002",
				"home_team": "Home B",
				"away_team": "Away B",
				"market": "moneyline",
				"period": "full_game",
				"bet": "home",
				"line": None,
				"price": -120,
				"edge": 2.0,
				"book": "book_c",
				"rec_type": "Moneyline",
				"rec_code": "ML",
				"pred_margin": 4.0,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "3001",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Home A",
				"away_team": "Away A",
				"pred_total": 155.0,
				"pred_margin": 3.0,
				"market_total": 150.5,
				"start_time": "2026-03-07T19:00:00Z",
			},
			{
				"game_id": "3002",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Home B",
				"away_team": "Away B",
				"pred_total": 142.0,
				"pred_margin": 4.0,
				"market_total": 141.0,
				"start_time": "2026-03-07T21:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get(f"/api/recommendations?date={date_str}")

	assert resp.status_code == 200
	payload = resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []
	ou_rows = [r for r in rows if str(r.get("game_id")) == "3001" and str(r.get("code") or r.get("rec_code") or "").upper() == "OU"]
	assert len(ou_rows) == 1
	assert float(ou_rows[0].get("edge")) == pytest.approx(4.5)


def test_api_recommendations_adds_basketball_matchup_logic(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-19"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4101",
				"home_team": "Matchup Home",
				"away_team": "Matchup Away",
				"market": "spreads",
				"period": "full_game",
				"bet": "Matchup Home",
				"selection": "Matchup Home",
				"line": -4.5,
				"price": -110,
				"edge": 2.5,
				"book": "book_a",
				"rec_type": "Spread",
				"rec_code": "ATS",
				"pred_margin": 7.0,
			},
			{
				"date": date_str,
				"game_id": "4102",
				"home_team": "Tempo Home",
				"away_team": "Tempo Away",
				"market": "totals",
				"period": "full_game",
				"bet": "Over",
				"selection": "Over",
				"line": 149.5,
				"price": -110,
				"edge": 3.0,
				"book": "book_b",
				"rec_type": "Totals",
				"rec_code": "OU",
				"pred_total": 155.0,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4101",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Matchup Home",
				"away_team": "Matchup Away",
				"pred_total": 141.0,
				"pred_margin": 7.0,
				"market_total": 140.5,
				"start_time": "2026-03-19T19:00:00Z",
			},
			{
				"game_id": "4102",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Tempo Home",
				"away_team": "Tempo Away",
				"pred_total": 155.0,
				"pred_margin": 2.0,
				"market_total": 149.5,
				"start_time": "2026-03-19T21:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4101",
				"home_team": "Matchup Home",
				"away_team": "Matchup Away",
				"home_off_rating": 112.0,
				"away_off_rating": 103.0,
				"home_def_rating": 96.0,
				"away_def_rating": 101.0,
				"pace_game_est": 68.0,
				"home_ppp_mu": 1.12,
				"away_ppp_mu": 1.00,
				"home_ppp_allowed_mu": 0.94,
				"away_ppp_allowed_mu": 1.05,
				"rest_home": 4.0,
				"rest_away": 1.0,
			},
			{
				"game_id": "4102",
				"home_team": "Tempo Home",
				"away_team": "Tempo Away",
				"home_off_rating": 111.0,
				"away_off_rating": 110.0,
				"home_def_rating": 102.0,
				"away_def_rating": 101.0,
				"pace_game_est": 74.5,
				"home_ppp_mu": 1.11,
				"away_ppp_mu": 1.10,
				"home_ppp_allowed_mu": 1.02,
				"away_ppp_allowed_mu": 1.03,
				"rest_home": 2.0,
				"rest_away": 2.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get(f"/api/recommendations?date={date_str}")

	assert resp.status_code == 200
	payload = resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []

	ats_row = next(r for r in rows if str(r.get("game_id")) == "4101" and str(r.get("code") or r.get("rec_code") or "").upper() == "ATS")
	ou_row = next(r for r in rows if str(r.get("game_id")) == "4102" and str(r.get("code") or r.get("rec_code") or "").upper() == "OU")

	assert ats_row.get("basketball_source") == f"features_{date_str}.csv"
	assert float(ats_row.get("basketball_matchup_score")) > 0.0
	assert "efficiency matchup" in str(ats_row.get("basketball_summary") or "").lower()
	assert isinstance(ats_row.get("basketball_reasons"), list) and ats_row.get("basketball_reasons")

	assert ou_row.get("basketball_source") == f"features_{date_str}.csv"
	assert float(ou_row.get("basketball_matchup_score")) > 0.0
	assert any(
		phrase in str(ou_row.get("basketball_summary") or "")
		for phrase in ("Tempo projects faster", "Feature-based scoring", "model total sits")
	)
	assert isinstance(ou_row.get("basketball_reasons"), list) and ou_row.get("basketball_reasons")


def test_api_recommendations_per_game_prioritizes_basketball_then_value(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-20"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4201",
				"home_team": "Priority Home",
				"away_team": "Priority Away",
				"market": "spreads",
				"period": "full_game",
				"bet": "Priority Home",
				"selection": "Priority Home",
				"line": -4.5,
				"price": -110,
				"edge": 2.5,
				"confidence": 0.78,
				"book": "book_a",
				"rec_type": "Spread",
				"rec_code": "ATS",
				"pred_margin": 8.0,
			},
			{
				"date": date_str,
				"game_id": "4201",
				"home_team": "Priority Home",
				"away_team": "Priority Away",
				"market": "totals",
				"period": "full_game",
				"bet": "Over",
				"selection": "Over",
				"line": 144.5,
				"price": -110,
				"edge": 7.0,
				"confidence": 0.72,
				"book": "book_b",
				"rec_type": "Totals",
				"rec_code": "OU",
				"pred_total": 145.0,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4201",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Priority Home",
				"away_team": "Priority Away",
				"pred_total": 145.0,
				"pred_margin": 8.0,
				"market_total": 144.5,
				"start_time": "2026-03-20T19:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4201",
				"home_team": "Priority Home",
				"away_team": "Priority Away",
				"home_off_rating": 113.0,
				"away_off_rating": 103.0,
				"home_def_rating": 95.0,
				"away_def_rating": 102.0,
				"pace_game_est": 70.0,
				"home_ppp_mu": 1.12,
				"away_ppp_mu": 1.00,
				"home_ppp_allowed_mu": 0.95,
				"away_ppp_allowed_mu": 1.06,
				"rest_home": 4.0,
				"rest_away": 1.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get(f"/api/recommendations?date={date_str}&per_game=1")

	assert resp.status_code == 200
	payload = resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []
	assert len(rows) == 1
	row = rows[0]

	assert str(row.get("code") or row.get("rec_code") or "").upper() == "ATS"
	assert float(row.get("recommendation_priority_score")) > float(row.get("value_support_score"))
	assert float(row.get("basketball_priority_score")) > float(row.get("value_support_score"))
	assert row.get("basketball_summary")


def test_recommendations_page_renders_basketball_rationale(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-21"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4301",
				"home_team": "Render Home",
				"away_team": "Render Away",
				"market": "spreads",
				"period": "full_game",
				"bet": "Render Home",
				"selection": "Render Home",
				"line": -3.5,
				"price": -110,
				"edge": 2.0,
				"confidence": 0.77,
				"book": "book_a",
				"rec_type": "Spread",
				"rec_code": "ATS",
				"pred_margin": 6.0,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4301",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Render Home",
				"away_team": "Render Away",
				"pred_total": 140.0,
				"pred_margin": 6.0,
				"market_total": 139.5,
				"start_time": "2026-03-21T19:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4301",
				"home_team": "Render Home",
				"away_team": "Render Away",
				"home_off_rating": 112.0,
				"away_off_rating": 104.0,
				"home_def_rating": 95.0,
				"away_def_rating": 101.0,
				"pace_game_est": 69.0,
				"home_ppp_mu": 1.11,
				"away_ppp_mu": 1.01,
				"home_ppp_allowed_mu": 0.95,
				"away_ppp_allowed_mu": 1.05,
				"rest_home": 3.0,
				"rest_away": 1.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True
	with app_module.app.test_client() as client:
		resp = client.get(f"/recommendations?date={date_str}")

	assert resp.status_code == 200
	html = resp.get_data(as_text=True)
	assert "efficiency matchup" in html.lower()
	assert "Overall" in html
	assert "Render Home" in html


def test_api_recommendations_normalizes_incomplete_rows_and_grouped_hides_null_angles(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-22"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4401",
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"market": "spreads",
				"period": "full_game",
				"bet": "Normalize Home",
				"selection": "Normalize Home",
				"line": -4.5,
				"price": -110,
				"edge": 2.2,
				"confidence": 0.74,
				"book": "book_a",
				"rec_type": "Spread",
				"rec_code": "ATS",
				"pred_margin": 7.0,
			},
			{
				"date": date_str,
				"game_id": "4401",
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"market": "moneyline",
				"period": "full_game",
				"pick": "Normalize Home ML",
				"bet": float("nan"),
				"selection": float("nan"),
				"line": None,
				"line_value": -145.0,
				"price": None,
				"edge": 5.5,
				"confidence": 0.81,
				"book": "book_b",
				"rec_type": float("nan"),
				"rec_code": float("nan"),
				"pred_margin": 7.0,
			},
			{
				"date": date_str,
				"game_id": "4401",
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"market": "totals",
				"period": "full_game",
				"pick": "Under",
				"bet": float("nan"),
				"selection": float("nan"),
				"line": None,
				"line_value": 141.5,
				"price": None,
				"edge": 6.0,
				"confidence": 0.79,
				"book": "book_c",
				"rec_type": float("nan"),
				"rec_code": float("nan"),
				"pred_total": 136.0,
			},
			{
				"date": date_str,
				"game_id": "4401",
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"market": "spreads",
				"period": "2h",
				"pick": "Normalize Away +0.5",
				"bet": float("nan"),
				"selection": float("nan"),
				"line": None,
				"line_value": 0.5,
				"price": None,
				"edge": 9.0,
				"confidence": 0.86,
				"book": "book_d",
				"rec_type": float("nan"),
				"rec_code": float("nan"),
				"pred_margin": -1.0,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4401",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"pred_total": 136.0,
				"pred_margin": 7.0,
				"market_total": 141.5,
				"start_time": "2026-03-22T19:00:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4401",
				"home_team": "Normalize Home",
				"away_team": "Normalize Away",
				"home_off_rating": 114.0,
				"away_off_rating": 103.0,
				"home_def_rating": 95.0,
				"away_def_rating": 101.0,
				"pace_game_est": 68.5,
				"home_ppp_mu": 1.13,
				"away_ppp_mu": 1.01,
				"home_ppp_allowed_mu": 0.95,
				"away_ppp_allowed_mu": 1.05,
				"rest_home": 3.0,
				"rest_away": 1.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True
	with app_module.app.test_client() as client:
		api_resp = client.get(f"/api/recommendations?date={date_str}")

	assert api_resp.status_code == 200
	payload = api_resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []

	ats_rows = [r for r in rows if str(r.get("code") or r.get("rec_code") or "").upper() == "ATS"]
	ou_rows = [r for r in rows if str(r.get("code") or r.get("rec_code") or "").upper() == "OU"]
	ml_rows = [r for r in rows if str(r.get("code") or r.get("rec_code") or "").upper() == "ML"]

	assert len(ats_rows) == 2
	assert any(str(r.get("period") or "").lower() == "full_game" for r in ats_rows)
	assert any(str(r.get("period") or "").lower() == "2h" for r in ats_rows)
	assert len(ou_rows) == 1
	assert len(ml_rows) == 1

	ml_row = ml_rows[0]
	assert ml_row.get("bet_label") == "Normalize Home ML"
	assert ml_row.get("selection") == "Normalize Home"
	assert float(ml_row.get("price")) == pytest.approx(-145.0)
	assert ml_row.get("basketball_summary")

	ou_row = ou_rows[0]
	assert ou_row.get("bet_label") == "Under 141.5"
	assert ou_row.get("selection") == "Under"
	assert ou_row.get("basketball_summary")

	assert not any(str(r.get("bet_label") or "").lower() == "nan" for r in rows)
	assert not any(str(r.get("code") or r.get("rec_code") or "").strip().lower() in ("", "nan", "none", "null") for r in rows)

	with app_module.app.test_client() as client:
		page_resp = client.get(f"/recommendations?date={date_str}")

	assert page_resp.status_code == 200
	html = page_resp.get_data(as_text=True)
	assert "Normalize Home ML" in html
	assert "Under 141.5" in html
	assert "Normalize Home -4.5" in html
	assert "Normalize Away +0.5" not in html
	assert ">None<" not in html
	assert ">nan<" not in html.lower()


def test_api_recommendations_repairs_synthetic_ats_lines_and_skips_bad_full_game_total_sentence(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-23"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4501",
				"home_team": "Michigan Wolverines",
				"away_team": "Howard Bison",
				"market": "totals",
				"period": "full_game",
				"bet": "Under",
				"selection": "Under",
				"line": 151.5,
				"price": -110,
				"edge": 74.3,
				"confidence": 0.79,
				"book": "book_totals",
				"rec_type": "Totals",
				"rec_code": "OU",
				"pred_total": 77.22501716613769,
				"pred_margin": 24.206571197509767,
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4501",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Michigan Wolverines",
				"away_team": "Howard Bison",
				"pred_total": 77.22501716613769,
				"pred_margin": 24.206571197509767,
				"market_total": 151.5,
				"start_time": "2026-03-23T23:10:00Z",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4501",
				"home_team": "Michigan Wolverines",
				"away_team": "Howard Bison",
				"home_off_rating": 102.0,
				"away_off_rating": 107.0,
				"home_def_rating": 94.0,
				"away_def_rating": 93.0,
				"pace_game_est": 68.0,
				"home_ppp_mu": 1.245830359772067,
				"away_ppp_mu": 1.169375749417494,
				"home_ppp_allowed_mu": 1.0071233323844564,
				"away_ppp_allowed_mu": 0.930657496087942,
				"rest_home": 4.0,
				"rest_away": 2.0,
			},
			{
				"game_id": "4502",
				"home_team": "Tempo Median Home",
				"away_team": "Tempo Median Away",
				"home_off_rating": 111.0,
				"away_off_rating": 109.0,
				"home_def_rating": 101.0,
				"away_def_rating": 100.0,
				"pace_game_est": 74.0,
				"home_ppp_mu": 1.17,
				"away_ppp_mu": 1.15,
				"home_ppp_allowed_mu": 1.03,
				"away_ppp_allowed_mu": 1.02,
				"rest_home": 2.0,
				"rest_away": 2.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	(tmp_path / "picks").mkdir()
	pd.DataFrame(
		[
			{
				"game_id": "4501",
				"home_team": "Michigan Wolverines",
				"away_team": "Howard Bison",
				"ats_side": "home",
				"closing_spread_home": -3.0,
				"spread_home": -3.0,
				"_pred_margin_blend": 24.206571197509767,
			},
		]
	).to_csv(tmp_path / "picks" / f"ats_picks_{date_str}.csv", index=False)

	empty_cwd = tmp_path / "cwd_root"
	empty_cwd.mkdir()
	pd.DataFrame(
		[
			{
				"game_id": "4501",
				"market": "spreads",
				"period": "full_game",
				"book": "BetMGM",
				"home_team": "Michigan Wolverines",
				"away_team": "Howard Bison",
				"home_spread": -30.5,
				"away_spread": 30.5,
				"home_spread_price": -115.0,
				"away_spread_price": -105.0,
				"pair_key": "michiganwolverines::howardbison",
			},
		]
	).to_csv(tmp_path / f"align_period_{date_str}_edges.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	monkeypatch.setattr(app_module.os, "getcwd", lambda: str(empty_cwd))
	app_module.app.testing = True

	with app_module.app.test_client() as client:
		api_resp = client.get(f"/api/recommendations?date={date_str}")

	assert api_resp.status_code == 200
	payload = api_resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []

	ats_row = next(r for r in rows if str(r.get("game_id") or "") == "4501" and str(r.get("code") or r.get("rec_code") or "").upper() == "ATS")
	ou_row = next(r for r in rows if str(r.get("game_id") or "") == "4501" and str(r.get("code") or r.get("rec_code") or "").upper() == "OU")

	assert float(ats_row.get("line")) == pytest.approx(-30.5)
	assert ats_row.get("bet_label") == "Michigan Wolverines -30.5"
	assert float(ats_row.get("price")) == pytest.approx(-115.0)
	assert ats_row.get("book") == "BetMGM"

	summary = str(ou_row.get("basketball_summary") or "")
	assert "Tempo projects slower" in summary
	assert ("Feature-based scoring" in summary) or ("combined scoring environment" in summary)
	assert "74.3 points below the number" not in summary

	with app_module.app.test_client() as client:
		page_resp = client.get(f"/recommendations?date={date_str}")

	assert page_resp.status_code == 200
	html = page_resp.get_data(as_text=True)
	assert "Michigan Wolverines -30.5" in html
	assert "Michigan Wolverines -3.0" not in html


def test_api_recommendations_keeps_moneyline_rows_for_ohio_style_team_names(monkeypatch, tmp_path: Path):
	app_module = importlib.import_module("app")
	date_str = "2026-03-24"

	pd.DataFrame(
		[
			{
				"date": date_str,
				"game_id": "4601",
				"home_team": "Ohio State Buckeyes",
				"away_team": "TCU Horned Frogs",
				"market": "moneyline",
				"period": "full_game",
				"book": "BetUS",
				"pick": "Ohio State Buckeyes ML",
				"edge": 5.076717728589794,
				"line_value": -135.0,
				"predicted_value": 2.990022563934326,
				"fair_price": -153.2876568840106,
				"start_time_iso": "2026-03-24T16:15:00Z",
				"start_time_local": "2026-03-24 12:15",
				"start_tz_abbr": "EDT",
			},
		]
	).to_csv(tmp_path / "picks_raw.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4601",
				"date": date_str,
				"display_date": date_str,
				"home_team": "Ohio State Buckeyes",
				"away_team": "TCU Horned Frogs",
				"pred_total": 146.2,
				"pred_margin": 2.990022563934326,
				"market_total": 146.0,
				"start_time": "2026-03-24T16:15:00Z",
				"start_time_iso": "2026-03-24T16:15:00Z",
				"start_time_local": "2026-03-24 12:15",
				"start_tz_abbr": "EDT",
			},
		]
	).to_csv(tmp_path / f"predictions_display_{date_str}.csv", index=False)

	pd.DataFrame(
		[
			{
				"game_id": "4601",
				"home_team": "Ohio State Buckeyes",
				"away_team": "TCU Horned Frogs",
				"home_off_rating": 114.0,
				"away_off_rating": 109.0,
				"home_def_rating": 99.0,
				"away_def_rating": 101.0,
				"pace_game_est": 69.0,
				"home_ppp_mu": 1.14,
				"away_ppp_mu": 1.08,
				"home_ppp_allowed_mu": 0.99,
				"away_ppp_allowed_mu": 1.03,
				"rest_home": 2.0,
				"rest_away": 1.0,
			},
		]
	).to_csv(tmp_path / f"features_{date_str}.csv", index=False)

	monkeypatch.setattr(app_module, "OUT", tmp_path)
	app_module.app.testing = True

	with app_module.app.test_client() as client:
		api_resp = client.get(f"/api/recommendations?date={date_str}")

	assert api_resp.status_code == 200
	payload = api_resp.get_json() or {}
	rows = payload.get("data") or payload.get("rows") or []

	ml_rows = [r for r in rows if str(r.get("code") or r.get("rec_code") or "").upper() == "ML"]
	assert len(ml_rows) == 1

	ml_row = ml_rows[0]
	assert ml_row.get("bet_label") == "Ohio State Buckeyes ML"
	assert ml_row.get("selection") == "Ohio State Buckeyes"
	assert float(ml_row.get("price")) == pytest.approx(-135.0)
	assert float(ml_row.get("pred_margin")) == pytest.approx(2.990022563934326)

	bad_moneyline_rows = [
		r for r in rows
		if str(r.get("market") or "").lower() == "moneyline"
		and str(r.get("code") or r.get("rec_code") or "").upper() != "ML"
	]
	assert not bad_moneyline_rows
	assert not any(str(r.get("bet_label") or "") == "Over -135.0" for r in rows)

	with app_module.app.test_client() as client:
		page_resp = client.get(f"/recommendations?date={date_str}")

	assert page_resp.status_code == 200
	html = page_resp.get_data(as_text=True)
	assert "Ohio State Buckeyes ML" in html
	assert "Over -135.0" not in html
