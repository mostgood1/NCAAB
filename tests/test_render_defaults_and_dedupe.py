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
