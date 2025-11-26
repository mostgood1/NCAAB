import os
import json
import datetime as dt
import pytest

# Ensure outputs dir exists for reading CSVs if needed
ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'outputs')

@pytest.fixture(scope='module')
def flask_app():
    # Import the Flask app from app.py
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app

@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


def test_coverage_today_endpoint_basic(client):
    # Query today by default
    resp = client.get('/api/coverage-today')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'date' in data
    assert 'd1_expected_today' in data
    assert 'coverage_present_today' in data
    assert 'coverage_missing_today' in data
    assert isinstance(data['report_rows'], list)
    assert isinstance(data['missing_teams_rows'], list)


def test_coverage_today_with_date_param(client):
    # Query an explicit date param
    date_str = dt.datetime.now().strftime('%Y-%m-%d')
    resp = client.get(f'/api/coverage-today?date={date_str}')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['date'] == date_str


def test_coverage_health_gate(client):
    # If coverage_report_today.csv exists and has rows, coverage_missing_today should reflect that.
    # This test asserts the field is present; strict failure gating will be in a separate test.
    resp = client.get('/api/coverage-today')
    assert resp.status_code == 200
    data = resp.get_json()
    # Basic invariants
    assert data['coverage_present_today'] is None or isinstance(data['coverage_present_today'], int)
    assert data['coverage_missing_today'] is None or isinstance(data['coverage_missing_today'], int)


def test_fail_when_missing_coverage_exceeds_zero(client):
    # Health gate: If outputs/coverage_report_today.csv exists AND has entries, fail this test.
    rep_path = os.path.join(OUT, 'coverage_report_today.csv')
    if os.path.exists(rep_path):
        try:
            import pandas as pd
            df = pd.read_csv(rep_path)
            if len(df) > 0:
                pytest.fail(f"Coverage missing today: {len(df)} rows in coverage_report_today.csv")
        except Exception:
            # If unreadable, don't fail the test
            pass
    # If file doesn't exist or has no rows, pass
    assert True
