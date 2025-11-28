import json
import types
import pytest

# Import the Flask app
import importlib
app_module = importlib.import_module('app')
app = getattr(app_module, 'app')
app.testing = True

@pytest.fixture(scope='module')
def client():
    with app.test_client() as c:
        yield c


def test_results_dates(client):
    resp = client.get('/api/results_dates')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert 'dates' in data
    assert isinstance(data['dates'], list)
    assert 'latest' in data


def test_results_by_date_missing_param(client):
    resp = client.get('/api/results_by_date')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data.get('error') == 'missing date param'


def test_results_by_date_not_found(client):
    resp = client.get('/api/results_by_date?date=1900-01-01')
    # Either 404 not found, or 200 with empty rows if file exists unexpectedly
    assert resp.status_code in (200, 404)
    data = resp.get_json()
    if resp.status_code == 404:
        assert data.get('error') == 'results file not found'
    else:
        assert 'rows' in data
        assert isinstance(data['rows'], list)


def test_finalize_hint_shape(client):
    resp = client.get('/api/finalize_hint')
    assert resp.status_code == 200
    data = resp.get_json()
    for key in ['date', 'rows', 'final', 'pending', 'started', 'ready']:
        assert key in data


def test_stake_sheet_dates(client):
    resp = client.get('/api/stake_sheet_dates')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'dates' in data
    assert 'latest' in data


def test_stake_sheets_default(client):
    resp = client.get('/api/stake_sheets')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'date' in data
    assert 'file' in data
    assert 'count' in data
    assert 'rows' in data
    assert isinstance(data['rows'], list)
