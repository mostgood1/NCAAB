import importlib
import pytest

app_module = importlib.import_module('app')
app = getattr(app_module, 'app')
app.testing = True

@pytest.fixture(scope='module')
def client():
    with app.test_client() as c:
        yield c


def test_backtest_summary(client):
    resp = client.get('/api/backtest-summary')
    assert resp.status_code in (200, 404)
    data = resp.get_json()
    assert 'status' in data
    if resp.status_code == 200:
        assert data['status'] == 'ok'
        assert 'summary' in data


def test_backtest_page(client):
    resp = client.get('/backtest')
    assert resp.status_code == 200


def test_display_hash_diff(client):
    resp = client.get('/api/display_hash_diff')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'date' in data
    assert 'current_hash' in data
    assert 'match' in data


def test_download_display_predictions(client):
    # Without date, this may 404 if no file exists; accept 200 or 404
    resp = client.get('/download/display-predictions')
    assert resp.status_code in (200, 404)
