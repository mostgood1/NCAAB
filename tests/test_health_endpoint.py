import importlib
import pytest

app_module = importlib.import_module('app')
app = getattr(app_module, 'app')
app.testing = True

@pytest.fixture(scope='module')
def client():
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    for key in ['status', 'providers', 'display_hash', 'results_latest', 'stake_latest', 'finalize']:
        assert key in data
    assert isinstance(data['providers'], list)
    # finalize can be None or dict
    if data['finalize'] is not None:
        f = data['finalize']
        for k in ['date','rows','final','pending','started','ready']:
            assert k in f
