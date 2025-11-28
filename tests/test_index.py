import importlib
import pytest

app_module = importlib.import_module('app')
app = getattr(app_module, 'app')
app.testing = True

@pytest.fixture(scope='module')
def client():
    with app.test_client() as c:
        yield c


def test_index_page(client):
    resp = client.get('/')
    assert resp.status_code == 200

def test_home_page(client):
    resp = client.get('/home')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert '/stake-archive' in html
    assert '/results-archive' in html
    assert '/display-archive' in html
    assert '/coverage' in html
    assert '/calibration' in html
