import os
import io
import csv

def test_predictions_ingest_auth_fail(client):
    # Missing token should fail
    resp = client.post('/api/predictions_ingest')
    assert resp.status_code in (401,403)

def test_predictions_ingest_success(client):
    token = os.environ.get('NCAAB_PREDICTIONS_INGEST_TOKEN', 'testtoken123')
    os.environ['NCAAB_PREDICTIONS_INGEST_TOKEN'] = token
    # Build minimal CSV in-memory
    rows = [
        {'game_id':'G1','pred_total':140.5,'pred_margin':3.2,'date':'2099-12-31'},
        {'game_id':'G2','pred_total':141.0,'pred_margin':-1.4,'date':'2099-12-31'},
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    buf.seek(0)
    data = {
        'file': (io.BytesIO(buf.read().encode('utf-8')), 'preds.csv')
    }
    resp = client.post('/api/predictions_ingest?force=1', headers={'X-Ingest-Token': token}, data=data, content_type='multipart/form-data')
    js = resp.get_json()
    assert resp.status_code == 200
    assert js.get('ok') is True
    assert js.get('rows') == 2
    assert js.get('date') == '2099-12-31'
    assert js.get('primary_written') is True