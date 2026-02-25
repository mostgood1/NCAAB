import json
import importlib
from pathlib import Path

import pandas as pd
import pytest


app_module = importlib.import_module('app')
app = getattr(app_module, 'app')
app.testing = True


@pytest.fixture(scope='module')
def client():
    with app.test_client() as c:
        yield c


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def test_live_lens_analytics_range_aggregates_totals_and_ats(client):
    out_dir = Path(getattr(app_module, 'OUT'))
    date = '1900-01-02'

    sig_p = out_dir / f'live_lens_signals_{date}.jsonl'
    res_p = out_dir / 'daily_results' / f'results_{date}.csv'

    # Synthetic: 2 games, 1 totals BET + 1 ATS BET each.
    signals = [
        {
            'game_id': 'g1',
            'kind': 'total',
            'lens': 'fg',
            'side': 'over',
            'live_line': 148.0,
            'is_bet': True,
            'edge': 0.03,
            'elapsed': 18,
            'remaining': 22,
            'driver': 'pace_up',
            'driver_tags': ['pace_up'],
            'ts': '1900-01-02T01:00:00Z',
        },
        {
            'game_id': 'g1',
            'kind': 'ats',
            'lens': 'fg',
            'side': 'home',
            'live_line': -3.0,
            'is_bet': True,
            'edge': 0.02,
            'elapsed': 18,
            'remaining': 22,
            'driver': 'foul_rate',
            'driver_tags': ['foul_rate'],
            'ts': '1900-01-02T01:00:10Z',
        },
        {
            'game_id': 'g2',
            'kind': 'total',
            'lens': 'fg',
            'side': 'under',
            'live_line': 125.0,
            'is_bet': True,
            'edge': 0.04,
            'elapsed': 26,
            'remaining': 14,
            'driver': 'tempo_down',
            'driver_tags': ['tempo_down'],
            'ts': '1900-01-02T02:00:00Z',
        },
        {
            'game_id': 'g2',
            'kind': 'ats',
            'lens': 'fg',
            'side': 'away',
            'live_line': 2.0,
            'is_bet': True,
            'edge': 0.01,
            'elapsed': 26,
            'remaining': 14,
            'driver': 'turnovers',
            'driver_tags': ['turnovers'],
            'ts': '1900-01-02T02:00:10Z',
        },
    ]

    results = pd.DataFrame(
        [
            {'game_id': 'g1', 'completed': True, 'actual_total': 150, 'actual_margin': 5},
            {'game_id': 'g2', 'completed': True, 'actual_total': 120, 'actual_margin': -4},
        ]
    )

    # Write artifacts
    _write_jsonl(sig_p, signals)
    res_p.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(res_p, index=False)

    try:
        resp = client.get(
            f'/api/live_lens_analytics?start={date}&end={date}&include_rows=1&max_rows=100&full_game_only=1'
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ok'] is True
        assert data['status'] == 'ok'
        assert data['dates'] == [date]

        overall = data['overall']
        assert overall['totals']['status'] == 'ok'
        assert overall['totals']['n_settled'] == 2
        assert overall['totals']['wins'] == 2
        assert overall['totals']['losses'] == 0

        # Tag recap fields (non-breaking additions)
        assert 'by_driver_tag' in overall['totals']
        assert 'by_driver_tag_full' in overall['totals']
        assert 'by_driver_tag_canonical' in overall['totals']
        assert 'by_driver_tag_type' in overall['totals']
        assert isinstance(overall['totals']['by_driver_tag_type'], list)

        # Driver recap fields (non-breaking additions)
        assert 'by_driver' in overall['totals']
        assert 'by_driver_full' in overall['totals']
        assert isinstance(overall['totals']['by_driver'], list)
        assert isinstance(overall['totals']['by_driver_full'], list)

        assert overall['ats']['status'] == 'ok'
        assert overall['ats']['n_settled'] == 2
        assert overall['ats']['wins'] == 2
        assert overall['ats']['losses'] == 0

        assert 'by_driver_tag' in overall['ats']
        assert 'by_driver_tag_full' in overall['ats']
        assert 'by_driver_tag_canonical' in overall['ats']
        assert 'by_driver_tag_type' in overall['ats']
        assert isinstance(overall['ats']['by_driver_tag_type'], list)

        assert 'by_driver' in overall['ats']
        assert 'by_driver_full' in overall['ats']
        assert isinstance(overall['ats']['by_driver'], list)
        assert isinstance(overall['ats']['by_driver_full'], list)

        assert 'per_day' in data
        assert isinstance(data['per_day'], list)
        assert len(data['per_day']) == 1

        hist = data.get('history')
        assert hist is not None
        assert hist.get('count') == 4
        assert isinstance(hist.get('rows'), list)
        assert len(hist.get('rows')) == 4
    finally:
        # Cleanup
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
