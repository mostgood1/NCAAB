import json
from pathlib import Path
import pandas as pd

from scripts.stake_simulation import simulate, OUT

def setup_module(module):
    # Create minimal outputs for TEST date
    (OUT / 'daily_results').mkdir(parents=True, exist_ok=True)
    # Fake single game
    games = pd.DataFrame([
        {
            'game_id': 'TEST:away:home',
            'home_score': 80,
            'away_score': 70,
        }
    ])
    games.to_csv(OUT / 'daily_results' / 'results_TEST.csv', index=False)
    closing = pd.DataFrame([
        {
            'game_id': 'TEST:away:home',
            'closing_total': 145.5,
            'closing_spread_home': -3.5,
            'closing_ml_home': -140,
            'closing_ml_away': 120,
        }
    ])
    closing.to_csv(OUT / 'games_with_closing_TEST.csv', index=False)
    payload = {
        'date': 'TEST',
        'bets_detail': {
            'totals': [{'game_id': 'TEST:away:home', 'edge_total': 5.0, 'stake': 1.0}],
            'spread': [{'game_id': 'TEST:away:home', 'edge_margin': 4.0, 'stake': 1.0}],
            'moneyline': [{'game_id': 'TEST:away:home', 'edge_prob': 0.05, 'stake': 1.0, 'p_model': 0.65, 'p_implied': 0.58}],
        }
    }
    (OUT / 'backtest_metrics_TEST.json').write_text(json.dumps(payload), encoding='utf-8')


def test_simulate_kelly(tmp_path):
    simulate('TEST', 'kelly', [0.5], [])
    out = json.loads((OUT / 'stake_simulation_TEST.json').read_text(encoding='utf-8'))
    assert out['date'] == 'TEST'
    assert out['regimes']
    kelly = out['regimes'][0]
    assert kelly['mode'] == 'kelly'
    assert 'roi' in kelly


def test_simulate_flat(tmp_path):
    simulate('TEST', 'flat', [], [1.0])
    out = json.loads((OUT / 'stake_simulation_TEST.json').read_text(encoding='utf-8'))
    assert any(r['mode']=='flat' for r in out['regimes'])
