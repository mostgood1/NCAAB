import pandas as pd
import numpy as np
from pathlib import Path

from ncaab_model.cli import edge_persistence as edge_persistence_cmd
from typer.testing import CliRunner
from ncaab_model.cli import app as typer_app

runner = CliRunner()

def _make_temp_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p

def test_edge_persistence_basic(tmp_path):
    # Synthetic 3 games with stable edges
    games = pd.DataFrame({
        'game_id': ['1','2','3'],
        'date': ['2025-11-01','2025-11-01','2025-11-01'],
        'home_score': [80, 75, 70],
        'away_score': [70, 65, 60],
    })
    preds = pd.DataFrame({
        'game_id': ['1','2','3'],
        'date': ['2025-11-01','2025-11-01','2025-11-01'],
        'pred_total': [150, 140, 130],
        'pred_margin': [10, 10, 10],
    })
    last = pd.DataFrame({
        'game_id': ['1','2','3'],
        'date_game': ['2025-11-01','2025-11-01','2025-11-01'],
        'market': ['totals','totals','totals'],
        'period': ['full_game','full_game','full_game'],
        'total': [148, 138, 128],  # early edges: +2,+2,+2
        'home_spread': [-5, -5, -5],
    })
    closing = pd.DataFrame({
        'game_id': ['1','2','3'],
        'date_game': ['2025-11-01','2025-11-01','2025-11-01'],
        'market': ['totals','totals','totals'],
        'period': ['full_game','full_game','full_game'],
        'total': [149, 139, 129],  # closing edges: +1,+1,+1
        'home_spread': [-5, -5, -5],
    })
    gp = _make_temp_csv(tmp_path, 'games.csv', games)
    lp = _make_temp_csv(tmp_path, 'last.csv', last)
    cp = _make_temp_csv(tmp_path, 'closing.csv', closing)
    pp = _make_temp_csv(tmp_path, 'preds.csv', preds)

    out = tmp_path / 'summary.json'
    result = runner.invoke(typer_app, [
        'edge-persistence', str(gp), str(lp), str(cp), str(pp), '--total-threshold', '1.0', '--out', str(out)
    ])
    assert result.exit_code == 0, result.output
    js = out.read_text(encoding='utf-8')
    assert 'correlation_total' in js
    # Correlation should be perfect (edges scale uniformly) -> 1.0
    # Because all early edges 2, closing edges 1 -> correlation 1
    import json
    data = json.loads(js)
    # Correlation may be None due to constant vectors; expect None here (treated as missing)
    assert data['correlation_total'] is None
    assert data['sign_retention_total'] == 1.0
    # Decay ratio median should be 0.5 (1/2)
    assert data['median_magnitude_ratio_total'] == 0.5
