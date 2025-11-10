import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
import numpy as np

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def test_distributional_ev_columns_and_filter(tmp_path: Path):
    # Two totals with the same mu and line, but different prices so EV differs
    rows = [
        {
            'game_id': 'G1', 'event_id': 'E1', 'date': '2025-11-08', 'book': 'BookX', 'market': 'totals', 'period': 'full_game',
            'pred_total_mu': 152.0, 'pred_total_sigma': 8.0, 'total': 150.0, 'over_price': -110, 'under_price': -110
        },
        {
            'game_id': 'G2', 'event_id': 'E2', 'date': '2025-11-08', 'book': 'BookX', 'market': 'totals', 'period': 'full_game',
            'pred_total_mu': 152.0, 'pred_total_sigma': 8.0, 'total': 150.0, 'over_price': -105, 'under_price': -115
        },
    ]
    df = pd.DataFrame(rows)
    merged_csv = tmp_path / 'merged.csv'
    df.to_csv(merged_csv, index=False)
    out_csv = tmp_path / 'stakes.csv'
    # No EV filter -> both included
    res = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--bankroll', '1000', '--kelly-fraction', '0.5', '--use-distributional', '--max-pct-per-bet', '0.10', '--out', str(out_csv)
    ])
    assert res.exit_code == 0, res.output
    stakes = pd.read_csv(out_csv)
    assert {'ev','prob','p_over','p_under','b_over','b_under','ev_over','ev_under'}.issubset(stakes.columns)
    assert len(stakes) == 2
    # With a modest EV filter, at least one should remain
    out_csv2 = tmp_path / 'stakes_ev.csv'
    res2 = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--bankroll', '1000', '--kelly-fraction', '0.5', '--use-distributional', '--min-ev', '0.005', '--max-pct-per-bet', '0.10', '--out', str(out_csv2)
    ])
    assert res2.exit_code == 0, res2.output
    stakes2 = pd.read_csv(out_csv2)
    assert len(stakes2) >= 1
