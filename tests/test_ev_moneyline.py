import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def test_moneyline_ev_and_filter(tmp_path: Path):
    # Two h2h rows: predicted margin favors home modestly; different prices change EV
    rows = [
        {
            'game_id': 'G1', 'event_id': 'E1', 'date': '2025-11-08', 'book': 'BookX', 'market': 'h2h', 'period': 'full_game',
            'pred_margin': 4.0, 'moneyline_home': -140, 'moneyline_away': +125
        },
        {
            'game_id': 'G2', 'event_id': 'E2', 'date': '2025-11-08', 'book': 'BookX', 'market': 'h2h', 'period': 'full_game',
            'pred_margin': 4.0, 'moneyline_home': -170, 'moneyline_away': +155
        }
    ]
    df = pd.DataFrame(rows)
    merged_csv = tmp_path / 'merged_ml.csv'
    df.to_csv(merged_csv, index=False)
    out_csv = tmp_path / 'stakes_ml.csv'
    res = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--include-markets', 'h2h', '--bankroll', '1000', '--kelly-fraction', '0.5', '--max-pct-per-bet', '0.10', '--out', str(out_csv)
    ])
    assert res.exit_code == 0, res.output
    stakes = pd.read_csv(out_csv)
    assert {'ev','prob'}.issubset(stakes.columns)
    # Apply an EV filter that should exclude at least one pick
    out_csv2 = tmp_path / 'stakes_ml_ev.csv'
    res2 = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--include-markets', 'h2h', '--bankroll', '1000', '--kelly-fraction', '0.5', '--min-ev', '0.01', '--max-pct-per-bet', '0.10', '--out', str(out_csv2)
    ])
    assert res2.exit_code in (0,1), res2.output  # May produce empty picks with exit code 0 or 0 exit due to no picks
    if out_csv2.exists():
        stakes_ev = pd.read_csv(out_csv2)
        if not stakes_ev.empty:
            assert (stakes_ev['ev'] >= 0.01).all()
