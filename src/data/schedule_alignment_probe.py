import argparse, datetime as dt, json, pathlib, sys
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"

def load_games_curr(date_str: str) -> pd.DataFrame:
    p = OUT / 'games_curr.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    if 'date' in df.columns:
        df = df[df['date'].astype(str) == date_str]
    return df

def load_enriched(date_str: str) -> pd.DataFrame:
    p = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_espn_ids(date_str: str) -> set[str]:
    # Prefer subset if available to avoid inflating parity target beyond curated slate
    p_subset = OUT / f'schedule_espn_subset_{date_str}.json'
    p_full = OUT / f'schedule_espn_ids_{date_str}.json'
    p = p_subset if p_subset.exists() else p_full
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return set(map(str, data.get('game_ids', [])))
    except Exception:
        return set()

def analyze(date_str: str) -> dict:
    games = load_games_curr(date_str)
    enriched = load_enriched(date_str)
    espn_ids = load_espn_ids(date_str)
    games_ids = set(map(str, games['game_id'])) if 'game_id' in games.columns else set()
    enriched_ids = set(map(str, enriched['game_id'])) if 'game_id' in enriched.columns else set()
    placeholder_cols = {'home_team','away_team'} & set(games.columns)
    placeholder_rows = 0
    if placeholder_cols:
        placeholder_rows = int(((games['home_team'] == 'TBD') | (games['away_team'] == 'TBD')).sum())
    espn_missing = sorted(espn_ids - games_ids) if espn_ids else []
    local_extras = sorted(games_ids - espn_ids) if espn_ids else []
    parity_ok = espn_ids and len(espn_missing) == 0 and len(local_extras) == 0 and placeholder_rows == 0
    return {
        'date': date_str,
        'games_curr_rows': len(games),
        'enriched_rows': len(enriched),
        'espn_ids_count': len(espn_ids),
        'games_curr_ids': sorted(games_ids),
        'enriched_ids': sorted(enriched_ids),
        'espn_missing_ids': espn_missing,
        'local_extra_ids': local_extras,
        'placeholder_rows': placeholder_rows,
        'parity_ok': parity_ok
    }

def _json_default(o):
    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    if isinstance(o, set):
        return sorted(list(o))
    return str(o)

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description='Generate schedule alignment probe JSON')
    ap.add_argument('--date', help='Date YYYY-MM-DD (defaults to today)')
    ap.add_argument('--fail-on-mismatch', action='store_true', help='Exit 1 if parity not achieved')
    args = ap.parse_args(argv)
    date_str = args.date or dt.date.today().isoformat()
    report = analyze(date_str)
    out_path = OUT / f'schedule_alignment_probe_{date_str}.json'
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f'Wrote {out_path} parity_ok={report["parity_ok"]} placeholders={report["placeholder_rows"]} missing={len(report["espn_missing_ids"])}')
    if args.fail_on_mismatch and not report['parity_ok']:
        return 1
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
