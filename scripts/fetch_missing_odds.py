import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests

try:
    from rapidfuzz import fuzz  # type: ignore
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False

OUT = Path('outputs')
SPORT_KEY = 'basketball_ncaab'
API_BASE = 'https://api.the-odds-api.com/v4'

NORMALIZATION_REPLACEMENTS = [
    (" university", ""), (" univ", ""), (" state", " st"), ("\.", ""), ("&", " and "),
    ("-", " "), ("  ", " ")
]

def norm(name: str) -> str:
    if not isinstance(name, str):
        return ''
    n = name.lower().strip()
    for a, b in NORMALIZATION_REPLACEMENTS:
        n = n.replace(a, b)
    return " ".join(n.split())

def score_pair(ht_query: str, at_query: str, ev_home: str, ev_away: str) -> float:
    hq, aq = norm(ht_query), norm(at_query)
    eh, ea = norm(ev_home), norm(ev_away)
    if HAVE_FUZZ:
        s_dir = 0.5 * (fuzz.WRatio(hq, eh) + fuzz.WRatio(aq, ea))
        s_swap = 0.5 * (fuzz.WRatio(hq, ea) + fuzz.WRatio(aq, eh))
    else:
        def basic(a,b):
            return 100 if a == b or a in b or b in a else 0
        s_dir = 0.5 * (basic(hq, eh) + basic(aq, ea))
        s_swap = 0.5 * (basic(hq, ea) + basic(aq, eh))
    return max(s_dir, s_swap)

def fetch_events(api_key: str):
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'totals,spreads',
        'oddsFormat': 'american'
    }
    url = f"{API_BASE}/sports/{SPORT_KEY}/odds"
    try:
        resp = requests.get(url, params=params, timeout=25)
    except Exception as e:
        print('Request error:', e)
        return []
    if resp.status_code != 200:
        print('API error', resp.status_code, resp.text[:160])
        return []
    try:
        data = resp.json()
    except Exception:
        print('Parse error')
        return []
    return data if isinstance(data, list) else []

def extract_consensus(event) -> dict:
    books = event.get('bookmakers', []) or []
    totals = []
    spreads_home = []
    home_team = event.get('home_team')
    for bk in books:
        for m in bk.get('markets', []):
            key = m.get('key')
            outs = m.get('outcomes', []) or []
            if key == 'totals':
                pts = [o.get('point') for o in outs if isinstance(o.get('point'), (int,float))]
                if pts:
                    # store single representative (median of points if multiple)
                    totals.append(float(pd.Series(pts).median()))
            elif key == 'spreads':
                for o in outs:
                    if o.get('name') == home_team and isinstance(o.get('point'), (int,float)):
                        spreads_home.append(float(o.get('point')))
    cons_total = None
    cons_spread = None
    # Require at least 2 books consensus; filter outliers by IQR
    if len(totals) >= 2:
        s = pd.Series(totals)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        filt = s[(s >= q1 - 1.5*iqr) & (s <= q3 + 1.5*iqr)]
        cons_total = float(filt.median()) if not filt.empty else float(s.median())
    if len(spreads_home) >= 2:
        s2 = pd.Series(spreads_home)
        q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
        iqr = q3 - q1
        filt2 = s2[(s2 >= q1 - 1.5*iqr) & (s2 <= q3 + 1.5*iqr)]
        cons_spread = float(filt2.median()) if not filt2.empty else float(s2.median())
    # Single-book fallback (still real, but mark low_consensus)
    basis_flags = []
    if cons_total is None and totals:
        cons_total = float(pd.Series(totals).median())
        basis_flags.append('single_book_total')
    if cons_spread is None and spreads_home:
        cons_spread = float(pd.Series(spreads_home).median())
        basis_flags.append('single_book_spread')
    basis = 'consensus_multi' if not basis_flags else 'consensus_' + '_'.join(basis_flags)
    return {'market_total': cons_total, 'spread_home': cons_spread, 'consensus_basis': basis, 'books_count': len(books)}

def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    miss_path = OUT / f'missing_real_coverage_{date_str}.csv'
    if not miss_path.exists():
        print('Missing coverage file not found:', miss_path)
        return
    miss_df = pd.read_csv(miss_path)
    if miss_df.empty:
        print('No missing rows; nothing to fetch.')
        return
    for col in ['home_team','away_team']:
        if col not in miss_df.columns:
            print('Missing column:', col)
            return
    api_key = os.getenv('THEODDSAPI_KEY') or os.getenv('ODDS_API_KEY')
    if not api_key:
        print('No API key (THEODDSAPI_KEY) provided.')
        return
    events = fetch_events(api_key)
    if not events:
        print('No events returned from API.')
        return
    # Filter events by date (commence_time ISO) matching date_str
    filtered = []
    for ev in events:
        ct = ev.get('commence_time')
        if isinstance(ct, str) and ct[:10] == date_str:
            filtered.append(ev)
    if not filtered:
        # fallback: use all if none match date (could be early API window)
        filtered = events
    results = []
    THRESHOLDS = [65, 58, 52]  # progressive relax
    for _, row in miss_df.iterrows():
        ht = str(row.get('home_team','')).strip()
        at = str(row.get('away_team','')).strip()
        best_ev = None
        best_score = -1.0
        # Progressive threshold search
        for thresh in THRESHOLDS:
            for ev in filtered:
                eh = ev.get('home_team','')
                ea = ev.get('away_team','')
                sc = score_pair(ht, at, eh, ea)
                if sc > best_score:
                    best_score = sc
                    best_ev = ev
            if best_score >= thresh:
                break
        if best_ev and best_score >= THRESHOLDS[-1]:
            lines = extract_consensus(best_ev)
            rec = {
                'game_id': row.get('game_id'),
                'home_team': ht,
                'away_team': at,
                'match_score': best_score,
                'api_event_id': best_ev.get('id'),
                'market_total_fetched': lines.get('market_total'),
                'spread_home_fetched': lines.get('spread_home'),
                'fetched_basis': lines.get('consensus_basis'),
                'books_count': lines.get('books_count')
            }
            results.append(rec)
        else:
            results.append({
                'game_id': row.get('game_id'),
                'home_team': ht,
                'away_team': at,
                'match_score': best_score,
                'api_event_id': None,
                'market_total_fetched': None,
                'spread_home_fetched': None,
                'fetched_basis': 'no_match',
                'books_count': 0
            })
    out_df = pd.DataFrame(results)
    out_path = OUT / f'missing_odds_fetched_{date_str}.csv'
    out_df.to_csv(out_path, index=False)
    print('Fetched odds rows (including unmatched):', len(out_df), 'written to', out_path)
    fills_total = int(out_df['market_total_fetched'].notna().sum())
    fills_spread = int(out_df['spread_home_fetched'].notna().sum())
    print('Real total fills:', fills_total, 'Real spread fills:', fills_spread)
    out_json = OUT / f'missing_odds_fetched_{date_str}.json'
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({'date': date_str, 'records': results, 'fills_total': fills_total, 'fills_spread': fills_spread}, fh, indent=2)
    print('JSON written:', out_json)

if __name__ == '__main__':
    main()
