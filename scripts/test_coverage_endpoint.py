import json
from app import app

def main():
    with app.test_client() as c:
        rv = c.get('/api/coverage_today')
        print('Status:', rv.status_code)
        data = rv.get_json()
        print('Keys:', sorted(list(data.keys())))
        print('Counts:', data.get('status_counts'))
        print('Rows:', data.get('rows'))
        print('Non-placeholder:', data.get('non_placeholder_games'))
        # Show first 3 games sample
        games = data.get('games', [])
        print('Sample games:', games[:3])

if __name__ == '__main__':
    main()
