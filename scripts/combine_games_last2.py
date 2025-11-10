import pandas as pd
from pathlib import Path

paths = [
    'outputs/games_2023.csv',
    'outputs/games_2024.csv'
]

dfs = []
for p in paths:
    pth = Path(p)
    if pth.exists():
        df = pd.read_csv(pth)
        dfs.append(df)
    else:
        print(f"Warning: {p} not found")

if not dfs:
    raise SystemExit('No games files found to combine')

all_df = pd.concat(dfs, ignore_index=True)
if 'game_id' in all_df.columns:
    all_df['game_id'] = all_df['game_id'].astype(str)
    all_df = all_df.drop_duplicates(subset=['game_id'])

out_path = Path('outputs/games_last2.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
all_df.to_csv(out_path, index=False)
print(f'Wrote {out_path} with {len(all_df)} rows')
