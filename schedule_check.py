import os, pandas as pd
OUT='outputs'
today='2025-11-27'
path=os.path.join(OUT,'games_curr.csv')
if not os.path.exists(path):
    print('games_curr.csv missing')
    raise SystemExit(1)
sched=pd.read_csv(path)
if 'date' in sched.columns:
    sched['date']=pd.to_datetime(sched['date'],errors='coerce').dt.strftime('%Y-%m-%d')
    sched=sched[sched['date']==today]
if 'home_team' not in sched.columns and 'home' in sched.columns:
    sched['home_team']=sched['home']
if 'away_team' not in sched.columns and 'away' in sched.columns:
    sched['away_team']=sched['away']
print('Total schedule rows for date:',len(sched))
for r in sched[['game_id','home_team','away_team']].to_dict('records'):
    print(r)
