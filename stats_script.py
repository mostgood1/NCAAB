from src.modeling.data import build_training_frame
X,_,_ = build_training_frame()
import json
team = [c for c in X.columns if c.startswith(" home_team_\) or c.startswith(\away_team_\)]
diff = [c for c in X.columns if c.startswith(\diff_\)]
probes = [\home_team_season_off_ppg\,\away_team_season_off_ppg\,\diff_season_off_ppg\,\home_team_rest_days\,\diff_rest_days\]
stats = { \total_cols\: len(X.columns), \team_cols_count\: len(team), \diff_cols_count\: len(diff), \first_30\: X.columns[:30].tolist(), \team_samples\: team[:15], \diff_samples\: diff[:15], \probes\: { p: (None if p not in X.columns else int(X[p].notna().sum())) for p in probes } }
print(json.dumps(stats, indent=2))
