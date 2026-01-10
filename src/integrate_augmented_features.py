import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

AUG_RECENT = OUT / 'features_augmented_recent.csv'

TARGETS = [
    OUT / 'features_curr.csv',
    OUT / 'features_all.csv',
    OUT / 'features_last2.csv',
]

AUG_SUFFIX = '_augmented'


def integrate_one(target: Path, aug: pd.DataFrame) -> Path | None:
    if not target.exists():
        return None
    try:
        base = pd.read_csv(target)
    except Exception:
        return None
    if base.empty or aug.empty:
        return None
    # Merge on game_id; avoid duplicating columns
    key = 'game_id'
    common = set(base.columns) & set(aug.columns)
    aug_cols = [c for c in aug.columns if c not in common or c == key]
    merged = base.merge(aug[aug_cols], on=key, how='left')
    out = target.with_name(target.stem + AUG_SUFFIX + target.suffix)
    merged.to_csv(out, index=False)
    return out


def run() -> dict:
    if not AUG_RECENT.exists():
        return {'status': 'no_augmented_file'}
    aug = pd.read_csv(AUG_RECENT)
    results = {}
    for t in TARGETS:
        out = integrate_one(t, aug)
        if out:
            results[t.name] = str(out)
    # Also produce a standalone enriched preview
    prev = OUT / 'features_merged_recent.csv'
    aug.to_csv(prev, index=False)
    results['preview'] = str(prev)
    results['rows'] = int(len(aug))
    # Integrate into per-date feature files
    per_date_files = sorted([p for p in OUT.glob('features_*.csv') if 'augmented' not in p.name and p.name not in {t.name for t in TARGETS}])
    integrated_count = 0
    for base_path in per_date_files:
        try:
            base = pd.read_csv(base_path)
        except Exception:
            continue
        if base.empty:
            continue
        # infer date from filename
        date_str = base_path.stem.replace('features_', '')
        # filter augmented to this date when present
        aug_d = aug.copy()
        if 'date' in aug_d.columns:
            try:
                aug_d['date'] = pd.to_datetime(aug_d['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception:
                aug_d['date'] = aug_d['date'].astype(str)
            aug_d = aug_d[aug_d['date'].astype(str) == date_str]
        # primary join on game_id
        if 'game_id' in base.columns and 'game_id' in aug_d.columns:
            base['game_id'] = base['game_id'].astype(str)
            aug_d['game_id'] = aug_d['game_id'].astype(str)
            merged = base.merge(aug_d[[c for c in aug_d.columns if c not in base.columns or c == 'game_id']], on='game_id', how='left')
        else:
            # fallback join via enriched mapping to add team names for alignment
            enrich_path = OUT / f'predictions_unified_enriched_{date_str}.csv'
            merged = base.copy()
            if enrich_path.exists():
                try:
                    enrich = pd.read_csv(enrich_path)
                    for c in ('game_id','date','home_team','away_team'):
                        if c in enrich.columns:
                            enrich[c] = enrich[c].astype(str)
                    if 'game_id' in merged.columns:
                        merged['game_id'] = merged['game_id'].astype(str)
                        merged = merged.merge(enrich[['game_id','date','home_team','away_team']], on='game_id', how='left')
                    # attempt join on teams+date
                    if {'home_team','away_team','date'}.issubset(set(merged.columns)) and {'home_team','away_team','date'}.issubset(set(aug_d.columns)):
                        merged = merged.merge(aug_d[['home_team','away_team','date'] + [c for c in aug_d.columns if c not in {'home_team','away_team','date'}]], on=['home_team','away_team','date'], how='left')
                except Exception:
                    pass
        out_path = base_path.with_name(base_path.stem + '_augmented' + base_path.suffix)
        merged.to_csv(out_path, index=False)
        integrated_count += 1
    # Integrate team rolling features by home/away team and date
    tr_path = OUT / 'team_rolling_features_recent.csv'
    if tr_path.exists():
        try:
            tr = pd.read_csv(tr_path)
            for c in ('team','date'):
                if c in tr.columns:
                    tr[c] = tr[c].astype(str)
        except Exception:
            tr = pd.DataFrame()
        added_count = 0
        if not tr.empty:
            for base_path in per_date_files:
                try:
                    base = pd.read_csv(base_path)
                except Exception:
                    continue
                if base.empty:
                    continue
                date_str = base_path.stem.replace('features_', '')
                if 'date' in base.columns:
                    base['date'] = base['date'].astype(str)
                else:
                    base['date'] = str(date_str)
                # Join home team rolling
                ht_cols = ['team','date','pace5','ts5','rate3p5','to_rate5','drb_rate5']
                tr_h = tr[ht_cols].rename(columns={'team':'home_team', 'pace5':'home_pace5','ts5':'home_ts5','rate3p5':'home_rate3p5','to_rate5':'home_to_rate5','drb_rate5':'home_drb_rate5'})
                merged = base.merge(tr_h, on=['home_team','date'], how='left') if 'home_team' in base.columns else base.copy()
                # Join away team rolling
                tr_a = tr[ht_cols].rename(columns={'team':'away_team', 'pace5':'away_pace5','ts5':'away_ts5','rate3p5':'away_rate3p5','to_rate5':'away_to_rate5','drb_rate5':'away_drb_rate5'})
                merged = merged.merge(tr_a, on=['away_team','date'], how='left') if 'away_team' in merged.columns else merged
                out_path = base_path.with_name(base_path.stem + '_augmented' + base_path.suffix)
                merged.to_csv(out_path, index=False)
                added_count += 1
        results['team_rolling_integrated'] = added_count
    results['per_date_integrated'] = integrated_count
    return results


def main():
    res = run()
    print(res)

if __name__ == '__main__':
    main()
