import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

print("Loading data...")
# Data is in ../data relative to Model_Training
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

summary_df = pd.read_csv(os.path.join(data_dir, "all_season_summary.csv"))
batting_df = pd.read_csv(os.path.join(data_dir, "all_season_batting_card.csv"))
bowling_df = pd.read_csv(os.path.join(data_dir, "all_season_bowling_card.csv"))

print("Processing venue metadata...")
venues = summary_df['venue_name'].dropna().unique()

venue_meta = []
for v in venues:
    v_lower = v.lower()
    meta = {'venue_name': v, 'city': 'Unknown', 'ground_size': 'M', 'coastal': 'N', 'altitude_m': 50, 'home_team': 'Unknown'}
    if 'wankhede' in v_lower or 'brabourne' in v_lower or 'dy patil' in v_lower:
        meta.update({'city': 'Mumbai', 'ground_size': 'S', 'coastal': 'Y', 'altitude_m': 14, 'home_team': 'MI'})
    elif 'eden gardens' in v_lower:
        meta.update({'city': 'Kolkata', 'ground_size': 'M', 'coastal': 'Y', 'altitude_m': 9, 'home_team': 'KKR'})
    elif 'chidambaram' in v_lower or 'chepauk' in v_lower:
        meta.update({'city': 'Chennai', 'ground_size': 'M', 'coastal': 'Y', 'altitude_m': 6, 'home_team': 'CSK'})
    elif 'chinnaswamy' in v_lower:
        meta.update({'city': 'Bengaluru', 'ground_size': 'S', 'coastal': 'N', 'altitude_m': 920, 'home_team': 'RCB'})
    elif 'rajiv gandhi' in v_lower or 'uppal' in v_lower:
        meta.update({'city': 'Hyderabad', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 542, 'home_team': 'SRH'})
    elif 'narendra modi' in v_lower or 'motera' in v_lower:
        meta.update({'city': 'Ahmedabad', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 55, 'home_team': 'GT'})
    elif 'arun jaitley' in v_lower or 'feroz shah kotla' in v_lower:
        meta.update({'city': 'Delhi', 'ground_size': 'S', 'coastal': 'N', 'altitude_m': 216, 'home_team': 'DC'})
    elif 'sawai mansingh' in v_lower:
        meta.update({'city': 'Jaipur', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 431, 'home_team': 'RR'})
    elif 'bindra' in v_lower or 'mohali' in v_lower or 'punjab cricket association' in v_lower:
        meta.update({'city': 'Chandigarh', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 318, 'home_team': 'PBKS'})
    elif 'ekana' in v_lower or 'lucknow' in v_lower:
        meta.update({'city': 'Lucknow', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 123, 'home_team': 'LSG'})
    elif 'barsapara' in v_lower or 'guwahati' in v_lower:
        meta.update({'city': 'Guwahati', 'ground_size': 'M', 'coastal': 'N', 'altitude_m': 55, 'home_team': 'RR'})
    elif 'mca' in v_lower or 'pune' in v_lower:
        meta.update({'city': 'Pune', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 560, 'home_team': 'CSK'})
    elif 'dharamshala' in v_lower or 'himachal' in v_lower:
        meta.update({'city': 'Dharamshala', 'ground_size': 'M', 'coastal': 'N', 'altitude_m': 1457, 'home_team': 'PBKS'})
    elif 'indore' in v_lower or 'holkar' in v_lower:
        meta.update({'city': 'Indore', 'ground_size': 'S', 'coastal': 'N', 'altitude_m': 553, 'home_team': 'PBKS'})
    elif 'visakhapatnam' in v_lower:
        meta.update({'city': 'Visakhapatnam', 'ground_size': 'M', 'coastal': 'Y', 'altitude_m': 5, 'home_team': 'DC'})
    elif 'raipur' in v_lower:
        meta.update({'city': 'Raipur', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 298, 'home_team': 'DC'})
    elif 'cuttack' in v_lower or 'barabati' in v_lower:
        meta.update({'city': 'Cuttack', 'ground_size': 'L', 'coastal': 'N', 'altitude_m': 36, 'home_team': 'DC'})
    
    venue_meta.append(meta)

venue_df = pd.DataFrame(venue_meta)
venue_df.to_csv(os.path.join(data_dir, 'venue_metadata.csv'), index=False)
print("Saved venue metadata to data/venue_metadata.csv")

print("Engineering features...")
summary_df['1st_inning_score'] = pd.to_numeric(summary_df['1st_inning_score'], errors='coerce')
venue_stats = summary_df.groupby('venue_name').agg(
    avg_first_innings_score=('1st_inning_score', 'mean'),
    total_matches=('id', 'count')
).reset_index()

def get_bat_first_team(row):
    if row['decision'] == 'BAT FIRST':
        return row['toss_won']
    elif row['decision'] == 'BOWL FIRST':
        return row['home_team'] if row['toss_won'] == row['away_team'] else row['away_team']
    return None

summary_df['bat_first_team'] = summary_df.apply(get_bat_first_team, axis=1)
summary_df['bat_second_team'] = summary_df.apply(lambda row: row['away_team'] if row['home_team'] == row['bat_first_team'] else row['home_team'], axis=1)

summary_df['chase_win'] = (summary_df['winner'] == summary_df['bat_second_team']).astype(int)
chase_stats = summary_df.groupby('venue_name')['chase_win'].mean().reset_index().rename(columns={'chase_win': 'chase_win_pct'})
venue_stats = venue_stats.merge(chase_stats, on='venue_name', how='left')

# Wickets per match
wickets_per_match = bowling_df.groupby(['match_id', 'venue'])['wickets'].sum().reset_index()
avg_wickets = wickets_per_match.groupby('venue')['wickets'].mean().reset_index().rename(columns={'wickets': 'avg_wickets_per_match', 'venue': 'venue_name'})
venue_stats = venue_stats.merge(avg_wickets, on='venue_name', how='left')

# Synthetic Spin Wicket PCT
np.random.seed(42)
venue_stats['spin_wicket_pct_proxy'] = np.random.uniform(0.3, 0.6, size=len(venue_stats))

# Save venue stats so the prediction script can use them without recomputing
venue_stats.to_csv(os.path.join(data_dir, 'venue_stats.csv'), index=False)
print("Saved venue stats to data/venue_stats.csv")

summary_df['home_win'] = (summary_df['winner'] == summary_df['home_team']).astype(int)
home_adv = summary_df.groupby(['season', 'home_team']).agg(
    home_matches=('id', 'count'),
    home_wins=('home_win', 'sum')
).reset_index()
home_adv['home_win_rate'] = home_adv['home_wins'] / home_adv['home_matches']
home_adv = home_adv.rename(columns={'home_team': 'team'})

team_matches = []
for idx, row in summary_df.iterrows():
    if pd.notna(row['home_team']):
        team_matches.append({'season': row['season'], 'team': row['home_team'], 'toss_won': int(row['toss_won'] == row['home_team']), 'decision': row['decision']})
    if pd.notna(row['away_team']):
        team_matches.append({'season': row['season'], 'team': row['away_team'], 'toss_won': int(row['toss_won'] == row['away_team']), 'decision': row['decision']})

tm_df = pd.DataFrame(team_matches)
toss_stats = tm_df.groupby(['season', 'team']).agg(
    total_matches=('toss_won', 'count'),
    toss_wins=('toss_won', 'sum')
).reset_index()
toss_stats['toss_win_rate'] = toss_stats['toss_wins'] / toss_stats['total_matches']

bat_first_decisions = tm_df[(tm_df['toss_won'] == 1) & (tm_df['decision'] == 'BAT FIRST')].groupby(['season', 'team']).size().reset_index(name='bat_first_decisions')
toss_stats = toss_stats.merge(bat_first_decisions, on=['season', 'team'], how='left')
toss_stats['bat_first_decisions'] = toss_stats['bat_first_decisions'].fillna(0)
toss_stats['bat_first_decision_rate'] = toss_stats['bat_first_decisions'] / toss_stats['toss_wins'].replace(0, 1)

summary_df['start_date'] = pd.to_datetime(summary_df['start_date'], format='mixed', errors='coerce')
summary_df['month'] = summary_df['start_date'].dt.month
summary_df['is_late_season'] = summary_df['month'].isin([5, 6]).astype(int)

late_season = pd.DataFrame(team_matches)
late_season['is_late_season'] = [summary_df.loc[idx // 2, 'is_late_season'] if (idx//2) < len(summary_df) else 0 for idx in range(len(late_season))]
late_season_stats = late_season.groupby(['season', 'team'])['is_late_season'].mean().reset_index().rename(columns={'is_late_season': 'late_season_match_pct'})

batting = batting_df.copy()
batting["season"] = batting["season"].dropna().astype(int)
batting["strikeRate"] = pd.to_numeric(batting["strikeRate"], errors="coerce")
batting = batting.rename(columns={"current_innings": "team"})
batting = batting.dropna(subset=["team", "runs", "ballsFaced"])

bat_team = batting.groupby(["season", "team"]).agg(
    total_runs=("runs", "sum"),
    total_balls=("ballsFaced", "sum"),
    avg_strike_rate=("strikeRate", "mean")
).reset_index()

final_features = bat_team.merge(home_adv, on=['season', 'team'], how='left').fillna(0)
final_features = final_features.merge(toss_stats[['season', 'team', 'toss_win_rate', 'bat_first_decision_rate']], on=['season', 'team'], how='left')
final_features = final_features.merge(late_season_stats, on=['season', 'team'], how='left')

final_features.to_csv(os.path.join(data_dir, 'enhanced_team_season_features.csv'), index=False)
print("Saved enhanced features to data/enhanced_team_season_features.csv")

print("Preparing training data...")
train_df = summary_df.dropna(subset=['winner', 'home_team', 'away_team', 'venue_name']).copy()

train_df = train_df.merge(venue_stats[['venue_name', 'avg_first_innings_score', 'chase_win_pct', 'avg_wickets_per_match', 'spin_wicket_pct_proxy']], on='venue_name', how='left')
train_df = train_df.merge(venue_df[['venue_name', 'ground_size', 'coastal']], on='venue_name', how='left')

train_df['ground_size_encode'] = train_df['ground_size'].map({'S': 1, 'M': 2, 'L': 3}).fillna(2)
train_df['coastal_encode'] = train_df['coastal'].map({'Y': 1, 'N': 0}).fillna(0)

# Apply venue adjustment: normalize team total_runs by venue_avg_score
final_features['team_avg_score'] = final_features['total_runs'] / final_features['home_matches'].replace(0, 7) * 2 # Approximating 14 matches total

final_features_home = final_features.rename(columns={c: f"home_{c}" for c in final_features.columns if c not in ['season', 'team']})
train_df = train_df.merge(final_features_home, left_on=['season', 'home_team'], right_on=['season', 'team'], how='left').drop(columns=['team'])

final_features_away = final_features.rename(columns={c: f"away_{c}" for c in final_features.columns if c not in ['season', 'team']})
train_df = train_df.merge(final_features_away, left_on=['season', 'away_team'], right_on=['season', 'team'], how='left').drop(columns=['team'])

train_df['home_venue_adjusted_batting'] = train_df['home_team_avg_score'] / train_df['avg_first_innings_score'].replace(0, 160)
train_df['away_venue_adjusted_batting'] = train_df['away_team_avg_score'] / train_df['avg_first_innings_score'].replace(0, 160)

train_df = train_df.fillna(0)
train_df['target'] = (train_df['winner'] == train_df['home_team']).astype(int)

features = [
    'avg_first_innings_score', 'chase_win_pct', 'avg_wickets_per_match', 'spin_wicket_pct_proxy',
    'ground_size_encode', 'coastal_encode',
    'home_home_win_rate', 'home_toss_win_rate', 'home_bat_first_decision_rate', 'home_late_season_match_pct', 'home_venue_adjusted_batting',
    'away_home_win_rate', 'away_toss_win_rate', 'away_bat_first_decision_rate', 'away_late_season_match_pct', 'away_venue_adjusted_batting'
]

X = train_df[features]
y = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Step 1 & 3: Train models, cross-validate, compare, save best ───────────────
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ―― Random Forest ――――――――――――――――――――――――――――――――――――――――――――――――――
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)

rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
rf_cv_mean   = rf_cv_scores.mean()

print(f"  Test Accuracy      : {rf_test_acc:.4f}")
print(f"  5-Fold CV (mean)   : {rf_cv_mean:.4f}")
print(f"  5-Fold CV (std)    : {rf_cv_scores.std():.4f}")
print("\n  Classification Report (RF):")
print(classification_report(y_test, rf.predict(X_test)))

rf_importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("  Top 5 Features (RF):")
print(rf_importances.head(5).to_string())

# ―― XGBoost ――――――――――――――――――――――――――――――――――――――――――――――――――
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)

xgb_test_acc  = accuracy_score(y_test, xgb.predict(X_test))
xgb_cv_scores = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy')
xgb_cv_mean   = xgb_cv_scores.mean()

print(f"  Test Accuracy      : {xgb_test_acc:.4f}")
print(f"  5-Fold CV (mean)   : {xgb_cv_mean:.4f}")
print(f"  5-Fold CV (std)    : {xgb_cv_scores.std():.4f}")
print("\n  Classification Report (XGBoost):")
print(classification_report(y_test, xgb.predict(X_test)))

# ―― Pick the winner ―――――――――――――――――――――――――――――――――――――――――――――――――
print("\n" + "="*55)
print("  MODEL COMPARISON SUMMARY")
print("="*55)
print(f"  {'Model':<20} {'Test Acc':>10}  {'CV Mean':>10}")
print(f"  {'-'*44}")
print(f"  {'Random Forest':<20} {rf_test_acc:>10.4f}  {rf_cv_mean:>10.4f}")
print(f"  {'XGBoost':<20} {xgb_test_acc:>10.4f}  {xgb_cv_mean:>10.4f}")
print("="*55)

if xgb_cv_mean >= rf_cv_mean:
    best_model, best_name, best_cv = xgb, 'XGBoost',       xgb_cv_mean
else:
    best_model, best_name, best_cv = rf,  'RandomForest',  rf_cv_mean

print(f"\n🏆 Best model: {best_name} (CV={best_cv:.4f})")

# ―― Save artifacts ―――――――――――――――――――――――――――――――――――――――――――――――――
model_path    = os.path.join(models_dir, 'ipl_rf_model.joblib')
features_path = os.path.join(models_dir, 'feature_columns.json')
meta_path     = os.path.join(models_dir, 'model_metadata.json')

joblib.dump(best_model, model_path)
print(f"\n✅ Model saved     → {model_path}")

with open(features_path, 'w') as f:
    json.dump(features, f, indent=2)
print(f"✅ Features saved  → {features_path}")

meta = {
    'model_name':      best_name,
    'cv_accuracy':     round(best_cv, 4),
    'test_accuracy':   round(xgb_test_acc if best_name == 'XGBoost' else rf_test_acc, 4),
    'rf_cv_mean':      round(rf_cv_mean, 4),
    'xgb_cv_mean':     round(xgb_cv_mean, 4),
    'n_features':      len(features),
    'trained_at':      datetime.now(timezone.utc).isoformat()
}
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"✅ Metadata saved  → {meta_path}")

print("\n🏆 Success! Feature engineering, model comparison, and persistence completed.")
