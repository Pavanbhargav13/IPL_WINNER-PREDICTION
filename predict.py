"""
predict.py — IPL Match Winner Predictor
========================================
Usage:
    python predict.py --home MI --away CSK --venue "Wankhede Stadium"

Or run interactively (no args):
    python predict.py
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(ROOT, 'models')
DATA_DIR    = os.path.join(ROOT, 'data')

MODEL_PATH    = os.path.join(MODELS_DIR, 'ipl_rf_model.joblib')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_columns.json')
VENUE_STATS   = os.path.join(DATA_DIR,   'venue_stats.csv')
VENUE_META    = os.path.join(DATA_DIR,   'venue_metadata.csv')
TEAM_FEATURES = os.path.join(DATA_DIR,   'enhanced_team_season_features.csv')

# ── Known IPL team abbreviations ───────────────────────────────────────────────
TEAM_ALIASES = {
    'MI':   ['MI', 'Mumbai Indians', 'Mumbai'],
    'CSK':  ['CSK', 'Chennai Super Kings', 'Chennai'],
    'RCB':  ['RCB', 'Royal Challengers Bangalore', 'Royal Challengers Bengaluru', 'Bangalore', 'Bengaluru'],
    'KKR':  ['KKR', 'Kolkata Knight Riders', 'Kolkata'],
    'SRH':  ['SRH', 'Sunrisers Hyderabad', 'Hyderabad'],
    'DC':   ['DC', 'Delhi Capitals', 'Delhi', 'Delhi Daredevils', 'DD'],
    'RR':   ['RR', 'Rajasthan Royals', 'Rajasthan'],
    'PBKS': ['PBKS', 'Punjab Kings', 'Punjab', 'Kings XI Punjab', 'KXIP'],
    'GT':   ['GT', 'Gujarat Titans', 'Gujarat'],
    'LSG':  ['LSG', 'Lucknow Super Giants', 'Lucknow'],
}

def normalise_team(name: str) -> str:
    """Return the canonical team abbreviation for a given name."""
    name_stripped = name.strip()
    for abbr, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if alias.lower() == name_stripped.lower():
                return abbr
    return name_stripped  # return as-is and let validation catch it


def load_artifacts():
    """Load model, feature list, and lookup tables."""
    for path, label in [(MODEL_PATH, 'ipl_rf_model.joblib'),
                        (FEATURES_PATH, 'feature_columns.json'),
                        (VENUE_STATS,   'venue_stats.csv'),
                        (VENUE_META,    'venue_metadata.csv'),
                        (TEAM_FEATURES, 'enhanced_team_season_features.csv')]:
        if not os.path.exists(path):
            sys.exit(f"❌  Missing required file: {label}\n"
                     f"   Run the training script first:\n"
                     f"   python Model_Training/feature_engineering_and_training.py")

    model        = joblib.load(MODEL_PATH)
    features     = json.load(open(FEATURES_PATH))
    venue_stats  = pd.read_csv(VENUE_STATS)
    venue_meta   = pd.read_csv(VENUE_META)
    team_feats   = pd.read_csv(TEAM_FEATURES)
    return model, features, venue_stats, venue_meta, team_feats


def find_venue(name: str, venue_stats: pd.DataFrame, venue_meta: pd.DataFrame):
    """Fuzzy-match a venue name from the known venues."""
    name_lower = name.lower().strip()

    # Exact match first
    vs_match = venue_stats[venue_stats['venue_name'].str.lower() == name_lower]
    if not vs_match.empty:
        return vs_match.iloc[0], venue_meta[venue_meta['venue_name'].str.lower() == name_lower].iloc[0]

    # Partial match
    vs_match = venue_stats[venue_stats['venue_name'].str.lower().str.contains(name_lower, na=False)]
    if not vs_match.empty:
        matched_name = vs_match.iloc[0]['venue_name']
        vm_match = venue_meta[venue_meta['venue_name'] == matched_name]
        return vs_match.iloc[0], vm_match.iloc[0] if not vm_match.empty else pd.Series(dtype=object)

    return None, None


def get_team_features(team: str, team_feats: pd.DataFrame) -> pd.Series:
    """Return the most recent season's features for a team."""
    tf = team_feats[team_feats['team'] == team]
    if tf.empty:
        # Return zeros if team not found (new team or name mismatch)
        return pd.Series({'home_win_rate': 0, 'toss_win_rate': 0,
                          'bat_first_decision_rate': 0, 'late_season_match_pct': 0,
                          'team_avg_score': 0})
    return tf.sort_values('season').iloc[-1]  # latest season


def predict_winner(home_team: str, away_team: str, venue_name: str):
    """Core prediction function. Returns a dict with win probabilities."""
    model, features, venue_stats, venue_meta, team_feats = load_artifacts()

    # Normalise team names
    home_abbr = normalise_team(home_team)
    away_abbr = normalise_team(away_team)

    # Validate teams
    known_teams = list(TEAM_ALIASES.keys())
    for abbr, original in [(home_abbr, home_team), (away_abbr, away_team)]:
        if abbr not in known_teams:
            print(f"⚠️  '{original}' not in known teams. Known teams: {', '.join(known_teams)}")

    # Venue lookup
    v_stats, v_meta = find_venue(venue_name, venue_stats, venue_meta)
    if v_stats is None:
        print(f"⚠️  Venue '{venue_name}' not found. Using average venue stats.")
        v_stats = venue_stats.mean(numeric_only=True)
        v_meta  = pd.Series({'ground_size': 'M', 'coastal': 'N'})

    # Team feature lookup
    home_tf = get_team_features(home_abbr, team_feats)
    away_tf = get_team_features(away_abbr, team_feats)

    # Ground size encoding
    gs_map   = {'S': 1, 'M': 2, 'L': 3}
    avg_score = v_stats.get('avg_first_innings_score', 160) or 160

    row = {
        'avg_first_innings_score':    v_stats.get('avg_first_innings_score', 160),
        'chase_win_pct':              v_stats.get('chase_win_pct', 0.5),
        'avg_wickets_per_match':      v_stats.get('avg_wickets_per_match', 12),
        'spin_wicket_pct_proxy':      v_stats.get('spin_wicket_pct_proxy', 0.45),
        'ground_size_encode':         gs_map.get(v_meta.get('ground_size', 'M'), 2),
        'coastal_encode':             1 if v_meta.get('coastal', 'N') == 'Y' else 0,

        'home_home_win_rate':         home_tf.get('home_win_rate', 0),
        'home_toss_win_rate':         home_tf.get('toss_win_rate', 0),
        'home_bat_first_decision_rate': home_tf.get('bat_first_decision_rate', 0),
        'home_late_season_match_pct': home_tf.get('late_season_match_pct', 0),
        'home_venue_adjusted_batting': (home_tf.get('team_avg_score', 0) or 0) / avg_score,

        'away_home_win_rate':         away_tf.get('home_win_rate', 0),
        'away_toss_win_rate':         away_tf.get('toss_win_rate', 0),
        'away_bat_first_decision_rate': away_tf.get('bat_first_decision_rate', 0),
        'away_late_season_match_pct': away_tf.get('late_season_match_pct', 0),
        'away_venue_adjusted_batting': (away_tf.get('team_avg_score', 0) or 0) / avg_score,
    }

    X    = pd.DataFrame([row])[features]
    prob = model.predict_proba(X)[0]

    return {
        'home_team':      home_abbr,
        'away_team':      away_abbr,
        'venue':          v_stats.get('venue_name', venue_name),
        'home_win_prob':  round(float(prob[1]) * 100, 1),
        'away_win_prob':  round(float(prob[0]) * 100, 1),
    }


def print_result(result: dict):
    """Pretty-print the prediction result."""
    home  = result['home_team']
    away  = result['away_team']
    h_pct = result['home_win_prob']
    a_pct = result['away_win_prob']
    winner = home if h_pct >= a_pct else away

    bar_width = 40
    home_bar  = int(bar_width * h_pct / 100)
    away_bar  = bar_width - home_bar

    print("\n" + "═" * 55)
    print(f"  🏏  IPL MATCH PREDICTION")
    print("═" * 55)
    print(f"  📍 Venue : {result['venue']}")
    print(f"  🏠 Home  : {home}  vs  🚌 Away : {away}")
    print("─" * 55)
    print(f"  {home:<6}  {'█' * home_bar}{'░' * away_bar}  {away}")
    print(f"  {h_pct:>5.1f}%  {'':^{bar_width}}  {a_pct:.1f}%")
    print("─" * 55)
    print(f"  🏆  Predicted Winner : {winner}")
    print("═" * 55 + "\n")


def interactive_mode():
    """Prompt user for inputs if no CLI args are given."""
    print("\n🏏  IPL Match Predictor  (type 'quit' to exit)\n")

    # Show available teams
    print("Available teams:", ', '.join(TEAM_ALIASES.keys()))
    print()

    while True:
        home  = input("Enter HOME team  : ").strip()
        if home.lower() == 'quit':
            break
        away  = input("Enter AWAY team  : ").strip()
        venue = input("Enter venue name : ").strip()

        result = predict_winner(home, away, venue)
        print_result(result)

        again = input("Predict another? (y/n): ").strip().lower()
        if again != 'y':
            break


def main():
    parser = argparse.ArgumentParser(description="IPL Match Winner Predictor")
    parser.add_argument('--home',  type=str, help='Home team name or abbreviation')
    parser.add_argument('--away',  type=str, help='Away team name or abbreviation')
    parser.add_argument('--venue', type=str, help='Venue name (partial match supported)')
    args = parser.parse_args()

    if args.home and args.away and args.venue:
        result = predict_winner(args.home, args.away, args.venue)
        print_result(result)
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
