import os
import json
import pandas as pd
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FRONTEND_DATA_DIR = os.path.join(ROOT, "frontend", "src", "data")

def extract_all_players():
    print("Reading batting and bowling cards...")
    bat_df = pd.read_csv(os.path.join(DATA_DIR, "all_season_batting_card.csv"))
    bowl_df = pd.read_csv(os.path.join(DATA_DIR, "all_season_bowling_card.csv"))

    print("Extracting unique players...")
    # Get player names and teams from batting
    batters = bat_df.groupby('fullName').agg({'current_innings': 'first'}).reset_index()
    batters.rename(columns={'current_innings': 'team'}, inplace=True)
    batters['is_batter'] = True

    # Get player names and teams from bowling
    bowlers = bowl_df.groupby('fullName').agg({'bowling_team': 'first'}).reset_index()
    bowlers.rename(columns={'bowling_team': 'team'}, inplace=True)
    bowlers['is_bowler'] = True

    # Merge
    players = pd.merge(batters, bowlers, on='fullName', how='outer', suffixes=('_bat', '_bowl'))
    
    # Resolve team (prefer bat team, fallback to bowl team)
    players['team'] = players['team_bat'].combine_first(players['team_bowl'])

    players_list = []
    
    random.seed(42) # For consistent credits
    
    for idx, row in players.iterrows():
        name = row['fullName']
        if pd.isna(name):
            continue
            
        team = row['team'] if not pd.isna(row['team']) else "UNK"
        
        # Determine role
        is_bat = row['is_batter'] == True
        is_bowl = row['is_bowler'] == True
        
        if is_bat and is_bowl:
            role = 'ALL'
        elif is_bowl:
            role = 'BOWL'
        elif is_bat:
            role = 'BAT'
        else:
            role = 'UNK'

        # Generate a random credit between 7.0 and 10.5
        credits = round(random.uniform(7.0, 10.5) * 2) / 2 # rounds to nearest 0.5
        
        players_list.append({
            "id": int(idx + 1),
            "name": name,
            "role": role,
            "team": team,
            "credits": credits
        })

    print(f"Extracted {len(players_list)} unique players.")

    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)
    out_path = os.path.join(FRONTEND_DATA_DIR, "players.json")
    
    with open(out_path, 'w') as f:
        json.dump(players_list, f, indent=2)
        
    print(f"Successfully wrote players to {out_path}")

if __name__ == "__main__":
    extract_all_players()
