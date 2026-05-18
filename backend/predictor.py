"""
predictor.py — Core prediction logic, loaded once at startup.
Wraps the existing predict.py logic into a class for FastAPI use.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT, "models")
DATA_DIR    = os.path.join(ROOT, "data")

MODEL_PATH    = os.path.join(MODELS_DIR, "ipl_rf_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
META_PATH     = os.path.join(MODELS_DIR, "model_metadata.json")
VENUE_STATS   = os.path.join(DATA_DIR,   "venue_stats.csv")
VENUE_META    = os.path.join(DATA_DIR,   "venue_metadata.csv")
TEAM_FEATURES = os.path.join(DATA_DIR,   "enhanced_team_season_features.csv")

# ── Team metadata ──────────────────────────────────────────────────────────────
TEAM_ALIASES = {
    "MI":   ["MI", "Mumbai Indians", "Mumbai"],
    "CSK":  ["CSK", "Chennai Super Kings", "Chennai"],
    "RCB":  ["RCB", "Royal Challengers Bangalore", "Royal Challengers Bengaluru", "Bangalore", "Bengaluru"],
    "KKR":  ["KKR", "Kolkata Knight Riders", "Kolkata"],
    "SRH":  ["SRH", "Sunrisers Hyderabad", "Hyderabad"],
    "DC":   ["DC", "Delhi Capitals", "Delhi", "Delhi Daredevils", "DD"],
    "RR":   ["RR", "Rajasthan Royals", "Rajasthan"],
    "PBKS": ["PBKS", "Punjab Kings", "Punjab", "Kings XI Punjab", "KXIP"],
    "GT":   ["GT", "Gujarat Titans", "Gujarat"],
    "LSG":  ["LSG", "Lucknow Super Giants", "Lucknow"],
}

TEAM_FULL_NAMES = {
    "MI":   "Mumbai Indians",
    "CSK":  "Chennai Super Kings",
    "RCB":  "Royal Challengers Bengaluru",
    "KKR":  "Kolkata Knight Riders",
    "SRH":  "Sunrisers Hyderabad",
    "DC":   "Delhi Capitals",
    "RR":   "Rajasthan Royals",
    "PBKS": "Punjab Kings",
    "GT":   "Gujarat Titans",
    "LSG":  "Lucknow Super Giants",
}

TEAM_COLORS = {
    "MI":   "#004BA0",
    "CSK":  "#F9CD05",
    "RCB":  "#EC1C24",
    "KKR":  "#3A225D",
    "SRH":  "#F7A721",
    "DC":   "#0078BC",
    "RR":   "#EA1A85",
    "PBKS": "#AA4545",
    "GT":   "#1C1C1C",
    "LSG":  "#A2FFBE",
}

# ── Venue intelligence ─────────────────────────────────────────────────────────
VENUE_INTELLIGENCE = {
    "wankhede":         {"pitch_type": "Pace/Batting",  "toss_advice": "Chase — dew factor in evenings", "description": "High-scoring coastal ground. Dew plays a big role in evening matches."},
    "chepauk":          {"pitch_type": "Spin",           "toss_advice": "Bat first — pitch deteriorates", "description": "Classic spin track. Pitch slows down significantly after 10 overs."},
    "chidambaram":      {"pitch_type": "Spin",           "toss_advice": "Bat first — pitch deteriorates", "description": "Classic spin track. Pitch slows down significantly after 10 overs."},
    "chinnaswamy":      {"pitch_type": "Batting",        "toss_advice": "Chase — small ground, par scores high", "description": "Smallest effective boundary. 180+ is the norm here."},
    "eden gardens":     {"pitch_type": "Balanced",       "toss_advice": "Chase — dew on humid nights", "description": "Iconic ground. Good for both pace and spin. Dew can be a factor."},
    "narendra modi":    {"pitch_type": "Pace",           "toss_advice": "Bat first — pitch good early", "description": "World's largest cricket stadium. Bouncy track early on."},
    "motera":           {"pitch_type": "Pace",           "toss_advice": "Bat first — pitch good early", "description": "World's largest cricket stadium. Bouncy track early on."},
    "arun jaitley":     {"pitch_type": "Batting",        "toss_advice": "Chase — flat track, high scores", "description": "Flat deck. Teams regularly post 180+. Spinners get little help."},
    "feroz shah kotla": {"pitch_type": "Batting",        "toss_advice": "Chase — flat track, high scores", "description": "Flat deck. Teams regularly post 180+."},
    "sawai mansingh":   {"pitch_type": "Batting",        "toss_advice": "Bat first — smaller ground", "description": "Shorter boundaries in Jaipur. Explosive batting conditions."},
    "rajiv gandhi":     {"pitch_type": "Batting",        "toss_advice": "Chase — coastal evening dew", "description": "Good batting surface in Hyderabad with coastal influence."},
    "uppal":            {"pitch_type": "Batting",        "toss_advice": "Chase — coastal evening dew", "description": "Good batting surface in Hyderabad with coastal influence."},
    "ekana":            {"pitch_type": "Balanced",       "toss_advice": "Bat first — slight pitch assistance early", "description": "LSG's home. Balanced track that rewards both batters and bowlers."},
    "lucknow":          {"pitch_type": "Balanced",       "toss_advice": "Bat first — slight pitch assistance early", "description": "LSG's home. Balanced track."},
    "punjab cricket":   {"pitch_type": "Pace",           "toss_advice": "Bat first — seam movement early", "description": "Mohali's seam-friendly track. Pace bowlers dominate first 6 overs."},
    "mohali":           {"pitch_type": "Pace",           "toss_advice": "Bat first — seam movement early", "description": "Mohali's seam-friendly track."},
    "bindra":           {"pitch_type": "Pace",           "toss_advice": "Bat first — seam movement early", "description": "Mohali's seam-friendly track."},
    "dy patil":         {"pitch_type": "Batting",        "toss_advice": "Chase — neutral venue, high-scoring", "description": "Neutral venue. Very flat track — perfect for batting."},
    "brabourne":        {"pitch_type": "Batting",        "toss_advice": "Chase — dew factor at night", "description": "Neutral Mumbai venue. Coastal dew in evening games."},
}

DEFAULT_VENUE_INTEL = {
    "pitch_type":    "Balanced",
    "toss_advice":   "Toss-up — conditions unclear",
    "description":   "Neutral conditions. Match-up between teams will decide the winner.",
}


class IPLPredictor:
    """Singleton-style predictor class loaded once at FastAPI startup."""

    def __init__(self):
        self.model       = joblib.load(MODEL_PATH)
        self.features    = json.load(open(FEATURES_PATH))
        self.venue_stats = pd.read_csv(VENUE_STATS)
        self.venue_meta  = pd.read_csv(VENUE_META)
        self.team_feats  = pd.read_csv(TEAM_FEATURES)

        # Load model metadata
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                self.meta = json.load(f)
        else:
            self.meta = {"model_name": "Unknown", "cv_accuracy": 0.0, "trained_at": "Unknown"}

        # Try to get feature importances for SHAP-like values
        try:
            self.feature_importances = dict(
                zip(self.features, self.model.feature_importances_)
            )
        except AttributeError:
            self.feature_importances = {f: 1 / len(self.features) for f in self.features}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def normalise_team(self, name: str) -> str:
        name_s = name.strip()
        for abbr, aliases in TEAM_ALIASES.items():
            for alias in aliases:
                if alias.lower() == name_s.lower():
                    return abbr
        return name_s

    def find_venue(self, name: str) -> Tuple[pd.Series, pd.Series]:
        name_lower = name.lower().strip()
        vs = self.venue_stats
        vm = self.venue_meta

        exact = vs[vs["venue_name"].str.lower() == name_lower]
        if not exact.empty:
            n = exact.iloc[0]["venue_name"]
            vm_m = vm[vm["venue_name"] == n]
            return exact.iloc[0], vm_m.iloc[0] if not vm_m.empty else pd.Series(dtype=object)

        partial = vs[vs["venue_name"].str.lower().str.contains(name_lower, na=False)]
        if not partial.empty:
            n = partial.iloc[0]["venue_name"]
            vm_m = vm[vm["venue_name"] == n]
            return partial.iloc[0], vm_m.iloc[0] if not vm_m.empty else pd.Series(dtype=object)

        return pd.Series(dtype=object), pd.Series(dtype=object)

    def get_team_features(self, team: str) -> pd.Series:
        tf = self.team_feats[self.team_feats["team"] == team]
        if tf.empty:
            return pd.Series({
                "home_win_rate": 0.0, "toss_win_rate": 0.5,
                "bat_first_decision_rate": 0.5, "late_season_match_pct": 0.2,
                "team_avg_score": 160.0,
            })
        return tf.sort_values("season").iloc[-1]

    def build_feature_row(self, home_abbr: str, away_abbr: str,
                          v_stats: pd.Series, v_meta: pd.Series) -> dict:
        gs_map   = {"S": 1, "M": 2, "L": 3}
        avg_score = float(v_stats.get("avg_first_innings_score", 160) or 160)
        home_tf  = self.get_team_features(home_abbr)
        away_tf  = self.get_team_features(away_abbr)

        return {
            "avg_first_innings_score":        avg_score,
            "chase_win_pct":                  float(v_stats.get("chase_win_pct", 0.5) or 0.5),
            "avg_wickets_per_match":           float(v_stats.get("avg_wickets_per_match", 12) or 12),
            "spin_wicket_pct_proxy":           float(v_stats.get("spin_wicket_pct_proxy", 0.45) or 0.45),
            "ground_size_encode":              gs_map.get(str(v_meta.get("ground_size", "M")), 2),
            "coastal_encode":                  1 if str(v_meta.get("coastal", "N")) == "Y" else 0,

            "home_home_win_rate":              float(home_tf.get("home_win_rate", 0) or 0),
            "home_toss_win_rate":              float(home_tf.get("toss_win_rate", 0) or 0),
            "home_bat_first_decision_rate":    float(home_tf.get("bat_first_decision_rate", 0) or 0),
            "home_late_season_match_pct":      float(home_tf.get("late_season_match_pct", 0) or 0),
            "home_venue_adjusted_batting":     float(home_tf.get("team_avg_score", 0) or 0) / avg_score,

            "away_home_win_rate":              float(away_tf.get("home_win_rate", 0) or 0),
            "away_toss_win_rate":              float(away_tf.get("toss_win_rate", 0) or 0),
            "away_bat_first_decision_rate":    float(away_tf.get("bat_first_decision_rate", 0) or 0),
            "away_late_season_match_pct":      float(away_tf.get("late_season_match_pct", 0) or 0),
            "away_venue_adjusted_batting":     float(away_tf.get("team_avg_score", 0) or 0) / avg_score,
        }

    def get_shap_approximation(self, row: dict, home_prob: float) -> dict:
        """Approximate feature contributions using feature importances × feature values."""
        shap_vals = {}
        total_importance = sum(self.feature_importances.values())
        for feat, imp in self.feature_importances.items():
            val = row.get(feat, 0)
            # Scale contribution by both importance and probability
            contribution = round((imp / total_importance) * home_prob * 100, 2)
            shap_vals[feat] = contribution
        return shap_vals

    def get_venue_intelligence(self, venue_name: str) -> dict:
        vl = venue_name.lower()
        for key, intel in VENUE_INTELLIGENCE.items():
            if key in vl:
                return intel
        return DEFAULT_VENUE_INTEL

    def assign_badge(self, prob: float) -> str:
        if prob >= 65:
            return "🔥 Hot Favourite"
        elif prob >= 55:
            return "📈 Slight Edge"
        elif prob >= 45:
            return "⚖️ Even Match"
        elif prob >= 35:
            return "📉 Underdog"
        else:
            return "💀 Long Shot"

    # ── Core predictions ───────────────────────────────────────────────────────

    def predict_h2h(self, home_team: str, away_team: str, venue_name: str) -> dict:
        home_abbr = self.normalise_team(home_team)
        away_abbr = self.normalise_team(away_team)

        v_stats, v_meta = self.find_venue(venue_name)

        if v_stats.empty:
            v_stats = self.venue_stats.mean(numeric_only=True)
            v_stats["venue_name"] = venue_name
            v_meta  = pd.Series({"ground_size": "M", "coastal": "N"})

        row  = self.build_feature_row(home_abbr, away_abbr, v_stats, v_meta)
        X    = pd.DataFrame([row])[self.features]
        prob = self.model.predict_proba(X)[0]

        home_prob = round(float(prob[1]) * 100, 1)
        away_prob = round(float(prob[0]) * 100, 1)
        winner    = home_abbr if home_prob >= away_prob else away_abbr

        intel = self.get_venue_intelligence(str(v_stats.get("venue_name", venue_name)))

        # Home advantage modifier (typical +6-8%)
        home_adv = float(row.get("home_home_win_rate", 0))
        away_adv = float(row.get("away_home_win_rate", 0))
        modifier = round((home_adv - away_adv) * 100, 1)

        return {
            "home_team":              home_abbr,
            "away_team":              away_abbr,
            "home_full":              TEAM_FULL_NAMES.get(home_abbr, home_abbr),
            "away_full":              TEAM_FULL_NAMES.get(away_abbr, away_abbr),
            "venue":                  str(v_stats.get("venue_name", venue_name)),
            "home_win_prob":          home_prob,
            "away_win_prob":          away_prob,
            "winner":                 winner,
            "pitch_type":             intel["pitch_type"],
            "avg_first_innings_score": float(row["avg_first_innings_score"]),
            "chase_win_pct":          round(float(row["chase_win_pct"]) * 100, 1),
            "ground_size":            str(v_meta.get("ground_size", "M")),
            "coastal":                str(v_meta.get("coastal", "N")) == "Y",
            "home_advantage_modifier": modifier,
            "toss_advice":            intel["toss_advice"],
            "venue_description":      intel["description"],
        }

    def predict_season(self) -> dict:
        """Predict win probabilities for all 10 teams against a balanced field."""
        teams   = list(TEAM_ALIASES.keys())
        results = []

        for team in teams:
            opponents = [t for t in teams if t != team]
            probs = []
            for opp in opponents:
                # Use a neutral venue (average stats) for season predictions
                avg_vstats = self.venue_stats.mean(numeric_only=True)
                avg_vstats["venue_name"] = "Neutral"
                avg_vmeta  = pd.Series({"ground_size": "M", "coastal": "N"})
                row  = self.build_feature_row(team, opp, avg_vstats, avg_vmeta)
                X    = pd.DataFrame([row])[self.features]
                prob = self.model.predict_proba(X)[0]
                probs.append(float(prob[1]))  # prob of team winning as home

            avg_prob  = round(np.mean(probs) * 100, 1)
            team_tf   = self.get_team_features(team)
            shap_vals = self.get_shap_approximation(
                self.build_feature_row(team, "CSK", self.venue_stats.mean(numeric_only=True), pd.Series({"ground_size": "M", "coastal": "N"})),
                avg_prob / 100
            )

            results.append({
                "team":             team,
                "team_full_name":   TEAM_FULL_NAMES.get(team, team),
                "team_color":       TEAM_COLORS.get(team, "#888888"),
                "win_probability":  avg_prob,
                "implied_odds":     round(100 / avg_prob, 2) if avg_prob > 0 else 99.0,
                "badge":            self.assign_badge(avg_prob),
                "home_win_rate":    round(float(team_tf.get("home_win_rate", 0)) * 100, 1),
                "toss_win_rate":    round(float(team_tf.get("toss_win_rate", 0)) * 100, 1),
                "avg_score":        round(float(team_tf.get("team_avg_score", 0)), 1),
                "shap_values":      shap_vals,
            })

        # Sort by win probability descending, add rank
        results.sort(key=lambda x: x["win_probability"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "predictions": results,
            "model_name":  self.meta.get("model_name", "Unknown"),
            "model_accuracy": self.meta.get("cv_accuracy", 0.0),
            "trained_at":  self.meta.get("trained_at", "Unknown"),
        }

    def predict_whatif(self, home_team: str, away_team: str, venue_name: str,
                       overrides: dict) -> dict:
        """Predict with custom stat overrides from sliders."""
        home_abbr = self.normalise_team(home_team)
        away_abbr = self.normalise_team(away_team)

        v_stats, v_meta = self.find_venue(venue_name)
        if v_stats.empty:
            v_stats = self.venue_stats.mean(numeric_only=True)
            v_stats["venue_name"] = venue_name
            v_meta  = pd.Series({"ground_size": "M", "coastal": "N"})

        # Baseline
        baseline_row = self.build_feature_row(home_abbr, away_abbr, v_stats, v_meta)
        X_base       = pd.DataFrame([baseline_row])[self.features]
        base_prob    = self.model.predict_proba(X_base)[0]
        base_home    = float(base_prob[1]) * 100

        # Apply overrides
        modified_row = dict(baseline_row)
        avg_score    = float(v_stats.get("avg_first_innings_score", 160) or 160)

        if overrides.get("home_batting_avg") is not None:
            modified_row["home_venue_adjusted_batting"] = overrides["home_batting_avg"] / avg_score
        if overrides.get("home_nrr") is not None:
            nrr_boost = (overrides["home_nrr"] + 1.5) / 3.0  # normalise 0→1
            modified_row["home_home_win_rate"] = min(1.0, baseline_row["home_home_win_rate"] + nrr_boost * 0.1)
        if overrides.get("away_batting_avg") is not None:
            modified_row["away_venue_adjusted_batting"] = overrides["away_batting_avg"] / avg_score
        if overrides.get("away_nrr") is not None:
            nrr_boost = (overrides["away_nrr"] + 1.5) / 3.0
            modified_row["away_home_win_rate"] = min(1.0, baseline_row["away_home_win_rate"] + nrr_boost * 0.1)

        X_mod    = pd.DataFrame([modified_row])[self.features]
        mod_prob = self.model.predict_proba(X_mod)[0]
        mod_home = round(float(mod_prob[1]) * 100, 1)
        mod_away = round(float(mod_prob[0]) * 100, 1)

        return {
            "home_team":      home_abbr,
            "away_team":      away_abbr,
            "venue":          str(v_stats.get("venue_name", venue_name)),
            "home_win_prob":  mod_home,
            "away_win_prob":  mod_away,
            "winner":         home_abbr if mod_home >= mod_away else away_abbr,
            "delta_home":     round(mod_home - base_home, 1),
            "delta_away":     round(mod_away - (100 - base_home), 1),
        }

    def predict_dream11(self, home_team: str, away_team: str,
                         venue_name: str, selected_players: list) -> dict:
        """Generate a Dream11 squad-based win prediction with strategy tips."""
        home_abbr = self.normalise_team(home_team)
        away_abbr = self.normalise_team(away_team)

        result = self.predict_h2h(home_abbr, away_abbr, venue_name)

        home_prob = result["home_win_prob"]
        away_prob = result["away_win_prob"]
        pitch     = result["pitch_type"]
        venue     = result["venue"]

        # Build a narrative based on pitch and probabilities
        if pitch == "Spin":
            narrative = (f"The {venue} track is a classic spin-friendly surface. "
                         f"Your squad will benefit from picking extra spinners. "
                         f"Batters who play spin well are key.")
            tips = [
                "Pick 2-3 specialist spinners from your squad",
                "Avoid pace-heavy all-rounders",
                "Target batters with high spin-play average",
                "Consider a batting spinner as your utility pick",
            ]
        elif pitch == "Pace" or "Pace" in pitch:
            narrative = (f"Pacers rule at {venue}. Early wickets will set the tone. "
                         f"A pace-heavy squad gives you a strategic edge here.")
            tips = [
                "Pick your fastest bowlers as priority",
                "Batting powerplay is crucial — top-order batters are premium",
                "Avoid front-foot players who struggle against short-pitch",
                "A captain who is a batting all-rounder maximises points",
            ]
        else:
            narrative = (f"{venue} offers balanced conditions — both batting and bowling contribute. "
                         f"An all-round squad with mixed skills will perform well here.")
            tips = [
                "Balance your squad — 3 pace + 2 spin is ideal",
                "Pick the in-form captain regardless of specialty",
                "All-rounders are gold here — extra point opportunities",
                "Check recent form, not just career stats",
            ]

        # Captain tip based on pitch and winning team
        winner   = result["winner"]
        captain_tip = (
            f"Pick the {TEAM_FULL_NAMES.get(winner, winner)} captain or their leading batter as your Dream11 captain. "
            f"{'Spin-savvy' if pitch == 'Spin' else 'Explosive top-order'} players perform best at {venue}."
        )

        return {
            "home_team":         home_abbr,
            "away_team":         away_abbr,
            "venue":             venue,
            "win_probability":   home_prob if home_abbr == winner else away_prob,
            "predicted_winner":  winner,
            "strategy_narrative": narrative,
            "captain_tip":       captain_tip,
            "tips":              tips,
            "pitch_type":        pitch,
            "toss_advice":       result["toss_advice"],
        }
