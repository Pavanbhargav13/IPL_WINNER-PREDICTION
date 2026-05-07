# 🏏 IPL Winner Prediction using Machine Learning

> Predict IPL match outcomes using historical team performance, venue analytics, and toss intelligence.

---

## 📌 Project Overview

Every IPL season raises the same question — *which team will win?*  
This project uses machine learning to predict the **winner of an IPL match** based on historical team-level and venue-level features.

---

## 🎯 Problem Statement

Given two IPL teams and a venue, predict which team is more likely to win — using aggregated features derived from player statistics, venue history, toss decisions, and home advantage.

---

## 📊 Data Used

| File | Description |
|---|---|
| `all_season_summary.csv` | Match-level results, toss info, venues |
| `all_season_batting_card.csv` | Player batting stats per match |
| `all_season_bowling_card.csv` | Player bowling stats per match |
| `points_table.csv` | Season standings |

Data spans **multiple IPL seasons** (2022–2024+).

---

## 🧪 Feature Engineering

Features are split into **venue-level** and **team-level** categories:

### 🏟️ Venue Features
- `avg_first_innings_score` — historical average 1st innings total at the venue
- `chase_win_pct` — % of matches won by chasing team at this venue
- `avg_wickets_per_match` — how bowling-friendly the pitch is
- `spin_wicket_pct_proxy` — synthetic proxy for spin-friendliness
- `ground_size_encode` — Small / Medium / Large (1/2/3)
- `coastal_encode` — whether the venue is coastal (affects dew)

### 👥 Team Features (home & away)
- `home_win_rate` — season-level win rate at home ground
- `toss_win_rate` — how often the team wins the toss
- `bat_first_decision_rate` — tendency to bat first after winning toss
- `late_season_match_pct` — proportion of matches played in May/June (knockouts)
- `venue_adjusted_batting` — team's average score normalised by venue difficulty

---

## 🧠 Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Train/Test Split | 80% / 20% |
| Target | `1` if Home Team wins, `0` if Away Team wins |
| Saved as | `models/ipl_rf_model.joblib` |

---

## 📁 Project Structure

```
IPL_WINNER-PREDICTION-main/
├── data/
│   ├── all_season_summary.csv
│   ├── all_season_batting_card.csv
│   ├── all_season_bowling_card.csv
│   ├── venue_metadata.csv          # ✅ auto-generated
│   └── enhanced_team_season_features.csv  # ✅ auto-generated
├── models/
│   ├── ipl_rf_model.joblib         # ✅ trained model (auto-saved)
│   └── feature_columns.json        # ✅ feature order (auto-saved)
├── Model_Training/
│   └── feature_engineering_and_training.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training pipeline
python Model_Training/feature_engineering_and_training.py
```

This will:
- Engineer all features from raw CSVs
- Train a Random Forest classifier
- Save the model to `models/ipl_rf_model.joblib`
- Save the feature list to `models/feature_columns.json`

---

## 🛠️ Technologies Used

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Joblib

---

## 🗺️ Roadmap

- [x] Data loading & preprocessing
- [x] Venue metadata engineering
- [x] Team-level feature engineering
- [x] Random Forest model training
- [x] Model persistence (save/load)
- [ ] Prediction script (input teams → output winner probability)
- [ ] Model improvement (XGBoost, cross-validation, form features)
- [ ] Web UI for live predictions

---

## 👤 Author

**PAVAN BHARGAV MN**
