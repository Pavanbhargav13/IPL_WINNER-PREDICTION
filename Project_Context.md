# Project Context

## What This Project Is

IPL Winner Prediction is a machine learning web application that predicts the probability of each IPL team winning a season. The model is trained on historical batting stats, bowling stats, match summaries, and points table data spanning 2008 to 2025. The frontend transforms these predictions into an interactive, cinematic experience.

This document is the single source of truth for anyone building or extending this project. Read this before touching any code.

---

## Author

**Pavan Bhargav MN**
CSE (AI & ML), Semester VI, P.E.S. College of Engineering

---

## Problem Statement

Predict the IPL season winner using aggregated team-level features derived from player batting and bowling statistics. The model outputs a probability score per team — not a binary winner prediction — so users can see how likely each team is relative to others.

---

## Data

### Real Data
- Batting card data (player-level, per season)
- Bowling card data (player-level, per season)
- Match summaries (result, runs, wickets, venue, toss)
- Points table (played, won, lost, NRR, points)
- Seasons covered: 2008 to 2023

### Synthetic Data (2024 and 2025)
Generated programmatically because structured public datasets were unavailable for these seasons. The generation methodology:

- Team strength priors set from historical win rates (MI: 0.80, CSK: 0.78, RCB: 0.72, etc.)
- 2024 winner: KKR — encoded as +0.08 strength boost in simulation
- 2025 winner: **RCB** — encoded as +0.08 strength boost (**corrected from MI**)
- Batting stats drawn from Normal distribution scaled by team strength
- Bowling stats (economy, wickets) drawn from Normal/Poisson distributions
- Match outcomes determined by Bernoulli trial with home advantage factored in

### Synthetic CSV Files
| File | Rows | Description |
|------|------|-------------|
| `batting_2024_2025_synthetic.csv` | 220 | Player batting stats, both seasons |
| `bowling_2024_2025_synthetic.csv` | 140 | Player bowling stats, both seasons |
| `match_summary_2024_2025_synthetic.csv` | 120 | Match results with NRR |
| `points_table_2024_2025_synthetic.csv` | 20 | Season points table |
| `venue_metadata.csv` | 12 | Static ground metadata |
| `match_venue_2024_2025.csv` | 480+ | Match data with venue feature columns |
| `venue_season_features_2024_2025.csv` | 20 | Season-level venue-adjusted features |

---

## Model

- **Type:** Supervised classification (scikit-learn)
- **Output:** predict_proba() — probability array per team, sums to ~1.0
- **Validation:** Season-wise hold-out (train on 2008–2022, validate on 2023)
- **Serialisation:** pickle — `ipl_model.pkl` and `scaler.pkl`
- **Planned upgrade:** XGBoost + SHAP explainability

### IPL 2026 Predictions (current model output)
| Team | Win Probability |
|------|----------------|
| CSK  | 42.1% |
| GT   | 27.5% |
| MI   | 19.2% |
| LSG  | 4.4%  |
| RCB  | 2.6%  |
| DC   | 1.7%  |
| RR   | 1.1%  |
| KKR  | 0.6%  |
| SRH  | 0.6%  |
| PBKS | 0.2%  |

---

## Venue & Environment Features

The current model is blind to venue. A batting average of 28 at Wankhede (small flat ground) looks identical to 28 at Chepauk (slow turner). Venue-adjusted features are being added in the next model iteration.

### 12 New Feature Columns
- `home_win_rate` — wins at home / total home matches
- `home_away_delta` — home_win_rate minus away_win_rate
- `venue_adj_batting_avg` — raw avg divided by (venue avg / 170)
- `venue_adj_economy` — raw economy multiplied by (venue avg / 170)
- `chase_win_pct_avg` — weighted average chase win % across team venues
- `spin_exposure_avg` — average spin wicket % across venues played
- `ground_size_score` — S=1, M=2, L=3, averaged across venues
- `coastal_match_pct` — % matches at coastal venues
- `altitude_score` — average altitude across venues
- `toss_win_rate` — toss wins per season
- `late_season_match_pct` — % matches in May
- `late_season_win_rate` — win rate in May matches only

---

## IPL Winner History (Corrected)

| Year | Winner |
|------|--------|
| 2008 | Rajasthan Royals |
| 2009 | Deccan Chargers |
| 2010 | Chennai Super Kings |
| 2011 | Chennai Super Kings |
| 2012 | Kolkata Knight Riders |
| 2013 | Mumbai Indians |
| 2014 | Kolkata Knight Riders |
| 2015 | Mumbai Indians |
| 2016 | Sunrisers Hyderabad |
| 2017 | Mumbai Indians |
| 2018 | Chennai Super Kings |
| 2019 | Mumbai Indians |
| 2020 | Mumbai Indians |
| 2021 | Chennai Super Kings |
| 2022 | Gujarat Titans |
| 2023 | Chennai Super Kings |
| 2024 | Kolkata Knight Riders |
| 2025 | **Royal Challengers Bengaluru** ← corrected |

---

## Design Philosophy

Dark-first UI. Amber (#f4a623) as the single primary accent. Monospace numbers everywhere. Data density without clutter. Progressive disclosure — summary on load, detail on interaction.

Fonts: **Syne** for all display/UI text, **JetBrains Mono** for all numeric data.

---

## Deployment Targets

- Backend: Render (free tier) — FastAPI app
- Frontend: Vercel (free tier) — React + Vite app
- Automation: n8n — self-hosted or cloud, existing instance

---

## Related Documents

- `Architecture.md` — system design, data flow, component tree
- `Coding_Rules.md` — code style, naming conventions, component patterns
- `Feature_Log.md` — all features, status, owner, notes
