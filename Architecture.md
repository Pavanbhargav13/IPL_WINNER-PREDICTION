# Architecture

## System Overview

```
User Browser
    │
    ▼
React + Vite (Vercel)
    │  fetch / POST
    ▼
FastAPI (Render)
    │               │
    ▼               ▼
scikit-learn    n8n webhook
pickle model    (email automation)
                    │
                    ▼
                Gmail → User inbox
```

---

## Frontend Architecture

### Stack
- React 18 + Vite 5
- React Router 6 (URL state for shareable links)
- Chart.js 4 (bar chart, radar chart)
- Embla Carousel 8 (team card carousel)
- Framer Motion 11 (panel transitions, card animations)
- GSAP + ScrollTrigger 3 (scroll reveals)
- VanillaTilt.js (3D card tilt on podium)
- html2canvas (PNG export)

### Page / Route Structure

```
/                    → Dashboard (home)
/dream11             → Dream11 Team Predictor
/h2h                 → Head-to-Head Match Predictor
/whatif              → What-If Scenario Engine
/?team=CSK           → Dashboard with CSK pre-selected (shareable)
```

### Component Tree

```
App
├── Layout
│   ├── TopBar
│   └── CustomCursor
│
├── Dashboard (/)
│   ├── HeroSection
│   │   ├── PodiumCard × 3        (parallax + 3D tilt)
│   │   └── ParallaxLayer × 3
│   ├── TeamList
│   │   └── TeamRow × 10          (scroll reveal, staggered)
│   ├── ProbabilityChart          (Chart.js bar)
│   ├── DetailPanel               (slide-in on team click)
│   │   ├── BigProbNumber         (count-up animation)
│   │   ├── GaugeBar
│   │   ├── TierBadge
│   │   └── ShapExplainer
│   └── TeamCarousel              (Embla)
│       └── TeamCard × 10
│
├── Dream11 (/dream11)
│   ├── MatchSelector             (dropdown)
│   ├── VenueSelector             (dropdown + VenueCard)
│   ├── PlayerGrid
│   │   └── PlayerCard × N        (flip on select, fly-left on predict)
│   ├── SelectionFooter           (counter, captain/vc badges)
│   ├── PredictButton
│   └── ResultPanel               (slides in from right)
│       ├── WinProbability        (count-up)
│       ├── StrategyReport
│       │   ├── WinCondition
│       │   ├── VenueStrategy
│       │   ├── KeyMatchup
│       │   └── RiskFactors
│       └── EmailForm             (n8n trigger)
│
├── HeadToHead (/h2h)
│   ├── TeamSelectorA
│   ├── VsBadge
│   ├── TeamSelectorB
│   ├── VenueSelector
│   ├── PredictButton
│   ├── SplitResultBar            (animated 50/50 → actual split)
│   └── VenueConditionsCard
│
└── WhatIf (/whatif)
    ├── SliderPanel
    │   ├── BattingAvgSlider
    │   ├── EconomySlider
    │   ├── NRRSlider
    │   ├── HomeWinRateSlider
    │   └── VenueDropdown
    └── LiveProbabilityBars × 10
```

---

## State Management

No external state library (no Redux, no Zustand). Pure React state is sufficient.

### State locations

| State | Location | Tool |
|-------|----------|------|
| Selected team | `Dashboard` component | `useState` |
| Dream11 selected players | `Dream11` component | `useState` (array) |
| Slider values | `WhatIf` component | `useState` (object) |
| Probabilities from API | Per-page component | `useState` + `useEffect` |
| URL team param | React Router | `useSearchParams` |
| Last selected team (refresh) | Browser | `sessionStorage` |
| Scenario cache | `WhatIf` component | `useRef` (JS Map) |

### Data flow — dashboard

```
useEffect on mount
    → GET /api/predict
    → setProbabilities(data)
    → TeamList, PodiumCards, Chart all read from probabilities state

User clicks TeamRow
    → setSelectedTeam(team)
    → DetailPanel re-renders with new team data
    → URL updates to ?team=CSK via useSearchParams
    → countUp() fires on BigProbNumber
```

### Data flow — what-if

```
User moves slider
    → setSliderValues(prev => ({...prev, batting_avg: newVal}))
    → debounce 300ms
    → POST /api/predict/custom with sliderValues
    → setWhatIfProbabilities(response)
    → LiveProbabilityBars re-render with transition
```

---

## Backend Architecture

### Stack
- FastAPI 0.110+
- Uvicorn 0.27+ (ASGI server)
- scikit-learn (model inference)
- pickle (model serialisation)
- httpx (async HTTP calls to n8n)
- SHAP (optional, for /explain endpoint)

### File structure

```
backend/
├── main.py              # FastAPI app, all routes
├── model/
│   ├── ipl_model.pkl    # trained classifier
│   └── scaler.pkl       # fitted StandardScaler
├── data/
│   ├── venue_metadata.csv
│   └── season_defaults.json  # average feature values per team
├── schemas.py           # Pydantic request/response models
├── predict.py           # prediction logic, feature merging
├── explain.py           # SHAP wrapper
└── requirements.txt
```

### API Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/predict` | GET | — | `{ CSK: 0.421, GT: 0.274, ... }` |
| `/api/predict/custom` | POST | `{ batting_avg: 32, economy: 7.8, venue: "Wankhede" }` | `{ CSK: 0.44, GT: 0.25, ... }` |
| `/api/predict/h2h` | POST | `{ team_a: "CSK", team_b: "MI", venue: "Chepauk" }` | `{ team_a_prob: 0.67, team_b_prob: 0.33, strategy: {...} }` |
| `/api/predict/dream11` | POST | `{ players: [...], venue: "Wankhede" }` | `{ win_prob: 0.58, strategy: {...} }` |
| `/api/venues` | GET | — | Array of 12 venue objects |
| `/api/teams` | GET | — | Array of 10 team metadata objects |
| `/api/history` | GET | `?from=2008&to=2025` | Array of `{ year, winner, code }` |
| `/api/explain/{team}` | GET | team code in path | `{ feature: shap_value, ... }` |
| `/api/email/report` | POST | Report payload | `{ status: "sent" }` |
| `/health` | GET | — | `{ status: "ok", model_loaded: true }` |

### Startup pattern

```python
# main.py
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# load once, stay in memory
model  = pickle.load(open("model/ipl_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
TEAMS  = ["CSK","GT","MI","LSG","RCB","DC","RR","KKR","SRH","PBKS"]
```

---

## n8n Automation Flow

```
FastAPI POST /api/email/report
    │
    │  JSON payload:
    │  { email, team_name, win_pct, strategy,
    │    venue, player_list, toss_tip, opponent }
    │
    ▼
n8n Webhook node (POST trigger)
    │
    ▼
Function node (build HTML email)
    │  Transforms JSON → styled HTML template
    │  Subject: "Your IPL Dream11 Prediction — {team} vs {opp} at {venue}"
    │
    ▼
Gmail node (send email)
    │  To: payload.email
    │  From: configured Gmail account
    │
    ▼
Respond to Webhook node
    │  { success: true, message_id: "..." }
    │
    ▼
FastAPI receives 200 → returns { status: "sent" } to frontend
```

---

## Venue Metadata Schema

```json
{
  "venue": "Wankhede Stadium",
  "city": "Mumbai",
  "home_team": "MI",
  "ground_size": "S",
  "coastal": true,
  "altitude_m": 14,
  "avg_1st_innings": 182,
  "spin_wkt_pct": 28,
  "chase_win_pct": 52,
  "capacity": 33000
}
```

---

## Environment Variables

### Backend (.env)
```
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/ipl-report
MODEL_PATH=model/ipl_model.pkl
SCALER_PATH=model/scaler.pkl
```

### Frontend (.env)
```
VITE_API_URL=https://your-render-app.onrender.com
```

---

## Deployment

### Backend — Render
1. Push backend folder to GitHub
2. New Web Service on Render → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard

### Frontend — Vercel
1. Push frontend folder to GitHub
2. Import project on Vercel → auto-detects Vite
3. Set `VITE_API_URL` in environment variables
4. Deploy — Vercel handles build and CDN
