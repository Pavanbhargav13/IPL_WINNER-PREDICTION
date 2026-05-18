# Feature Log

Tracks every user-facing feature, its status, priority, build phase, and implementation notes.

**Status key:**
- `planned` — not started
- `in-progress` — currently being built
- `done` — complete and working
- `blocked` — waiting on dependency

**Source key:**
- `core` — part of the original dashboard design
- `pavan` — Pavan's direct idea from product discussion

---

## Module 1 — Season Prediction Dashboard

---

### F01 — Win Probability Display

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P0 — build first |
| Phase | Phase 2 |
| Source | `core` |
| Estimate | 3 hours |

**What it does:**
Podium cards for the top 3 teams. Full ranked list for all 10 teams with probability bars, percentage, and implied odds. Bar chart showing distribution.

**UI components needed:**
- `PodiumCard` × 3
- `TeamRow` × 10
- `ProbabilityChart` (Chart.js)

**Motion:**
- Numbers count up from 0 on page load (easeOutQuart, 1.2s)
- Team list rows stagger in with 80ms delay using scroll reveal

**API:** `GET /api/predict`

---

### F02 — Team Detail Panel

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P0 |
| Phase | Phase 2 |
| Source | `core` |
| Estimate | 3 hours |

**What it does:**
Clicking any team slides in the right panel. Shows big win probability number, implied betting odds, tier badge (hot favourite / in contention / long shot), gauge bar filling to team colour, and SHAP feature explanation.

**Tier logic:**
- Hot favourite: probability ≥ 15%
- In contention: probability ≥ 3%
- Long shot: probability < 3%

**Motion:**
- Panel slides in from right with Framer Motion layout animation
- Big number counts up
- Gauge bar fills with team colour using CSS transition

**API:** `GET /api/explain/{team}` for SHAP values

---

### F03 — Team Probability Carousel

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 2 |
| Source | `core` |
| Estimate | 2 hours |

**What it does:**
Horizontal swipeable carousel of all 10 team cards. Touch on mobile, arrow buttons on desktop. Active card snaps to centre, scales 1.05x, gets coloured border. Adjacent cards dim to 0.6 opacity. Dot indicators below.

**Library:** Embla Carousel 8 — `npm install embla-carousel-react`

---

## Module 2 — Dream11 Team Predictor

---

### F04 — Player Card Picker

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 5 hours |

**What it does:**
User picks 11 players from a card grid. Each card shows player name, role badge (BAT / BOWL / AR / WK), team colour strip, and recent form score (last 5 matches — green good, amber average, red poor). Selected cards flip using CSS 3D rotateY. Counter shows X / 11 selected. Captain and vice-captain can be starred separately.

**Card dimensions:** 140px × 180px

**CSS flip pattern:**
```css
.card { perspective: 600px; }
.card-inner { transition: transform 0.4s; transform-style: preserve-3d; }
.card.selected .card-inner { transform: rotateY(180deg); }
```

**Dependency:** Match selector (F04a) and venue selector must be completed first

---

### F05 — Cards Fly Left + Result Reveal

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 3 |
| Source | `pavan` |
| Estimate | 3 hours |

**What it does:**
When user hits Predict, all 11 player cards animate translateX(-120%) with 50ms stagger per card. After 400ms, result panel slides in from right. This is the signature animation of the app.

**Animation detail:**
- Each card gets CSS variable `--index` set as its position (0–10)
- `animation-delay: calc(var(--index) * 50ms)` on `.card-exit`
- Result panel uses Framer Motion `AnimatePresence` with `x: "100%"` initial

**Must feel:** Sports broadcast transition — clean, fast, confident.

---

### F06 — Strategy Prediction Report

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 4 hours |

**What it does:**
After Dream11 prediction, result panel shows full strategy report. Sections:
1. Win probability (animated count-up)
2. Win condition — how the team needs to play
3. Venue strategy — ground-specific game plan
4. Key matchup — most important individual contest
5. Toss recommendation — bat or bowl with reason
6. Risk factors — what could go wrong

**API:** `POST /api/predict/dream11`

**Request body:**
```json
{
  "players": ["CSK_1", "CSK_3", "MI_2", ...],
  "venue": "Wankhede Stadium",
  "match": "CSK vs MI"
}
```

---

### F07 — Email Report via n8n

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P2 |
| Phase | Phase 5 |
| Source | `pavan` |
| Estimate | 3 hours (includes n8n workflow) |

**What it does:**
After the prediction report appears, user enters their email and hits Send Report. Button shows loading spinner, then "Sent!" confirmation. n8n fires in background. Email arrives within 10 seconds.

**User flow:**
1. User sees result panel
2. Email input and Send Report button at bottom
3. Click → `POST /api/email/report`
4. FastAPI queues background task → calls n8n webhook
5. n8n formats HTML email → sends via Gmail node
6. Frontend shows success toast

**n8n workflow nodes:** Webhook → Function (HTML builder) → Gmail → Respond

**Email subject:** `Your IPL Dream11 Prediction — {team_a} vs {team_b} at {venue}`

---

## Module 3 — Head-to-Head Match Predictor

---

### F08 — Team and Venue Selector

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 2 hours |

**What it does:**
Two team dropdowns side by side with team badge and full name. VS badge in the centre. Venue dropdown below showing ground name and city. Single Predict button.

**API:** `GET /api/teams` for dropdown options, `GET /api/venues` for venue list

---

### F09 — Animated H2H Split Result

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 3 hours |

**What it does:**
Horizontal bar starts at 50/50. Animates outward over 1.2s to actual split (e.g. CSK 67% | MI 33%). Each side fills with team colour. Winning side gets subtle pulse. Numbers count up.

**Animation:** CSS width transition from 50% to actual value, triggered after 200ms delay on result render.

**API:** `POST /api/predict/h2h`

---

### F10 — Venue Conditions Card

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P2 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 1 hour |

**What it does:**
Card below H2H result showing selected venue characteristics: pitch type, average first innings score, chase win %, spin vs pace wicket balance, ground size.

**Data source:** `venue_metadata.csv` loaded at backend startup — static, no DB needed.

---

## Module 4 — What-If Scenario Engine

---

### F11 — Live Stat Sliders

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P1 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 4 hours |

**What it does:**
5 sliders: batting average (15–45), economy rate (6.5–12.0), NRR (-1.5 to +1.5), home win rate (20–85%), and a venue dropdown. Model re-predicts live via FastAPI. All 10 probability bars update with smooth CSS transition.

**Critical rule:** Debounce at 300ms minimum. Do not fire API on every pixel of slider movement.

**API:** `POST /api/predict/custom`

---

### F12 — Venue Switcher in What-If

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P2 |
| Phase | Phase 4 |
| Source | `pavan` |
| Estimate | 1 hour (extension of F11) |

**What it does:**
Venue dropdown added to the what-if panel. Switching between Chepauk, Wankhede, and Eden Gardens changes the venue-related features passed to the model, shifting probabilities based on pitch and home advantage.

**Dependency:** F11 must be complete. Venue is added as an additional POST body field.

---

## Module 5 — Share & Export

---

### F13 — Shareable PNG Prediction Card

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P2 |
| Phase | Phase 5 |
| Source | `core` |
| Estimate | 2 hours |

**What it does:**
One-click export button generates a styled PNG card showing top-3 predictions. Built using html2canvas rendering the card component to canvas, then triggering a download.

**Library:** `html2canvas` — `npm install html2canvas`

**Card content:** Top-3 teams, probabilities, IPL 2026 label, author name.

---

### F14 — Email Full Report

| Attribute | Value |
|-----------|-------|
| Status | `planned` |
| Priority | P2 |
| Phase | Phase 5 |
| Source | `pavan` |
| Estimate | Covered under F07 (same n8n workflow) |

**What it does:**
Email delivery of the full Dream11 prediction report. Covered entirely by F07 implementation. Listed separately because it is surfaced as a standalone "Email Report" button in the export module.

---

## Motion Effects (cross-cutting)

These are not standalone features but implementations that enhance multiple modules.

| Effect | Phase | Estimate | Status | Notes |
|--------|-------|----------|--------|-------|
| Count-up animation | Phase 2 | 1 hr | `planned` | Implement first — highest ROI |
| Scroll reveal (team list) | Phase 2 | 2 hr | `planned` | IntersectionObserver, no library |
| Dream11 card fly-left | Phase 3 | 3 hr | `planned` | CSS stagger, see Coding_Rules.md |
| Mouse parallax (podium) | Phase 3 | 3 hr | `planned` | Disable below 768px |
| 3D card tilt (podium) | Phase 3 | 30 min | `planned` | VanillaTilt.js — do this early, fast win |
| Custom glowing cursor | Phase 3 | 1 hr | `planned` | Desktop only |

---

## Build Phase Summary

| Phase | Features | Estimate |
|-------|----------|----------|
| Phase 1 — Foundation | Export model, FastAPI skeleton, React + Vite setup | 4–6 hrs |
| Phase 2 — Core Dashboard | F01, F02, F03, count-up, scroll reveal | 8–10 hrs |
| Phase 3 — Motion | Parallax, 3D tilt, cursor, card fly animation | 6–8 hrs |
| Phase 4 — New Features | F04–F12 | 14–16 hrs |
| Phase 5 — Automation + Deploy | F07/F14, F13, Render, Vercel | 6–8 hrs |
| **Total** | **14 features** | **~40–50 hrs** |

---

## Known Corrections

| Item | Old value | Corrected value | Applied |
|------|-----------|-----------------|---------|
| 2025 IPL winner | MI | **RCB** | Yes — all synthetic data regenerated |
| RCB team strength prior | 0.60 | **0.72** | Yes — reflects title win |
