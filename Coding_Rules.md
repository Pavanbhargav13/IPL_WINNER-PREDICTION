# Coding Rules

Rules for every developer (and AI assistant) working on this project. Follow these without exception.

---

## General

- Write clean, readable code over clever code
- No commented-out dead code in commits
- Every component has one clear responsibility
- If a file exceeds 200 lines, split it
- No `console.log` left in production code — use a `debug.ts` utility flag instead

---

## Naming Conventions

### Files and folders
```
components/         PascalCase        TeamCard.tsx, PodiumSection.tsx
hooks/              camelCase prefix  useProbabilities.ts, useCountUp.ts
utils/              camelCase         formatProbability.ts, debounce.ts
pages/              PascalCase        Dashboard.tsx, Dream11.tsx
styles/             camelCase.css     teamCard.module.css
```

### Variables and functions
```js
// Components — PascalCase
const TeamCard = () => {}

// Hooks — camelCase with use prefix
const useProbabilities = () => {}

// Event handlers — handle prefix
const handleTeamClick = (team) => {}
const handleSliderChange = (value) => {}

// Booleans — is/has/should prefix
const isSelected = true
const hasLoaded = false
const shouldAnimate = true

// Constants — SCREAMING_SNAKE
const IPL_BASELINE_SCORE = 170
const TEAM_COLORS = { CSK: "#f4c430", MI: "#005daa" }

// API response objects — camelCase
const winProbability = data.win_probability
```

---

## React Rules

### Component structure — always in this order
```tsx
// 1. Imports
import React, { useState, useEffect } from "react"
import styles from "./TeamCard.module.css"

// 2. Types/interfaces
interface TeamCardProps {
  team: Team
  isSelected: boolean
  onSelect: (team: Team) => void
}

// 3. Component
const TeamCard = ({ team, isSelected, onSelect }: TeamCardProps) => {

  // 4. State
  const [isHovered, setIsHovered] = useState(false)

  // 5. Effects
  useEffect(() => {
    // effect logic
  }, [dependency])

  // 6. Handlers
  const handleClick = () => {
    onSelect(team)
  }

  // 7. Render
  return (
    <div className={styles.card} onClick={handleClick}>
      {/* content */}
    </div>
  )
}

// 8. Export
export default TeamCard
```

### Do not
- Do not use class components
- Do not use index as key in lists that reorder — use team code or stable ID
- Do not fetch data inside deeply nested child components — fetch at page level, pass as props
- Do not put business logic in JSX — extract to handlers or utils
- Do not use inline styles except for dynamic values (team colors, animation transforms)

---

## CSS / Styling Rules

### CSS variables — define once in `globals.css`, use everywhere
```css
:root {
  --color-bg:          #0a0c10;
  --color-surface:     #0f1117;
  --color-border:      #1e2130;
  --color-amber:       #f4a623;
  --color-orange:      #e8460b;
  --color-text:        #e8e6df;
  --color-muted:       #6b6e7a;
  --font-display:      'Syne', sans-serif;
  --font-mono:         'JetBrains Mono', monospace;
  --radius-card:       12px;
  --radius-badge:      20px;
  --transition-fast:   0.15s ease;
  --transition-medium: 0.4s ease;
  --transition-slow:   0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
```

### Rules
- Use CSS Modules for component-scoped styles
- Never use `!important`
- Never hardcode colors inline — always use CSS variables or `TEAM_COLORS` constant
- All transitions use the CSS variable — never hardcode `0.3s ease` directly
- Mobile breakpoint: 768px. Disable parallax, tilt, custom cursor below this

### Team color usage
```tsx
// correct — use constant
<div style={{ borderColor: TEAM_COLORS[team.code] }} />

// wrong — hardcode
<div style={{ borderColor: "#f4c430" }} />
```

---

## Animation Rules

### Count-up numbers
```ts
// Always use this exact pattern — easeOutQuart only
export function countUp(el: HTMLElement, target: number, duration = 1200) {
  const start = performance.now()
  const tick = (now: number) => {
    const t = Math.min((now - start) / duration, 1)
    const ease = 1 - Math.pow(1 - t, 4) // easeOutQuart
    el.textContent = (target * ease).toFixed(1) + "%"
    if (t < 1) requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
}
```

### Parallax
```ts
// Always check isMobile before attaching
const isMobile = window.innerWidth < 768
if (!isMobile) {
  document.addEventListener("mousemove", handleParallax)
}
```

### Dream11 card exit animation
```css
.card-exit {
  animation: flyLeft 0.45s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

@keyframes flyLeft {
  to {
    transform: translateX(-120%);
    opacity: 0;
  }
}
```
- Stagger: 50ms delay per card using `animation-delay: calc(var(--index) * 50ms)`
- Set `--index` as a CSS variable on each card element

### Scroll reveal
```ts
// Use IntersectionObserver — never scroll event listener
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.classList.add("revealed")
        observer.unobserve(entry.target)
      }, i * 80) // 80ms stagger
    }
  })
}, { threshold: 0.15 })
```

---

## API / Data Fetching Rules

### Always debounce slider calls
```ts
// Minimum 300ms debounce on any call triggered by a slider
const debouncedPredict = useMemo(
  () => debounce(async (features: FeatureOverrides) => {
    const res = await fetch(`${API_URL}/api/predict/custom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    })
    const data = await res.json()
    setWhatIfProbabilities(data)
  }, 300),
  []
)
```

### Always handle loading and error states
```tsx
// Every API call needs these three states
const [probabilities, setProbabilities] = useState<ProbMap | null>(null)
const [isLoading, setIsLoading] = useState(true)
const [error, setError] = useState<string | null>(null)
```

### API base URL — always from env
```ts
const API_URL = import.meta.env.VITE_API_URL
// Never hardcode http://localhost:8000 in component files
```

---

## Backend / Python Rules

### FastAPI
- All route handlers are `async def`
- All Pydantic models go in `schemas.py` — not inline in `main.py`
- Use `Optional` for all override fields in custom predict schema
- Return plain dicts for probability responses — no wrapper objects needed

```python
# correct
@app.get("/api/predict")
async def predict():
    probs = run_inference(SEASON_DEFAULTS)
    return dict(zip(TEAMS, probs))

# wrong — unnecessary wrapper
@app.get("/api/predict")
async def predict():
    return { "data": { "probabilities": dict(zip(TEAMS, probs)) } }
```

### Model inference — always goes through `predict.py`
```python
# predict.py — single source for all inference
def run_inference(features: dict) -> list[float]:
    X = scaler.transform([list(features.values())])
    return model.predict_proba(X)[0].tolist()
```

### n8n call — always fire and return fast
```python
# Do not await the n8n response before sending 200 to frontend
# Use background tasks
from fastapi import BackgroundTasks

@app.post("/api/email/report")
async def send_report(payload: ReportPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(call_n8n_webhook, payload)
    return { "status": "queued" }
```

---

## Git Commit Format

```
feat: add Dream11 player card flip animation
fix: correct 2025 winner to RCB in synthetic data
style: update team card hover border colour
refactor: extract countUp into shared hook
docs: update Architecture.md with n8n flow
```

Prefixes: `feat`, `fix`, `style`, `refactor`, `docs`, `chore`, `test`

---

## What Never to Do

- Never hardcode probability values in the frontend — always fetch from API
- Never store the pickle model in the frontend repo
- Never commit `.env` files
- Never use `any` type in TypeScript — define proper interfaces
- Never add `console.log` calls in components — use the debug flag utility
- Never attach `mousemove` or `scroll` listeners without cleaning them up in `useEffect` return
- Never skip the debounce on slider-triggered API calls
