"""
main.py — FastAPI application for IPL Winner Prediction
=========================================================
Start with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /api/predict           → Season-level predictions for all 10 teams
    POST /api/predict/h2h       → Head-to-Head: two teams at a venue
    POST /api/predict/whatif    → What-If sliders
    POST /api/predict/dream11   → Dream11 squad strategy
    GET  /api/venues            → All known venues list
    GET  /api/teams             → All teams metadata
    GET  /api/health            → Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import httpx

from backend.predictor import IPLPredictor, TEAM_ALIASES, TEAM_FULL_NAMES, TEAM_COLORS
from backend.schemas import H2HRequest, WhatIfRequest, Dream11Request, EmailReportRequest

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global predictor instance ──────────────────────────────────────────────────
predictor: IPLPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global predictor
    logger.info("🏏 Loading IPL prediction model...")
    try:
        predictor = IPLPredictor()
        logger.info(f"✅ Model loaded: {predictor.meta.get('model_name')} "
                    f"(CV Acc: {predictor.meta.get('cv_accuracy')})")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    yield
    logger.info("🏏 Shutting down IPL prediction server...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IPL Winner Prediction API",
    description="ML-powered IPL match winner prediction with venue intelligence and Dream11 strategy.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS — allow Vite dev server ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":     "ok",
        "model":      predictor.meta.get("model_name"),
        "accuracy":   predictor.meta.get("cv_accuracy"),
        "trained_at": predictor.meta.get("trained_at"),
    }


# ── Teams Metadata ─────────────────────────────────────────────────────────────
@app.get("/api/teams", tags=["Data"])
def get_teams():
    """Return all 10 IPL teams with full names, colors, and aliases."""
    teams = []
    for abbr, aliases in TEAM_ALIASES.items():
        teams.append({
            "abbr":      abbr,
            "full_name": TEAM_FULL_NAMES.get(abbr, abbr),
            "color":     TEAM_COLORS.get(abbr, "#888888"),
            "aliases":   aliases,
        })
    return {"teams": teams}


# ── Venues List ────────────────────────────────────────────────────────────────
@app.get("/api/venues", tags=["Data"])
def get_venues():
    """Return all known venues from the dataset."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    venues = predictor.venue_stats["venue_name"].dropna().tolist()
    return {"venues": venues}


# ── Season Predictions (all 10 teams) ─────────────────────────────────────────
@app.get("/api/predict", tags=["Prediction"])
def predict_season():
    """
    Predict win probabilities for all 10 IPL teams for the current season.
    Teams are ranked by their average win probability across all matchups.
    Returns podium + full ranked list with SHAP-approximate feature contributions.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_season()
        return result
    except Exception as e:
        logger.error(f"Season prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Head-to-Head Predictor ─────────────────────────────────────────────────────
@app.post("/api/predict/h2h", tags=["Prediction"])
def predict_h2h(req: H2HRequest):
    """
    Head-to-head prediction: Team A vs Team B at a specific venue.
    Returns win probabilities, venue intelligence, pitch type, and toss advice.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.home_team.upper() == req.away_team.upper():
        raise HTTPException(status_code=400, detail="Home and Away teams must be different")
    try:
        result = predictor.predict_h2h(req.home_team, req.away_team, req.venue)
        return result
    except Exception as e:
        logger.error(f"H2H prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── What-If Scenario Engine ────────────────────────────────────────────────────
@app.post("/api/predict/whatif", tags=["Prediction"])
def predict_whatif(req: WhatIfRequest):
    """
    What-If prediction with custom stat overrides.
    Sliders can adjust batting avg, economy rate, and NRR for both teams.
    Returns modified probability + delta from baseline.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        overrides = {
            "home_batting_avg":   req.home_batting_avg,
            "home_economy_rate":  req.home_economy_rate,
            "home_nrr":           req.home_nrr,
            "away_batting_avg":   req.away_batting_avg,
            "away_economy_rate":  req.away_economy_rate,
            "away_nrr":           req.away_nrr,
        }
        result = predictor.predict_whatif(req.home_team, req.away_team, req.venue, overrides)
        return result
    except Exception as e:
        logger.error(f"What-If prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Dream11 Strategy ───────────────────────────────────────────────────────────
@app.post("/api/predict/dream11", tags=["Prediction"])
def predict_dream11(req: Dream11Request):
    """
    Dream11 squad-based prediction.
    Given 11 selected players + match context, returns win probability,
    strategy narrative, captain recommendation, and pitch-based tips.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_dream11(
            req.home_team, req.away_team, req.venue, req.selected_players
        )
        return result
    except Exception as e:
        logger.error(f"Dream11 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Email Report (n8n Automation) ───────────────────────────────────────────────
async def trigger_n8n_webhook(email: str, report_data: dict):
    """Background task to trigger n8n webhook for sending an email."""
    webhook_url = "http://localhost:5678/webhook/ipl-report" # Default local n8n URL
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json={
                "email": email,
                "report": report_data
            }, timeout=10.0)
            if resp.status_code == 200:
                logger.info(f"✅ Successfully triggered n8n webhook for {email}")
            else:
                logger.error(f"❌ n8n webhook failed with status {resp.status_code}: {resp.text}")
    except httpx.RequestError as e:
        logger.warning(f"⚠️ Could not reach n8n webhook at {webhook_url}. Is n8n running? Error: {e}")
        # Not failing the main request since this is a background task

@app.post("/api/email/report", tags=["Automation"])
async def send_email_report(req: EmailReportRequest, background_tasks: BackgroundTasks):
    """
    Queue a background task to send the prediction report via email using n8n.
    Returns immediately so the frontend UI doesn't hang.
    """
    logger.info(f"Queueing email report for {req.email}")
    background_tasks.add_task(trigger_n8n_webhook, req.email, req.report_data)
    return {"message": f"Email queued for {req.email}", "status": "success"}


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "message": "🏏 IPL Winner Prediction API v2.0",
        "docs":    "/docs",
        "health":  "/api/health",
    }
