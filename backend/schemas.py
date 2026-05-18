"""
schemas.py — Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request Schemas ────────────────────────────────────────────────────────────

class H2HRequest(BaseModel):
    home_team: str = Field(..., example="MI")
    away_team: str = Field(..., example="CSK")
    venue: str = Field(..., example="Wankhede Stadium")


class WhatIfRequest(BaseModel):
    home_team: str = Field(..., example="MI")
    away_team: str = Field(..., example="CSK")
    venue: str = Field(..., example="Wankhede Stadium")
    # Overrides — sliders
    home_batting_avg: Optional[float] = Field(None, ge=20, le=60)
    home_economy_rate: Optional[float] = Field(None, ge=6.0, le=12.0)
    home_nrr: Optional[float] = Field(None, ge=-1.5, le=1.5)
    away_batting_avg: Optional[float] = Field(None, ge=20, le=60)
    away_economy_rate: Optional[float] = Field(None, ge=6.0, le=12.0)
    away_nrr: Optional[float] = Field(None, ge=-1.5, le=1.5)


class Dream11Request(BaseModel):
    home_team: str
    away_team: str
    venue: str
    selected_players: List[str] = Field(..., min_items=11, max_items=11)


class EmailReportRequest(BaseModel):
    email: str
    home_team: str
    away_team: str
    venue: str
    home_win_prob: float
    away_win_prob: float
    pitch_type: Optional[str] = None
    toss_advice: Optional[str] = None


# ── Response Schemas ───────────────────────────────────────────────────────────

class TeamPrediction(BaseModel):
    rank: int
    team: str
    team_full_name: str
    win_probability: float
    implied_odds: float
    badge: str                    # "🔥 Hot Favourite" / "📉 Long Shot" etc.
    home_win_rate: float
    toss_win_rate: float
    avg_score: float
    shap_values: dict             # feature → contribution


class SeasonPredictionResponse(BaseModel):
    predictions: List[TeamPrediction]
    model_name: str
    model_accuracy: float
    trained_at: str


class H2HPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    venue: str
    home_win_prob: float
    away_win_prob: float
    winner: str
    # Venue intelligence
    pitch_type: str
    avg_first_innings_score: float
    chase_win_pct: float
    ground_size: str
    coastal: bool
    home_advantage_modifier: float
    toss_advice: str
    venue_description: str


class WhatIfResponse(BaseModel):
    home_team: str
    away_team: str
    venue: str
    home_win_prob: float
    away_win_prob: float
    winner: str
    delta_home: float             # change from baseline
    delta_away: float


class Dream11Response(BaseModel):
    home_team: str
    away_team: str
    venue: str
    win_probability: float
    strategy_narrative: str
    captain_tip: str
    tips: List[str]
