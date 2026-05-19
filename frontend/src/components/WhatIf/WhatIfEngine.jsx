import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Home, Plane, MapPin, RotateCcw, Sparkles, Sliders, ArrowRight, TrendingUp } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const TEAM_META = {
  RCB: { name: "Royal Challengers Bengaluru", color: "#EC1C24" },
  CSK: { name: "Chennai Super Kings", color: "#FFD700" },
  MI: { name: "Mumbai Indians", color: "#004BA0" },
  KKR: { name: "Kolkata Knight Riders", color: "#3A225D" },
  SRH: { name: "Sunrisers Hyderabad", color: "#FF8225" },
  GT: { name: "Gujarat Titans", color: "#1B365D" },
  RR: { name: "Rajasthan Royals", color: "#FF69B4" },
  LSG: { name: "Lucknow Super Giants", color: "#00A2E8" },
  PBKS: { name: "Punjab Kings", color: "#ED1B24" },
  DC: { name: "Delhi Capitals", color: "#134285" }
};

export default function WhatIfEngine() {
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  
  const [homeTeam, setHomeTeam] = useState('RCB');
  const [awayTeam, setAwayTeam] = useState('CSK');
  const [venue, setVenue] = useState('M. Chinnaswamy Stadium');
  
  // Overrides
  const [homeBattingAvg, setHomeBattingAvg] = useState(175);
  const [homeEconomyRate, setHomeEconomyRate] = useState(8.2);
  const [homeNrr, setHomeNrr] = useState(0.2);
  
  const [awayBattingAvg, setAwayBattingAvg] = useState(172);
  const [awayEconomyRate, setAwayEconomyRate] = useState(8.5);
  const [awayNrr, setAwayNrr] = useState(0.1);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch teams and venues on mount
  useEffect(() => {
    fetch(`${API_BASE}/teams`)
      .then(res => res.json())
      .then(data => {
        if (data.teams && data.teams.length > 0) setTeams(data.teams);
      })
      .catch(err => {
        console.warn("Using local fallback teams in What-If:", err);
        setTeams(Object.keys(TEAM_META).map(key => ({ abbr: key, full_name: TEAM_META[key].name })));
      });

    fetch(`${API_BASE}/venues`)
      .then(res => res.json())
      .then(data => {
        if (data.venues && data.venues.length > 0) {
          setVenues(data.venues);
          setVenue(data.venues[0]);
        } else {
          setVenues([
            "M. Chinnaswamy Stadium", "MA Chidambaram Stadium", "Wankhede Stadium",
            "Eden Gardens", "Narendra Modi Stadium", "Arjun Jaitley Stadium"
          ]);
        }
      })
      .catch(err => {
        console.warn("Using local fallback venues in What-If:", err);
        setVenues([
          "M. Chinnaswamy Stadium", "MA Chidambaram Stadium", "Wankhede Stadium",
          "Eden Gardens", "Narendra Modi Stadium", "Arjun Jaitley Stadium"
        ]);
      });
  }, []);

  // Debounced API call for sliders
  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      if (homeTeam && awayTeam && venue) {
        fetchPrediction();
      }
    }, 400);

    return () => clearTimeout(delayDebounceFn);
  }, [homeTeam, awayTeam, venue, homeBattingAvg, homeEconomyRate, homeNrr, awayBattingAvg, awayEconomyRate, awayNrr]);

  const fetchPrediction = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          venue: venue,
          home_batting_avg: homeBattingAvg,
          home_economy_rate: homeEconomyRate,
          home_nrr: homeNrr,
          away_batting_avg: awayBattingAvg,
          away_economy_rate: awayEconomyRate,
          away_nrr: awayNrr
        })
      });
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.warn("Calculating fallback slider offsets locally:", err);
      // Fallback calculation so the sliders respond instantly in real-time even offline!
      const baseHomeProb = 50 + (homeBattingAvg - awayBattingAvg) * 0.4 - (homeEconomyRate - awayEconomyRate) * 5 + (homeNrr - awayNrr) * 15;
      const homeProb = Math.min(95, Math.max(5, Math.round(baseHomeProb)));
      const awayProb = 100 - homeProb;

      setPrediction({
        home_team: homeTeam,
        away_team: awayTeam,
        venue,
        home_win_prob: homeProb,
        away_win_prob: awayProb,
        winner: homeProb > awayProb ? homeTeam : awayTeam,
        delta_home: Math.round(homeProb - 50),
        delta_away: Math.round(awayProb - 50)
      });
    } finally {
      setLoading(false);
    }
  };

  const resetToReal = () => {
    setHomeBattingAvg(175);
    setHomeEconomyRate(8.2);
    setHomeNrr(0.2);
    setAwayBattingAvg(172);
    setAwayEconomyRate(8.5);
    setAwayNrr(0.1);
  };

  const homeColor = TEAM_META[homeTeam]?.color || '#FF6915';
  const awayColor = TEAM_META[awayTeam]?.color || '#FFFBF4';

  return (
    <div className="main-content pos-dashboard-viewport" style={{ background: '#11120D', color: '#FFFBF4', padding: '2rem', overflowY: 'auto' }}>
      
      {/* Editorial Header */}
      <div style={{ marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: '#FF6915', fontWeight: '700', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
          <Sparkles size={13} />
          WHAT-IF SIMULATION ENGINE
        </div>
        <h1 className="editorial-title" style={{ fontSize: '2.4rem', fontWeight: '900', color: '#FFFBF4', margin: '0.5rem 0 0.2rem', textTransform: 'uppercase', letterSpacing: '-0.02em' }}>
          Parameter Tuner
        </h1>
        <p style={{ color: '#D8CFBC', margin: 0, fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>
          SLIDE TEAM METRICS TO COMPUTE PROBABILITY DELTAS IN REAL-TIME
        </p>
      </div>

      {/* ── 3-COLUMN SELECTION GRID (SAME AS H2H) ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.2fr', gap: '1.5rem', marginBottom: '2.5rem' }}>
        
        {/* Card 1: Home Team Selection */}
        <div className="pos-card" style={{ padding: '1.5rem', borderLeft: `4px solid ${homeColor}`, boxShadow: `0 15px 30px ${homeColor}15` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <div style={{ background: `${homeColor}22`, color: homeColor, padding: '0.5rem', borderRadius: '10px' }}>
              <Home size={18} />
            </div>
            <div>
              <span className="pos-card-eyebrow" style={{ color: homeColor }}>HOME HOST</span>
              <h3 style={{ fontSize: '0.9rem', fontWeight: '800', margin: 0 }}>Select Primary Host</h3>
            </div>
          </div>
          
          <select 
            value={homeTeam} 
            onChange={e => setHomeTeam(e.target.value)}
            style={{ width: '100%', padding: '0.85rem 1rem', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', color: '#FFFBF4', border: '1px solid rgba(255,255,255,0.08)', outline: 'none', cursor: 'pointer', fontSize: '0.85rem', fontWeight: '700' }}
          >
            {teams.map(t => <option key={t.abbr} value={t.abbr} style={{ background: '#1C1D17', color: '#fff' }}>{t.full_name || t.abbr} ({t.abbr})</option>)}
          </select>
        </div>

        {/* Card 2: Away Team Selection */}
        <div className="pos-card" style={{ padding: '1.5rem', borderLeft: `4px solid ${awayColor}`, boxShadow: `0 15px 30px ${awayColor}15` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <div style={{ background: `${awayColor}22`, color: awayColor, padding: '0.5rem', borderRadius: '10px' }}>
              <Plane size={18} />
            </div>
            <div>
              <span className="pos-card-eyebrow" style={{ color: awayColor }}>AWAY CHALLENGER</span>
              <h3 style={{ fontSize: '0.9rem', fontWeight: '800', margin: 0 }}>Select Traveling Challenger</h3>
            </div>
          </div>
          
          <select 
            value={awayTeam} 
            onChange={e => setAwayTeam(e.target.value)}
            style={{ width: '100%', padding: '0.85rem 1rem', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', color: '#FFFBF4', border: '1px solid rgba(255,255,255,0.08)', outline: 'none', cursor: 'pointer', fontSize: '0.85rem', fontWeight: '700' }}
          >
            {teams.map(t => <option key={t.abbr} value={t.abbr} style={{ background: '#1C1D17', color: '#fff' }}>{t.full_name || t.abbr} ({t.abbr})</option>)}
          </select>
        </div>

        {/* Card 3: Venue Arena Location */}
        <div className="pos-card pos-gradient-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <div style={{ background: 'rgba(255, 105, 21, 0.15)', color: '#FF6915', padding: '0.5rem', borderRadius: '10px' }}>
              <MapPin size={18} />
            </div>
            <div>
              <span className="pos-card-eyebrow" style={{ color: '#FF6915' }}>CLASH ARENA</span>
              <h3 style={{ fontSize: '0.9rem', fontWeight: '800', margin: 0 }}>Venue Stadium Location</h3>
            </div>
          </div>
          
          <select 
            value={venue} 
            onChange={e => setVenue(e.target.value)}
            style={{ width: '100%', padding: '0.85rem 1rem', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', color: '#FFFBF4', border: '1px solid rgba(255,255,255,0.08)', outline: 'none', cursor: 'pointer', fontSize: '0.85rem', fontWeight: '700' }}
          >
            {venues.map(v => <option key={v} value={v} style={{ background: '#1C1D17', color: '#fff' }}>{v}</option>)}
          </select>
        </div>

      </div>

      {/* ── Interactive Tuning Sliders ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2.5rem' }}>
        
        {/* Home Stats Sliders */}
        <div className="pos-card" style={{ padding: '2rem', borderTop: `4px solid ${homeColor}` }}>
          <h3 style={{ color: homeColor, fontSize: '1.2rem', fontWeight: '900', margin: '0 0 1.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {homeTeam} Tactical Tuner
          </h3>
          
          <div className="slider-group" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Expected Runs Score</span>
              <span style={{ color: homeColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{homeBattingAvg} runs</span>
            </div>
            <input 
              type="range" min="120" max="230" step="1" 
              value={homeBattingAvg} 
              onChange={e => setHomeBattingAvg(Number(e.target.value))} 
              style={{ width: '100%', accentColor: homeColor }}
            />
          </div>

          <div className="slider-group" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Team Bowling Economy</span>
              <span style={{ color: homeColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{homeEconomyRate} rpo</span>
            </div>
            <input 
              type="range" min="5.5" max="13.0" step="0.1" 
              value={homeEconomyRate} 
              onChange={e => setHomeEconomyRate(Number(e.target.value))} 
              style={{ width: '100%', accentColor: homeColor }}
            />
          </div>

          <div className="slider-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Target Net Run Rate (NRR)</span>
              <span style={{ color: homeColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{homeNrr > 0 ? `+${homeNrr}` : homeNrr}</span>
            </div>
            <input 
              type="range" min="-2.0" max="2.0" step="0.05" 
              value={homeNrr} 
              onChange={e => setHomeNrr(Number(e.target.value))} 
              style={{ width: '100%', accentColor: homeColor }}
            />
          </div>
        </div>

        {/* Away Stats Sliders */}
        <div className="pos-card" style={{ padding: '2rem', borderTop: `4px solid ${awayColor}` }}>
          <h3 style={{ color: awayColor, fontSize: '1.2rem', fontWeight: '900', margin: '0 0 1.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {awayTeam} Tactical Tuner
          </h3>
          
          <div className="slider-group" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Expected Runs Score</span>
              <span style={{ color: awayColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{awayBattingAvg} runs</span>
            </div>
            <input 
              type="range" min="120" max="230" step="1" 
              value={awayBattingAvg} 
              onChange={e => setAwayBattingAvg(Number(e.target.value))} 
              style={{ width: '100%', accentColor: awayColor }}
            />
          </div>

          <div className="slider-group" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Team Bowling Economy</span>
              <span style={{ color: awayColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{awayEconomyRate} rpo</span>
            </div>
            <input 
              type="range" min="5.5" max="13.0" step="0.1" 
              value={awayEconomyRate} 
              onChange={e => setAwayEconomyRate(Number(e.target.value))} 
              style={{ width: '100%', accentColor: awayColor }}
            />
          </div>

          <div className="slider-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: '#D8CFBC', fontWeight: '600' }}>Target Net Run Rate (NRR)</span>
              <span style={{ color: awayColor, fontWeight: '800', fontFamily: 'var(--font-mono)' }}>{awayNrr > 0 ? `+${awayNrr}` : awayNrr}</span>
            </div>
            <input 
              type="range" min="-2.0" max="2.0" step="0.05" 
              value={awayNrr} 
              onChange={e => setAwayNrr(Number(e.target.value))} 
              style={{ width: '100%', accentColor: awayColor }}
            />
          </div>
        </div>

      </div>

      {/* Reset control */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2.5rem' }}>
        <button 
          onClick={resetToReal} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', color: '#FFFBF4', padding: '0.6rem 1.5rem', borderRadius: '10px', fontSize: '0.8rem', fontWeight: '700', cursor: 'pointer', transition: 'all 0.2s' }}
          className="header-icon-btn"
        >
          <RotateCcw size={13} /> Reset Baseline Metrics
        </button>
      </div>

      {/* ── LIVE SIMULATION RESULTS PANEL ── */}
      <AnimatePresence mode="wait">
        {prediction && (
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5, type: 'spring' }}
            className="pos-card"
            style={{ padding: '2.5rem', textAlign: 'center', background: 'linear-gradient(135deg, rgba(30, 31, 26, 0.4) 0%, rgba(17, 18, 13, 0.95) 100%)', borderLeft: '4px solid #FF6915' }}
          >
            <h3 style={{ fontSize: '1rem', fontWeight: '800', letterSpacing: '0.1em', color: '#FF6915', textTransform: 'uppercase', margin: '0 0 1.5rem' }}>
              Computed Win Probability Shifts
            </h3>
            
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '5rem', marginBottom: '1rem' }}>
              <div>
                <span style={{ fontSize: '1rem', fontWeight: '700', color: homeColor }}>{prediction.home_team}</span>
                <div className="mono" style={{ fontSize: '3.5rem', fontWeight: '950', color: homeColor, textShadow: `0 0 30px ${homeColor}33`, fontFamily: 'var(--font-mono)' }}>
                  {prediction.home_win_prob}%
                </div>
                <div style={{ fontSize: '0.85rem', fontWeight: '700', color: prediction.delta_home >= 0 ? '#4ADE80' : '#F87171', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.2rem' }}>
                  <TrendingUp size={12} style={{ transform: prediction.delta_home >= 0 ? 'none' : 'rotate(180deg)' }} />
                  {prediction.delta_home > 0 ? '+' : ''}{prediction.delta_home}% from baseline
                </div>
              </div>
              
              <div style={{ 
                width: '45px', height: '45px', borderRadius: '50%', background: 'rgba(255,255,255,0.03)', 
                border: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#D8CFBC', fontSize: '0.85rem', fontWeight: '700'
              }}>
                VS
              </div>
              
              <div>
                <span style={{ fontSize: '1rem', fontWeight: '700', color: awayColor }}>{prediction.away_team}</span>
                <div className="mono" style={{ fontSize: '3.5rem', fontWeight: '950', color: awayColor, textShadow: `0 0 30px ${awayColor}33`, fontFamily: 'var(--font-mono)' }}>
                  {prediction.away_win_prob}%
                </div>
                <div style={{ fontSize: '0.85rem', fontWeight: '700', color: prediction.delta_away >= 0 ? '#4ADE80' : '#F87171', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.2rem' }}>
                  <TrendingUp size={12} style={{ transform: prediction.delta_away >= 0 ? 'none' : 'rotate(180deg)' }} />
                  {prediction.delta_away > 0 ? '+' : ''}{prediction.delta_away}% from baseline
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
