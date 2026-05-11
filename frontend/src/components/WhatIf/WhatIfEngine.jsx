import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:8000/api';

export default function WhatIfEngine() {
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  
  const [homeTeam, setHomeTeam] = useState('MI');
  const [awayTeam, setAwayTeam] = useState('CSK');
  const [venue, setVenue] = useState('Wankhede Stadium');
  
  // Overrides
  const [homeBattingAvg, setHomeBattingAvg] = useState(30);
  const [homeEconomyRate, setHomeEconomyRate] = useState(8.5);
  const [homeNrr, setHomeNrr] = useState(0);
  
  const [awayBattingAvg, setAwayBattingAvg] = useState(30);
  const [awayEconomyRate, setAwayEconomyRate] = useState(8.5);
  const [awayNrr, setAwayNrr] = useState(0);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/teams`).then(res => res.json()).then(data => setTeams(data.teams));
    fetch(`${API_BASE}/venues`).then(res => res.json()).then(data => {
      setVenues(data.venues);
      if(data.venues.length > 0) setVenue(data.venues[0]);
    });
  }, []);

  // Debounced API call for sliders
  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      if (homeTeam && awayTeam && venue) {
        fetchPrediction();
      }
    }, 300);

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
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const resetToReal = () => {
    setHomeBattingAvg(30);
    setHomeEconomyRate(8.5);
    setHomeNrr(0);
    setAwayBattingAvg(30);
    setAwayEconomyRate(8.5);
    setAwayNrr(0);
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">What-If Scenario Engine</h1>
      </div>
      <p style={{ color: 'var(--text-muted-on-light)', marginBottom: '2rem' }}>Adjust team stats to see how win probabilities shift in real-time.</p>

      <div className="card-dark" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <div style={{ flex: 1 }}>
            <label style={{ color: 'var(--text-muted-on-dark)' }}>🏠 Home Team</label>
            <select value={homeTeam} onChange={e => setHomeTeam(e.target.value)} style={{ marginTop: '0.5rem' }}>
              {teams.map(t => <option key={t.abbr} value={t.abbr}>{t.abbr}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ color: 'var(--text-muted-on-dark)' }}>🚌 Away Team</label>
            <select value={awayTeam} onChange={e => setAwayTeam(e.target.value)} style={{ marginTop: '0.5rem' }}>
              {teams.map(t => <option key={t.abbr} value={t.abbr}>{t.abbr}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ color: 'var(--text-muted-on-dark)' }}>📍 Venue</label>
            <select value={venue} onChange={e => setVenue(e.target.value)} style={{ marginTop: '0.5rem' }}>
              {venues.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <div className="card-dark">
          <h3 className="text-cyan">{homeTeam} Stats</h3>
          
          <div style={{ marginTop: '1.5rem' }}>
            <label>Batting Average ({homeBattingAvg})</label>
            <input type="range" min="20" max="60" step="1" value={homeBattingAvg} onChange={e => setHomeBattingAvg(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
          <div style={{ marginTop: '1.5rem' }}>
            <label>Economy Rate ({homeEconomyRate})</label>
            <input type="range" min="6.0" max="12.0" step="0.1" value={homeEconomyRate} onChange={e => setHomeEconomyRate(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
          <div style={{ marginTop: '1.5rem' }}>
            <label>Net Run Rate (NRR) ({homeNrr})</label>
            <input type="range" min="-1.5" max="1.5" step="0.1" value={homeNrr} onChange={e => setHomeNrr(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
        </div>

        <div className="card-dark">
          <h3 className="text-accent">{awayTeam} Stats</h3>
          
          <div style={{ marginTop: '1.5rem' }}>
            <label>Batting Average ({awayBattingAvg})</label>
            <input type="range" min="20" max="60" step="1" value={awayBattingAvg} onChange={e => setAwayBattingAvg(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
          <div style={{ marginTop: '1.5rem' }}>
            <label>Economy Rate ({awayEconomyRate})</label>
            <input type="range" min="6.0" max="12.0" step="0.1" value={awayEconomyRate} onChange={e => setAwayEconomyRate(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
          <div style={{ marginTop: '1.5rem' }}>
            <label>Net Run Rate (NRR) ({awayNrr})</label>
            <input type="range" min="-1.5" max="1.5" step="0.1" value={awayNrr} onChange={e => setAwayNrr(Number(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <button onClick={resetToReal} style={{ background: 'var(--card-dark)' }}>Reset to Real Stats</button>
      </div>

      <AnimatePresence>
        {prediction && (
          <motion.div 
            className="card-dark text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h3 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Live Win Probability</h3>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '4rem', fontSize: '1.5rem' }}>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>{prediction.home_team}</span>
                <div className="mono" style={{ fontSize: '2.5rem', color: prediction.delta_home >= 0 ? 'var(--cyan)' : 'var(--rose)' }}>
                  {prediction.home_win_prob}%
                </div>
                <div style={{ fontSize: '1rem', color: prediction.delta_home >= 0 ? 'var(--cyan)' : 'var(--rose)' }}>
                  {prediction.delta_home > 0 ? '+' : ''}{prediction.delta_home}%
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center' }}>vs</div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>{prediction.away_team}</span>
                <div className="mono" style={{ fontSize: '2.5rem', color: prediction.delta_away >= 0 ? 'var(--cyan)' : 'var(--rose)' }}>
                  {prediction.away_win_prob}%
                </div>
                <div style={{ fontSize: '1rem', color: prediction.delta_away >= 0 ? 'var(--cyan)' : 'var(--rose)' }}>
                  {prediction.delta_away > 0 ? '+' : ''}{prediction.delta_away}%
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
