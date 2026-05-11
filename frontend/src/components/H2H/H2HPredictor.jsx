import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const API_BASE = 'http://localhost:8000/api';

export default function H2HPredictor() {
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  
  const [homeTeam, setHomeTeam] = useState('MI');
  const [awayTeam, setAwayTeam] = useState('CSK');
  const [venue, setVenue] = useState('Wankhede Stadium');
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch teams and venues on mount
  useEffect(() => {
    fetch(`${API_BASE}/teams`)
      .then(res => res.json())
      .then(data => setTeams(data.teams))
      .catch(err => console.error(err));

    fetch(`${API_BASE}/venues`)
      .then(res => res.json())
      .then(data => {
        setVenues(data.venues);
        if(data.venues.length > 0) setVenue(data.venues[0]);
      })
      .catch(err => console.error(err));
  }, []);

  const handlePredict = async () => {
    if (homeTeam === awayTeam) {
      setError("Home and Away teams must be different");
      return;
    }
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const response = await fetch(`${API_BASE}/predict/h2h`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, venue })
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed');
      
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Head-to-Head Predictor</h1>
      </div>
      
      <div className="card-dark" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted-on-dark)' }}>🏠 Home Team</label>
            <select 
              value={homeTeam} 
              onChange={e => setHomeTeam(e.target.value)}
              style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', background: 'var(--navy)', color: 'white', border: '1px solid var(--text-secondary)' }}
            >
              {teams.map(t => <option key={t.abbr} value={t.abbr}>{t.full_name} ({t.abbr})</option>)}
            </select>
          </div>
          
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted-on-dark)' }}>🚌 Away Team</label>
            <select 
              value={awayTeam} 
              onChange={e => setAwayTeam(e.target.value)}
            >
              {teams.map(t => <option key={t.abbr} value={t.abbr}>{t.full_name} ({t.abbr})</option>)}
            </select>
          </div>
          
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted-on-dark)' }}>📍 Venue</label>
            <select 
              value={venue} 
              onChange={e => setVenue(e.target.value)}
            >
              {venues.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          
          <button className="btn-accent" onClick={handlePredict} disabled={loading} style={{ height: '45px', flexShrink: 0 }}>
            {loading ? 'Predicting...' : 'Predict Match'}
          </button>
        </div>
        {error && <p className="text-red" style={{ marginTop: '1rem' }}>{error}</p>}
      </div>

      {/* Results Section */}
      {prediction && (
        <motion.div 
          className="card-dark"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, type: 'spring' }}
        >
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h3 style={{ margin: 0, fontSize: '2rem' }}>
              <span style={{ color: prediction.winner === prediction.home_team ? 'var(--gold)' : 'var(--text-secondary)' }}>{prediction.home_team}</span> 
              <span style={{ fontSize: '1rem', margin: '0 1rem' }}>vs</span> 
              <span style={{ color: prediction.winner === prediction.away_team ? 'var(--gold)' : 'var(--text-secondary)' }}>{prediction.away_team}</span>
            </h3>
            <p className="text-cyan">🏆 Winner: {prediction.winner === prediction.home_team ? prediction.home_full : prediction.away_full}</p>
          </div>

          {/* Animated Split Bar */}
          <div style={{ height: '30px', background: 'var(--navy)', borderRadius: '15px', overflow: 'hidden', display: 'flex', margin: '2rem 0' }}>
            <motion.div 
              style={{ background: 'var(--cyan)', display: 'flex', alignItems: 'center', paddingLeft: '1rem', fontWeight: 'bold' }}
              initial={{ width: '50%' }}
              animate={{ width: `${prediction.home_win_prob}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            >
              {prediction.home_win_prob}%
            </motion.div>
            <motion.div 
              style={{ background: 'var(--rose)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '1rem', fontWeight: 'bold' }}
              initial={{ width: '50%' }}
              animate={{ width: `${prediction.away_win_prob}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            >
              {prediction.away_win_prob}%
            </motion.div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
            <div>
              <h4 className="text-gold">🏟️ Venue Intelligence</h4>
              <p><strong>Pitch:</strong> {prediction.pitch_type}</p>
              <p><strong>Avg 1st Innings:</strong> {Math.round(prediction.avg_first_innings_score)}</p>
              <p><strong>Chase Win %:</strong> {prediction.chase_win_pct}%</p>
              <p><strong>Ground Size:</strong> {prediction.ground_size === 'S' ? 'Small' : prediction.ground_size === 'M' ? 'Medium' : 'Large'}</p>
            </div>
            <div>
              <h4 className="text-gold">💡 Strategy</h4>
              <p><strong>Toss Advice:</strong> {prediction.toss_advice}</p>
              <p><strong>Home Advantage:</strong> {prediction.home_advantage_modifier > 0 ? `+${prediction.home_advantage_modifier}% to ${prediction.home_team}` : 'Neutral/Negative'}</p>
              <p style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>"{prediction.venue_description}"</p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
