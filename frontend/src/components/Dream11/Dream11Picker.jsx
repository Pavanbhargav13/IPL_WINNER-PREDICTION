import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:8000/api';

// Dummy player pool for demo purposes
const PLAYER_POOL = [
  { id: 1, name: 'Virat Kohli', role: 'BAT', team: 'RCB', credits: 10.5 },
  { id: 2, name: 'Rohit Sharma', role: 'BAT', team: 'MI', credits: 10.0 },
  { id: 3, name: 'MS Dhoni', role: 'WK', team: 'CSK', credits: 9.0 },
  { id: 4, name: 'Jasprit Bumrah', role: 'BOWL', team: 'MI', credits: 9.5 },
  { id: 5, name: 'Rashid Khan', role: 'BOWL', team: 'GT', credits: 10.0 },
  { id: 6, name: 'Hardik Pandya', role: 'ALL', team: 'MI', credits: 9.5 },
  { id: 7, name: 'Ravindra Jadeja', role: 'ALL', team: 'CSK', credits: 9.0 },
  { id: 8, name: 'Suryakumar Yadav', role: 'BAT', team: 'MI', credits: 10.0 },
  { id: 9, name: 'Shubman Gill', role: 'BAT', team: 'GT', credits: 9.5 },
  { id: 10, name: 'Kagiso Rabada', role: 'BOWL', team: 'PBKS', credits: 9.0 },
  { id: 11, name: 'Trent Boult', role: 'BOWL', team: 'RR', credits: 9.0 },
  { id: 12, name: 'Glenn Maxwell', role: 'ALL', team: 'RCB', credits: 9.5 },
  { id: 13, name: 'Yuzvendra Chahal', role: 'BOWL', team: 'RR', credits: 9.0 },
  { id: 14, name: 'KL Rahul', role: 'WK', team: 'LSG', credits: 9.5 },
  { id: 15, name: 'Rishabh Pant', role: 'WK', team: 'DC', credits: 9.0 },
];

export default function Dream11Picker() {
  const [venues, setVenues] = useState([]);
  const [venue, setVenue] = useState('Wankhede Stadium');
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/venues`).then(res => res.json()).then(data => {
      setVenues(data.venues);
      if(data.venues.length > 0) setVenue(data.venues[0]);
    });
  }, []);

  const togglePlayer = (id) => {
    if (selectedPlayers.includes(id)) {
      setSelectedPlayers(selectedPlayers.filter(pid => pid !== id));
    } else {
      if (selectedPlayers.length < 11) {
        setSelectedPlayers([...selectedPlayers, id]);
      }
    }
  };

  const handlePredict = async () => {
    if (selectedPlayers.length !== 11) {
      alert("Please select exactly 11 players.");
      return;
    }
    
    setLoading(true);
    setPrediction(null);
    
    try {
      const playerNames = selectedPlayers.map(id => PLAYER_POOL.find(p => p.id === id).name);
      
      const response = await fetch(`${API_BASE}/predict/dream11`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: 'MI', // Dummy teams for demo
          away_team: 'CSK',
          venue: venue,
          selected_players: playerNames
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

  return (
    <div className="dream11-container">
      <h2>👤 Dream11 Squad Strategy</h2>
      <p style={{ color: 'var(--text-secondary)' }}>Select 11 players to see how they perform at a specific venue.</p>

      <div className="card" style={{ marginBottom: '2rem' }}>
        <label>📍 Match Venue</label>
        <select value={venue} onChange={e => setVenue(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: 'var(--navy)', color: 'white', marginTop: '0.5rem', marginBottom: '1rem' }}>
          {venues.map(v => <option key={v} value={v}>{v}</option>)}
        </select>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="text-cyan">{selectedPlayers.length}/11 Players Selected</span>
          <button onClick={handlePredict} disabled={selectedPlayers.length !== 11 || loading}>
            {loading ? 'Analyzing Squad...' : 'Generate Strategy'}
          </button>
        </div>
      </div>

      {prediction && (
        <motion.div className="card" style={{ marginBottom: '2rem', border: '1px solid var(--gold)' }} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
          <h3 className="text-gold">🤖 Strategy Report</h3>
          <p><strong>Pitch Intel:</strong> {prediction.pitch_type} - {prediction.strategy_narrative}</p>
          <p><strong>Captaincy Tip:</strong> {prediction.captain_tip}</p>
          <div style={{ marginTop: '1rem' }}>
            <strong>Actionable Tips:</strong>
            <ul style={{ paddingLeft: '1.5rem', color: 'var(--text-secondary)' }}>
              {prediction.tips.map((tip, i) => <li key={i} style={{ marginBottom: '0.5rem' }}>{tip}</li>)}
            </ul>
          </div>
        </motion.div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
        {PLAYER_POOL.map(player => {
          const isSelected = selectedPlayers.includes(player.id);
          return (
            <motion.div 
              key={player.id}
              className="card"
              style={{ 
                cursor: 'pointer', 
                background: isSelected ? 'var(--navy)' : 'var(--slate)',
                border: isSelected ? '2px solid var(--cyan)' : '1px solid rgba(255,255,255,0.05)',
                transformStyle: 'preserve-3d'
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => togglePlayer(player.id)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: 'bold' }}>{player.name}</span>
                <span className="text-gold">{player.credits}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                <span>{player.team}</span>
                <span>{player.role}</span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
