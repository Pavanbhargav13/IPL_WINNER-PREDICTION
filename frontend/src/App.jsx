import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import html2canvas from 'html2canvas';

// API Base URL (FastAPI)
const API_BASE = 'http://localhost:8000/api';

// --- Placeholder Components for Routes ---

function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/predict`)
      .then(res => res.json())
      .then(data => {
        setPredictions(data.predictions || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError('Failed to load predictions. Is the backend running?');
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-8 text-center"><h2 className="text-cyan">Loading Predictions...</h2></div>;
  if (error) return <div className="p-8 text-center text-rose"><h2>{error}</h2></div>;

  const handleExport = async () => {
    const element = document.getElementById('dashboard-export');
    if (!element) return;
    
    try {
      const canvas = await html2canvas(element, { backgroundColor: '#0B1120' });
      const dataUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = 'ipl-prediction-2025.png';
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Export failed', err);
    }
  };

  const top3 = predictions.slice(0, 3);
  const rest = predictions.slice(3);

  // Reorder top3 for visual podium: [2nd, 1st, 3rd]
  const podiumOrder = top3.length === 3 ? [top3[1], top3[0], top3[2]] : top3;

  return (
    <div className="dashboard">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2>🏆 Season Winner Predictions</h2>
        <button onClick={handleExport}>
          <span>📸</span> Export Card
        </button>
      </div>

      <div id="dashboard-export" style={{ padding: '2rem', background: 'var(--navy)', margin: '-2rem', borderRadius: '20px' }}>
        
        {/* Podium Section */}
        {podiumOrder.length === 3 && (
          <div className="podium-container">
            {podiumOrder.map((p, index) => {
              const isFirst = p.rank === 1;
              const isSecond = p.rank === 2;
              const podiumClass = isFirst ? 'podium-1' : isSecond ? 'podium-2' : 'podium-3';
              
              return (
                <motion.div 
                  key={p.team}
                  className={`card ${podiumClass}`}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.2, type: "spring" }}
                  style={{ width: '30%', textAlign: 'center' }}
                >
                  <div style={{ fontSize: isFirst ? '3rem' : '2rem', marginBottom: '1rem' }}>
                    {isFirst ? '🥇' : isSecond ? '🥈' : '🥉'}
                  </div>
                  <h3 style={{ margin: 0, fontSize: '1.5rem' }}>{p.team}</h3>
                  <div className="mono text-gold" style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: '1rem 0' }}>{p.win_probability}%</div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{p.badge}</p>
                </motion.div>
              );
            })}
          </div>
        )}

        {/* Rest of the teams */}
        <div className="grid-cards">
        {rest.map((p, index) => (
          <motion.div 
            key={p.team}
            className="card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            style={{ borderLeft: `4px solid ${p.team_color}` }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ margin: 0, fontSize: '1.2rem' }}>
                <span style={{ color: 'var(--text-secondary)', marginRight: '0.5rem' }}>#{p.rank}</span>
                {p.team}
              </h3>
              <span className="mono text-cyan" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{p.win_probability}%</span>
            </div>
            
            <div className="progress-container">
               <motion.div 
                  className="progress-fill" 
                  style={{ background: p.team_color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${p.win_probability}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
               />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginTop: '1.5rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Odds: <span className="mono text-primary">{p.implied_odds}</span></span>
              <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>{p.badge}</span>
            </div>
          </motion.div>
        ))}
        </div>
      </div>
    </div>
  );
}

import H2HPredictor from './components/H2H/H2HPredictor';
import WhatIfEngine from './components/WhatIf/WhatIfEngine';
import Dream11Picker from './components/Dream11/Dream11Picker';

// --- Main App Component ---

function App() {
  return (
    <Router>
      <div className="app-container">
        <header className="app-header glass">
          <h1 className="logo-text">🏏 IPL Oracle '25</h1>
          <nav className="nav-links">
            <Link to="/" className="nav-link">Dashboard</Link>
            <Link to="/h2h" className="nav-link">H2H Engine</Link>
            <Link to="/whatif" className="nav-link">What-If</Link>
            <Link to="/dream11" className="nav-link">Dream11</Link>
          </nav>
        </header>

        <main>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/h2h" element={<H2HPredictor />} />
              <Route path="/whatif" element={<WhatIfEngine />} />
              <Route path="/dream11" element={<Dream11Picker />} />
            </Routes>
          </AnimatePresence>
        </main>
      </div>
    </Router>
  );
}

export default App;
