import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

import H2HPredictor from './components/H2H/H2HPredictor';
import WhatIfEngine from './components/WhatIf/WhatIfEngine';
import Dream11Picker from './components/Dream11/Dream11Picker';
import html2canvas from 'html2canvas';

// API Base URL
const API_BASE = 'http://localhost:8000/api';

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

  const handleExport = async () => {
    const element = document.getElementById('dashboard-export');
    if (!element) return;
    try {
      const canvas = await html2canvas(element, { backgroundColor: '#EAECEF' });
      const dataUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = 'ipl-prediction-2025.png';
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Export failed', err);
    }
  };

  if (loading) return <div className="main-content"><h2 className="text-accent">Loading Data...</h2></div>;
  if (error) return <div className="main-content"><h2 className="text-red">{error}</h2></div>;

  const top3 = predictions.slice(0, 3);
  const rest = predictions.slice(3);

  return (
    <div className="main-content" id="dashboard-export">
      <div className="page-header">
        <h1 className="page-title">IPL Prediction Dashboard</h1>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn-accent" onClick={handleExport}>Refresh ↻</button>
          <button onClick={handleExport}>↓ Export</button>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Left Side - Overview Map (Simulated as Top 3) */}
        <div className="card-light" style={{ display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden', minHeight: '350px' }}>
          <h3 style={{ color: 'var(--text-muted-on-light)' }}>Championship Overview</h3>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem', zIndex: 10 }}>
            {top3.map((p, i) => (
              <div key={p.team}>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--text-dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {p.win_probability}% 
                  {i === 0 && <span className="text-green" style={{ fontSize: '1rem' }}>↑ +2.4%</span>}
                </div>
                <div style={{ color: 'var(--text-muted-on-light)' }}>{p.team} - {p.badge}</div>
              </div>
            ))}
          </div>
          
          {/* Decorative map-like element */}
          <div style={{ position: 'absolute', bottom: '-50px', right: '-50px', width: '400px', height: '300px', background: 'radial-gradient(circle, rgba(249,65,92,0.2) 0%, transparent 70%)', borderRadius: '50%', zIndex: 1 }}></div>
        </div>

        {/* Right Side - Key Teams (Top Contenders) */}
        <div className="card-dark" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <h3 style={{ color: 'var(--text-light)' }}>Key Contenders</h3>
            <span style={{ color: 'var(--text-muted-on-dark)' }}>↗</span>
          </div>
          <div style={{ color: 'var(--positive-green)', marginBottom: '1.5rem' }}>10 Total Teams</div>
          
          <div style={{ display: 'flex', alignItems: 'flex-end', height: '100%', gap: '1rem' }}>
            {top3.map((p, i) => (
              <div key={p.team} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <span style={{ marginBottom: '0.5rem', fontSize: '1.2rem' }}>{p.win_probability}%</span>
                <div style={{ width: '100%', height: `${p.win_probability * 2}px`, background: i === 0 ? 'var(--accent-red)' : 'var(--text-muted-on-dark)', borderRadius: '4px 4px 0 0', opacity: i === 0 ? 1 : 0.7 }}></div>
                <span style={{ fontSize: '0.8rem', marginTop: '0.5rem', color: 'var(--text-muted-on-dark)' }}>{p.team}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Left Side - Spend by Category (Simulated as Team Standings) */}
        <div className="card-dark" style={{ padding: '1.5rem 2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h3>Team Standings</h3>
            <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-muted-on-dark)', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-light)' }}>All</span>
              <span>Batting</span>
              <span>Bowling</span>
            </div>
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted-on-dark)', fontSize: '0.75rem', marginBottom: '1rem', textTransform: 'uppercase', paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ width: '150px' }}>Team</span>
            <span style={{ width: '100px' }}>Win Prob</span>
            <span style={{ width: '100px' }}>Implied Odds</span>
            <span style={{ width: '100px' }}>Home Win %</span>
          </div>

          {rest.map((p, index) => (
            <div key={p.team} className="list-row">
              <span style={{ width: '150px', fontWeight: '500' }}>{p.team_full_name}</span>
              <span style={{ width: '100px' }} className="mono">{p.win_probability}%</span>
              <span style={{ width: '100px' }} className="mono">{p.implied_odds}</span>
              <span style={{ width: '100px', color: p.home_win_rate > 50 ? 'var(--positive-green)' : 'var(--text-muted-on-dark)' }}>
                {p.home_win_rate}% {p.home_win_rate > 50 ? '↑' : '↓'}
              </span>
            </div>
          ))}
        </div>

        {/* Right Side - Total Invoice (Simulated as Probability Chart) */}
        <div className="card-dark">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h3>Season Win Probability %</h3>
            <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-muted-on-dark)', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-light)' }}>Teams</span>
            </div>
          </div>
          
          <div style={{ height: '200px', display: 'flex', alignItems: 'flex-end', gap: '10px', marginTop: '3rem' }}>
            {predictions.map(p => (
              <div key={p.team} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div style={{ width: '100%', height: `${p.win_probability * 3}px`, background: 'var(--text-muted-on-dark)', position: 'relative' }}>
                  <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--accent-red)' }}></div>
                </div>
                <span style={{ fontSize: '0.7rem', marginTop: '0.5rem', color: 'var(--text-muted-on-dark)' }}>{p.team}</span>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '2rem', padding: '1rem', background: 'linear-gradient(90deg, rgba(249,65,92,0.2) 0%, transparent 100%)', borderRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Upgrade to access advanced predictions</span>
            <button className="btn-accent" style={{ padding: '0.4rem 1rem' }}>Get Pro</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Navigation wrapper
function SidebarNav() {
  const location = useLocation();
  return (
    <div className="sidebar">
      <div style={{ color: 'var(--text-dark)', fontWeight: 'bold', fontSize: '1.5rem', marginTop: '0.5rem' }}>🏏</div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '2rem' }}>
        <Link to="/" className={`sidebar-icon ${location.pathname === '/' ? 'active' : ''}`}>📊</Link>
        <Link to="/h2h" className={`sidebar-icon ${location.pathname === '/h2h' ? 'active' : ''}`}>⚔️</Link>
        <Link to="/whatif" className={`sidebar-icon ${location.pathname === '/whatif' ? 'active' : ''}`}>🎛️</Link>
        <Link to="/dream11" className={`sidebar-icon ${location.pathname === '/dream11' ? 'active' : ''}`}>👤</Link>
      </div>
      
      <div style={{ marginTop: 'auto' }}>
        <div className="sidebar-icon">⚙️</div>
        <div className="sidebar-icon">🚪</div>
      </div>
    </div>
  );
}

function TopNav() {
  const location = useLocation();
  return (
    <div className="top-nav">
      <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Dashboard</Link>
      <Link to="/h2h" className={`nav-link ${location.pathname === '/h2h' ? 'active' : ''}`}>Head to Head</Link>
      <Link to="/whatif" className={`nav-link ${location.pathname === '/whatif' ? 'active' : ''}`}>What-If Scenario</Link>
      <Link to="/dream11" className={`nav-link ${location.pathname === '/dream11' ? 'active' : ''}`}>Dream11 Squad</Link>
      <Link to="#" className="nav-link">Reports & Analytics</Link>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <div className="app-window">
        <SidebarNav />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '1.5rem 3rem 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <TopNav />
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div style={{ background: 'rgba(0,0,0,0.05)', padding: '0.5rem', borderRadius: '20px', display: 'flex', gap: '0.5rem' }}>
                <span style={{ cursor: 'pointer' }}>☀️</span>
                <span style={{ opacity: 0.5, cursor: 'pointer' }}>🌙</span>
              </div>
              <div style={{ width: '35px', height: '35px', borderRadius: '50%', background: 'var(--card-dark)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>🔔</div>
              <div style={{ width: '35px', height: '35px', borderRadius: '50%', background: 'var(--accent-red)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>PB</div>
            </div>
          </div>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/h2h" element={<H2HPredictor />} />
              <Route path="/whatif" element={<WhatIfEngine />} />
              <Route path="/dream11" element={<Dream11Picker />} />
            </Routes>
          </AnimatePresence>
        </div>
      </div>
    </Router>
  );
}
