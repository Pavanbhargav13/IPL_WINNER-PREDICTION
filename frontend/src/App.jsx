import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Swords,
  SlidersHorizontal,
  Users,
  Settings,
  Sun,
  Moon,
  Bell,
  Trophy,
} from 'lucide-react';

import Dashboard from './components/Dashboard/Dashboard';
import H2HPredictor from './components/H2H/H2HPredictor';
import WhatIfEngine from './components/WhatIf/WhatIfEngine';
import Dream11Picker from './components/Dream11/Dream11Picker';
import Cursor from './components/Cursor';

// ── Sidebar Navigation ───────────────────────────────────────────────────────
function SidebarNav() {
  const location = useLocation();
  return (
    <div className="sidebar">
      <div style={{ color: 'var(--accent-red)', marginTop: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Trophy size={28} strokeWidth={2.2} />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '2rem' }}>
        <Link to="/" className={`sidebar-icon ${location.pathname === '/' ? 'active' : ''}`} title="Dashboard">
          <BarChart3 size={20} />
        </Link>
        <Link to="/h2h" className={`sidebar-icon ${location.pathname === '/h2h' ? 'active' : ''}`} title="Head to Head">
          <Swords size={20} />
        </Link>
        <Link to="/whatif" className={`sidebar-icon ${location.pathname === '/whatif' ? 'active' : ''}`} title="What-If">
          <SlidersHorizontal size={20} />
        </Link>
        <Link to="/dream11" className={`sidebar-icon ${location.pathname === '/dream11' ? 'active' : ''}`} title="Dream11 Squad">
          <Users size={20} />
        </Link>
      </div>

      <div style={{ marginTop: 'auto' }}>
        <div className="sidebar-icon" title="Settings">
          <Settings size={20} />
        </div>
      </div>
    </div>
  );
}

// ── Top Navigation Bar ────────────────────────────────────────────────────────
function TopNav() {
  const location = useLocation();
  return (
    <div className="top-nav">
      <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Dashboard</Link>
      <Link to="/h2h" className={`nav-link ${location.pathname === '/h2h' ? 'active' : ''}`}>Head to Head</Link>
      <Link to="/whatif" className={`nav-link ${location.pathname === '/whatif' ? 'active' : ''}`}>What-If Scenario</Link>
      <Link to="/dream11" className={`nav-link ${location.pathname === '/dream11' ? 'active' : ''}`}>Dream11 Squad</Link>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <Router>
      <Cursor />
      <div className="app-window">
        <SidebarNav />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Header row */}
          <div style={{
            padding: '1.5rem 3rem 0',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            flexShrink: 0,
          }}>
            <TopNav />
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <div className="header-toggle-group">
                <span className="header-toggle-btn active"><Sun size={14} /></span>
                <span className="header-toggle-btn"><Moon size={14} /></span>
              </div>
              <div className="header-icon-btn">
                <Bell size={16} />
              </div>
              <div style={{
                width: '35px', height: '35px', borderRadius: '50%',
                background: 'var(--accent-red)', color: 'white',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: 'bold', fontSize: '0.8rem',
              }}>PB</div>
            </div>
          </div>

          {/* Route content */}
          <div style={{ flex: 1, overflow: 'hidden' }}>
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
      </div>
    </Router>
  );
}
