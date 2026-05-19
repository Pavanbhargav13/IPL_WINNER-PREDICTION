import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import {
  BarChart3, Swords, SlidersHorizontal, Users,
  Settings, Sun, Moon, Bell, Trophy,
} from 'lucide-react';

import { ThemeProvider, useTheme } from './context/ThemeContext';
import LandingPage    from './components/Landing/LandingPage';
import OnboardingQuiz from './components/Onboarding/OnboardingQuiz';
import Dashboard      from './components/Dashboard/Dashboard';
import H2HPredictor   from './components/H2H/H2HPredictor';
import WhatIfEngine   from './components/WhatIf/WhatIfEngine';
import Dream11Picker  from './components/Dream11/Dream11Picker';
import Cursor         from './components/Cursor';

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
function SidebarNav({ onOpenSettings }) {
  const location = useLocation();
  const { theme } = useTheme();
  const accentColor = theme?.accent || 'var(--team-accent, white)';

  return (
    <div className="sidebar">
      <div style={{ color: accentColor, marginTop: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Trophy size={28} strokeWidth={2.2} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '2rem' }}>
        {[
          { to: '/app',        icon: BarChart3,        label: 'Dashboard'   },
          { to: '/app/h2h',    icon: Swords,           label: 'Head to Head'},
          { to: '/app/whatif', icon: SlidersHorizontal, label: 'What-If'    },
          { to: '/app/dream11',icon: Users,            label: 'Dream11'     },
        ].map(({ to, icon: Icon, label }) => (
          <Link
            key={to}
            to={to}
            className={`sidebar-icon ${location.pathname === to ? 'active' : ''}`}
            title={label}
            style={location.pathname === to ? { background: accentColor, color: 'white' } : {}}
          >
            <Icon size={20} />
          </Link>
        ))}
      </div>
      <div style={{ marginTop: 'auto' }}>
        <div className="sidebar-icon" title="Settings" onClick={onOpenSettings} style={{ cursor: 'pointer' }}>
          <Settings size={20} />
        </div>
      </div>
    </div>
  );
}

/* ── Top Nav ─────────────────────────────────────────────────────────────── */
function TopNav() {
  const location = useLocation();
  const { theme } = useTheme();
  const accentColor = theme?.accent;

  return (
    <div className="top-nav">
      {[
        { to: '/app',        label: 'Dashboard'      },
        { to: '/app/h2h',    label: 'Head to Head'   },
        { to: '/app/whatif', label: 'What-If'        },
        { to: '/app/dream11',label: 'Dream11 Squad'  },
      ].map(({ to, label }) => (
        <Link
          key={to}
          to={to}
          className={`nav-link ${location.pathname === to ? 'active' : ''}`}
          style={location.pathname === to && accentColor
            ? { background: accentColor, color: 'white' }
            : {}}
        >
          {label}
        </Link>
      ))}
    </div>
  );
}

/* ── Main App Shell (after quiz) ─────────────────────────────────────────── */
function AppShell() {
  const { selectedTeam, theme, clearTeam } = useTheme();
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);

  const clearAiMemory = () => {
    localStorage.removeItem('ai_memories');
    alert('AI Chat Memory has been cleared successfully.');
    setIsSettingsOpen(false);
  };

  return (
    <div className="app-window">
      <SidebarNav onOpenSettings={() => setIsSettingsOpen(true)} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
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
            <div className="header-icon-btn"><Bell size={16} /></div>
            {/* Team badge — shows chosen team */}
            {theme && (
              <div
                title={`${theme.name} — Click to change team`}
                onClick={clearTeam}
                style={{
                  padding: '0.25rem 0.75rem',
                  borderRadius: '20px',
                  background: theme.accent,
                  color: 'white',
                  fontWeight: '700',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  letterSpacing: '0.05em',
                  transition: 'opacity 0.2s'
                }}
                onMouseOver={(e) => e.target.style.opacity = 0.8}
                onMouseOut={(e) => e.target.style.opacity = 1}
              >
                {theme.abbr}
              </div>
            )}
            <div style={{
              width: '35px', height: '35px', borderRadius: '50%',
              background: theme?.accent || '#333',
              color: 'white', display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontWeight: 'bold', fontSize: '0.8rem',
            }}>PB</div>
          </div>
        </div>

        {/* Page content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/"       element={<Dashboard />} />
              <Route path="/h2h"    element={<H2HPredictor />} />
              <Route path="/whatif" element={<WhatIfEngine />} />
              <Route path="/dream11" element={<Dream11Picker />} />
            </Routes>
          </AnimatePresence>
        </div>
      </div>

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          <div className="pos-card" style={{ padding: '2.5rem', width: '400px', maxWidth: '90%' }}>
            <h2 style={{ margin: '0 0 1.5rem', fontSize: '1.5rem', fontWeight: '800', color: '#FFFBF4' }}>Settings</h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <button 
                onClick={clearTeam} 
                className="epic-btn-secondary" 
                style={{ width: '100%', justifyContent: 'center' }}
              >
                Reset Team DNA Quiz
              </button>
              
              <button 
                onClick={clearAiMemory} 
                className="epic-btn-secondary" 
                style={{ width: '100%', justifyContent: 'center', color: '#F87171', borderColor: 'rgba(248,113,113,0.3)' }}
              >
                Erase AI Bot Memory
              </button>

              <button 
                onClick={() => setIsSettingsOpen(false)} 
                className="epic-btn-primary" 
                style={{ width: '100%', justifyContent: 'center', marginTop: '1rem' }}
              >
                Close Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Root ────────────────────────────────────────────────────────────────── */
function AppRoutes() {
  return (
    <>
      <Cursor />
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/"     element={<LandingPage />} />
          <Route path="/quiz" element={<OnboardingQuiz />} />
          <Route path="/app/*" element={<AppShell />} />
          {/* catch-all → landing */}
          <Route path="*"     element={<Navigate to="/" replace />} />
        </Routes>
      </AnimatePresence>
    </>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppRoutes />
      </Router>
    </ThemeProvider>
  );
}
