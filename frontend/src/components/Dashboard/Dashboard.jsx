import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, RefreshCw, AlertTriangle } from 'lucide-react';
import html2canvas from 'html2canvas';

import PodiumCard from './PodiumCard';
import TeamRow from './TeamRow';
import TeamDetailPanel from './TeamDetailPanel';
import TeamCarousel from './TeamCarousel';
import AnimatedNumber from '../AnimatedNumber';

const API_BASE = 'http://localhost:8000/api';

const PODIUM_ORDER = [1, 0, 2]; // display: 2nd | 1st | 3rd

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const dashboardRef = useRef(null);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const res  = await fetch(`${API_BASE}/predict`);
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch (err) {
      console.error(err);
      setError('Failed to load predictions. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, []);

  const exportAsPNG = async () => {
    if (!dashboardRef.current) return;
    try {
      const canvas = await html2canvas(dashboardRef.current, {
        backgroundColor: '#1a1617',
        scale: 2,
      });
      const link = document.createElement('a');
      link.download = `ipl-predictions-${new Date().toISOString().split('T')[0]}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (err) {
      console.error('Export failed', err);
    }
  };

  const handleTeamClick = (team) => {
    if (selectedTeam?.team === team.team) {
      setSelectedTeam(null);
    } else {
      // Attach rank for the detail panel
      const idx = predictions.findIndex(p => p.team === team.team);
      setSelectedTeam({ ...team, _rank: idx });
    }
  };

  // ── Loading & Error States ──────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="main-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
          style={{ textAlign: 'center' }}
        >
          <div style={{
            width: '48px', height: '48px', border: '3px solid rgba(249,65,92,0.2)',
            borderTopColor: 'var(--accent-red)', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite', margin: '0 auto 1rem',
          }} />
          <h2 className="text-accent" style={{ margin: 0 }}>Loading Predictions…</h2>
          <p style={{ color: 'var(--text-muted-on-light)', marginTop: '0.5rem' }}>Crunching the numbers for IPL 2026</p>
        </motion.div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="main-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <AlertTriangle size={48} color="var(--negative-red)" />
          </div>
          <h2 className="text-red" style={{ margin: 0 }}>{error}</h2>
          <button className="btn-accent" onClick={fetchPredictions} style={{ marginTop: '1rem' }}>
            <RefreshCw size={14} /> Retry
          </button>
        </div>
      </div>
    );
  }

  const top3 = predictions.slice(0, 3);
  const rest = predictions.slice(3);

  return (
    <div className="main-content" style={{ position: 'relative' }}>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Championship Dashboard</h1>
          <p style={{ color: 'var(--text-muted-on-light)', margin: '0.25rem 0 0', fontSize: '0.9rem' }}>
            IPL 2026 season win probabilities — powered by ML
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={exportAsPNG} style={{ background: 'var(--card-dark)' }}>
            <Camera size={14} /> Export PNG
          </button>
          <button className="btn-accent" onClick={fetchPredictions}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      <div ref={dashboardRef}>
        {/* ═══════════════════════════════════════════════════════════════════════
            F01 — PODIUM: Top 3 Teams
            ═══════════════════════════════════════════════════════════════════════ */}
        <div style={{
          display: 'flex', justifyContent: 'center', alignItems: 'flex-end',
          gap: '1.25rem', marginBottom: '2.5rem', paddingTop: '1rem',
        }}>
          {PODIUM_ORDER.map((rank) => {
            const team = top3[rank];
            if (!team) return null;
            return (
              <motion.div
                key={team.team}
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: rank * 0.15, ease: 'easeOut' }}
              >
                <PodiumCard
                  team={team}
                  rank={rank}
                  onClick={handleTeamClick}
                  isSelected={selectedTeam?.team === team.team}
                />
              </motion.div>
            );
          })}
        </div>

        {/* ═══════════════════════════════════════════════════════════════════════
            F03 — CAROUSEL: All 10 team cards
            ═══════════════════════════════════════════════════════════════════════ */}
        <div className="card-dark" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0 }}>Team Explorer</h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted-on-dark)' }}>
              Swipe or click to explore all 10 teams
            </span>
          </div>
          <TeamCarousel
            teams={predictions}
            onTeamSelect={handleTeamClick}
            selectedTeam={selectedTeam}
          />
        </div>

        {/* ═══════════════════════════════════════════════════════════════════════
            F01 — RANKED LIST + BAR CHART
            ═══════════════════════════════════════════════════════════════════════ */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
          {/* Full Ranked Standings */}
          <div className="card-dark" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Season Standings</h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted-on-dark)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                {predictions.length} teams
              </span>
            </div>

            {/* Column headers */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '1rem',
              padding: '0.4rem 1rem', marginBottom: '0.25rem',
              fontSize: '0.7rem', color: 'var(--text-muted-on-dark)',
              textTransform: 'uppercase', letterSpacing: '0.06em',
              borderBottom: '1px solid rgba(255,255,255,0.06)',
            }}>
              <span style={{ width: '24px', textAlign: 'right' }}>#</span>
              <span style={{ width: '10px' }} />
              <span style={{ width: '200px' }}>Team</span>
              <span style={{ flex: 1 }}>Probability</span>
              <span style={{ width: '56px', textAlign: 'right' }}>Win %</span>
              <span style={{ width: '60px', textAlign: 'right' }}>Odds</span>
            </div>

            {predictions.map((team, i) => (
              <TeamRow
                key={team.team}
                team={team}
                rank={i}
                delay={i * 80}
                onClick={handleTeamClick}
                isSelected={selectedTeam?.team === team.team}
              />
            ))}
          </div>

          {/* Bar Chart — Probability Distribution */}
          <div className="card-dark" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ margin: '0 0 1.5rem' }}>Win Probability Distribution</h3>
            <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end', gap: '6px' }}>
              {predictions.map((team, i) => {
                const maxProb  = predictions[0]?.win_probability || 1;
                const barH     = Math.max(4, (team.win_probability / maxProb) * 100);
                const isActive = selectedTeam?.team === team.team;
                return (
                  <div
                    key={team.team}
                    onClick={() => handleTeamClick(team)}
                    style={{
                      flex: 1, display: 'flex', flexDirection: 'column',
                      alignItems: 'center', cursor: 'pointer',
                    }}
                  >
                    {/* Probability label */}
                    <span style={{
                      fontSize: '0.65rem', color: 'var(--text-muted-on-dark)',
                      fontFamily: 'var(--font-mono)', marginBottom: '0.25rem',
                    }}>
                      {team.win_probability.toFixed(1)}
                    </span>
                    {/* Bar */}
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: `${barH}%` }}
                      transition={{ duration: 1.2, delay: i * 0.05, ease: 'easeOut' }}
                      style={{
                        width: '100%', minHeight: '4px',
                        background: isActive
                          ? team.color
                          : `linear-gradient(180deg, ${team.color}cc, ${team.color}55)`,
                        borderRadius: '4px 4px 0 0',
                        boxShadow: isActive ? `0 0 12px ${team.color}88` : 'none',
                        transition: 'box-shadow 0.3s',
                      }}
                    />
                    {/* Team abbreviation */}
                    <span style={{
                      fontSize: '0.65rem', marginTop: '0.4rem',
                      color: isActive ? team.color : 'var(--text-muted-on-dark)',
                      fontWeight: isActive ? '700' : '400',
                      transition: 'color 0.3s',
                    }}>
                      {team.team}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          F02 — TEAM DETAIL PANEL (slides in from right)
          ═══════════════════════════════════════════════════════════════════════ */}
      <TeamDetailPanel
        team={selectedTeam}
        onClose={() => setSelectedTeam(null)}
      />
    </div>
  );
}
