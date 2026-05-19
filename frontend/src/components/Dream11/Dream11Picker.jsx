import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MapPin, Search, BrainCircuit, Mail, Send, CheckCircle2, Star, Crown, Flame, Target, Zap } from 'lucide-react';
import AnimatedNumber from '../AnimatedNumber';
import PLAYER_POOL from '../../data/players.json';

const API_BASE = 'http://localhost:8000/api';

/* ── Role Config ──────────────────────────────────────────────────────────── */
const ROLE_CONFIG = {
  BAT:  { label: 'BAT',  color: '#22D3EE', bg: 'rgba(34,211,238,0.12)' },
  BOWL: { label: 'BOWL', color: '#F87171', bg: 'rgba(248,113,113,0.12)' },
  ALL:  { label: 'AR',   color: '#A78BFA', bg: 'rgba(167,139,250,0.12)' },
  WK:   { label: 'WK',   color: '#FBBF24', bg: 'rgba(251,191,36,0.12)'  },
};

/* ── Form score generator (deterministic by player id) ────────────────────── */
function getFormDots(id, credits) {
  // Seed based on player id for consistency
  const seed = id * 7919;
  const dots = [];
  for (let i = 0; i < 5; i++) {
    const val = ((seed * (i + 1) * 1103515245 + 12345) >>> 0) % 100;
    // Better players skew toward green
    const threshold = credits >= 9.5 ? 55 : credits >= 8.0 ? 40 : 30;
    if (val > threshold)      dots.push('green');
    else if (val > threshold / 2) dots.push('amber');
    else                      dots.push('red');
  }
  return dots;
}

const DOT_COLOR = { green: '#4ADE80', amber: '#FBBF24', red: '#F87171' };

const TEAM_COLORS = {
  CSK: "#FFD700",
  MI: "#004BA0",
  RCB: "#EC1C24",
  KKR: "#3A225D",
  SRH: "#FF8225",
  GT: "#1B365D",
  RR: "#FF69B4",
  LSG: "#00A2E8",
  PBKS: "#ED1B24",
  DC: "#134285"
};

/* ── Individual Player Card (3D flip) ─────────────────────────────────────── */
function PlayerCard({ player, isSelected, isCaptain, isViceCaptain, onToggle, onSetCaptain, onSetViceCaptain, showResult }) {
  const roleConfig = ROLE_CONFIG[player.role] || ROLE_CONFIG.BAT;
  const formDots   = useMemo(() => getFormDots(player.id, player.credits), [player.id, player.credits]);
  
  const teamColor = TEAM_COLORS[player.team] || '#FF6915';
  const lastName = player.name.split(' ').pop().toUpperCase();
  const firstName = player.name.split(' ')[0].toUpperCase();

  // Generate hot streak score based on green form dots
  const hotStreakScore = formDots.filter(d => d === 'green').length * 20 + 20;

  const handleCardClick = (e) => {
    if (e.target.closest('.role-action-btn')) return;
    onToggle(player.id);
  };

  const getPlayerSilhouette = (role, color) => {
    if (role === 'BAT') {
      return (
        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', opacity: 0.85 }}>
          <circle cx="50" cy="22" r="8" fill={color} />
          <path d="M42,32 L58,32 L54,75 L46,75 Z" fill={color} />
          <path d="M42,32 L22,48 L27,53 L45,38 Z" fill={color} />
          <path d="M58,32 L78,48 L73,53 L55,38 Z" fill={color} />
          {/* Bat */}
          <rect x="74" y="20" width="6" height="35" rx="1" fill={color} transform="rotate(35, 74, 20)" />
        </svg>
      );
    } else if (role === 'BOWL') {
      return (
        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', opacity: 0.85 }}>
          <circle cx="60" cy="20" r="8" fill={color} />
          <path d="M45,30 L55,30 L50,75 L45,75 Z" fill={color} />
          {/* Bowling arm high up */}
          <path d="M55,30 Q40,10 50,0 L58,5 Q48,15 60,30 Z" fill={color} />
          {/* Ball */}
          <circle cx="48" cy="2" r="3.5" fill="#FFF" />
        </svg>
      );
    } else {
      // Wicket Keeper / All Rounder
      return (
        <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%', opacity: 0.85 }}>
          <circle cx="50" cy="20" r="8" fill={color} />
          <path d="M40,30 L60,30 L55,75 L45,75 Z" fill={color} />
          <path d="M40,30 L25,50 L30,55 L43,35 Z" fill={color} />
          <path d="M60,30 L75,50 L70,55 L57,35 Z" fill={color} />
        </svg>
      );
    }
  };

  return (
    <div
      className={`flip-card ${showResult && isSelected ? 'card-exit' : ''}`}
      style={{ '--index': player._index }}
      onClick={handleCardClick}
    >
      <div className={`flip-card-inner ${isSelected ? 'is-flipped' : ''}`}>

        {/* ── FRONT FACE (Premium Trading Card) ── */}
        <div className="flip-card-front" style={{ background: '#0F100D', border: `1px solid ${teamColor}33`, overflow: 'hidden', padding: 0 }}>
          
          {/* Team Color Top Accent */}
          <div style={{ height: '5px', background: teamColor, width: '100%' }} />

          {/* Large Background Typography (Last Name) */}
          <div style={{
            position: 'absolute', top: '10%', left: '5%', width: '90%',
            fontSize: '2.5rem', fontWeight: '950', color: `${teamColor}0b`,
            textTransform: 'uppercase', fontStyle: 'italic', pointerEvents: 'none',
            lineHeight: 1, zIndex: 1, wordBreak: 'break-all'
          }}>
            {lastName}
          </div>

          {/* Player Silhouette (AI/Illustration cutout effect) */}
          <div style={{
            position: 'absolute', top: '25%', left: '10%', width: '80%', height: '50%',
            zIndex: 2, display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            {getPlayerSilhouette(player.role, teamColor)}
          </div>

          {/* Top Left Team Badge */}
          <div style={{
            position: 'absolute', top: '15px', left: '15px',
            fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: '900',
            color: '#FFFBF4', background: `${teamColor}33`, padding: '0.2rem 0.5rem',
            borderRadius: '6px', border: `1px solid ${teamColor}55`, zIndex: 3
          }}>
            {player.team}
          </div>

          {/* Circle Overlay Badges (Stats & Role) */}
          <div style={{
            position: 'absolute', right: '12px', top: '15px',
            display: 'flex', flexDirection: 'column', gap: '0.4rem', zIndex: 3
          }}>
            {/* Role Icon */}
            <div 
              title={`Role: ${player.role}`}
              style={{
                width: '26px', height: '26px', borderRadius: '50%',
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: roleConfig.color
              }}
            >
              <Target size={12} />
            </div>

            {/* Form Heat/Flame Icon */}
            <div 
              title={`Form Rating: ${hotStreakScore}%`}
              style={{
                width: '26px', height: '26px', borderRadius: '50%',
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FF6915'
              }}
            >
              <Flame size={12} />
            </div>

            {/* Credits Icon */}
            <div 
              title={`Credits: ${player.credits}`}
              style={{
                width: '26px', height: '26px', borderRadius: '50%',
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FBBF24'
              }}
            >
              <Zap size={12} />
            </div>
          </div>

          {/* Footer Info Area */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0, width: '100%',
            padding: '1.25rem 1.25rem 1rem', background: 'linear-gradient(0deg, #070805 60%, transparent 100%)',
            zIndex: 4
          }}>
            <div style={{ fontSize: '0.62rem', fontWeight: '800', color: `${teamColor}`, letterSpacing: '0.1em' }}>
              {firstName}
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: '900', color: '#FFFBF4', lineHeight: 1.1, margin: '2px 0 6px' }}>
              {player.name}
            </div>
            
            {/* Form Dots */}
            <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
              {formDots.map((dot, i) => (
                <span
                  key={i}
                  style={{
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: DOT_COLOR[dot], boxShadow: `0 0 4px ${DOT_COLOR[dot]}88`
                  }}
                />
              ))}
              <span style={{ fontSize: '0.65rem', color: '#888', marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontWeight: '700' }}>
                {player.credits} CR
              </span>
            </div>
          </div>

        </div>

        {/* ── BACK FACE (selected state) ── */}
        <div className="flip-card-back" style={{ background: '#0F100D', border: `2px solid ${teamColor}` }}>
          {/* Glow orb */}
          <div className="flip-card-glow" style={{ background: `radial-gradient(circle, ${teamColor}33, transparent 70%)` }} />

          <div className="flip-card-back-name">{player.name}</div>
          <div className="flip-card-back-team" style={{ color: teamColor }}>{player.team}</div>

          {/* Captain / Vice-Captain buttons */}
          <div className="flip-card-actions">
            <button
              className={`role-action-btn ${isCaptain ? 'role-action-btn--active-c' : ''}`}
              title="Set as Captain"
              onClick={(e) => { e.stopPropagation(); onSetCaptain(player.id); }}
            >
              <Crown size={13} />
              <span>C</span>
            </button>
            <button
              className={`role-action-btn ${isViceCaptain ? 'role-action-btn--active-vc' : ''}`}
              title="Set as Vice-Captain"
              onClick={(e) => { e.stopPropagation(); onSetViceCaptain(player.id); }}
            >
              <Star size={13} />
              <span>VC</span>
            </button>
          </div>

          <div className="flip-card-back-credits">{player.credits} cr</div>
        </div>
      </div>
    </div>
  );
}

/* ── Main Dream11 Picker ─────────────────────────────────────────────────── */
export default function Dream11Picker() {
  const [venues, setVenues]               = useState([]);
  const [venue, setVenue]                 = useState('Wankhede Stadium');
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [captain, setCaptain]             = useState(null);
  const [viceCaptain, setViceCaptain]     = useState(null);
  const [searchQuery, setSearchQuery]     = useState('');

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading]       = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [email, setEmail]           = useState('');
  const [emailStatus, setEmailStatus] = useState('');

  useEffect(() => {
    fetch(`${API_BASE}/venues`)
      .then(res => res.json())
      .then(data => {
        setVenues(data.venues);
        if (data.venues.length > 0) setVenue(data.venues[0]);
      })
      .catch(() => {});
  }, []);

  /* ── Player selection ─────────────────────────────────────────────────── */
  const togglePlayer = (id) => {
    if (selectedPlayers.includes(id)) {
      setSelectedPlayers(prev => prev.filter(pid => pid !== id));
      if (captain === id)     setCaptain(null);
      if (viceCaptain === id) setViceCaptain(null);
    } else {
      if (selectedPlayers.length < 11) {
        setSelectedPlayers(prev => [...prev, id]);
      }
    }
  };

  const handleSetCaptain = (id) => {
    if (viceCaptain === id) setViceCaptain(null);
    setCaptain(prev => prev === id ? null : id);
  };

  const handleSetViceCaptain = (id) => {
    if (captain === id) setCaptain(null);
    setViceCaptain(prev => prev === id ? null : id);
  };

  /* ── Filtered player list ─────────────────────────────────────────────── */
  const filteredPlayers = useMemo(() => {
    let filtered = PLAYER_POOL;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q)
      );
    }
    const selected   = PLAYER_POOL.filter(p => selectedPlayers.includes(p.id));
    const topMatches = filtered.filter(p => !selectedPlayers.includes(p.id)).slice(0, 50);
    return [...selected, ...topMatches].map((p, i) => ({ ...p, _index: i }));
  }, [searchQuery, selectedPlayers]);

  /* ── Prediction ───────────────────────────────────────────────────────── */
  const handlePredict = async () => {
    if (selectedPlayers.length !== 11) {
      alert('Please select exactly 11 players.');
      return;
    }
    setLoading(true);
    setShowResult(true);
    setPrediction(null);

    try {
      const playerNames = selectedPlayers.map(id => PLAYER_POOL.find(p => p.id === id).name);
      const response = await fetch(`${API_BASE}/predict/dream11`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: 'MI',
          away_team: 'CSK',
          venue,
          selected_players: playerNames,
        }),
      });
      const data = await response.json();
      setTimeout(() => setPrediction(data), 500);
    } catch (err) {
      alert('Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSendEmail = async () => {
    if (!email) { alert('Please enter your email.'); return; }
    setEmailStatus('sending');
    try {
      const res = await fetch(`${API_BASE}/email/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, report_data: prediction || {} }),
      });
      setEmailStatus(res.ok ? 'sent' : 'error');
    } catch {
      setEmailStatus('error');
    }
  };

  const progress     = (selectedPlayers.length / 11) * 100;
  const isSquadFull  = selectedPlayers.length === 11;

  /* ── Render ───────────────────────────────────────────────────────────── */
  return (
    <div className="main-content">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Dream11 Squad Builder</h1>
          <p style={{ color: 'var(--text-muted-on-light)', margin: '0.25rem 0 0', fontSize: '0.9rem' }}>
            Pick 11 players · Set Captain &amp; Vice-Captain · Generate strategy
          </p>
        </div>
      </div>

      {/* Controls row */}
      <div className="card-dark" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label className="field-label"><MapPin size={14} /> Match Venue</label>
            <select
              value={venue}
              onChange={e => setVenue(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', background: 'var(--navy)', color: 'white', marginTop: '0.5rem' }}
            >
              {venues.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label className="field-label"><Search size={14} /> Search Players</label>
            <input
              type="text"
              placeholder="Search by name or team…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                width: '100%', padding: '0.5rem', marginTop: '0.5rem',
                background: 'var(--app-bg)', color: '#111',
                border: '1px solid var(--text-muted-on-dark)', borderRadius: '4px',
              }}
            />
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
            <span style={{ color: isSquadFull ? 'var(--positive-green)' : 'var(--accent-red)', fontWeight: '600', fontSize: '0.9rem' }}>
              {selectedPlayers.length}/11 Players Selected
            </span>
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem', color: 'var(--text-muted-on-dark)' }}>
              {captain     && <span><Crown size={12} style={{ color: '#F59E0B', marginRight: 3 }} />C set</span>}
              {viceCaptain && <span><Star  size={12} style={{ color: '#A78BFA', marginRight: 3 }} />VC set</span>}
              {!captain    && selectedPlayers.length > 0 && <span style={{ color: '#F59E0B' }}>Set Captain ↓</span>}
            </div>
          </div>
          <div style={{ height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '99px', overflow: 'hidden' }}>
            <div style={{
              height: '100%', borderRadius: '99px',
              background: isSquadFull
                ? 'linear-gradient(90deg, #4ADE80, #22C55E)'
                : 'linear-gradient(90deg, var(--accent-red), #FF6B8A)',
              width: `${progress}%`,
              transition: 'width 0.4s cubic-bezier(0.4,0,0.2,1)',
              boxShadow: isSquadFull ? '0 0 12px rgba(74,222,128,0.5)' : '0 0 8px rgba(249,65,92,0.4)',
            }} />
          </div>
        </div>

        {/* Role legend */}
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          {Object.entries(ROLE_CONFIG).map(([key, cfg]) => (
            <span key={key} style={{
              fontSize: '0.7rem', fontWeight: '600', padding: '0.15rem 0.6rem',
              borderRadius: '20px', color: cfg.color, background: cfg.bg,
            }}>{cfg.label}</span>
          ))}
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted-on-dark)', marginLeft: 'auto', alignSelf: 'center' }}>
            Click card to select · Flip back to set C/VC
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            className="btn-accent"
            onClick={handlePredict}
            disabled={!isSquadFull || loading}
          >
            <BrainCircuit size={16} />
            {loading ? 'Analyzing…' : 'Generate Strategy'}
          </button>
        </div>
      </div>

      {/* Strategy result panel */}
      {prediction && !loading && (
        <motion.div
          className="card-dark"
          style={{ marginBottom: '2rem', border: '1px solid var(--accent-red)' }}
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ type: 'spring', damping: 20 }}
        >
          <h3 className="text-accent" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BrainCircuit size={18} /> Strategy Report
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: '10px', padding: '1rem' }}>
              <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Win Probability</div>
              <AnimatedNumber
                value={prediction.win_prob * 100 || 58}
                style={{ fontSize: '2rem', color: 'var(--positive-green)', fontWeight: '700', fontFamily: 'var(--font-mono)' }}
              />
            </div>
            <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: '10px', padding: '1rem' }}>
              <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Pitch Type</div>
              <div style={{ fontWeight: '600', marginTop: '0.25rem' }}>{prediction.pitch_type || '—'}</div>
            </div>
          </div>

          {prediction.captain_tip && (
            <div style={{ background: 'rgba(249,65,92,0.08)', borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1rem', borderLeft: '3px solid var(--accent-red)' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted-on-dark)', display: 'block' }}>Captaincy Tip</span>
              <span style={{ fontWeight: '600' }}>{prediction.captain_tip}</span>
            </div>
          )}

          {prediction.strategy_narrative && (
            <p style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.9rem', marginBottom: '1rem' }}>
              {prediction.strategy_narrative}
            </p>
          )}

          {prediction.tips && prediction.tips.length > 0 && (
            <div style={{ marginBottom: '1.25rem' }}>
              <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>Actionable Tips</div>
              <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-muted-on-dark)', margin: 0 }}>
                {prediction.tips.map((tip, i) => (
                  <li key={i} style={{ marginBottom: '0.4rem', fontSize: '0.9rem' }}>{tip}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Email section */}
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--navy)', borderRadius: '8px' }}>
            <h4 style={{ color: 'var(--text-muted-on-dark)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Mail size={16} /> Email Full Report
            </h4>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="email"
                placeholder="your@email.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                style={{ flex: 1, padding: '0.5rem', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(255,255,255,0.06)', color: 'white' }}
              />
              <button
                className="btn-accent"
                onClick={handleSendEmail}
                disabled={emailStatus === 'sending'}
              >
                {emailStatus === 'sending' ? <><Send size={14} /> Sending…</> :
                 emailStatus === 'sent'    ? <><CheckCircle2 size={14} /> Sent!</> :
                                             <><Send size={14} /> Send via n8n</>}
              </button>
            </div>
            {emailStatus === 'error' && (
              <p style={{ color: 'var(--accent-red)', fontSize: '0.82rem', marginTop: '0.5rem' }}>
                Failed to trigger email workflow. Is the backend running?
              </p>
            )}
          </div>
        </motion.div>
      )}

      {/* Player grid — 140×180px cards per spec */}
      <div className="player-grid">
        {filteredPlayers.map((player) => {
          const isSelected    = selectedPlayers.includes(player.id);
          const shouldHide    = showResult && !isSelected;
          if (shouldHide) return null;

          return (
            <PlayerCard
              key={player.id}
              player={player}
              isSelected={isSelected}
              isCaptain={captain === player.id}
              isViceCaptain={viceCaptain === player.id}
              onToggle={togglePlayer}
              onSetCaptain={handleSetCaptain}
              onSetViceCaptain={handleSetViceCaptain}
              showResult={showResult}
            />
          );
        })}
      </div>
    </div>
  );
}
