import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Flame, Zap, Dices } from 'lucide-react';
import AnimatedNumber from '../AnimatedNumber';

const TIER_CONFIG = {
  favourite:  { label: 'Hot Favourite',  icon: Flame, bg: '#F9415C', glow: '#F9415C55' },
  contention: { label: 'In Contention', icon: Zap,   bg: '#4ADE80', glow: '#4ADE8055' },
  longshot:   { label: 'Long Shot',      icon: Dices,  bg: '#71717A', glow: '#71717A44' },
};

function getTier(prob) {
  if (prob >= 15) return 'favourite';
  if (prob >= 3)  return 'contention';
  return 'longshot';
}

// Simulated SHAP-approximate feature contributions from the predictor
const SHAP_LABELS = {
  batting_avg:          'Batting Average',
  bowling_economy:      'Bowling Economy',
  nrr:                  'Net Run Rate',
  home_win_rate:        'Home Win Rate',
  home_away_delta:      'Home/Away Delta',
  venue_adj_batting:    'Venue-Adj Batting',
  late_season_win_rate: 'Late Season Form',
};

export default function TeamDetailPanel({ team, onClose }) {
  const gaugeRef  = useRef(null);
  const [gaugeW, setGaugeW] = useState(0);

  // Animate gauge bar after mount
  useEffect(() => {
    if (!team) return;
    setGaugeW(0);
    const t = setTimeout(() => setGaugeW(team.win_probability), 80);
    return () => clearTimeout(t);
  }, [team]);

  if (!team) return null;

  const tier    = getTier(team.win_probability);
  const tierCfg = TIER_CONFIG[tier];
  const TierIcon = tierCfg.icon;
  const shap    = team.feature_contributions || {};

  // Sort SHAP features by absolute value desc
  const shapEntries = Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 6);
  const maxShap = Math.max(...shapEntries.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <AnimatePresence>
      {team && (
        <motion.div
          key={team.team}
          initial={{ x: '100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '100%', opacity: 0 }}
          transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          style={{
            position: 'absolute', top: 0, right: 0,
            width: '360px', height: '100%',
            background: 'var(--card-dark)',
            borderLeft: `2px solid ${team.color}55`,
            boxShadow: `-20px 0 60px rgba(0,0,0,0.4), 0 0 40px ${team.color}22`,
            display: 'flex', flexDirection: 'column',
            overflowY: 'auto', zIndex: 100,
            padding: '2rem 1.5rem',
          }}
        >
          {/* Close button */}
          <button
            onClick={onClose}
            style={{
              position: 'absolute', top: '1rem', right: '1rem',
              background: 'rgba(255,255,255,0.08)', color: 'white',
              width: '32px', height: '32px', padding: 0,
              borderRadius: '50%', fontSize: '1rem',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', border: 'none',
            }}
          >
            <X size={16} />
          </button>

          {/* Team header */}
          <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            <div style={{
              fontSize: '3rem', fontWeight: '900',
              color: team.color,
              textShadow: `0 0 30px ${team.color}66`,
              fontFamily: 'var(--font-sans)',
            }}>
              {team.team}
            </div>
            <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.85rem' }}>
              {team.team_full_name}
            </div>
          </div>

          {/* Big probability number */}
          <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
            <AnimatedNumber
              value={team.win_probability}
              duration={1000}
              style={{
                fontSize: '4rem', fontWeight: '700',
                color: 'var(--text-light)',
                fontFamily: 'var(--font-mono)',
                lineHeight: 1,
              }}
            />
            <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.8rem', marginTop: '0.25rem' }}>
              season win probability
            </div>
          </div>

          {/* Gauge bar */}
          <div ref={gaugeRef} style={{
            height: '10px', background: 'rgba(255,255,255,0.08)',
            borderRadius: '99px', overflow: 'hidden', marginBottom: '1.5rem',
          }}>
            <div style={{
              height: '100%', borderRadius: '99px',
              background: `linear-gradient(90deg, ${team.color}, ${team.color}99)`,
              width: `${gaugeW}%`,
              transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
              boxShadow: `0 0 12px ${team.color}`,
            }} />
          </div>

          {/* Tier badge */}
          <div style={{
            textAlign: 'center', marginBottom: '1.5rem',
          }}>
            <span style={{
              background: tierCfg.bg, color: 'white',
              padding: '0.3rem 1rem', borderRadius: '20px',
              fontSize: '0.8rem', fontWeight: '700',
              boxShadow: `0 0 16px ${tierCfg.glow}`,
              display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
            }}>
              <TierIcon size={14} />
              {tierCfg.label}
            </span>
          </div>

          {/* Stats row */}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr',
            gap: '0.75rem', marginBottom: '1.5rem',
          }}>
            {[
              { label: 'Implied Odds',  value: team.implied_odds },
              { label: 'Home Win %',    value: `${team.home_win_rate}%` },
              { label: 'Badge',         value: team.badge },
              { label: 'Rank',          value: `#${(team._rank || 0) + 1}` },
            ].map(({ label, value }) => (
              <div key={label} style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '10px', padding: '0.75rem',
              }}>
                <div style={{ color: 'var(--text-muted-on-dark)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {label}
                </div>
                <div style={{ color: 'var(--text-light)', fontWeight: '600', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                  {value}
                </div>
              </div>
            ))}
          </div>

          {/* SHAP feature explanation */}
          {shapEntries.length > 0 && (
            <div>
              <div style={{
                color: 'var(--text-muted-on-dark)', fontSize: '0.75rem',
                textTransform: 'uppercase', letterSpacing: '0.08em',
                marginBottom: '1rem', borderTop: '1px solid rgba(255,255,255,0.08)',
                paddingTop: '1rem',
              }}>
                Feature Contributions
              </div>
              {shapEntries.map(([key, val]) => {
                const pct     = (Math.abs(val) / maxShap) * 100;
                const isPos   = val >= 0;
                const barClr  = isPos ? '#4ADE80' : '#F87171';
                const label   = SHAP_LABELS[key] || key.replace(/_/g, ' ');
                return (
                  <div key={key} style={{ marginBottom: '0.65rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted-on-dark)' }}>{label}</span>
                      <span style={{ fontSize: '0.78rem', color: barClr, fontFamily: 'var(--font-mono)' }}>
                        {isPos ? '+' : ''}{val.toFixed(3)}
                      </span>
                    </div>
                    <div style={{
                      height: '4px', background: 'rgba(255,255,255,0.06)',
                      borderRadius: '99px', overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%', background: barClr,
                        width: `${pct}%`, borderRadius: '99px',
                        transition: 'width 0.8s ease',
                        boxShadow: `0 0 6px ${barClr}99`,
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
