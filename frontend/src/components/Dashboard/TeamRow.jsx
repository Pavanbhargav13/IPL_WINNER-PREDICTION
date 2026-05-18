import React, { useEffect, useRef } from 'react';
import AnimatedNumber from '../AnimatedNumber';

export default function TeamRow({ team, rank, onClick, isSelected, delay = 0 }) {
  const rowRef = useRef(null);

  // Scroll-reveal via IntersectionObserver (no library)
  useEffect(() => {
    const el = rowRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.style.opacity    = '1';
          el.style.transform  = 'translateY(0)';
          obs.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const maxProb  = 100; // bar is relative to max 100%
  const barWidth = Math.max(2, team.win_probability);

  return (
    <div
      ref={rowRef}
      onClick={() => onClick(team)}
      style={{
        opacity: 0,
        transform: 'translateY(16px)',
        transition: `opacity 0.4s ease ${delay}ms, transform 0.4s ease ${delay}ms`,
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        padding: '0.75rem 1rem',
        borderRadius: '12px',
        cursor: 'pointer',
        background: isSelected ? `${team.color}18` : 'transparent',
        border: isSelected ? `1px solid ${team.color}55` : '1px solid transparent',
        transition: `opacity 0.4s ease ${delay}ms, transform 0.4s ease ${delay}ms, background 0.2s, border 0.2s`,
      }}
      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent'; }}
    >
      {/* Rank number */}
      <span style={{
        width: '24px', textAlign: 'right',
        fontSize: '0.8rem', fontWeight: '600',
        color: rank <= 2 ? 'var(--accent-red)' : 'var(--text-muted-on-dark)',
        flexShrink: 0,
      }}>
        {rank + 1}
      </span>

      {/* Color dot */}
      <div style={{
        width: '10px', height: '10px', borderRadius: '50%',
        background: team.color, flexShrink: 0,
        boxShadow: `0 0 6px ${team.color}88`,
      }} />

      {/* Team name */}
      <span style={{
        width: '200px', fontWeight: '600', fontSize: '0.9rem',
        color: 'var(--text-light)', flexShrink: 0,
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {team.team_full_name || team.team}
      </span>

      {/* Probability bar */}
      <div style={{
        flex: 1, height: '6px',
        background: 'rgba(255,255,255,0.06)',
        borderRadius: '99px', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', borderRadius: '99px',
          background: team.color,
          width: `${barWidth}%`,
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
          boxShadow: `0 0 8px ${team.color}88`,
        }} />
      </div>

      {/* Win probability */}
      <AnimatedNumber
        value={team.win_probability}
        duration={1200}
        style={{
          width: '56px', textAlign: 'right',
          fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: '600',
          color: 'var(--text-light)', flexShrink: 0,
        }}
      />

      {/* Implied odds */}
      <span style={{
        width: '60px', textAlign: 'right',
        fontFamily: 'var(--font-mono)', fontSize: '0.78rem',
        color: 'var(--text-muted-on-dark)', flexShrink: 0,
      }}>
        {team.implied_odds}
      </span>
    </div>
  );
}
