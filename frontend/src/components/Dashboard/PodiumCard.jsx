import React, { useRef, useEffect } from 'react';
import VanillaTilt from 'vanilla-tilt';
import { Flame, Zap, Dices } from 'lucide-react';
import AnimatedNumber from '../AnimatedNumber';

const TIER_CONFIG = {
  favourite:   { label: 'Hot Favourite', icon: Flame,  bg: 'rgba(249,65,92,0.15)', color: '#F9415C' },
  contention:  { label: 'In Contention', icon: Zap,    bg: 'rgba(74,222,128,0.12)', color: '#4ADE80' },
  longshot:    { label: 'Long Shot',     icon: Dices,   bg: 'rgba(161,161,170,0.12)', color: '#A1A1AA' },
};

function getTier(prob) {
  if (prob >= 15) return 'favourite';
  if (prob >= 3)  return 'contention';
  return 'longshot';
}

const MEDAL_STYLES = [
  { bg: 'linear-gradient(135deg, #FFD700, #FFA500)', shadow: '0 2px 12px rgba(255,215,0,0.4)', text: '#7C5800' },
  { bg: 'linear-gradient(135deg, #C0C0C0, #A8A8A8)', shadow: '0 2px 12px rgba(192,192,192,0.4)', text: '#555' },
  { bg: 'linear-gradient(135deg, #CD7F32, #A0522D)', shadow: '0 2px 12px rgba(205,127,50,0.4)', text: '#5C3A1E' },
];

const PODIUM_HEIGHTS = ['260px', '220px', '200px'];
const PODIUM_SCALES  = [1, 0.97, 0.94];

export default function PodiumCard({ team, rank, onClick, isSelected }) {
  const cardRef = useRef(null);
  const tier    = getTier(team.win_probability);
  const tierCfg = TIER_CONFIG[tier];
  const TierIcon = tierCfg.icon;
  const medal   = MEDAL_STYLES[rank];

  useEffect(() => {
    if (!cardRef.current || window.innerWidth < 768) return;
    VanillaTilt.init(cardRef.current, {
      max: 12, speed: 400, glare: true, 'max-glare': 0.15, scale: 1.03,
    });
    return () => cardRef.current?.vanillaTilt?.destroy();
  }, []);

  return (
    <div
      ref={cardRef}
      onClick={() => onClick(team)}
      style={{
        transformStyle: 'preserve-3d',
        cursor: 'pointer',
        minWidth: '180px',
        flex: 1,
        maxWidth: '240px',
      }}
    >
      <div
        style={{
          height: PODIUM_HEIGHTS[rank],
          transform: `scale(${PODIUM_SCALES[rank]})`,
          transformOrigin: 'bottom center',
          background: isSelected
            ? `linear-gradient(135deg, ${team.color}44, ${team.color}22)`
            : 'var(--card-dark)',
          border: isSelected
            ? `2px solid ${team.color}`
            : `1px solid ${team.color}33`,
          borderRadius: '20px',
          padding: '1.5rem 1rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: isSelected
            ? `0 0 30px ${team.color}44, 0 20px 40px rgba(0,0,0,0.3)`
            : '0 20px 40px rgba(0,0,0,0.3)',
          transition: 'border 0.3s, box-shadow 0.3s',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Glow orb */}
        <div style={{
          position: 'absolute', bottom: '-40px', left: '50%',
          transform: 'translateX(-50%)',
          width: '120px', height: '120px',
          background: `radial-gradient(circle, ${team.color}33 0%, transparent 70%)`,
          borderRadius: '50%', pointerEvents: 'none',
        }} />

        {/* Medal badge */}
        <div style={{
          width: '42px', height: '42px', borderRadius: '50%',
          background: medal.bg, boxShadow: medal.shadow,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontWeight: '900', fontSize: '1.1rem', color: medal.text,
          fontFamily: 'var(--font-mono)',
        }}>
          {rank + 1}
        </div>

        {/* Team abbreviation */}
        <div style={{
          fontSize: '2rem', fontWeight: '800',
          color: team.color,
          letterSpacing: '-1px',
          fontFamily: 'var(--font-sans)',
        }}>
          {team.team}
        </div>

        {/* Win probability */}
        <AnimatedNumber
          value={team.win_probability}
          duration={1400}
          style={{
            fontSize: '2.2rem', fontWeight: '700',
            color: 'var(--text-light)',
            fontFamily: 'var(--font-mono)',
          }}
        />

        {/* Implied odds */}
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted-on-dark)' }}>
          {team.implied_odds} odds
        </div>

        {/* Tier badge */}
        <div style={{
          background: tierCfg.bg, color: tierCfg.color,
          padding: '0.2rem 0.6rem', borderRadius: '20px',
          fontSize: '0.7rem', fontWeight: '600', letterSpacing: '0.02em',
          display: 'flex', alignItems: 'center', gap: '0.3rem',
        }}>
          <TierIcon size={12} />
          {tierCfg.label}
        </div>
      </div>
    </div>
  );
}
