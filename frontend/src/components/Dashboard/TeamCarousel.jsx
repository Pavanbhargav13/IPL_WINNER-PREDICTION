import React, { useCallback, useEffect, useState } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import { ChevronLeft, ChevronRight, Flame, Zap, Dices } from 'lucide-react';
import AnimatedNumber from '../AnimatedNumber';

const TIER_CONFIG = {
  favourite:  { label: 'Favourite',  icon: Flame, color: '#F9415C' },
  contention: { label: 'Contention', icon: Zap,   color: '#4ADE80' },
  longshot:   { label: 'Long Shot',  icon: Dices,  color: '#71717A' },
};

function getTier(prob) {
  if (prob >= 15) return TIER_CONFIG.favourite;
  if (prob >= 3)  return TIER_CONFIG.contention;
  return TIER_CONFIG.longshot;
}

export default function TeamCarousel({ teams, onTeamSelect, selectedTeam }) {
  const [emblaRef, emblaApi] = useEmblaCarousel({
    align: 'center',
    containScroll: 'trimSnaps',
    loop: true,
  });

  const [selectedIndex, setSelectedIndex] = useState(0);
  const [scrollSnaps, setScrollSnaps]     = useState([]);

  const handleSelect = useCallback(() => {
    if (!emblaApi) return;
    setSelectedIndex(emblaApi.selectedScrollSnap());
  }, [emblaApi]);

  useEffect(() => {
    if (!emblaApi) return;
    setScrollSnaps(emblaApi.scrollSnapList());
    emblaApi.on('select', handleSelect);
    handleSelect();
    return () => emblaApi.off('select', handleSelect);
  }, [emblaApi, handleSelect]);

  const scrollTo   = useCallback((i) => emblaApi && emblaApi.scrollTo(i), [emblaApi]);
  const scrollPrev = useCallback(() => emblaApi && emblaApi.scrollPrev(), [emblaApi]);
  const scrollNext = useCallback(() => emblaApi && emblaApi.scrollNext(), [emblaApi]);

  if (!teams || teams.length === 0) return null;

  return (
    <div style={{ position: 'relative', marginBottom: '0.5rem' }}>
      {/* Arrow — Prev */}
      <button
        onClick={scrollPrev}
        aria-label="Previous team"
        style={{
          position: 'absolute', left: '0', top: '50%',
          transform: 'translateY(-50%)', zIndex: 10,
          background: 'rgba(35,35,37,0.9)',
          border: '1px solid rgba(255,255,255,0.1)',
          color: 'white', width: '36px', height: '36px',
          borderRadius: '50%', padding: 0,
          cursor: 'pointer', fontSize: '1rem',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          backdropFilter: 'blur(4px)',
        }}
      >
        <ChevronLeft size={18} />
      </button>

      {/* Viewport */}
      <div ref={emblaRef} style={{ overflow: 'hidden', paddingInline: '3rem' }}>
        <div style={{ display: 'flex', gap: '1rem', userSelect: 'none' }}>
          {teams.map((team, i) => {
            const isActive   = i === selectedIndex;
            const tier       = getTier(team.win_probability);
            const TierIcon   = tier.icon;
            const isChosen   = selectedTeam?.team === team.team;

            return (
              <div
                key={team.team}
                onClick={() => {
                  scrollTo(i);
                  if (onTeamSelect) onTeamSelect(team);
                }}
                style={{
                  flex: '0 0 auto',
                  width: '180px',
                  cursor: 'pointer',
                  transition: 'transform 0.35s ease, opacity 0.35s ease',
                  transform: isActive ? 'scale(1.05)' : 'scale(0.95)',
                  opacity:   isActive ? 1 : 0.55,
                }}
              >
                <div style={{
                  background: 'var(--card-dark)',
                  border: isChosen
                    ? `2px solid ${team.color}`
                    : isActive
                    ? `1px solid ${team.color}77`
                    : '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '16px',
                  padding: '1.25rem 1rem',
                  textAlign: 'center',
                  boxShadow: isActive
                    ? `0 0 24px ${team.color}44, 0 10px 30px rgba(0,0,0,0.3)`
                    : '0 4px 16px rgba(0,0,0,0.2)',
                  transition: 'border 0.3s, box-shadow 0.3s',
                }}>
                  {/* Team colour dot */}
                  <div style={{
                    width: '10px', height: '10px', borderRadius: '50%',
                    background: team.color,
                    boxShadow: `0 0 8px ${team.color}`,
                    margin: '0 auto 0.5rem',
                  }} />

                  {/* Team abbrev */}
                  <div style={{
                    fontSize: '1.5rem', fontWeight: '800',
                    color: isActive ? team.color : 'var(--text-light)',
                    fontFamily: 'var(--font-sans)',
                    transition: 'color 0.3s',
                  }}>
                    {team.team}
                  </div>

                  {/* Full name */}
                  <div style={{
                    fontSize: '0.65rem', color: 'var(--text-muted-on-dark)',
                    marginBottom: '0.75rem',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {team.team_full_name || ''}
                  </div>

                  {/* Probability */}
                  <AnimatedNumber
                    value={team.win_probability}
                    duration={1000}
                    style={{
                      fontSize: '1.6rem', fontWeight: '700',
                      color: 'var(--text-light)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  />

                  {/* Tier */}
                  <div style={{
                    marginTop: '0.5rem',
                    fontSize: '0.65rem', fontWeight: '600',
                    color: tier.color,
                    background: `${tier.color}18`,
                    borderRadius: '20px', padding: '0.2rem 0.5rem',
                    display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
                  }}>
                    <TierIcon size={11} />
                    {tier.label}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Arrow — Next */}
      <button
        onClick={scrollNext}
        aria-label="Next team"
        style={{
          position: 'absolute', right: '0', top: '50%',
          transform: 'translateY(-50%)', zIndex: 10,
          background: 'rgba(35,35,37,0.9)',
          border: '1px solid rgba(255,255,255,0.1)',
          color: 'white', width: '36px', height: '36px',
          borderRadius: '50%', padding: 0,
          cursor: 'pointer', fontSize: '1rem',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          backdropFilter: 'blur(4px)',
        }}
      >
        <ChevronRight size={18} />
      </button>

      {/* Dot indicators */}
      <div style={{
        display: 'flex', justifyContent: 'center',
        gap: '0.4rem', marginTop: '1rem',
      }}>
        {scrollSnaps.map((_, i) => (
          <button
            key={i}
            onClick={() => scrollTo(i)}
            aria-label={`Go to team ${i + 1}`}
            style={{
              width:  i === selectedIndex ? '20px' : '8px',
              height: '8px',
              borderRadius: '99px',
              background: i === selectedIndex
                ? (teams[i]?.color || 'var(--accent-red)')
                : 'rgba(255,255,255,0.2)',
              border: 'none', padding: 0, cursor: 'pointer',
              transition: 'width 0.3s ease, background 0.3s ease',
            }}
          />
        ))}
      </div>
    </div>
  );
}
