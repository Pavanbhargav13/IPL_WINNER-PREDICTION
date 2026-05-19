import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Play, ArrowRight, ShieldAlert, Sparkles, Cpu, CheckCircle2, TrendingUp, BrainCircuit, Activity, Github, Linkedin, Mail } from 'lucide-react';

const TEAM_JERSEYS = [
  '#FFD700', '#004BA0', '#EC1C24', '#3A225D', '#FF8225', 
  '#134285', '#FF69B4', '#00A2E8', '#ED1B24', '#1B365D'
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [showBlast, setShowBlast] = useState(false);
  const { scrollY } = useScroll();
  
  // Parallax effects for the hero section
  const yBg = useTransform(scrollY, [0, 1000], [0, 300]);
  const opacityHero = useTransform(scrollY, [0, 500], [1, 0]);

  const handleStart = () => {
    setShowBlast(true);
    setTimeout(() => {
      navigate('/quiz');
    }, 1500);
  };

  return (
    <div className="landing-epic-root">
      
      {/* ====================================================================
          1. EPIC HERO SECTION (100vh)
          ==================================================================== */}
      <motion.section className="epic-hero-section" style={{ opacity: opacityHero }}>
        {/* ── REALISTIC STADIUM BACKGROUND ── */}
        <motion.div 
          className="epic-stadium-bg" 
          style={{ 
            backgroundImage: 'url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?ixlib=rb-4.0.3&auto=format&fit=crop&w=2560&q=80")',
            y: yBg 
          }}
        >
          <div className="epic-stadium-overlay" />
          <div className="epic-stadium-fog" />
        </motion.div>

        {/* ── CINEMATIC LIGHT BEAMS ── */}
        <div className="epic-spotlight left" />
        <div className="epic-spotlight right" />
        <div className="epic-spotlight center" />

        {/* ── THE 11 FIELDING PLAYERS ── */}
        <div className="epic-fielders-layer">
          {[...Array(11)].map((_, i) => (
            <motion.div 
              key={i}
              className="epic-fielder"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1, duration: 1, type: "spring" }}
              style={{
                left: `${15 + (i * 7) + (i % 2 === 0 ? 3 : -3)}%`,
                bottom: `${30 + (i % 3) * 5}%`,
                transform: `scale(${0.7 + (i % 3) * 0.15})`
              }}
            >
              <svg viewBox="0 0 100 200" width="40" height="80">
                <ellipse cx="50" cy="195" rx="30" ry="5" fill="#000" opacity="0.6" />
                <rect x="35" y="60" width="30" height="60" rx="10" fill={TEAM_JERSEYS[i % TEAM_JERSEYS.length]} />
                <rect x="35" y="110" width="12" height="80" rx="5" fill="#FFFBF4" />
                <rect x="53" y="110" width="12" height="80" rx="5" fill="#FFFBF4" />
                <rect x="20" y="65" width="10" height="50" rx="5" fill={TEAM_JERSEYS[i % TEAM_JERSEYS.length]} transform="rotate(15, 25, 65)" />
                <rect x="70" y="65" width="10" height="50" rx="5" fill={TEAM_JERSEYS[i % TEAM_JERSEYS.length]} transform="rotate(-15, 75, 65)" />
                <circle cx="50" cy="35" r="15" fill="#FFE0BD" />
              </svg>
            </motion.div>
          ))}
        </div>

        {/* ── ACTION FOREGROUND: BATSMAN HITTING & BOWLER BOWLING ── */}
        <div className="epic-action-foreground">
          <motion.div className="epic-batsman" initial={{ x: -100, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 1, ease: "easeOut" }}>
            <svg viewBox="0 0 250 250" width="300" height="300">
              <ellipse cx="120" cy="230" rx="70" ry="10" fill="#000" opacity="0.7" />
              <path d="M120,130 L160,220 L180,215 L140,125 Z" fill="#FFFBF4" />
              <path d="M100,135 L60,225 L40,215 L90,125 Z" fill="#FFFBF4" />
              <rect x="145" y="150" width="22" height="65" rx="5" fill="#D8CFBC" transform="rotate(-25, 156, 182)" />
              <rect x="55" y="150" width="22" height="65" rx="5" fill="#D8CFBC" transform="rotate(25, 66, 182)" />
              <path d="M85,60 L145,55 L130,140 L90,135 Z" fill="#EC1C24" />
              <path d="M85,60 Q115,45 145,55 L135,90 Q115,80 95,90 Z" fill="#FFD700" />
              <g transform="rotate(-55, 60, 100)">
                <rect x="50" y="30" width="8" height="40" rx="3" fill="#D8CFBC" />
                <path d="M44,70 L64,70 L68,170 Q68,180 54,180 Q40,180 40,170 Z" fill="#FFFBF4" />
              </g>
              <circle cx="120" cy="30" r="18" fill="#FFFBF4" />
              <path d="M100,25 C100,10 140,10 140,25 C140,30 100,30 100,25 Z" fill="#1B365D" />
            </svg>
          </motion.div>

          <motion.div className="epic-bowler" initial={{ x: 100, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 1, ease: "easeOut" }}>
            <svg viewBox="0 0 250 250" width="250" height="250">
              <ellipse cx="120" cy="240" rx="40" ry="6" fill="#000" opacity="0.4" />
              <path d="M120,120 L160,190 L175,185 L135,110 Z" fill="#FFFBF4" />
              <path d="M110,120 L80,170 L65,160 L100,110 Z" fill="#FFFBF4" />
              <path d="M95,50 L145,60 L130,130 L100,120 Z" fill="#004BA0" />
              <path d="M105,70 Q120,70 135,80" stroke="#FFD700" strokeWidth="4" fill="none" />
              <path d="M95,60 Q40,20 80,0 L90,10 Q60,30 100,60 Z" fill="#FFE0BD" />
              <circle cx="85" cy="5" r="8" fill="#FFF" />
              <circle cx="130" cy="30" r="15" fill="#FFE0BD" />
            </svg>
          </motion.div>
          
          {showBlast && (
            <motion.div className="epic-ball-blast" initial={{ x: "-10vw", scale: 0.5, opacity: 1 }} animate={{ x: "100vw", scale: 3, opacity: 0 }} transition={{ duration: 0.6, ease: "easeIn" }} />
          )}
        </div>

        {/* ── HERO TEXT & UI DECK ── */}
        <div className="epic-ui-deck">
          <motion.div className="epic-tag" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Cpu size={14} /> IPL PREDICTIVE AI ENGINE 2026
          </motion.div>

          <motion.h1 className="epic-title" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            WELCOME TO THE <br />
            <span className="epic-gradient-text">ARENA.</span>
          </motion.h1>

          <motion.p className="epic-desc" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
            Ground-level simulation. Advanced Machine Learning. Uncover what it takes to win in the ultimate championship forecaster.
          </motion.p>

          <motion.div className="epic-actions" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
            <button className="epic-btn-primary" onClick={handleStart}>
              <Sparkles size={18} /> Configure Cricket DNA
            </button>
            <button className="epic-btn-secondary" onClick={() => navigate('/app')}>
              Bypass to Dashboard <ArrowRight size={16} />
            </button>
          </motion.div>
        </div>
      </motion.section>

      {/* ====================================================================
          2. FEATURES SECTION
          ==================================================================== */}
      <section className="epic-content-section" style={{ background: '#0B0C0A' }}>
        <div className="epic-section-header">
          <div className="epic-section-eyebrow">The Engine</div>
          <h2 className="epic-section-title">Powered by Intelligence.</h2>
        </div>

        <div className="epic-feature-grid">
          <div className="epic-feature-card">
            <div className="epic-feature-icon"><BrainCircuit size={28} /></div>
            <h3>Deep Learning Models</h3>
            <p>Our backend Random Forest models are trained on thousands of historical IPL matches, adapting to pitch conditions and toss decisions in real-time.</p>
          </div>
          <div className="epic-feature-card">
            <div className="epic-feature-icon"><Activity size={28} /></div>
            <h3>Live What-If Scenarios</h3>
            <p>Adjust team batting averages and net run rates to instantly compute dynamic probability deltas based on ground-level match modifiers.</p>
          </div>
          <div className="epic-feature-card">
            <div className="epic-feature-icon"><TrendingUp size={28} /></div>
            <h3>Dream11 Integration</h3>
            <p>Select your squad and get an AI-generated strategy narrative. Discover the perfect captaincy pick based on the clash arena's specific pitch type.</p>
          </div>
        </div>
      </section>

      {/* ====================================================================
          3. PRICING SECTION
          ==================================================================== */}
      <section className="epic-content-section">
        <div className="epic-section-header">
          <div className="epic-section-eyebrow">Membership</div>
          <h2 className="epic-section-title">Choose Your Tier.</h2>
        </div>

        <div className="epic-pricing-grid">
          
          <div className="epic-pricing-card">
            <h3 style={{ fontSize: '1.5rem', color: '#FFFBF4', margin: '0 0 0.5rem' }}>Rookie</h3>
            <p style={{ color: '#D8CFBC', margin: 0, fontSize: '0.95rem' }}>Basic predictions for casual fans.</p>
            <div className="epic-price">Free <span>/ season</span></div>
            
            <ul className="epic-pricing-features">
              <li><CheckCircle2 size={16} color="#4ADE80" /> Basic Head-to-Head Odds</li>
              <li><CheckCircle2 size={16} color="#4ADE80" /> Standard Match Previews</li>
              <li style={{ opacity: 0.5 }}><ShieldAlert size={16} /> No What-If Engine Access</li>
              <li style={{ opacity: 0.5 }}><ShieldAlert size={16} /> No Dream11 Integrations</li>
            </ul>

            <button className="epic-btn-secondary" style={{ width: '100%', justifyContent: 'center' }}>
              Start Free
            </button>
          </div>

          <div className="epic-pricing-card pro">
            <div className="epic-pricing-badge">MOST POPULAR</div>
            <h3 style={{ fontSize: '1.5rem', color: '#FF6915', margin: '0 0 0.5rem' }}>Analyst Pro</h3>
            <p style={{ color: '#D8CFBC', margin: 0, fontSize: '0.95rem' }}>Full ML suite for fantasy experts.</p>
            <div className="epic-price">$9.99 <span>/ season</span></div>
            
            <ul className="epic-pricing-features">
              <li><CheckCircle2 size={16} color="#FF6915" /> Advanced What-If Parameter Tuner</li>
              <li><CheckCircle2 size={16} color="#FF6915" /> Unlimited Dream11 Squad Optimizations</li>
              <li><CheckCircle2 size={16} color="#FF6915" /> Real-time Pitch Intelligence Modifiers</li>
              <li><CheckCircle2 size={16} color="#FF6915" /> AI Chatbot Conversational Memory</li>
            </ul>

            <button className="epic-btn-primary" style={{ width: '100%', justifyContent: 'center' }}>
              Upgrade to Pro
            </button>
          </div>

        </div>
      </section>

      {/* ====================================================================
          4. FOOTER
          ==================================================================== */}
      <footer className="epic-footer">
        <div className="epic-footer-content">
          <div className="epic-footer-brand">IPL FORECASTER 2026</div>
          <div className="epic-footer-links">
            <a href="mailto:contact@example.com" className="epic-footer-link">
              <Mail size={18} /> Email
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noreferrer" className="epic-footer-link">
              <Linkedin size={18} /> LinkedIn
            </a>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="epic-footer-link">
              <Github size={18} /> GitHub
            </a>
          </div>
        </div>
        <div className="epic-footer-bottom">
          &copy; {new Date().getFullYear()} IPL Championship Forecaster. All rights reserved.
        </div>
      </footer>

    </div>
  );
}
