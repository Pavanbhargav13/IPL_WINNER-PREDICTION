import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ArrowLeft, Sparkles } from 'lucide-react';
import { useTheme, TEAM_THEMES } from '../../context/ThemeContext';

const QUESTIONS = [
  {
    id: 1,
    question: "Which city's cricket energy calls to you?",
    subtitle: "Your roots shape your game",
    options: [
      { text: "Mumbai — Maximum city, maximum cricket",     scores: { MI: 3, DC: 1 } },
      { text: "Chennai — Where loyalty is a religion",      scores: { CSK: 3, GT: 1 } },
      { text: "Kolkata — The grand stage, loudest ground",  scores: { KKR: 3, RR: 1 } },
      { text: "Bengaluru — Passionate and relentless",      scores: { RCB: 3, PBKS: 1 } },
    ],
  },
  {
    id: 2,
    question: "How should a great cricket team play?",
    subtitle: "Philosophy reveals character",
    options: [
      { text: "Ruthless power — smash everything from ball one", scores: { MI: 2, SRH: 2, PBKS: 1 } },
      { text: "Tactical chess — plan every single delivery",     scores: { CSK: 2, GT: 2, DC: 1 } },
      { text: "Fearless instinct — trust gut over data",         scores: { RCB: 2, RR: 2, KKR: 1 } },
      { text: "Relentless spin — outthink the batsman",          scores: { RR: 2, DC: 2, LSG: 1 } },
    ],
  },
  {
    id: 3,
    question: "Pick your all-time IPL legend",
    subtitle: "The player you'd build a team around",
    options: [
      { text: "Rohit Sharma — The Hitman, ice in his veins",   scores: { MI: 3, DC: 1 } },
      { text: "MS Dhoni — Captain Cool, never panics",         scores: { CSK: 3, GT: 1 } },
      { text: "Virat Kohli — The Chase Master, never quits",   scores: { RCB: 3, PBKS: 1 } },
      { text: "Gautam Gambhir — The Fighter, pure grit",       scores: { KKR: 2, DC: 1, MI: 1 } },
    ],
  },
  {
    id: 4,
    question: "Final over. 22 needed. You are...",
    subtitle: "Pressure reveals who you really are",
    options: [
      { text: "Already calculating which balls to target",     scores: { CSK: 2, GT: 2, MI: 1 } },
      { text: "Walking in to smash every ball for six",        scores: { RCB: 2, SRH: 2, PBKS: 1 } },
      { text: "Backing your bowlers to defend anything",       scores: { KKR: 2, DC: 2, LSG: 1 } },
      { text: "Believing in miracles — always",                scores: { RR: 3, RCB: 1 } },
    ],
  },
  {
    id: 5,
    question: "Your perfect IPL match atmosphere?",
    subtitle: "Where do you truly belong?",
    options: [
      { text: "100,000 voices shaking the Wankhede",          scores: { MI: 3, RCB: 1 } },
      { text: "The yellow sea at Chepauk — pure devotion",     scores: { CSK: 3 } },
      { text: "Eden Gardens — the loudest colosseum on Earth", scores: { KKR: 3, SRH: 1 } },
      { text: "A ferocious home crowd willing you on",         scores: { RR: 2, GT: 2, PBKS: 1 } },
    ],
  },
  {
    id: 6,
    question: "Which jersey calls to you?",
    subtitle: "Sometimes you just know",
    options: [
      { text: "Deep navy blue — composed, powerful",           scores: { MI: 3, GT: 1 } },
      { text: "Blazing yellow — iconic, eternal",              scores: { CSK: 3 } },
      { text: "Bold red and black — fire and fury",            scores: { RCB: 3 } },
      { text: "Electric purple and gold — regal, fearless",    scores: { KKR: 3, RR: 1 } },
    ],
  },
];

function computeTeam(answers) {
  const totals = {};
  Object.keys(TEAM_THEMES).forEach(t => { totals[t] = 0; });
  answers.forEach(ans => {
    Object.entries(ans).forEach(([team, pts]) => {
      totals[team] = (totals[team] || 0) + pts;
    });
  });
  return Object.entries(totals).sort((a, b) => b[1] - a[1])[0][0];
}

export default function OnboardingQuiz() {
  const navigate = useNavigate();
  const { chooseTeam } = useTheme();

  const [step, setStep] = useState(0); // 0 = intro, 1..6 = quiz questions, 7 = result
  const [answers, setAnswers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [resultTeam, setResultTeam] = useState(null);

  const handleStart = () => setStep(1);

  const handleSelectOption = (qIdx, optIdx) => {
    setSelected(optIdx);
  };

  const handleNext = () => {
    if (selected === null) return;
    const newAnswers = [...answers, QUESTIONS[step - 1].options[selected].scores];
    setAnswers(newAnswers);
    setSelected(null);

    if (step < QUESTIONS.length) {
      setStep(step + 1);
    } else {
      const team = computeTeam(newAnswers);
      setResultTeam(team);
      chooseTeam(team);
      setStep(QUESTIONS.length + 1);
    }
  };

  const handleBack = () => {
    if (step > 1) {
      setAnswers(prev => prev.slice(0, -1));
      setSelected(null);
      setStep(step - 1);
    }
  };

  const handleEnterApp = () => navigate('/app');

  return (
    <div className="quiz-carousel-container">
      <div className="quiz-bg-glow" />

      <div className="quiz-card-stack-wrapper">
        <AnimatePresence mode="popLayout">
          
          {/* INTRO SCREEN */}
          {step === 0 && (
            <motion.div
              key="intro"
              className="quiz-card-white"
              initial={{ opacity: 0, scale: 0.9, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, x: -300, rotate: -10 }}
              transition={{ type: "spring", stiffness: 260, damping: 25 }}
            >
              <div className="quiz-intro-label">TEAM SELECTION</div>
              <h1 className="quiz-intro-title">What's your cricket DNA?</h1>
              <p className="quiz-intro-sub">
                Answer 6 questions to configure your custom app branding, color theme, and analytics defaults.
              </p>
              
              <button className="quiz-white-cta" onClick={handleStart}>
                Begin DNA Configuration <ChevronRight size={18} />
              </button>
              <button 
                className="quiz-white-back" 
                onClick={() => navigate('/app')} 
                style={{ marginTop: '1.5rem', alignSelf: 'center' }}
              >
                Skip configuration
              </button>
            </motion.div>
          )}

          {/* QUESTIONS STACK */}
          {step >= 1 && step <= QUESTIONS.length && (
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
              {QUESTIONS.map((q, idx) => {
                const qNum = idx + 1;
                // Only render the current card and the one immediately behind it for performance
                if (qNum < step || qNum > step + 2) return null;

                const isCurrent = qNum === step;
                const offset = qNum - step; // 0 for current, 1 for behind, 2 for further behind

                return (
                  <motion.div
                    key={q.id}
                    className="quiz-card-white"
                    style={{
                      position: 'absolute',
                      top: 0, left: 0,
                      pointerEvents: isCurrent ? 'auto' : 'none'
                    }}
                    initial={{ opacity: 0, scale: 0.9, y: 20 }}
                    animate={{
                      y: offset * 15,
                      scale: 1 - offset * 0.04,
                      opacity: 1 - offset * 0.35,
                      zIndex: 100 - offset
                    }}
                    exit={isCurrent ? { x: -500, rotate: -15, opacity: 0 } : undefined}
                    transition={{ type: "spring", stiffness: 300, damping: 28 }}
                  >
                    {/* Back control */}
                    {isCurrent && (
                      <button className="quiz-white-back" onClick={handleBack}>
                        <ArrowLeft size={14} /> Back
                      </button>
                    )}

                    <div className="quiz-white-progress">
                      QUESTION {qNum} OF {QUESTIONS.length}
                    </div>
                    <div style={{ fontSize: '0.82rem', color: '#8A8A85', fontWeight: '600', marginBottom: '0.2rem' }}>
                      {q.subtitle}
                    </div>
                    <h2 style={{ fontSize: '1.45rem', fontWeight: '900', color: '#FFFBF4', margin: '0 0 1.25rem', letterSpacing: '-0.02em', lineHeight: '1.25' }}>
                      {q.question}
                    </h2>

                    <div className="quiz-options" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {q.options.map((opt, optIdx) => {
                        const isSelected = isCurrent && selected === optIdx;
                        return (
                          <button
                            key={optIdx}
                            className={`quiz-white-option ${isSelected ? 'selected' : ''}`}
                            onClick={() => isCurrent && handleSelectOption(idx, optIdx)}
                          >
                            <span className="quiz-white-option-letter">
                              {String.fromCharCode(65 + optIdx)}
                            </span>
                            <span className="quiz-white-option-text">{opt.text}</span>
                          </button>
                        );
                      })}
                    </div>

                    {isCurrent && (
                      <button 
                        className={`quiz-white-cta ${selected === null ? 'disabled' : ''}`} 
                        onClick={handleNext}
                        disabled={selected === null}
                      >
                        {qNum === QUESTIONS.length ? 'Reveal My Team' : 'Next'} <ChevronRight size={18} />
                      </button>
                    )}
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* RESULT SCREEN */}
          {step === QUESTIONS.length + 1 && resultTeam && (
            <motion.div
              key="result"
              className="quiz-card-white"
              initial={{ opacity: 0, scale: 0.9, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              transition={{ type: "spring", stiffness: 260, damping: 25 }}
              style={{ textAlign: 'center', justifyContent: 'center' }}
            >
              <div className="quiz-intro-label">QUIZ COMPLETE</div>
              <div style={{ fontSize: '0.95rem', color: '#D8CFBC', fontWeight: '700' }}>YOUR MATCHED TEAM IS</div>
              
              <motion.div 
                style={{ 
                  fontSize: '4.5rem', fontWeight: '950', 
                  color: TEAM_THEMES[resultTeam]?.accent,
                  margin: '1rem 0 0.5rem'
                }}
                initial={{ scale: 0.5 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 350, damping: 15 }}
              >
                {resultTeam}
              </motion.div>

              <h2 style={{ fontSize: '1.6rem', fontWeight: '900', color: '#FFFBF4', margin: '0 0 1.5rem' }}>
                {TEAM_THEMES[resultTeam]?.name}
              </h2>

              <p style={{ color: '#D8CFBC', fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '2.5rem' }}>
                The dashboard theme has been adapted to match your team's visual identity.
              </p>

              <button className="quiz-white-cta" onClick={handleEnterApp}>
                Enter Dashboard <ChevronRight size={18} />
              </button>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
