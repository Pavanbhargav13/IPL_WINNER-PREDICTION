import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ArrowLeft } from 'lucide-react';
import { useTheme, TEAM_THEMES } from '../../context/ThemeContext';

/* ── Quiz Data ───────────────────────────────────────────────────────────── */
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

/* ── Scoring Algorithm ───────────────────────────────────────────────────── */
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

/* ── Question slide variants ─────────────────────────────────────────────── */
const slideVariants = {
  enter: { x: 80, opacity: 0 },
  center: { x: 0,  opacity: 1 },
  exit:  { x: -80, opacity: 0 },
};

/* ── Main Component ──────────────────────────────────────────────────────── */
export default function OnboardingQuiz() {
  const navigate           = useNavigate();
  const { chooseTeam }     = useTheme();

  const [step, setStep]    = useState(0);          // 0 = intro, 1–6 = questions, 7 = result
  const [answers, setAnswers] = useState([]);       // array of score objects
  const [selected, setSelected] = useState(null);  // index of chosen option
  const [resultTeam, setResultTeam] = useState(null);

  const qIndex   = step - 1;                        // 0-based question index
  const question = QUESTIONS[qIndex];
  const progress = step === 0 ? 0 : (step / QUESTIONS.length) * 100;

  /* ── Handlers ──────────────────────────────────────────────────────────── */
  const handleStart = () => setStep(1);

  const handleSelect = (optionIndex) => {
    setSelected(optionIndex);
  };

  const handleNext = () => {
    if (selected === null) return;
    const newAnswers = [...answers, question.options[selected].scores];
    setAnswers(newAnswers);
    setSelected(null);

    if (step < QUESTIONS.length) {
      setStep(step + 1);
    } else {
      // Compute result
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

  /* ── Render ────────────────────────────────────────────────────────────── */
  return (
    <div className="quiz-root">

      {/* Progress bar */}
      {step > 0 && step <= QUESTIONS.length && (
        <div className="quiz-progress-bar">
          <motion.div
            className="quiz-progress-fill"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      )}

      <AnimatePresence mode="wait">

        {/* ── INTRO ── */}
        {step === 0 && (
          <motion.div
            key="intro"
            className="quiz-screen"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.6 }}
          >
            <div className="quiz-intro-label">TEAM SELECTION</div>
            <h1 className="quiz-intro-title">What's your<br />cricket DNA?</h1>
            <p className="quiz-intro-sub">
              Answer 6 questions. We'll reveal your IPL team —
              <br />and tailor the entire app to your colors.
            </p>
            <div className="quiz-intro-dots">
              {QUESTIONS.map((_, i) => (
                <div key={i} className="quiz-intro-dot" />
              ))}
            </div>
            <motion.button
              className="quiz-cta"
              onClick={handleStart}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              Begin <ChevronRight size={18} />
            </motion.button>
            <button className="quiz-skip" onClick={() => navigate('/app')}>
              Skip — go to predictions
            </button>
          </motion.div>
        )}

        {/* ── QUESTIONS ── */}
        {step >= 1 && step <= QUESTIONS.length && (
          <motion.div
            key={`q-${step}`}
            className="quiz-screen"
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
          >
            {/* Back button */}
            {step > 1 && (
              <button className="quiz-back" onClick={handleBack}>
                <ArrowLeft size={16} /> Back
              </button>
            )}

            {/* Step counter */}
            <div className="quiz-step-count">
              {step} / {QUESTIONS.length}
            </div>

            <div className="quiz-subtitle">{question.subtitle}</div>
            <h2 className="quiz-question">{question.question}</h2>

            <div className="quiz-options">
              {question.options.map((opt, i) => (
                <motion.button
                  key={i}
                  className={`quiz-option ${selected === i ? 'quiz-option--selected' : ''}`}
                  onClick={() => handleSelect(i)}
                  whileHover={{ x: 6 }}
                  whileTap={{ scale: 0.98 }}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.07 }}
                >
                  <span className="quiz-option-letter">
                    {String.fromCharCode(65 + i)}
                  </span>
                  <span className="quiz-option-text">{opt.text}</span>
                  {selected === i && (
                    <motion.div
                      className="quiz-option-tick"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 400 }}
                    >✓</motion.div>
                  )}
                </motion.button>
              ))}
            </div>

            <motion.button
              className={`quiz-cta ${selected === null ? 'quiz-cta--disabled' : ''}`}
              onClick={handleNext}
              disabled={selected === null}
              whileHover={selected !== null ? { scale: 1.03 } : {}}
              whileTap={selected !== null ? { scale: 0.97 } : {}}
            >
              {step === QUESTIONS.length ? 'Reveal My Team' : 'Next'}
              <ChevronRight size={18} />
            </motion.button>
          </motion.div>
        )}

        {/* ── RESULT ── */}
        {step === QUESTIONS.length + 1 && resultTeam && (
          <motion.div
            key="result"
            className="quiz-screen quiz-result"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          >
            {/* Color flood from center */}
            <motion.div
              className="result-flood"
              style={{ background: TEAM_THEMES[resultTeam]?.accent }}
              initial={{ scale: 0, opacity: 0.8, borderRadius: '50%' }}
              animate={{ scale: 20, opacity: 0 }}
              transition={{ duration: 1.4, ease: 'easeOut' }}
            />

            <div className="result-label">YOUR TEAM IS</div>

            <motion.div
              className="result-team-abbr"
              style={{ color: TEAM_THEMES[resultTeam]?.accent }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              {resultTeam}
            </motion.div>

            <motion.div
              className="result-team-name"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              {TEAM_THEMES[resultTeam]?.name}
            </motion.div>

            <motion.p
              className="result-sub"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
            >
              The app is now tuned to your colors.
              <br />Your prediction engine awaits.
            </motion.p>

            <motion.button
              className="quiz-cta"
              onClick={handleEnterApp}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              whileHover={{ scale: 1.04 }}
              style={{ borderColor: TEAM_THEMES[resultTeam]?.accent, color: TEAM_THEMES[resultTeam]?.accent }}
            >
              Enter the Prediction Engine <ChevronRight size={18} />
            </motion.button>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
}
