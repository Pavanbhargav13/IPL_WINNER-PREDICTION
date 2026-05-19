import React, { createContext, useContext, useState, useEffect } from 'react';

/* ── IPL Team Themes ──────────────────────────────────────────────────────── */
export const TEAM_THEMES = {
  MI:   { name: 'Mumbai Indians',              accent: '#0057A8', secondary: '#C8A84B', abbr: 'MI'   },
  CSK:  { name: 'Chennai Super Kings',         accent: '#F4C430', secondary: '#1A3C8F', abbr: 'CSK'  },
  RCB:  { name: 'Royal Challengers Bengaluru', accent: '#CC0000', secondary: '#B8860B', abbr: 'RCB'  },
  KKR:  { name: 'Kolkata Knight Riders',       accent: '#7B2FBE', secondary: '#F5A623', abbr: 'KKR'  },
  SRH:  { name: 'Sunrisers Hyderabad',         accent: '#FF6B1A', secondary: '#000000', abbr: 'SRH'  },
  DC:   { name: 'Delhi Capitals',              accent: '#17479E', secondary: '#EF1C25', abbr: 'DC'   },
  PBKS: { name: 'Punjab Kings',                accent: '#CC0000', secondary: '#BEBEBE', abbr: 'PBKS' },
  RR:   { name: 'Rajasthan Royals',            accent: '#E91E8C', secondary: '#2153A0', abbr: 'RR'   },
  GT:   { name: 'Gujarat Titans',              accent: '#1D3461', secondary: '#D4AF37', abbr: 'GT'   },
  LSG:  { name: 'Lucknow Super Giants',        accent: '#45C4C6', secondary: '#1B1B1B', abbr: 'LSG'  },
};

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const [selectedTeam, setSelectedTeam] = useState(() => {
    return localStorage.getItem('ipl_team') || null;
  });

  // Apply team accent color as CSS variable globally
  useEffect(() => {
    const root = document.documentElement;
    if (selectedTeam && TEAM_THEMES[selectedTeam]) {
      const theme = TEAM_THEMES[selectedTeam];
      root.style.setProperty('--team-accent', theme.accent);
      root.style.setProperty('--team-secondary', theme.secondary);
      root.style.setProperty('--team-name', `"${theme.name}"`);
    } else {
      // Default accent before team is chosen
      root.style.setProperty('--team-accent', '#FFFFFF');
      root.style.setProperty('--team-secondary', '#888888');
    }
  }, [selectedTeam]);

  const chooseTeam = (teamCode) => {
    localStorage.setItem('ipl_team', teamCode);
    setSelectedTeam(teamCode);
  };

  const clearTeam = () => {
    localStorage.removeItem('ipl_team');
    setSelectedTeam(null);
  };

  return (
    <ThemeContext.Provider value={{ selectedTeam, chooseTeam, clearTeam, theme: TEAM_THEMES[selectedTeam] || null }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
