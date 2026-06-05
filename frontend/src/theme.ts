import { createTheme, type ThemeOptions } from '@mui/material/styles';

// ── Brand palette ─────────────────────────────────────────────────
// Databricks-aligned colours. The primary mark is "Lava" (#FF3621);
// dark surfaces ride a deep navy/teal underbase. Variable names keep
// the legacy ``--accent-yellow`` keys so we don't have to chase every
// component — the *value* shifts to Lava red.

const LAVA = '#FF3621';
const LAVA_HOT = '#FF6B47';
const LAVA_DEEP = '#C42A18';
const OAT = '#EEEDE9';
const NAVY = '#0A1014';
const NAVY_2 = '#0E1A22';
const NAVY_3 = '#13242E';
const TEAL = '#00A972';
const SKY = '#3FA9F5';

// ── Shared tokens ────────────────────────────────────────────────
const sharedTypography: ThemeOptions['typography'] = {
  fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
  fontSize: 15,
  button: {
    fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
    textTransform: 'none' as const,
    fontWeight: 600,
    letterSpacing: '-0.005em',
  },
  h1: { fontWeight: 800, letterSpacing: '-0.03em' },
  h2: { fontWeight: 800, letterSpacing: '-0.02em' },
  h3: { fontWeight: 700, letterSpacing: '-0.015em' },
};

const sharedComponents: ThemeOptions['components'] = {
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: '10px',
        fontWeight: 600,
        transition: 'transform 0.06s ease, background 0.12s ease, box-shadow 0.12s ease',
        '&:hover': { transform: 'translateY(-1px)' },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: { backgroundImage: 'none' },
    },
  },
};

const sharedShape: ThemeOptions['shape'] = { borderRadius: 12 };

// ── Dark palette (Databricks Lava on Navy) ────────────────────────
const darkVars = {
  '--bg': NAVY,
  '--panel': NAVY_2,
  '--surface': NAVY_3,
  '--text': OAT,
  '--muted-text': '#8FA0AE',
  '--accent': LAVA,
  '--accent-hot': LAVA_HOT,
  '--accent-deep': LAVA_DEEP,
  // Legacy alias kept so existing components don't break — value is Lava.
  '--accent-yellow': LAVA,
  '--accent-yellow-weak': 'rgba(255,54,33,0.10)',
  '--accent-green': TEAL,
  '--accent-red': '#E05A4F',
  '--accent-sky': SKY,
  '--shadow-1': '0 8px 24px rgba(0,0,0,0.55)',
  '--shadow-glow': '0 0 0 1px rgba(255,54,33,0.25), 0 12px 40px rgba(255,54,33,0.18)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(255,54,33,0.18)',
  '--border': 'rgba(255,255,255,0.06)',
  '--border-hover': 'rgba(255,255,255,0.14)',
  '--code-bg': 'rgba(0,0,0,0.45)',
  '--tool-bg': 'rgba(255,255,255,0.02)',
  '--tool-border': 'rgba(255,255,255,0.06)',
  '--hover-bg': 'rgba(255,255,255,0.05)',
  '--composer-bg': 'rgba(255,255,255,0.02)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(255,255,255,0.018), transparent)',
  '--body-gradient':
    'radial-gradient(1100px 600px at 70% -10%, rgba(255,54,33,0.10), transparent 60%),' +
    'radial-gradient(900px 500px at 0% 100%, rgba(0,169,114,0.06), transparent 60%),' +
    'linear-gradient(180deg, #0A1014 0%, #070C10 100%)',
  '--hero-gradient':
    'radial-gradient(900px 500px at 50% -10%, rgba(255,107,71,0.18), transparent 60%),' +
    'linear-gradient(180deg, rgba(14,26,34,0.4), rgba(10,16,20,0.95))',
  '--scrollbar-thumb': '#2C3A45',
  '--success-icon': TEAL,
  '--error-icon': '#F87171',
  '--clickable-text': 'rgba(255, 255, 255, 0.92)',
  '--clickable-underline': 'rgba(255,255,255,0.3)',
  '--code-panel-bg': '#070C10',
  '--tab-active-bg': 'rgba(255,255,255,0.08)',
  '--tab-active-border': 'rgba(255,54,33,0.35)',
  '--tab-hover-bg': 'rgba(255,255,255,0.05)',
  '--tab-close-hover': 'rgba(255,255,255,0.1)',
  '--plan-bg': 'rgba(0,0,0,0.25)',
} as const;

// ── Light palette ────────────────────────────────────────────────
const lightVars = {
  '--bg': '#FAFAF7',
  '--panel': '#FFFFFF',
  '--surface': '#F2F1EC',
  '--text': '#1B2A33',
  '--muted-text': '#5C6770',
  '--accent': LAVA,
  '--accent-hot': LAVA_HOT,
  '--accent-deep': LAVA_DEEP,
  '--accent-yellow': LAVA,
  '--accent-yellow-weak': 'rgba(255,54,33,0.10)',
  '--accent-green': '#00875A',
  '--accent-red': '#DC2626',
  '--accent-sky': '#1F6FCF',
  '--shadow-1': '0 6px 18px rgba(20,30,40,0.10)',
  '--shadow-glow': '0 0 0 1px rgba(255,54,33,0.25), 0 12px 40px rgba(255,54,33,0.16)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(255,54,33,0.18)',
  '--border': 'rgba(20,30,40,0.10)',
  '--border-hover': 'rgba(20,30,40,0.20)',
  '--code-bg': 'rgba(20,30,40,0.05)',
  '--tool-bg': 'rgba(20,30,40,0.03)',
  '--tool-border': 'rgba(20,30,40,0.08)',
  '--hover-bg': 'rgba(20,30,40,0.05)',
  '--composer-bg': 'rgba(20,30,40,0.02)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(20,30,40,0.02), transparent)',
  '--body-gradient':
    'radial-gradient(900px 500px at 70% -10%, rgba(255,54,33,0.08), transparent 60%),' +
    'linear-gradient(180deg, #FAFAF7 0%, #F2F1EC 100%)',
  '--hero-gradient':
    'radial-gradient(900px 500px at 50% -10%, rgba(255,54,33,0.10), transparent 60%),' +
    'linear-gradient(180deg, #FFFFFF 0%, #FAFAF7 100%)',
  '--scrollbar-thumb': '#C4C8CC',
  '--success-icon': '#00875A',
  '--error-icon': '#DC2626',
  '--clickable-text': 'rgba(20,30,40,0.85)',
  '--clickable-underline': 'rgba(20,30,40,0.25)',
  '--code-panel-bg': '#F5F4EF',
  '--tab-active-bg': 'rgba(20,30,40,0.06)',
  '--tab-active-border': 'rgba(255,54,33,0.4)',
  '--tab-hover-bg': 'rgba(20,30,40,0.04)',
  '--tab-close-hover': 'rgba(20,30,40,0.08)',
  '--plan-bg': 'rgba(20,30,40,0.03)',
} as const;

// ── Shared CSS baseline (scrollbar, code, brand-logo) ────────────
function makeCssBaseline(vars: Record<string, string>) {
  return {
    styleOverrides: {
      ':root': vars,
      body: {
        background: 'var(--body-gradient)',
        color: 'var(--text)',
        scrollbarWidth: 'thin' as const,
        '&::-webkit-scrollbar': { width: '8px', height: '8px' },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'var(--scrollbar-thumb)',
          borderRadius: '2px',
        },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
      'code, pre': {
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace',
      },
      '.brand-logo': {
        position: 'relative' as const,
        padding: '6px',
        borderRadius: '8px',
        '&::after': {
          content: '""',
          position: 'absolute' as const,
          inset: '-6px',
          borderRadius: '10px',
          background: 'var(--accent-yellow-weak)',
          zIndex: -1,
          pointerEvents: 'none' as const,
        },
      },
      '@keyframes auroraDrift': {
        '0%, 100%': { transform: 'translate3d(0,0,0) scale(1)' },
        '50%': { transform: 'translate3d(2%, -1%, 0) scale(1.05)' },
      },
      '@keyframes lavaPulse': {
        '0%, 100%': { boxShadow: '0 0 0 0 rgba(255,54,33,0.45)' },
        '50%': { boxShadow: '0 0 0 16px rgba(255,54,33,0)' },
      },
    },
  };
}

function makeDrawer() {
  return {
    styleOverrides: {
      paper: {
        backgroundColor: 'var(--panel)',
        borderRight: '1px solid var(--border)',
      },
    },
  };
}

function makeTextField() {
  return {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 'var(--radius-md)',
          '& fieldset': { borderColor: 'var(--border)' },
          '&:hover fieldset': { borderColor: 'var(--border-hover)' },
          '&.Mui-focused fieldset': {
            borderColor: 'var(--accent)',
            borderWidth: '1px',
            boxShadow: 'var(--focus)',
          },
        },
      },
    },
  };
}

// ── Theme builders ───────────────────────────────────────────────
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: LAVA, light: LAVA_HOT, dark: LAVA_DEEP, contrastText: '#fff' },
    secondary: { main: TEAL, contrastText: '#fff' },
    background: { default: NAVY, paper: NAVY_2 },
    text: { primary: OAT, secondary: '#8FA0AE' },
    divider: 'rgba(255,255,255,0.06)',
    success: { main: TEAL },
    error: { main: '#E05A4F' },
    warning: { main: LAVA_HOT },
    info: { main: SKY },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(darkVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: LAVA, light: LAVA_HOT, dark: LAVA_DEEP, contrastText: '#fff' },
    secondary: { main: '#00875A' },
    background: { default: '#FAFAF7', paper: '#FFFFFF' },
    text: { primary: '#1B2A33', secondary: '#5C6770' },
    divider: 'rgba(20,30,40,0.10)',
    success: { main: '#00875A' },
    error: { main: '#DC2626' },
    warning: { main: LAVA_HOT },
    info: { main: '#1F6FCF' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(lightVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

export default darkTheme;
