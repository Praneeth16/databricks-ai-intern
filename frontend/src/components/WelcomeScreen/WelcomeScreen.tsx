import { useState, useCallback, type ReactNode } from 'react';
import { Box, Typography, Button, CircularProgress, Alert, Chip, Stack } from '@mui/material';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import LoginIcon from '@mui/icons-material/Login';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import StorageIcon from '@mui/icons-material/Storage';
import ScienceIcon from '@mui/icons-material/Science';
import InsightsIcon from '@mui/icons-material/Insights';
import HubIcon from '@mui/icons-material/Hub';
import VerifiedIcon from '@mui/icons-material/Verified';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { triggerLogin } from '@/hooks/useAuth';

// ---------------------------------------------------------------------------
// Reusable building blocks
// ---------------------------------------------------------------------------

function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <Box
      component="img"
      src="/logo.svg"
      alt="ML Intern"
      sx={{ width: size, height: size, display: 'block' }}
    />
  );
}

function FeatureCard({
  icon,
  title,
  body,
}: {
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <Box
      sx={{
        position: 'relative',
        p: 2.5,
        borderRadius: 'var(--radius-md)',
        bgcolor: 'var(--surface)',
        border: '1px solid var(--border)',
        transition: 'border-color 0.2s, transform 0.2s',
        '&:hover': {
          borderColor: 'var(--border-hover)',
          transform: 'translateY(-2px)',
        },
      }}
    >
      <Box
        sx={{
          width: 36,
          height: 36,
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'var(--accent-yellow-weak)',
          color: 'var(--accent)',
          mb: 1.5,
        }}
      >
        {icon}
      </Box>
      <Typography
        variant="subtitle2"
        sx={{ fontWeight: 700, color: 'var(--text)', fontSize: '0.95rem', mb: 0.5 }}
      >
        {title}
      </Typography>
      <Typography
        variant="body2"
        sx={{ color: 'var(--muted-text)', fontSize: '0.8rem', lineHeight: 1.55 }}
      >
        {body}
      </Typography>
    </Box>
  );
}

function PipelineStep({
  num,
  label,
  isLast = false,
}: {
  num: string;
  label: string;
  isLast?: boolean;
}) {
  return (
    <Stack direction="row" alignItems="center" spacing={1.25} sx={{ minWidth: 0 }}>
      <Box
        sx={{
          flexShrink: 0,
          width: 26,
          height: 26,
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '0.72rem',
          fontWeight: 800,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--accent)',
          bgcolor: 'var(--accent-yellow-weak)',
          border: '1px solid rgba(255,54,33,0.25)',
        }}
      >
        {num}
      </Box>
      <Typography
        sx={{
          fontSize: '0.82rem',
          fontWeight: 600,
          color: 'var(--text)',
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </Typography>
      {!isLast && (
        <Box
          sx={{
            flexShrink: 0,
            width: 18,
            height: 1,
            bgcolor: 'var(--border-hover)',
          }}
        />
      )}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// WelcomeScreen — Databricks Apps-aware SaaS landing
// ---------------------------------------------------------------------------

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isAuthenticated = !!user?.authenticated;

  const handleStartSession = useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    setError(null);
    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setError(data.detail || 'Server is at capacity. Please try again later.');
        return;
      }
      if (response.status === 401) {
        triggerLogin();
        return;
      }
      if (!response.ok) {
        setError('Failed to create session. Please try again.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
    } catch {
      // redirect or transient — ignore
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, createSession, setPlan, clearPanel]);

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        overflowY: 'auto',
        background: 'var(--body-gradient)',
        position: 'relative',
      }}
    >
      {/* Ambient aurora layer */}
      <Box
        aria-hidden
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'radial-gradient(600px 320px at 80% 5%, rgba(255,107,71,0.18), transparent 60%),' +
            'radial-gradient(500px 280px at 5% 90%, rgba(0,169,114,0.10), transparent 60%)',
          animation: 'auroraDrift 22s ease-in-out infinite',
        }}
      />

      {/* Top bar */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: { xs: 3, md: 6 },
          py: 3,
          maxWidth: 1180,
          mx: 'auto',
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1.25}>
          <BrandMark size={28} />
          <Typography
            sx={{
              fontWeight: 800,
              fontSize: '1rem',
              letterSpacing: '-0.01em',
              color: 'var(--text)',
            }}
          >
            ML Intern
          </Typography>
          <Chip
            label="DATABRICKS"
            size="small"
            sx={{
              ml: 0.5,
              height: 20,
              fontSize: '0.62rem',
              fontWeight: 800,
              letterSpacing: '0.08em',
              color: 'var(--accent)',
              bgcolor: 'var(--accent-yellow-weak)',
              border: '1px solid rgba(255,54,33,0.25)',
              '& .MuiChip-label': { px: 1 },
            }}
          />
        </Stack>
        <Stack direction="row" alignItems="center" spacing={2}>
          <Typography
            sx={{
              display: { xs: 'none', md: 'block' },
              fontSize: '0.78rem',
              color: 'var(--muted-text)',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            v0.1 · preview
          </Typography>
          {isAuthenticated && user?.username && (
            <Chip
              label={user.username}
              size="small"
              sx={{
                bgcolor: 'var(--surface)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
                fontWeight: 600,
              }}
            />
          )}
        </Stack>
      </Box>

      {/* Hero */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          maxWidth: 1180,
          mx: 'auto',
          px: { xs: 3, md: 6 },
          pt: { xs: 4, md: 8 },
          pb: { xs: 5, md: 8 },
          textAlign: 'center',
        }}
      >
        <Chip
          icon={<AutoAwesomeIcon sx={{ fontSize: '14px !important', color: 'var(--accent) !important' }} />}
          label="Autonomous ML engineering on Databricks"
          sx={{
            mb: 3,
            bgcolor: 'var(--accent-yellow-weak)',
            border: '1px solid rgba(255,54,33,0.25)',
            color: 'var(--accent)',
            fontWeight: 600,
            fontSize: '0.74rem',
            letterSpacing: '0.01em',
          }}
        />
        <Typography
          sx={{
            fontSize: { xs: '2.2rem', md: '3.4rem' },
            fontWeight: 800,
            lineHeight: 1.05,
            letterSpacing: '-0.035em',
            color: 'var(--text)',
            mb: 2,
          }}
        >
          The agent that ships{' '}
          <Box component="span" sx={{ position: 'relative', display: 'inline-block' }}>
            <Box
              component="span"
              sx={{
                background: 'linear-gradient(135deg, #FF6B47 0%, #FF3621 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              ML models
            </Box>
          </Box>
          ,
          <br />
          not just chats about them.
        </Typography>
        <Typography
          sx={{
            maxWidth: 640,
            mx: 'auto',
            mb: 4.5,
            color: 'var(--muted-text)',
            fontSize: { xs: '0.95rem', md: '1.05rem' },
            lineHeight: 1.6,
          }}
        >
          ML Intern reads the literature, ingests Unity Catalog datasets, runs
          Mosaic AI fine-tune jobs, and registers the trained model — all
          inside your Databricks workspace, with full MLflow lineage.
        </Typography>

        {/* Primary CTA */}
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={1.5}
          justifyContent="center"
          alignItems="center"
          sx={{ mb: 4 }}
        >
          {isAuthenticated ? (
            <Button
              variant="contained"
              size="large"
              startIcon={
                isCreating ? (
                  <CircularProgress size={18} color="inherit" />
                ) : (
                  <RocketLaunchIcon />
                )
              }
              onClick={handleStartSession}
              disabled={isCreating}
              sx={{
                px: 4,
                py: 1.25,
                fontSize: '0.95rem',
                fontWeight: 700,
                bgcolor: 'var(--accent)',
                color: '#fff',
                boxShadow: 'var(--shadow-glow)',
                '&:hover': {
                  bgcolor: 'var(--accent-hot)',
                  boxShadow: 'var(--shadow-glow)',
                },
              }}
            >
              {isCreating ? 'Launching session…' : 'Start a session'}
            </Button>
          ) : (
            <Button
              variant="contained"
              size="large"
              startIcon={<LoginIcon />}
              onClick={() => triggerLogin()}
              sx={{
                px: 4,
                py: 1.25,
                fontSize: '0.95rem',
                fontWeight: 700,
                bgcolor: 'var(--accent)',
                color: '#fff',
                boxShadow: 'var(--shadow-glow)',
                '&:hover': {
                  bgcolor: 'var(--accent-hot)',
                  boxShadow: 'var(--shadow-glow)',
                },
              }}
            >
              Sign in to Databricks
            </Button>
          )}
          <Button
            component="a"
            href="https://docs.databricks.com/en/machine-learning/foundation-models/index.html"
            target="_blank"
            rel="noopener noreferrer"
            size="large"
            sx={{
              px: 3,
              py: 1.25,
              fontSize: '0.9rem',
              fontWeight: 600,
              color: 'var(--text)',
              border: '1px solid var(--border-hover)',
              '&:hover': { bgcolor: 'var(--hover-bg)', borderColor: 'var(--accent)' },
            }}
          >
            Foundation Model API docs
          </Button>
        </Stack>

        {/* Pipeline strip */}
        <Stack
          direction="row"
          spacing={1.25}
          justifyContent="center"
          flexWrap="wrap"
          rowGap={1.5}
          sx={{
            mb: 1,
            opacity: 0.92,
          }}
        >
          <PipelineStep num="01" label="Read papers" />
          <PipelineStep num="02" label="Ingest UC dataset" />
          <PipelineStep num="03" label="Run Mosaic AI job" />
          <PipelineStep num="04" label="Register UC model" isLast />
        </Stack>

        {error && (
          <Alert
            severity="warning"
            variant="outlined"
            onClose={() => setError(null)}
            sx={{
              mt: 4,
              maxWidth: 480,
              mx: 'auto',
              fontSize: '0.85rem',
              borderColor: 'var(--accent)',
              bgcolor: 'var(--accent-yellow-weak)',
              color: 'var(--text)',
            }}
          >
            {error}
          </Alert>
        )}
      </Box>

      {/* Feature grid */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          maxWidth: 1180,
          mx: 'auto',
          px: { xs: 3, md: 6 },
          pb: { xs: 4, md: 6 },
        }}
      >
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(4, 1fr)' },
            gap: 1.75,
          }}
        >
          <FeatureCard
            icon={<StorageIcon sx={{ fontSize: 20 }} />}
            title="Unity Catalog native"
            body="Datasets, volumes, registered models — all read and written through UC with full lineage."
          />
          <FeatureCard
            icon={<ScienceIcon sx={{ fontSize: 20 }} />}
            title="Mosaic AI fine-tune"
            body="Submit foundation-model fine-tunes by name. Auto-registers the trained model into UC."
          />
          <FeatureCard
            icon={<InsightsIcon sx={{ fontSize: 20 }} />}
            title="MLflow Tracing"
            body="Every turn, tool call, and model invocation is a span — replay any session later."
          />
          <FeatureCard
            icon={<HubIcon sx={{ fontSize: 20 }} />}
            title="AI Gateway routed"
            body="Claude, Llama, GPT-OSS — all routed through your gateway endpoints with rate-limit policy."
          />
        </Box>
      </Box>

      {/* Trust bar */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          maxWidth: 1180,
          mx: 'auto',
          px: { xs: 3, md: 6 },
          pb: 6,
        }}
      >
        <Box
          sx={{
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            bgcolor: 'var(--panel)',
            p: { xs: 2.5, md: 3 },
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            gap: { xs: 2, md: 4 },
            alignItems: { xs: 'flex-start', md: 'center' },
            justifyContent: 'space-between',
          }}
        >
          <Stack direction="row" alignItems="center" spacing={1.5}>
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'rgba(0,169,114,0.12)',
                color: 'var(--accent-green)',
              }}
            >
              <VerifiedIcon sx={{ fontSize: 18 }} />
            </Box>
            <Box>
              <Typography sx={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text)' }}>
                Runs entirely in your workspace
              </Typography>
              <Typography sx={{ fontSize: '0.75rem', color: 'var(--muted-text)' }}>
                Data never leaves Databricks · OAuth on-behalf-of your identity
              </Typography>
            </Box>
          </Stack>

          <Stack
            direction="row"
            spacing={3}
            sx={{ flexWrap: 'wrap', rowGap: 1, ml: { md: 'auto' } }}
            divider={
              <Box sx={{ width: 1, alignSelf: 'stretch', bgcolor: 'var(--border)' }} />
            }
          >
            <TrustStat label="Sessions in" value="Lakebase" />
            <TrustStat label="Telemetry via" value="MLflow Traces" />
            <TrustStat label="Compute" value="Serverless GPU" />
          </Stack>
        </Box>

        {/* Footer */}
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="center"
          spacing={1.25}
          sx={{ mt: 4, opacity: 0.55 }}
        >
          <BrandMark size={14} />
          <Typography sx={{ fontSize: '0.72rem', color: 'var(--muted-text)' }}>
            ML Intern · Built on Databricks Foundation Model API · MLflow · Unity Catalog
          </Typography>
        </Stack>
      </Box>
    </Box>
  );
}

function TrustStat({ label, value }: { label: string; value: string }) {
  return (
    <Box>
      <Typography
        sx={{
          fontSize: '0.65rem',
          color: 'var(--muted-text)',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          fontWeight: 700,
        }}
      >
        {label}
      </Typography>
      <Typography
        sx={{
          fontSize: '0.85rem',
          fontWeight: 700,
          color: 'var(--text)',
          fontFamily: 'JetBrains Mono, monospace',
        }}
      >
        {value}
      </Typography>
    </Box>
  );
}
