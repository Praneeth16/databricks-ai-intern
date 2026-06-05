import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import BoltOutlinedIcon from '@mui/icons-material/BoltOutlined';
import { useSessionStore } from '@/store/sessionStore';
import { apiFetch } from '@/utils/api';

const DEFAULT_CAP_USD = 5;

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'uncapped';
  if (value >= 100) return `$${value.toFixed(0)}`;
  return `$${value.toFixed(2).replace(/\.00$/, '')}`;
}

/**
 * YOLO auto-approval control. Mounted in the app top bar. Opens a dialog
 * with a toggle + cap field that PATCHes /api/session/{sid}/yolo and
 * persists the snapshot back into the session store so the rest of the
 * UI (chat banner, remaining-budget chip) sees one source of truth.
 */
export default function YoloControl() {
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const updateSessionYolo = useSessionStore((s) => s.updateSessionYolo);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) || null,
    [sessions, activeSessionId],
  );

  const [dialogOpen, setDialogOpen] = useState(false);
  const [capInput, setCapInput] = useState(String(DEFAULT_CAP_USD));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enabled = Boolean(activeSession?.autoApprovalEnabled);
  const disabled = !activeSessionId || activeSession?.expired || busy;
  const remaining = activeSession?.autoApprovalRemainingUsd ?? null;
  const cap = activeSession?.autoApprovalCostCapUsd ?? null;

  useEffect(() => {
    if (!activeSession) return;
    setCapInput(String(activeSession.autoApprovalCostCapUsd ?? DEFAULT_CAP_USD));
  }, [activeSession?.id, activeSession?.autoApprovalCostCapUsd]);

  async function patchPolicy(nextEnabled: boolean, nextCap?: number | null) {
    if (!activeSessionId) return null;
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { enabled: nextEnabled };
      if (nextCap !== undefined) body.cost_cap_usd = nextCap;
      const response = await apiFetch(`/api/session/${activeSessionId}/yolo`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        let detail = response.statusText || 'YOLO update failed';
        try {
          const body = await response.json();
          if (body?.detail) detail = String(body.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      const data = await response.json();
      updateSessionYolo(activeSessionId, data);
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'YOLO update failed');
      return null;
    } finally {
      setBusy(false);
    }
  }

  // Quick toggle without opening the dialog — flips the policy, keeps
  // the existing cap (or uses default on first turn-on).
  async function handleQuickToggle() {
    if (!activeSessionId) return;
    const nextEnabled = !enabled;
    const capToSend = nextEnabled
      ? (activeSession?.autoApprovalCostCapUsd ?? DEFAULT_CAP_USD)
      : undefined;
    await patchPolicy(nextEnabled, capToSend);
  }

  async function handleApplyFromDialog() {
    const raw = capInput.trim();
    let nextCap: number | null = DEFAULT_CAP_USD;
    if (raw === '') {
      nextCap = null;
    } else {
      const parsed = Number(raw);
      if (!Number.isFinite(parsed) || parsed < 0) {
        setError('Budget must be a non-negative number or blank for uncapped.');
        return;
      }
      nextCap = parsed;
    }
    const result = await patchPolicy(true, nextCap);
    if (result) setDialogOpen(false);
  }

  return (
    <>
      <Tooltip
        title={
          !activeSessionId
            ? 'No active session'
            : enabled
              ? `YOLO ON — ${money(remaining)} left of ${money(cap)}`
              : 'YOLO auto-approve OFF'
        }
      >
        <span>
          <IconButton
            onClick={handleQuickToggle}
            onContextMenu={(e) => {
              e.preventDefault();
              setDialogOpen(true);
            }}
            disabled={disabled}
            aria-label="Toggle YOLO auto-approval"
            sx={{
              p: 1,
              color: enabled ? 'var(--accent-yellow)' : 'var(--muted-text)',
              transition: 'color 0.2s',
              '&:hover': {
                color: 'var(--accent-yellow)',
                bgcolor: 'var(--hover-bg)',
              },
              '&.Mui-disabled': { opacity: 0.3 },
            }}
          >
            <BoltOutlinedIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>YOLO auto-approval</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2, color: 'var(--muted-text)' }}>
            Auto-approve tool calls under the budget cap. Destructive ops
            (volume rm, model alias mutate) and scheduled-job changes still
            prompt you regardless of policy.
          </Typography>

          <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
            Enable
          </Typography>
          <Switch
            checked={enabled}
            onChange={(e) => patchPolicy(e.target.checked)}
            disabled={busy}
          />

          <TextField
            label="Budget cap (USD)"
            value={capInput}
            onChange={(e) => setCapInput(e.target.value)}
            placeholder="blank for uncapped"
            fullWidth
            margin="normal"
            inputProps={{ inputMode: 'decimal' }}
            helperText={
              activeSession
                ? `Spent so far: ${money(activeSession.autoApprovalEstimatedSpendUsd ?? 0)}`
                : ''
            }
          />

          {error && (
            <Typography variant="body2" sx={{ color: 'var(--accent-red)', mt: 1 }}>
              {error}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)} disabled={busy}>
            Close
          </Button>
          <Button
            onClick={handleApplyFromDialog}
            variant="contained"
            disabled={busy}
            sx={{ bgcolor: 'var(--accent-yellow)', color: '#000' }}
          >
            Apply & enable
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
