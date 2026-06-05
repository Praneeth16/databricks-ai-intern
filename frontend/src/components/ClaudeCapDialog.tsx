import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Typography,
} from '@mui/material';
import type { PlanTier } from '@/hooks/useUserQuota';

interface ClaudeCapDialogProps {
  open: boolean;
  plan: PlanTier;
  cap: number;
  onClose: () => void;
  onUseFreeModel: () => void;
}

export default function ClaudeCapDialog({
  open,
  plan,
  cap,
  onClose,
  onUseFreeModel,
}: ClaudeCapDialogProps) {
  void plan;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' } },
      }}
      PaperProps={{
        sx: {
          bgcolor: 'var(--panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-1)',
          maxWidth: 460,
          mx: 2,
        },
      }}
    >
      <DialogTitle
        sx={{ color: 'var(--text)', fontWeight: 700, fontSize: '1rem', pt: 2.5, pb: 0, px: 3 }}
      >
        Claude rate limit reached
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <DialogContentText
          sx={{ color: 'var(--muted-text)', fontSize: '0.85rem', lineHeight: 1.6 }}
        >
          Databricks AI Gateway rate-limited this Claude endpoint after {cap}{' '}
          {cap === 1 ? 'request' : 'requests'}. Switch to Llama 3.3 70B or
          GPT-OSS 120B to keep going — both run on the same Foundation Model API
          with no extra setup.
        </DialogContentText>
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: '8px',
            bgcolor: 'var(--accent-yellow-weak)',
            border: '1px solid var(--border)',
          }}
        >
          <Typography
            variant="caption"
            sx={{ display: 'block', color: 'var(--muted-text)', fontSize: '0.78rem', lineHeight: 1.55 }}
          >
            Workspace admins can raise per-endpoint limits in the AI Gateway
            policy attached to the Claude serving endpoint.
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2, gap: 1 }}>
        <Button
          onClick={onUseFreeModel}
          variant="contained"
          size="small"
          sx={{
            fontSize: '0.82rem',
            px: 2.5,
            bgcolor: 'var(--accent-yellow)',
            color: '#000',
            textTransform: 'none',
            fontWeight: 700,
            boxShadow: 'none',
            '&:hover': { bgcolor: '#FFB340', boxShadow: 'none' },
          }}
        >
          Switch model
        </Button>
        <Button
          onClick={onClose}
          size="small"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.82rem',
            px: 2,
            textTransform: 'none',
            '&:hover': { bgcolor: 'var(--hover-bg)' },
          }}
        >
          Dismiss
        </Button>
      </DialogActions>
    </Dialog>
  );
}
