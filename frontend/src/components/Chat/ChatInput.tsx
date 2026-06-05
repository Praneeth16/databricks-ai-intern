import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import { Alert, Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip, LinearProgress, Tooltip } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import AddIcon from '@mui/icons-material/Add';
import { apiFetch, apiUpload } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import { useAgentStore } from '@/store/agentStore';
import { FIRST_FREE_MODEL_PATH } from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  recommended?: boolean;
}

// Models served by Databricks Foundation Model API. Avatar is a generated
// data URL (model initial on a coloured square) — no remote fetch.
const initialAvatar = (label: string, color: string) =>
  `data:image/svg+xml;utf8,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" rx="6" fill="${color}"/><text x="16" y="21" font-family="system-ui,sans-serif" font-size="16" font-weight="700" fill="#fff" text-anchor="middle">${label}</text></svg>`
  )}`;

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'claude-opus-4',
    name: 'Claude Opus 4',
    description: 'Databricks FMAPI',
    modelPath: 'databricks/databricks-claude-opus-4',
    avatarUrl: initialAvatar('C', '#FF3621'),
    recommended: true,
  },
  {
    id: 'claude-sonnet-4',
    name: 'Claude Sonnet 4',
    description: 'Databricks FMAPI',
    modelPath: 'databricks/databricks-claude-sonnet-4',
    avatarUrl: initialAvatar('C', '#FF6B47'),
  },
  {
    id: 'llama-3-3-70b',
    name: 'Llama 3.3 70B',
    description: 'Databricks FMAPI',
    modelPath: 'databricks/databricks-meta-llama-3-3-70b-instruct',
    avatarUrl: initialAvatar('L', '#3FA9F5'),
  },
  {
    id: 'gpt-oss-120b',
    name: 'GPT-OSS 120B',
    description: 'Databricks FMAPI',
    modelPath: 'databricks/databricks-gpt-oss-120b',
    avatarUrl: initialAvatar('G', '#00A972'),
  },
];

const findModelByPath = (path: string): ModelOption | undefined => {
  return MODEL_OPTIONS.find(m => m.modelPath === path || path?.includes(m.id));
};

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

interface DatasetUploadResponse {
  session_id: string;
  upload_id: string;
  filename: string;
  original_filename: string;
  volume_path: string;
  size_bytes: number;
  format: 'csv' | 'json' | 'jsonl' | 'parquet';
  read_snippet: string;
}

const MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024;
const DATASET_UPLOAD_ACCEPT = '.csv,.json,.jsonl,.parquet';
const DATASET_UPLOAD_EXTENSIONS = new Set(['csv', 'json', 'jsonl', 'parquet']);

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const isClaudeModel = (m: ModelOption) => m.modelPath.includes('claude');
const firstFreeModel = () => MODEL_OPTIONS.find(m => !isClaudeModel(m)) ?? MODEL_OPTIONS[0];

export default function ChatInput({ sessionId, onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [datasetUploadError, setDatasetUploadError] = useState<string | null>(null);
  const [datasetUploadSuccess, setDatasetUploadSuccess] = useState<string | null>(null);
  const [uploadedDatasets, setUploadedDatasets] = useState<DatasetUploadResponse[]>([]);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [datasetUploadProgress, setDatasetUploadProgress] = useState<number | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string>(MODEL_OPTIONS[0].id);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on Opus over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const lastSentRef = useRef<string>('');

  // Model is per-session: fetch this tab's current model every time the
  // session changes. Other tabs keep their own selections independently.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data?.model) {
          const model = findModelByPath(data.model);
          if (model) setSelectedModelId(model.id);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId]);

  const selectedModel = MODEL_OPTIONS.find(m => m.id === selectedModelId) || MODEL_OPTIONS[0];

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled && !isUploadingDataset) {
      lastSentRef.current = input;
      onSend(input);
      setInput('');
    }
  }, [input, disabled, isUploadingDataset, onSend]);

  const handleDatasetUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Single-file dataset upload through the FastAPI ``/datasets`` endpoint.
  // OBO threads through automatically (the apiFetch cookie path + the
  // browser's forwarded headers when running behind Databricks Apps).
  const uploadFile = useCallback(async (file: File) => {
    if (!sessionId) {
      setDatasetUploadError('Start a session before uploading a dataset.');
      return;
    }
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    if (!DATASET_UPLOAD_EXTENSIONS.has(ext)) {
      setDatasetUploadError('Only CSV, JSON, JSONL, and Parquet files are supported.');
      return;
    }
    if (file.size === 0) {
      setDatasetUploadError('Uploaded dataset file is empty.');
      return;
    }
    if (file.size > MAX_DATASET_UPLOAD_BYTES) {
      setDatasetUploadError(
        `Dataset files must be 100 MB or smaller. ${file.name} is ${formatBytes(file.size)}.`
      );
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setIsUploadingDataset(true);
    setDatasetUploadProgress(0);
    setDatasetUploadError(null);
    setDatasetUploadSuccess(null);
    try {
      const res = await apiUpload(`/api/session/${sessionId}/datasets`, formData, {
        onProgress: ({ percent }) => {
          setDatasetUploadProgress(percent !== null && percent < 100 ? percent : null);
        },
      });
      if (!res.ok) {
        let detail = 'Dataset upload failed.';
        try {
          const body = await res.json();
          if (body?.detail) detail = String(body.detail);
        } catch { /* ignore */ }
        setDatasetUploadError(detail);
        return;
      }
      const payload = await res.json() as DatasetUploadResponse;
      setUploadedDatasets((prev) => [payload, ...prev]);
      setDatasetUploadSuccess(`Uploaded ${payload.filename} to ${payload.volume_path}`);
    } catch (error) {
      setDatasetUploadError(error instanceof Error ? error.message : 'Dataset upload failed.');
    } finally {
      setIsUploadingDataset(false);
      setDatasetUploadProgress(null);
    }
  }, [sessionId]);

  const handleDatasetFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (file) await uploadFile(file);
    },
    [uploadFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (disabled || isProcessing || isUploadingDataset || !sessionId) return;
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, [disabled, isProcessing, isUploadingDataset, sessionId]);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled || isProcessing || isUploadingDataset || !sessionId) return;
    const file = e.dataTransfer.files?.[0];
    if (file) await uploadFile(file);
  }, [disabled, isProcessing, isUploadingDataset, sessionId, uploadFile]);

  useEffect(() => {
    if (!datasetUploadError) return;
    const t = window.setTimeout(() => setDatasetUploadError(null), 7000);
    return () => window.clearTimeout(t);
  }, [datasetUploadError]);

  useEffect(() => {
    if (!datasetUploadSuccess) return;
    const t = window.setTimeout(() => setDatasetUploadSuccess(null), 5000);
    return () => window.clearTimeout(t);
  }, [datasetUploadSuccess]);

  // When the chat transport reports a Claude-quota 429, restore the typed
  // text so the user doesn't lose their message.
  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current);
    }
  }, [claudeQuotaExhausted]);

  // Refresh the quota display whenever the session changes (user might
  // have started another tab that spent quota).
  useEffect(() => {
    if (sessionId) refreshQuota();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleModelClick = (event: React.MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    if (!sessionId) return;
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) setSelectedModelId(model.id);
    } catch { /* ignore */ }
  };

  // Dialog close: just clear the flag. The typed text is already restored.
  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  // "Use a free model" — switch the current session to Kimi (or the first
  // non-Anthropic option) and auto-retry the send that tripped the cap.
  const handleUseFreeModel = useCallback(async () => {
    setClaudeQuotaExhausted(false);
    if (!sessionId) return;
    const free = MODEL_OPTIONS.find(m => m.modelPath === FIRST_FREE_MODEL_PATH)
      ?? firstFreeModel();
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: free.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(free.id);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch { /* ignore */ }
  }, [sessionId, onSend, setClaudeQuotaExhausted]);

  // Hide the chip until the user has actually burned quota — an unused
  // Opus session shouldn't populate a counter.
  const claudeChip = (() => {
    if (!quota || quota.claudeUsedToday === 0) return null;
    if (quota.plan === 'free') {
      return quota.claudeRemaining > 0 ? 'Free today' : 'Pro only';
    }
    return `${quota.claudeUsedToday}/${quota.claudeDailyCap} today`;
  })();

  return (
    <Box
      sx={{
        pb: { xs: 2, md: 4 },
        pt: { xs: 1, md: 2 },
        position: 'relative',
        zIndex: 10,
      }}
    >
      <Box sx={{ maxWidth: '880px', mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 } }}>
        <Box
          className="composer"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          sx={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start',
            bgcolor: 'var(--composer-bg)',
            borderRadius: 'var(--radius-md)',
            p: '12px',
            border: '1px solid var(--border)',
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
            '&:focus-within': {
                borderColor: 'var(--accent-yellow)',
                boxShadow: 'var(--focus)',
            },
            ...(isDragOver ? {
              borderColor: 'var(--accent-yellow)',
              boxShadow: 'var(--focus)',
              bgcolor: 'rgba(255, 200, 80, 0.05)',
            } : {}),
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isProcessing}
            variant="standard"
            inputRef={inputRef}
            InputProps={{
                disableUnderline: true,
                sx: {
                    color: 'var(--text)',
                    fontSize: '15px',
                    fontFamily: 'inherit',
                    padding: 0,
                    lineHeight: 1.5,
                    minHeight: { xs: '44px', md: '56px' },
                    alignItems: 'flex-start',
                }
            }}
            sx={{
                flex: 1,
                '& .MuiInputBase-root': {
                    p: 0,
                    backgroundColor: 'transparent',
                },
                '& textarea': {
                    resize: 'none',
                    padding: '0 !important',
                }
            }}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept={DATASET_UPLOAD_ACCEPT}
            onChange={handleDatasetFileChange}
            style={{ display: 'none' }}
          />
          <Tooltip title="Upload dataset (CSV / JSON / JSONL / Parquet)">
            <span>
              <IconButton
                onClick={handleDatasetUploadClick}
                disabled={disabled || isProcessing || isUploadingDataset || !sessionId}
                sx={{
                  mt: 1,
                  p: 1,
                  borderRadius: '10px',
                  color: uploadedDatasets.length ? 'var(--accent-yellow)' : 'var(--muted-text)',
                  transition: 'all 0.2s',
                  '&:hover': {
                    color: 'var(--accent-yellow)',
                    bgcolor: 'var(--hover-bg)',
                  },
                  '&.Mui-disabled': { opacity: 0.3 },
                }}
                aria-label="Upload dataset"
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                mt: 1,
                p: 1.5,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                position: 'relative',
                '&:hover': {
                  bgcolor: 'var(--hover-bg)',
                  color: 'var(--accent-red)',
                },
              }}
            >
              <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={28} thickness={3} sx={{ color: 'inherit', position: 'absolute' }} />
                <StopIcon sx={{ fontSize: 16 }} />
              </Box>
            </IconButton>
          ) : (
            <IconButton
              onClick={handleSend}
              disabled={disabled || isUploadingDataset || !input.trim()}
              sx={{
                mt: 1,
                p: 1,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                '&:hover': {
                  color: 'var(--accent-yellow)',
                  bgcolor: 'var(--hover-bg)',
                },
                '&.Mui-disabled': {
                  opacity: 0.3,
                },
              }}
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
        {isUploadingDataset && (
          <Box sx={{ mt: 1, px: 0.5 }}>
            <LinearProgress
              variant={datasetUploadProgress === null ? 'indeterminate' : 'determinate'}
              value={datasetUploadProgress ?? 0}
              aria-label="Dataset upload progress"
              sx={{
                height: 4,
                borderRadius: 999,
                bgcolor: 'rgba(255,255,255,0.08)',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 999,
                  bgcolor: 'var(--accent-yellow)',
                },
              }}
            />
          </Box>
        )}
        {(datasetUploadError || datasetUploadSuccess) && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
            <Alert
              severity={datasetUploadError ? 'error' : 'success'}
              variant="filled"
              onClose={() => {
                setDatasetUploadError(null);
                setDatasetUploadSuccess(null);
              }}
              sx={{ fontSize: '0.8rem', maxWidth: 520, width: '100%' }}
            >
              {datasetUploadError ?? datasetUploadSuccess}
            </Alert>
          </Box>
        )}
        {uploadedDatasets.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, justifyContent: 'center', mt: 1 }}>
            {uploadedDatasets.map((dataset) => (
              <Chip
                key={dataset.upload_id}
                size="small"
                label={`Dataset: ${dataset.filename} (${formatBytes(dataset.size_bytes)})`}
                title={dataset.volume_path}
                sx={{
                  maxWidth: '100%',
                  bgcolor: 'rgba(255,255,255,0.08)',
                  color: 'var(--text)',
                  border: '1px solid var(--divider)',
                  '& .MuiChip-label': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            ))}
          </Box>
        )}

        {/* Powered By Badge */}
        <Box
          onClick={handleModelClick}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mt: 1.5,
            gap: 0.8,
            opacity: 0.6,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
            '&:hover': {
              opacity: 1
            }
          }}
        >
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            powered by
          </Typography>
          <img
            src={selectedModel.avatarUrl}
            alt={selectedModel.name}
            style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
          />
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600, letterSpacing: '0.02em' }}>
            {selectedModel.name}
          </Typography>
          <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
        </Box>

        {/* Model Selection Menu */}
        <Menu
          anchorEl={modelAnchorEl}
          open={Boolean(modelAnchorEl)}
          onClose={handleModelClose}
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'center',
          }}
          transformOrigin={{
            vertical: 'bottom',
            horizontal: 'center',
          }}
          slotProps={{
            paper: {
              sx: {
                bgcolor: 'var(--panel)',
                border: '1px solid var(--divider)',
                mb: 1,
                maxHeight: '400px',
              }
            }
          }}
        >
          {MODEL_OPTIONS.map((model) => (
            <MenuItem
              key={model.id}
              onClick={() => handleSelectModel(model)}
              selected={selectedModelId === model.id}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: 'rgba(255,255,255,0.05)',
                }
              }}
            >
              <ListItemIcon>
                <img
                  src={model.avatarUrl}
                  alt={model.name}
                  style={{ width: 24, height: 24, borderRadius: '4px', objectFit: 'cover' }}
                />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {model.name}
                    {model.recommended && (
                      <Chip
                        label="Recommended"
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'var(--accent-yellow)',
                          color: '#000',
                          fontWeight: 600,
                        }}
                      />
                    )}
                    {isClaudeModel(model) && claudeChip && (
                      <Chip
                        label={claudeChip}
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'rgba(255,255,255,0.08)',
                          color: 'var(--muted-text)',
                          fontWeight: 600,
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={model.description}
                secondaryTypographyProps={{
                  sx: { fontSize: '12px', color: 'var(--muted-text)' }
                }}
              />
            </MenuItem>
          ))}
        </Menu>

        <ClaudeCapDialog
          open={claudeQuotaExhausted}
          plan={quota?.plan ?? 'free'}
          cap={quota?.claudeDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
        />
      </Box>
    </Box>
  );
}
