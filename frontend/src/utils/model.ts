/**
 * Shared model-id constants used by session-create call sites.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py. LiteLLM-style ids — the
 * `databricks/` prefix routes to Foundation Model API + AI Gateway.
 */

export const CLAUDE_MODEL_PATH = 'databricks/databricks-claude-opus-4';
export const FIRST_FREE_MODEL_PATH = 'databricks/databricks-meta-llama-3-3-70b-instruct';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('claude');
}
