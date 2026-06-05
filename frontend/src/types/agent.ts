/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
  needsAttention: boolean;
  /** True when the backend no longer recognizes this session id (e.g.
   *  after a backend restart). The UI shows a recovery banner and
   *  disables input until the user chooses to restore-with-summary or
   *  start fresh. */
  expired?: boolean;
  /** YOLO auto-approval policy snapshot. Populated from the /yolo
   *  endpoint after a PATCH or on session restore. Fields default to
   *  off when missing. */
  autoApprovalEnabled?: boolean;
  autoApprovalCostCapUsd?: number | null;
  autoApprovalEstimatedSpendUsd?: number;
  autoApprovalRemainingUsd?: number | null;
}

export interface YoloPolicySnapshot {
  enabled: boolean;
  cost_cap_usd: number | null;
  estimated_spend_usd: number;
  remaining_usd: number | null;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
  orgMember?: boolean;
}
