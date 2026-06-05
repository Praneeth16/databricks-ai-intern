"""Pydantic models for API requests and responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OpType(str, Enum):
    """Operation types matching agent/core/agent_loop.py."""

    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


class Operation(BaseModel):
    """Operation to be submitted to the agent."""

    op_type: OpType
    data: dict[str, Any] | None = None


class Submission(BaseModel):
    """Submission wrapper with ID and operation."""

    id: str
    operation: Operation


class ToolApproval(BaseModel):
    """Approval decision for a single tool call."""

    tool_call_id: str
    approved: bool
    feedback: str | None = None
    edited_script: str | None = None


class ApprovalRequest(BaseModel):
    """Request to approve/reject tool calls."""

    session_id: str
    approvals: list[ToolApproval]


class SubmitRequest(BaseModel):
    """Request to submit user input."""

    session_id: str
    # Cap text size to prevent context-bloat / cost-amplification: a runaway
    # or malicious client could otherwise attach megabytes that then ride
    # along in every subsequent turn until /api/compact fires. 100k chars ≈
    # 25k tokens — well above any reasonable single prompt. Empty bodies
    # also fail validation (min_length=1) so the LLM never receives a
    # blank turn.
    text: str = Field(..., min_length=1, max_length=100_000)


class TruncateRequest(BaseModel):
    """Request to truncate conversation history to before a specific user message."""

    user_message_index: int


class SessionResponse(BaseModel):
    """Response when creating a new session."""

    session_id: str
    ready: bool = True


class PendingApprovalTool(BaseModel):
    """A tool waiting for user approval."""

    tool: str
    tool_call_id: str
    arguments: dict[str, Any] = {}


class SessionInfo(BaseModel):
    """Session metadata."""

    session_id: str
    created_at: str
    is_active: bool
    is_processing: bool = False
    message_count: int
    user_id: str = "dev"
    pending_approval: list[PendingApprovalTool] | None = None
    model: str | None = None


class YoloPolicyRequest(BaseModel):
    """PATCH body for ``/api/session/{sid}/yolo``."""

    enabled: bool
    # Capped at $10k just to stop a typo from authorising runaway spend.
    # Practical UX values are $1–$100.
    cost_cap_usd: float | None = Field(None, ge=0, le=10_000)


class YoloPolicyResponse(BaseModel):
    """Current YOLO state + remaining-budget snapshot for the UI."""

    enabled: bool
    cost_cap_usd: float | None
    estimated_spend_usd: float
    remaining_usd: float | None


class DatasetUploadResponse(BaseModel):
    """Response for a dataset file uploaded to a UC Volume."""

    session_id: str
    upload_id: str
    filename: str
    original_filename: str
    volume_path: str
    size_bytes: int
    format: str  # csv | json | jsonl | parquet
    read_snippet: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    active_sessions: int = 0
    max_sessions: int = 0


class LLMHealthResponse(BaseModel):
    """LLM provider health check response."""

    status: str  # "ok" | "error"
    model: str
    error: str | None = None
    error_type: str | None = None  # "auth" | "credits" | "rate_limit" | "network" | "unknown"
