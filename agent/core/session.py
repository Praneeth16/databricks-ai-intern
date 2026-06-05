import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.config import Config
from agent.context_manager.manager import ContextManager

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 200_000

# Public so agent.core.session_resume + the CLI slash command can share one
# fallback directory for offline (no-Lakebase) saves.
DEFAULT_SESSION_LOG_DIR = Path("session_logs")


def _get_max_tokens_safe(model_name: str) -> int:
    """Return the max input-context tokens for a model.

    Primary source: ``litellm.get_model_info(model)['max_input_tokens']`` —
    LiteLLM maintains an upstream catalog that knows Claude Opus 4.6 is
    1M, GPT-5 is 272k, Sonnet 4.5 is 200k, and so on. Strips any HF routing
    suffix / huggingface/ prefix so tagged ids ('moonshotai/Kimi-K2.6:cheapest')
    look up the bare model. Falls back to a conservative 200k default for
    models not in the catalog (typically HF-router-only models).
    """
    from litellm import get_model_info

    candidates = [model_name]
    stripped = model_name.removeprefix("huggingface/").split(":", 1)[0]
    if stripped != model_name:
        candidates.append(stripped)
    for candidate in candidates:
        try:
            info = get_model_info(candidate)
            max_input = info.get("max_input_tokens") if info else None
            if isinstance(max_input, int) and max_input > 0:
                return max_input
        except Exception:
            continue
    logger.info(
        "No litellm.get_model_info entry for %s, falling back to %d",
        model_name, _DEFAULT_MAX_TOKENS,
    )
    return _DEFAULT_MAX_TOKENS


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    RESUME = "resume"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None


class Session:
    """
    Maintains agent session state
    Similar to Session in codex-rs/core/src/codex.rs
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config | None = None,
        tool_router=None,
        context_manager: ContextManager | None = None,
        hf_token: str | None = None,
        local_mode: bool = False,
        stream: bool = True,
        databricks_user_token: str | None = None,
        user_email: str | None = None,
    ):
        self.hf_token: Optional[str] = hf_token
        # OBO token from the Apps proxy (X-Forwarded-Access-Token). Tools that
        # need to act as the end user (databricks_jobs, uc_volume, sandbox)
        # pick this up via getattr(session, "databricks_user_token", None).
        self.databricks_user_token: Optional[str] = databricks_user_token
        self.user_email: Optional[str] = user_email
        self.tool_router = tool_router
        self.stream = stream
        tool_specs = tool_router.get_tool_specs_for_llm() if tool_router else []
        self.context_manager = context_manager or ContextManager(
            model_max_tokens=_get_max_tokens_safe(config.model_name),
            compact_size=0.1,
            untouched_messages=5,
            tool_specs=tool_specs,
            hf_token=hf_token,
            local_mode=local_mode,
        )
        self.event_queue = event_queue
        self.session_id = str(uuid.uuid4())
        self.config = config or Config(
            model_name="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        self.is_running = True
        self._cancelled = asyncio.Event()
        self.pending_approval: Optional[dict[str, Any]] = None
        self.sandbox = None
        self._running_job_ids: set[str] = set()  # HF job IDs currently executing

        # Session trajectory logging
        self.logged_events: list[dict] = []
        self.session_start_time = datetime.now().isoformat()
        self.turn_count: int = 0
        self.last_auto_save_turn: int = 0
        # Stable local save path so heartbeat saves overwrite one file instead
        # of spamming session_logs/. ``_last_heartbeat_ts`` is owned by
        # ``agent.core.telemetry.HeartbeatSaver`` and lazily initialised there.
        self._local_save_path: Optional[str] = None
        self._last_heartbeat_ts: Optional[float] = None

        # Per-model probed reasoning-effort cache. Populated by the probe
        # on /model switch, read by ``effective_effort_for`` below. Keys are
        # raw model ids (including any ``:tag``). Values:
        #   str  → the effort level to send (may be a downgrade from the
        #          preference, e.g. "high" when user asked for "max")
        #   None → model rejected all efforts in the cascade; send no
        #          thinking params at all
        # Key absent → not probed yet; fall back to the raw preference.
        self.model_effective_effort: dict[str, str | None] = {}

        # Last plan_tool snapshot. Populated whenever the LLM calls
        # plan_tool; read by the no-tool continuation guard in agent_loop
        # so a text-only response that tries to stop while plan items are
        # still pending / in_progress gets one corrective retry instead of
        # quietly handing control back to the user.
        self.current_plan: list[dict[str, str]] = []

        # Pre-call cost accumulator (issue #16). Bumped by every completed
        # LLM call (via telemetry.record_llm_call) and every billable
        # tool submission (databricks_jobs, sandbox_create). The YOLO
        # auto-approval gate (#17) reads this to decide whether the next
        # call would exceed the user's cap. ``actual_cost_usd`` is the
        # post-hoc reconcile against system.billing.usage, lagged ~15min;
        # see ``reconcile_actual_cost`` below.
        self.total_cost_usd: float = 0.0
        self.actual_cost_usd: Optional[float] = None
        self._last_reconcile_ts: Optional[float] = None

        # YOLO auto-approval policy (#17). When enabled, tool calls whose
        # estimated cost plus the running total stay under ``cost_cap_usd``
        # bypass the human approval gate. ``estimated_spend_usd`` tracks
        # only the spend that the auto-approve path actually authorised,
        # so toggling YOLO off/on doesn't reset ``total_cost_usd``.
        self.auto_approval_enabled: bool = False
        self.auto_approval_cost_cap_usd: Optional[float] = None
        self.auto_approval_estimated_spend_usd: float = 0.0

    async def send_event(self, event: Event) -> None:
        """Send event back to client and log to trajectory"""
        await self.event_queue.put(event)

        # Log event to trajectory
        self.logged_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type,
                "data": event.data,
            }
        )

        # Mid-turn heartbeat flush (owned by telemetry module).
        from agent.core.telemetry import HeartbeatSaver
        HeartbeatSaver.maybe_fire(self)

    def cancel(self) -> None:
        """Signal cancellation to the running agent loop."""
        self._cancelled.set()

    def reset_cancel(self) -> None:
        """Clear the cancellation flag before a new run."""
        self._cancelled.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def update_model(self, model_name: str) -> None:
        """Switch the active model and update the context window limit."""
        self.config.model_name = model_name
        self.context_manager.model_max_tokens = _get_max_tokens_safe(model_name)

    def set_auto_approval_policy(
        self, *, enabled: bool, cost_cap_usd: Optional[float],
    ) -> None:
        """Apply a new YOLO config from the API. Doesn't touch the
        running spend so the budget is honoured across reconfigures."""
        self.auto_approval_enabled = bool(enabled)
        self.auto_approval_cost_cap_usd = (
            float(cost_cap_usd) if cost_cap_usd is not None else None
        )

    def add_auto_approval_estimated_spend(
        self, amount_usd: Optional[float],
    ) -> None:
        """Best-effort accumulator for spend authorised by the YOLO gate.

        Distinct from ``total_cost_usd`` so we can show "spent under YOLO"
        vs "spent total this session" in the UI. None-safe so the call
        site doesn't have to branch on a catalog-miss CostEstimate.
        """
        if amount_usd is None:
            return
        try:
            self.auto_approval_estimated_spend_usd = round(
                self.auto_approval_estimated_spend_usd + float(amount_usd), 6,
            )
        except (TypeError, ValueError):
            return

    def auto_approval_remaining_usd(self) -> Optional[float]:
        """Budget left under the YOLO cap (None when uncapped)."""
        if self.auto_approval_cost_cap_usd is None:
            return None
        return max(
            0.0,
            round(
                self.auto_approval_cost_cap_usd
                - self.auto_approval_estimated_spend_usd,
                6,
            ),
        )

    def add_estimated_spend(self, amount_usd: Optional[float]) -> None:
        """Best-effort accumulator for the pre-call cost estimate.

        None-safe so call sites that get a ``CostEstimate(None, ...)``
        from the estimator (unknown price) don't have to branch. The
        catalog-miss case is surfaced to the human via the YOLO policy;
        the accumulator just stays put.
        """
        if amount_usd is None:
            return
        try:
            self.total_cost_usd = round(self.total_cost_usd + float(amount_usd), 6)
        except (TypeError, ValueError):
            return

    async def reconcile_actual_cost(self) -> None:
        """Refresh ``actual_cost_usd`` from ``system.billing.usage`` for
        the workspace user. Best-effort; failure (no warehouse, no
        permissions) is logged at debug and the field stays put. Rate
        limited to one query per 60s to keep the warehouse load light.
        """
        import time as _t

        now = _t.monotonic()
        if (
            self._last_reconcile_ts is not None
            and now - self._last_reconcile_ts < 60
        ):
            return
        self._last_reconcile_ts = now
        if not self.user_email:
            return
        try:
            from agent.core import cost_estimation, db_client

            settings = db_client.resolve_settings(self.config)
            actual = await asyncio.to_thread(
                cost_estimation.query_actual_usd_for_user,
                settings, self.user_email,
            )
            if actual is not None:
                self.actual_cost_usd = round(float(actual), 4)
        except Exception as e:
            logger.debug("reconcile_actual_cost suppressed: %s", e)

    def effective_effort_for(self, model_name: str) -> str | None:
        """Resolve the effort level to actually send for ``model_name``.

        Returns the probed result when we have one (may be ``None`` meaning
        "model doesn't do thinking, strip it"), else the raw preference.
        Unknown-model case falls back to the preference so a stale cache
        from a prior ``/model`` can't poison research sub-calls that use a
        different model id.
        """
        if model_name in self.model_effective_effort:
            return self.model_effective_effort[model_name]
        return self.config.reasoning_effort

    def increment_turn(self) -> None:
        """Increment turn counter (called after each user interaction)"""
        self.turn_count += 1

    async def auto_save_if_needed(self) -> None:
        """Check if auto-save should trigger and save if so (completely non-blocking)"""
        if not self.config.save_sessions:
            return

        interval = self.config.auto_save_interval
        if interval <= 0:
            return

        turns_since_last_save = self.turn_count - self.last_auto_save_turn
        if turns_since_last_save >= interval:
            logger.info(f"Auto-saving session (turn {self.turn_count})...")
            self.save_trajectory_local()
            self.last_auto_save_turn = self.turn_count

    def get_trajectory(self) -> dict:
        """Serialize complete session trajectory for logging.

        ``user_id`` is stamped (user_email today, since that is the
        identity we resolve from Apps OBO + SDK chain) so the resume path
        can decide whether to continue or fork the saved conversation
        without an additional lookup.
        """
        tools: list = []
        if self.tool_router is not None:
            try:
                tools = self.tool_router.get_tool_specs_for_llm() or []
            except Exception:
                tools = []
        return {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "user_id": self.user_email,
            "messages": [msg.model_dump() for msg in self.context_manager.items],
            "events": self.logged_events,
            "tools": tools,
        }

    def save_trajectory_local(
        self,
        directory: str = "session_logs",
        upload_status: str = "pending",
        dataset_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save trajectory to local JSON file as backup with upload status

        Args:
            directory: Directory to save logs (default: "session_logs")
            upload_status: Status of upload attempt ("pending", "success", "failed")
            dataset_url: URL of dataset if upload succeeded

        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            log_dir = Path(directory)
            log_dir.mkdir(parents=True, exist_ok=True)

            trajectory = self.get_trajectory()

            # Scrub secrets at save time so session_logs/ never holds raw
            # tokens on disk — a log aggregator, crash dump, or filesystem
            # snapshot between heartbeats would otherwise leak them.
            try:
                from agent.core.redact import scrub
                for key in ("messages", "events", "tools"):
                    if key in trajectory:
                        trajectory[key] = scrub(trajectory[key])
            except Exception as _e:
                logger.debug("Redact-on-save failed (non-fatal): %s", _e)

            # Add upload metadata
            trajectory["upload_status"] = upload_status
            trajectory["upload_url"] = dataset_url
            trajectory["last_save_time"] = datetime.now().isoformat()

            # Reuse one stable path per session so heartbeat saves overwrite
            # the same file instead of creating a new timestamped file every
            # minute. The timestamp in the filename is kept for first-save
            # ordering; subsequent saves just rewrite that file.
            if self._local_save_path and Path(self._local_save_path).parent == log_dir:
                filepath = Path(self._local_save_path)
            else:
                filename = (
                    f"session_{self.session_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = log_dir / filename
                self._local_save_path = str(filepath)

            # Atomic-ish write: stage to .tmp then rename so a crash mid-write
            # doesn't leave a truncated JSON that breaks the retry scanner.
            tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(trajectory, f, indent=2)
            tmp_path.replace(filepath)

            # Mirror the trajectory into Lakebase so /resume can pick up the
            # same conversation from the frontend or another CLI install.
            # Best-effort — a Lakebase outage doesn't fail the local save.
            self._mirror_trajectory_to_lakebase(trajectory)

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save session locally: {e}")
            return None

    def _mirror_trajectory_to_lakebase(self, trajectory: dict) -> None:
        """Push the trajectory blob into ``ml_intern_sessions.trajectory``.

        Failure is intentionally suppressed: Lakebase may be unconfigured
        (local CLI dev), the pool may be down, or the user may be running
        in a context where the import isn't even available. The local
        filesystem copy is the durable artifact; Lakebase is the
        cross-device convenience layer.
        """
        if not self.user_email:
            # No identity we can attach this row to — skip rather than
            # writing under a synthetic "anonymous" user_id that breaks
            # the per-user listing query.
            return
        try:
            from backend import lakebase

            lakebase.save_trajectory(
                session_id=self.session_id,
                user_id=self.user_email,
                user_email=self.user_email,
                model_name=self.config.model_name,
                trajectory=trajectory,
            )
        except Exception as e:
            logger.debug("Lakebase mirror suppressed: %s", e)

    def update_local_save_status(
        self, filepath: str, upload_status: str, dataset_url: Optional[str] = None
    ) -> bool:
        """Update the upload status of an existing local save file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            data["upload_status"] = upload_status
            data["upload_url"] = dataset_url
            data["last_save_time"] = datetime.now().isoformat()

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Failed to update local save status: {e}")
            return False

