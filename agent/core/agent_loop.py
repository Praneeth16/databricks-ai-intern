"""loop
Main agent implementation with integrated tool system and MCP support
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from litellm import (
    ChatCompletionMessageToolCall,
    Message,
    acompletion,
    stream_chunk_builder,
)
from litellm.exceptions import ContextWindowExceededError

from agent.config import Config
from agent.core import db_client, telemetry, tracing
from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter

# CPU flavors that don't require explicit approval for the databricks_jobs
# script kind. Anything else is GPU-class compute and goes through the
# approval gate so the user sees it before launch.
_CPU_FLAVORS = {"cpu-basic", "cpu-upgrade"}

logger = logging.getLogger(__name__)

ToolCall = ChatCompletionMessageToolCall


def _validate_tool_args(tool_args: dict) -> tuple[bool, str | None]:
    """
    Validate tool arguments structure.

    Returns:
        (is_valid, error_message)
    """
    args = tool_args.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return (
            False,
            f"Tool call error: 'args' must be a JSON object, not a string. You passed: {repr(args)}",
        )
    if not isinstance(args, dict) and args is not None:
        return (
            False,
            f"Tool call error: 'args' must be a JSON object. You passed type: {type(args).__name__}",
        )
    return True, None


def _needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Check if a tool call requires user approval before execution."""
    # Yolo mode: skip all approvals
    if config and config.yolo_mode:
        return False

    # If args are malformed, skip approval (validation error will be shown later)
    args_valid, _ = _validate_tool_args(tool_args)
    if not args_valid:
        return False

    if tool_name == "sandbox_create":
        return True

    # sweep / research_loop fan out N training-job variants in parallel (the
    # loop across many rounds) — always confirm, they can consume many DBUs.
    if tool_name in ("sweep", "research_loop"):
        return True

    if tool_name == "databricks_jobs":
        operation = tool_args.get("operation", "")
        # Only the run / scheduled-run paths actually consume cluster compute.
        if operation not in {"run", "scheduled run"}:
            return False
        kind = (tool_args.get("kind") or "script").lower()
        # Mosaic AI fine-tune always needs explicit user approval — GPU-bound,
        # potentially many DBUs.
        if kind == "finetune":
            return True
        # Script / serverless kinds: skip approval for explicit cpu flavors when
        # the user has opted out of CPU confirmations. GPU jobs always confirm.
        flavor = (
            tool_args.get("hardware_flavor")
            or tool_args.get("flavor")
            or "cpu-basic"
        )
        is_cpu_job = flavor in _CPU_FLAVORS and not tool_args.get("node_type_id")
        if is_cpu_job:
            return bool(config and config.confirm_cpu_jobs)
        return True

    # uc_volume mutations land artifacts in durable storage; only the write/rm
    # paths require approval (read/ls are free).
    if tool_name == "uc_volume":
        return tool_args.get("operation") in {"write", "rm"}

    # repos: clone / pull / delete touch shared workspace state.
    if tool_name == "repos":
        return tool_args.get("operation") in {"clone", "delete", "pull"}

    # uc_model: alias rolls control which version production points at.
    if tool_name == "uc_model":
        return tool_args.get("operation") in {"set_alias", "delete_alias"}

    # hf_to_uc: data ingestion can be large and write a Delta table.
    if tool_name == "hf_to_uc":
        return True

    # model_serving: planning/querying/listing are free; building (serverless GPU
    # job), deploying (always-on GPU endpoint), benchmarking (load + cost), and
    # deletion all consume real money or change durable serving state.
    if tool_name == "model_serving":
        return tool_args.get("operation") in {"build_and_register", "deploy", "benchmark", "delete"}

    return False


# -- LLM retry constants --------------------------------------------------
_MAX_LLM_RETRIES = 3
_LLM_RETRY_DELAYS = [5, 15, 30]  # seconds between retries

# -- No-tool continuation guard (HF upstream #237) -----------------------
# Cap retry attempts when a text-only response tries to stop with plan
# items still unfinished. Two retries is enough to catch the common
# "I'll keep working" → forgot to call a tool pattern without burning
# tokens forever on a model that genuinely thinks it's done.
_NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT = 2


def _unfinished_plan_items(session: "Session") -> list[dict[str, str]]:
    """Return plan items still pending or in_progress on the session.

    Used by the no-tool continuation guard: if the agent emitted a
    text-only response (finish_reason='stop', zero tool calls) but the
    plan it tracked via plan_tool still has unfinished work, we inject
    a corrective prompt and retry instead of handing control back to
    the user prematurely.
    """
    plan = getattr(session, "current_plan", None) or []
    unfinished: list[dict[str, str]] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status in {"pending", "in_progress"}:
            unfinished.append(item)
    return unfinished


def _format_plan_items_for_guard(items: list[dict[str, str]], limit: int = 4) -> str:
    formatted: list[str] = []
    for item in items[:limit]:
        item_id = item.get("id") or "?"
        content = item.get("content") or "(unnamed task)"
        status = item.get("status") or "unknown"
        formatted.append(f"{item_id}. {content} [{status}]")
    if len(items) > limit:
        formatted.append(f"... and {len(items) - limit} more")
    return "; ".join(formatted)


def _no_tool_incomplete_plan_prompt(items: list[dict[str, str]]) -> str:
    summary = _format_plan_items_for_guard(items)
    return (
        "[SYSTEM: CONTINUATION GUARD] Your previous response ended without any "
        "tool calls, but the task is not complete. The current plan still has "
        f"unfinished items: {summary}. Do not return control to the user yet. "
        "Continue from the next unfinished item and make at least one tool call "
        "now. If you genuinely cannot continue, first use tools to inspect the "
        "state or verify the blocker."
    )


def _is_transient_error(error: Exception) -> bool:
    """Return True for errors that are likely transient and worth retrying."""
    err_str = str(error).lower()
    transient_patterns = [
        "timeout", "timed out",
        "429", "rate limit", "rate_limit",
        "503", "service unavailable",
        "502", "bad gateway",
        "500", "internal server error",
        "overloaded", "capacity",
        "connection reset", "connection refused", "connection error",
        "eof", "broken pipe",
    ]
    return any(pattern in err_str for pattern in transient_patterns)


def _is_effort_config_error(error: Exception) -> bool:
    """Catch the two 400s the effort probe also handles — thinking
    unsupported for this model, or the specific effort level invalid.

    This is our safety net for the case where ``/effort`` was changed
    mid-conversation (which clears the probe cache) and the new level
    doesn't work for the current model. We heal the cache and retry once.
    """
    from agent.core.effort_probe import _is_invalid_effort, _is_thinking_unsupported
    return _is_thinking_unsupported(error) or _is_invalid_effort(error)


async def _heal_effort_and_rebuild_params(
    session: Session, error: Exception, llm_params: dict,
) -> dict:
    """Update the session's effort cache based on ``error`` and return new
    llm_params. Called only when ``_is_effort_config_error(error)`` is True.

    Two branches:
      • thinking-unsupported → cache ``None`` for this model, next call
        strips thinking entirely
      • invalid-effort → re-run the full cascade probe; the result lands
        in the cache
    """
    from agent.core.effort_probe import ProbeInconclusive, _is_thinking_unsupported, probe_effort

    model = session.config.model_name
    if _is_thinking_unsupported(error):
        session.model_effective_effort[model] = None
        logger.info("healed: %s doesn't support thinking — stripped", model)
    else:
        try:
            outcome = await probe_effort(
                model, session.config.reasoning_effort, session.hf_token,
            )
            session.model_effective_effort[model] = outcome.effective_effort
            logger.info(
                "healed: %s effort cascade → %s", model, outcome.effective_effort,
            )
        except ProbeInconclusive:
            # Transient during healing — strip thinking for safety, next
            # call will either succeed or surface the real error.
            session.model_effective_effort[model] = None
            logger.info("healed: %s probe inconclusive — stripped", model)

    return _resolve_llm_params(
        model,
        session.hf_token,
        reasoning_effort=session.effective_effort_for(model),
    )


def _friendly_error_message(error: Exception) -> str | None:
    """Return a user-friendly message for known error types, or None to fall back to traceback."""
    err_str = str(error).lower()

    if "authentication" in err_str or "unauthorized" in err_str or "invalid x-api-key" in err_str:
        return (
            "Authentication failed — your API key is missing or invalid.\n\n"
            "To fix this, set the API key for your model provider:\n"
            "  • Anthropic:   export ANTHROPIC_API_KEY=sk-...\n"
            "  • OpenAI:      export OPENAI_API_KEY=sk-...\n"
            "  • HF Router:   export HF_TOKEN=hf_...\n\n"
            "You can also add it to a .env file in the project root.\n"
            "To switch models, use the /model command."
        )

    if "insufficient" in err_str and "credit" in err_str:
        return (
            "Insufficient API credits. Please check your account balance "
            "at your model provider's dashboard."
        )

    if "not supported by provider" in err_str or "no provider supports" in err_str:
        return (
            "The model isn't served by the provider you pinned.\n\n"
            "Drop the ':<provider>' suffix to let the HF router auto-pick a "
            "provider, or use '/model' (no arg) to see which providers host "
            "which models."
        )

    if "model_not_found" in err_str or (
        "model" in err_str
        and ("not found" in err_str or "does not exist" in err_str)
    ):
        return (
            "Model not found. Use '/model' to list suggestions, or paste an "
            "HF model id like 'MiniMaxAI/MiniMax-M2.7'. Availability is shown "
            "when you switch."
        )

    return None


async def _compact_and_notify(session: Session) -> None:
    """Run compaction and send event if context was reduced.

    Catches ``CompactionFailedError`` from ``ContextManager.compact`` and
    ends the session cleanly instead of letting the caller retry. Without
    this guard the retry loop pre-HF#213 burned LLM budget at ~$3/Opus
    attempt while the session never reached the upload path — invisible
    in telemetry.
    """
    from agent.context_manager.manager import CompactionFailedError

    cm = session.context_manager
    old_usage = cm.running_context_usage
    logger.debug(
        "Compaction check: usage=%d, max=%d, threshold=%d, needs_compact=%s",
        old_usage, cm.model_max_tokens, cm.compaction_threshold, cm.needs_compaction,
    )
    try:
        await cm.compact(
            model_name=session.config.model_name,
            tool_specs=session.tool_router.get_tool_specs_for_llm(),
            hf_token=session.hf_token,
            session=session,
        )
    except CompactionFailedError as e:
        logger.error(
            "Compaction failed for session %s: %s — terminating session",
            session.session_id, e,
        )
        await session.send_event(Event(
            event_type="session_terminated",
            data={
                "reason": "compaction_failed",
                "context_usage": cm.running_context_usage,
                "context_threshold": cm.compaction_threshold,
                "error": str(e)[:300],
                "user_message": (
                    "Your conversation has grown too large to continue. "
                    "The work you've done is saved — start a new session to keep going."
                ),
            },
        ))
        # Stop the agent loop. _run_session.finally will still fire
        # cleanup + trajectory save so the partial session is captured.
        session.is_running = False
        return

    new_usage = cm.running_context_usage
    if new_usage != old_usage:
        logger.warning(
            "Context compacted: %d -> %d tokens (max=%d, %d messages)",
            old_usage, new_usage, cm.model_max_tokens, len(cm.items),
        )
        await session.send_event(
            Event(
                event_type="compacted",
                data={"old_tokens": old_usage, "new_tokens": new_usage},
            )
        )


async def _cleanup_on_cancel(session: Session) -> None:
    """Kill sandbox processes and cancel Databricks job runs when interrupted."""
    # Kill active sandbox processes
    sandbox = getattr(session, "sandbox", None)
    if sandbox:
        try:
            await asyncio.to_thread(sandbox.kill_all)
            logger.info("Killed sandbox processes on cancel")
        except Exception as e:
            logger.warning("Failed to kill sandbox processes: %s", e)

    # Cancel running Databricks job runs (stored as run_id strings by
    # databricks_jobs_tool.py when it submits). Use the same OBO token the
    # job was submitted with so the audit log names the user and permission
    # checks line up with the original submission.
    job_ids = list(session._running_job_ids)
    if job_ids:
        try:
            from agent.core import db_client

            settings = db_client.resolve_settings(session.config)
            user_token = getattr(session, "databricks_user_token", None)
            if user_token and settings.host:
                wc = db_client.get_workspace_client_for_user(user_token, settings.host)
            else:
                wc = db_client.get_workspace_client(settings)
            for run_id in job_ids:
                try:
                    await asyncio.to_thread(wc.jobs.cancel_run, run_id=int(run_id))
                    logger.info("Cancelled Databricks run %s on interrupt", run_id)
                except Exception as e:
                    logger.warning("Failed to cancel Databricks run %s: %s", run_id, e)
        except Exception as e:
            logger.warning("Could not resolve WorkspaceClient for cancel: %s", e)
        session._running_job_ids.clear()


@dataclass
class LLMResult:
    """Result from an LLM call (streaming or non-streaming)."""
    content: str | None
    tool_calls_acc: dict[int, dict]
    token_count: int
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    thinking_blocks: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


def _extract_thinking_state(
    message: Any,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return provider reasoning fields that must be replayed after tool calls."""
    thinking_blocks = getattr(message, "thinking_blocks", None) or None
    reasoning_content = getattr(message, "reasoning_content", None) or None
    return thinking_blocks, reasoning_content


def _should_replay_thinking_state(model_name: str | None) -> bool:
    """Replay thinking metadata only for providers that accept it on input.

    Upstream gated this to ``anthropic/...`` (direct Anthropic API).

    Local extension: ``databricks/`` is also accepted, *but only when the
    served model is Claude-backed*. Databricks Foundation Model API + AI
    Gateway expose many models under the same prefix — Claude, Mistral,
    DBRX, Qwen, etc. Only Claude accepts Anthropic-shape
    ``thinking_blocks`` / ``reasoning_content`` on input messages; the
    rest reject them and tool use breaks after the first turn (Codex
    review caught this on the staged diff). We pattern-match the
    served-model name segment for ``claude`` so the gate stays
    schema-correct as new endpoints land.

    OpenAI-compatible schemas (HF router, generic OpenAI) reject
    ``reasoning_content`` in assistant history and fall through.
    """
    if not model_name:
        return False
    if model_name.startswith("anthropic/"):
        return True
    if model_name.startswith("databricks/"):
        # databricks/<endpoint-name> — Claude endpoints all carry "claude"
        # in the served-model name (e.g. databricks-claude-opus-4-7,
        # databricks-claude-3-7-sonnet). Anything else is non-Claude and
        # must not get Anthropic-shape replay metadata.
        endpoint = model_name.split("/", 1)[1]
        return "claude" in endpoint.lower()
    return False


def _assistant_message_from_result(
    llm_result: LLMResult,
    *,
    model_name: str | None,
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
) -> Message:
    """Build an assistant history message without dropping reasoning state.

    Anthropic extended-thinking + tool use requires ``thinking_blocks``
    from the prior assistant turn to be replayed on the next request,
    otherwise the API returns "Invalid signature in thinking block".
    Without this helper the loop rebuilt assistant history from only
    ``content`` + ``tool_calls`` and LiteLLM stripped the metadata.
    """
    kwargs: dict[str, Any] = {
        "role": "assistant",
        "content": llm_result.content,
    }
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    if _should_replay_thinking_state(model_name):
        if llm_result.thinking_blocks:
            kwargs["thinking_blocks"] = llm_result.thinking_blocks
        if llm_result.reasoning_content:
            kwargs["reasoning_content"] = llm_result.reasoning_content
    return Message(**kwargs)


async def _call_llm_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    """Call the LLM with streaming, emitting assistant_chunk events."""
    response = None
    _healed_effort = False  # one-shot safety net per call
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    for _llm_attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except Exception as e:
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(session, e, llm_params)
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": "Reasoning effort not supported for this model — adjusting and retrying."},
                ))
                continue
            if _llm_attempt < _MAX_LLM_RETRIES - 1 and _is_transient_error(e):
                _delay = _LLM_RETRY_DELAYS[_llm_attempt]
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %ds",
                    _llm_attempt + 1, _MAX_LLM_RETRIES, e, _delay,
                )
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": f"LLM connection error, retrying in {_delay}s..."},
                ))
                await asyncio.sleep(_delay)
                continue
            raise

    full_content = ""
    tool_calls_acc: dict[int, dict] = {}
    token_count = 0
    finish_reason = None
    final_usage_chunk = None
    # Buffered for stream_chunk_builder so we can rebuild thinking_blocks
    # / reasoning_content for providers that accept replay on the next turn
    # (Anthropic + Databricks FMAPI). Empty for everyone else — the cost
    # is one list reference per call.
    chunks: list[Any] = []

    async for chunk in response:
        chunks.append(chunk)
        if session.is_cancelled:
            tool_calls_acc.clear()
            break

        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            if hasattr(chunk, "usage") and chunk.usage:
                token_count = chunk.usage.total_tokens
                final_usage_chunk = chunk
            continue

        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            full_content += delta.content
            await session.send_event(
                Event(event_type="assistant_chunk", data={"content": delta.content})
            )

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        if hasattr(chunk, "usage") and chunk.usage:
            token_count = chunk.usage.total_tokens
            final_usage_chunk = chunk

    _llm_latency_ms = int((time.monotonic() - t_start) * 1000)
    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=final_usage_chunk,
        latency_ms=_llm_latency_ms,
        finish_reason=finish_reason,
    )
    tracing.record_llm_span(
        model=llm_params.get("model", session.config.model_name),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_tokens", 0),
        latency_ms=_llm_latency_ms,
    )

    # Recover thinking_blocks + reasoning_content from the streamed chunks
    # for replay on the next turn. ``stream_chunk_builder`` reassembles
    # the full message envelope including provider reasoning fields that
    # don't surface in the per-chunk delta. Gated by provider — see
    # ``_should_replay_thinking_state``.
    thinking_blocks = None
    reasoning_content = None
    if chunks and _should_replay_thinking_state(llm_params.get("model")):
        try:
            rebuilt = stream_chunk_builder(chunks, messages=messages)
            if rebuilt and getattr(rebuilt, "choices", None):
                rebuilt_msg = rebuilt.choices[0].message
                thinking_blocks, reasoning_content = _extract_thinking_state(rebuilt_msg)
        except Exception:
            logger.debug("Failed to rebuild streaming thinking state", exc_info=True)

    return LLMResult(
        content=full_content or None,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning_content,
    )


async def _call_llm_non_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    """Call the LLM without streaming, emit assistant_message at the end."""
    response = None
    _healed_effort = False
    messages, tools = with_prompt_caching(messages, tools, llm_params.get("model"))
    t_start = time.monotonic()
    for _llm_attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await acompletion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
                timeout=600,
                **llm_params,
            )
            break
        except ContextWindowExceededError:
            raise
        except Exception as e:
            if not _healed_effort and _is_effort_config_error(e):
                _healed_effort = True
                llm_params = await _heal_effort_and_rebuild_params(session, e, llm_params)
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": "Reasoning effort not supported for this model — adjusting and retrying."},
                ))
                continue
            if _llm_attempt < _MAX_LLM_RETRIES - 1 and _is_transient_error(e):
                _delay = _LLM_RETRY_DELAYS[_llm_attempt]
                logger.warning(
                    "Transient LLM error (attempt %d/%d): %s — retrying in %ds",
                    _llm_attempt + 1, _MAX_LLM_RETRIES, e, _delay,
                )
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "system", "log": f"LLM connection error, retrying in {_delay}s..."},
                ))
                await asyncio.sleep(_delay)
                continue
            raise

    choice = response.choices[0]
    message = choice.message
    content = message.content or None
    finish_reason = choice.finish_reason
    token_count = response.usage.total_tokens if response.usage else 0
    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    # Build tool_calls_acc in the same format as streaming
    tool_calls_acc: dict[int, dict] = {}
    if message.tool_calls:
        for idx, tc in enumerate(message.tool_calls):
            tool_calls_acc[idx] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }

    # Emit the full message as a single event
    if content:
        await session.send_event(
            Event(event_type="assistant_message", data={"content": content})
        )

    _llm_latency_ms = int((time.monotonic() - t_start) * 1000)
    usage = await telemetry.record_llm_call(
        session,
        model=llm_params.get("model", session.config.model_name),
        response=response,
        latency_ms=_llm_latency_ms,
        finish_reason=finish_reason,
    )
    tracing.record_llm_span(
        model=llm_params.get("model", session.config.model_name),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_tokens", 0),
        latency_ms=_llm_latency_ms,
    )

    return LLMResult(
        content=content,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
        usage=usage,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning_content,
    )


class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    async def _abandon_pending_approval(session: Session) -> None:
        """Cancel pending approval tools when the user continues the conversation.

        Injects rejection tool-result messages into the LLM context (so the
        history stays valid) and notifies the frontend that those tools were
        abandoned.
        """
        tool_calls = session.pending_approval.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.function.name
            abandon_msg = (
                "Task abandoned — user continued the conversation without approving."
            )

            # Keep LLM context valid: every tool_call needs a tool result
            tool_msg = Message(
                role="tool",
                content=abandon_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "abandoned",
                    },
                )
            )

        session.pending_approval = None
        logger.info("Abandoned %d pending approval tool(s)", len(tool_calls))

    @staticmethod
    async def run_agent(
        session: Session, text: str,
    ) -> str | None:
        """
        Handle user input (like user_input_or_turn in codex.rs:1291)
        Returns the final assistant response content, if any.
        """
        with tracing.trace_span(
            "agent_turn",
            attributes={
                "session_id": session.session_id,
                "turn_index": session.turn_count,
                "model": session.config.model_name,
                "input_chars": len(text or ""),
            },
        ):
            return await Handlers._run_agent_inner(session, text)

    @staticmethod
    async def _run_agent_inner(
        session: Session, text: str,
    ) -> str | None:
        # Clear any stale cancellation flag from a previous run
        session.reset_cancel()

        # If there's a pending approval and the user sent a new message,
        # abandon the pending tools so the LLM context stays valid.
        if text and session.pending_approval:
            await Handlers._abandon_pending_approval(session)

        # Add user message to history only if there's actual content
        if text:
            user_msg = Message(role="user", content=text)
            session.context_manager.add_message(user_msg)

        # Send event that we're processing
        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        # Agentic loop - continue until model doesn't call tools or max iterations is reached
        iteration = 0
        final_response = None
        errored = False
        max_iterations = session.config.max_iterations
        no_tool_incomplete_plan_retries = 0

        while max_iterations == -1 or iteration < max_iterations:
            # ── Cancellation check: before LLM call ──
            if session.is_cancelled:
                break

            # Compact before calling the LLM if context is near the limit.
            # _compact_and_notify sets session.is_running=False on
            # CompactionFailedError; exit before the LLM call would fire
            # with an over-threshold context (which would just re-trigger
            # the same failure path).
            await _compact_and_notify(session)
            if not session.is_running:
                break

            # Doom-loop detection: break out of repeated tool call patterns
            doom_prompt = check_for_doom_loop(session.context_manager.items)
            if doom_prompt:
                session.context_manager.add_message(
                    Message(role="user", content=doom_prompt)
                )
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={
                            "tool": "system",
                            "log": "Doom loop detected — injecting corrective prompt",
                        },
                    )
                )

            messages = session.context_manager.get_messages()
            tools = session.tool_router.get_tool_specs_for_llm()
            try:
                # ── Call the LLM (streaming or non-streaming) ──
                # Pull the per-model probed effort from the session cache when
                # available; fall back to the raw preference for models we
                # haven't probed yet (e.g. research sub-model).
                llm_params = _resolve_llm_params(
                    session.config.model_name,
                    session.hf_token,
                    reasoning_effort=session.effective_effort_for(session.config.model_name),
                )
                if session.stream:
                    llm_result = await _call_llm_streaming(session, messages, tools, llm_params)
                else:
                    llm_result = await _call_llm_non_streaming(session, messages, tools, llm_params)

                content = llm_result.content
                tool_calls_acc = llm_result.tool_calls_acc
                token_count = llm_result.token_count
                finish_reason = llm_result.finish_reason

                # If output was truncated, all tool call args are garbage.
                # Inject a system hint so the LLM retries with smaller content.
                if finish_reason == "length" and tool_calls_acc:
                    dropped_names = [
                        tc["function"]["name"]
                        for tc in tool_calls_acc.values()
                        if tc["function"]["name"]
                    ]
                    logger.warning(
                        "Output truncated (finish_reason=length) — dropping tool calls: %s",
                        dropped_names,
                    )
                    tool_calls_acc.clear()

                    # Tell the agent what happened so it can retry differently
                    truncation_hint = (
                        "Your previous response was truncated because the output hit the "
                        "token limit. The following tool calls were lost: "
                        f"{dropped_names}. "
                        "IMPORTANT: Do NOT retry with the same large content. Instead:\n"
                        "  • For 'write': use bash with cat<<'HEREDOC' to write the file, "
                        "or split into several smaller edit calls.\n"
                        "  • For other tools: reduce the size of your arguments or use bash."
                    )
                    if content:
                        assistant_msg = _assistant_message_from_result(
                            llm_result, model_name=llm_params.get("model"),
                        )
                        session.context_manager.add_message(assistant_msg, token_count)
                    session.context_manager.add_message(
                        Message(role="user", content=f"[SYSTEM: {truncation_hint}]")
                    )
                    if session.stream:
                        await session.send_event(
                            Event(event_type="assistant_stream_end", data={})
                        )
                    await session.send_event(
                        Event(
                            event_type="tool_log",
                            data={"tool": "system", "log": f"Output truncated — retrying with smaller content ({dropped_names})"},
                        )
                    )
                    iteration += 1
                    continue  # retry this iteration

                # Build tool_calls list from accumulated deltas
                tool_calls: list[ToolCall] = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_data = tool_calls_acc[idx]
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            type="function",
                            function={
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"],
                            },
                        )
                    )

                # Signal end of streaming to the frontend
                if session.stream:
                    await session.send_event(
                        Event(event_type="assistant_stream_end", data={})
                    )

                # If no tool calls, add assistant message and we're done.
                # Continuation guard (HF upstream #237): when the agent
                # planned work via plan_tool but the latest response is
                # text-only AND items are still pending/in_progress, retry
                # once with a corrective system message instead of handing
                # control back to the user prematurely. Capped to avoid
                # burning tokens on a model that genuinely thinks it's done.
                if not tool_calls:
                    unfinished_plan = _unfinished_plan_items(session)
                    if (
                        unfinished_plan
                        and no_tool_incomplete_plan_retries
                        < _NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT
                    ):
                        logger.info(
                            "No tool calls with unfinished plan; retrying agent turn "
                            "(attempt %d/%d)",
                            no_tool_incomplete_plan_retries + 1,
                            _NO_TOOL_INCOMPLETE_PLAN_RETRY_LIMIT,
                        )
                        if content:
                            assistant_msg = _assistant_message_from_result(
                                llm_result, model_name=llm_params.get("model"),
                            )
                            session.context_manager.add_message(
                                assistant_msg, token_count,
                            )
                        session.context_manager.add_message(
                            Message(
                                role="user",
                                content=_no_tool_incomplete_plan_prompt(unfinished_plan),
                            )
                        )
                        no_tool_incomplete_plan_retries += 1
                        await session.send_event(Event(
                            event_type="tool_log",
                            data={
                                "tool": "system",
                                "log": (
                                    "Plan still has unfinished items after a "
                                    "text-only response — retrying instead of "
                                    "returning to the prompt."
                                ),
                            },
                        ))
                        iteration += 1
                        continue

                    logger.debug(
                        "Agent loop ending: no tool calls. "
                        "finish_reason=%s, token_count=%d, "
                        "usage=%d, model_max_tokens=%d, "
                        "iteration=%d/%d, "
                        "response_text=%s",
                        finish_reason,
                        token_count,
                        session.context_manager.running_context_usage,
                        session.context_manager.model_max_tokens,
                        iteration,
                        max_iterations,
                        (content or "")[:500],
                    )
                    if content:
                        assistant_msg = _assistant_message_from_result(
                            llm_result, model_name=llm_params.get("model"),
                        )
                        session.context_manager.add_message(assistant_msg, token_count)
                        final_response = content
                    break

                # A successful tool call resets the no-tool retry counter so
                # the next genuine text-only stop isn't penalised by an
                # earlier guard hit.
                no_tool_incomplete_plan_retries = 0

                # Validate tool call args (one json.loads per call, once)
                # and split into good vs bad
                good_tools: list[tuple[ToolCall, str, dict]] = []
                bad_tools: list[ToolCall] = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        good_tools.append((tc, tc.function.name, args))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning(
                            "Malformed arguments for tool_call %s (%s) — skipping",
                            tc.id, tc.function.name,
                        )
                        tc.function.arguments = "{}"
                        bad_tools.append(tc)

                # Add assistant message with all tool calls to context.
                # Routed through ``_assistant_message_from_result`` so
                # thinking_blocks / reasoning_content survive on providers
                # that need them replayed on the next turn (Anthropic +
                # Databricks FMAPI). Without this the next request fails
                # with "Invalid signature in thinking block" under
                # extended-thinking + tool-use.
                assistant_msg = _assistant_message_from_result(
                    llm_result,
                    model_name=llm_params.get("model"),
                    tool_calls=tool_calls,
                )
                session.context_manager.add_message(assistant_msg, token_count)

                # Add error results for bad tool calls so the LLM
                # knows what happened and can retry differently
                for tc in bad_tools:
                    error_msg = (
                        f"ERROR: Tool call to '{tc.function.name}' had malformed JSON "
                        f"arguments and was NOT executed. Retry with smaller content — "
                        f"for 'write', split into multiple smaller writes using 'edit'."
                    )
                    session.context_manager.add_message(Message(
                        role="tool",
                        content=error_msg,
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    ))
                    await session.send_event(Event(
                        event_type="tool_call",
                        data={"tool": tc.function.name, "arguments": {}, "tool_call_id": tc.id},
                    ))
                    await session.send_event(Event(
                        event_type="tool_output",
                        data={"tool": tc.function.name, "tool_call_id": tc.id, "output": error_msg, "success": False},
                    ))

                # ── Cancellation check: before tool execution ──
                if session.is_cancelled:
                    break

                # Separate good tools into approval-required vs auto-execute.
                # Per-session YOLO policy (#17) layers on top: if the static
                # _needs_approval check says "ask", consult should_auto_approve
                # to see whether a budget-gated auto-approve still applies.
                from agent.core.approval_policy import should_auto_approve

                approval_required_tools: list[tuple[ToolCall, str, dict]] = []
                non_approval_tools: list[tuple[ToolCall, str, dict]] = []
                for tc, tool_name, tool_args in good_tools:
                    if _needs_approval(tool_name, tool_args, session.config):
                        approved, estimate, block_reason = should_auto_approve(
                            session, tool_name, tool_args,
                        )
                        if approved:
                            # Surface the auto-approval to the UI + accumulate
                            # spend so the next call sees the updated budget.
                            session.add_auto_approval_estimated_spend(
                                estimate.estimated_cost_usd
                            )
                            session.add_estimated_spend(estimate.estimated_cost_usd)
                            await session.send_event(Event(
                                event_type="tool_log",
                                data={
                                    "tool": "system",
                                    "log": (
                                        f"YOLO auto-approved {tool_name} "
                                        f"({estimate.label or '?'}): est "
                                        f"${(estimate.estimated_cost_usd or 0):.4f}. "
                                        f"Remaining budget: "
                                        f"{session.auto_approval_remaining_usd()}"
                                    ),
                                },
                            ))
                            non_approval_tools.append((tc, tool_name, tool_args))
                        else:
                            # Block reason already says why; surface it so the
                            # human approval UI can show "would exceed budget"
                            # etc. block_reason=None means "policy off".
                            if block_reason and session.auto_approval_enabled:
                                await session.send_event(Event(
                                    event_type="auto_approval_blocked",
                                    data={
                                        "tool_name": tool_name,
                                        "estimated_cost_usd": estimate.estimated_cost_usd,
                                        "remaining_usd": (
                                            session.auto_approval_remaining_usd()
                                        ),
                                        "reason": block_reason,
                                    },
                                ))
                            approval_required_tools.append((tc, tool_name, tool_args))
                    else:
                        non_approval_tools.append((tc, tool_name, tool_args))

                # Execute non-approval tools (in parallel when possible)
                if non_approval_tools:
                    # 1. Validate args upfront
                    parsed_tools: list[
                        tuple[ToolCall, str, dict, bool, str]
                    ] = []
                    for tc, tool_name, tool_args in non_approval_tools:
                        args_valid, error_msg = _validate_tool_args(tool_args)
                        parsed_tools.append(
                            (tc, tool_name, tool_args, args_valid, error_msg)
                        )

                    # 2. Send all tool_call events upfront (so frontend shows them all)
                    for tc, tool_name, tool_args, args_valid, _ in parsed_tools:
                        if args_valid:
                            await session.send_event(
                                Event(
                                    event_type="tool_call",
                                    data={
                                        "tool": tool_name,
                                        "arguments": tool_args,
                                        "tool_call_id": tc.id,
                                    },
                                )
                            )

                    # 3. Execute all valid tools in parallel, cancellable
                    async def _exec_tool(
                        tc: ToolCall,
                        name: str,
                        args: dict,
                        valid: bool,
                        err: str,
                    ) -> tuple[ToolCall, str, dict, str, bool]:
                        if not valid:
                            return (tc, name, args, err, False)
                        with tracing.trace_span(
                            f"tool.{name}",
                            attributes={
                                "tool": name,
                                "operation": (args or {}).get("operation"),
                                "tool_call_id": tc.id,
                            },
                        ):
                            out, ok = await session.tool_router.call_tool(
                                name, args, session=session, tool_call_id=tc.id
                            )
                        return (tc, name, args, out, ok)

                    gather_task = asyncio.ensure_future(asyncio.gather(
                        *[
                            _exec_tool(tc, name, args, valid, err)
                            for tc, name, args, valid, err in parsed_tools
                        ]
                    ))
                    cancel_task = asyncio.ensure_future(session._cancelled.wait())

                    done, _ = await asyncio.wait(
                        [gather_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if cancel_task in done:
                        gather_task.cancel()
                        try:
                            await gather_task
                        except asyncio.CancelledError:
                            pass
                        # Notify frontend that in-flight tools were cancelled
                        for tc, name, _args, valid, _ in parsed_tools:
                            if valid:
                                await session.send_event(Event(
                                    event_type="tool_state_change",
                                    data={"tool_call_id": tc.id, "tool": name, "state": "cancelled"},
                                ))
                        await _cleanup_on_cancel(session)
                        break

                    cancel_task.cancel()
                    results = gather_task.result()

                    # 4. Record results and send outputs (order preserved)
                    for tc, tool_name, tool_args, output, success in results:
                        tool_msg = Message(
                            role="tool",
                            content=output,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                        session.context_manager.add_message(tool_msg)

                        await session.send_event(
                            Event(
                                event_type="tool_output",
                                data={
                                    "tool": tool_name,
                                    "tool_call_id": tc.id,
                                    "output": output,
                                    "success": success,
                                },
                            )
                        )

                # If there are tools requiring approval, ask for batch approval
                if approval_required_tools:
                    # Prepare batch approval data
                    tools_data = []
                    for tc, tool_name, tool_args in approval_required_tools:
                        # Resolve sandbox file paths for hf_jobs scripts so the
                        # frontend can display & edit the actual file content.
                        if tool_name == "hf_jobs" and isinstance(tool_args.get("script"), str):
                            from agent.tools.sandbox_tool import resolve_sandbox_script
                            sandbox = getattr(session, "sandbox", None)
                            resolved, _ = await resolve_sandbox_script(sandbox, tool_args["script"])
                            if resolved:
                                tool_args = {**tool_args, "script": resolved}

                        tools_data.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "tool_call_id": tc.id,
                        })

                    await session.send_event(Event(
                        event_type="approval_required",
                        data={"tools": tools_data, "count": len(tools_data)},
                    ))

                    # Store all approval-requiring tools (ToolCall objects for execution)
                    session.pending_approval = {
                        "tool_calls": [tc for tc, _, _ in approval_required_tools],
                    }

                    # Return early - wait for EXEC_APPROVAL operation
                    return None

                iteration += 1

            except ContextWindowExceededError:
                # Force compact and retry this iteration.
                cm = session.context_manager
                logger.warning(
                    "ContextWindowExceededError at iteration %d — forcing compaction "
                    "(usage=%d, model_max_tokens=%d, messages=%d)",
                    iteration, cm.running_context_usage, cm.model_max_tokens, len(cm.items),
                )
                cm.running_context_usage = cm.model_max_tokens + 1
                await _compact_and_notify(session)
                # Same guard as the top-of-loop call: if compaction couldn't
                # bring us under threshold, _compact_and_notify has emitted
                # session_terminated and cleared is_running. Continuing would
                # just re-fire the LLM with the same too-big context.
                if not session.is_running:
                    break
                continue

            except Exception as e:
                import traceback

                error_msg = _friendly_error_message(e)
                if error_msg is None:
                    error_msg = str(e) + "\n" + traceback.format_exc()

                await session.send_event(
                    Event(
                        event_type="error",
                        data={"error": error_msg},
                    )
                )
                errored = True
                break

        if session.is_cancelled:
            await _cleanup_on_cancel(session)
            await session.send_event(Event(event_type="interrupted"))
        elif not errored:
            await session.send_event(
                Event(
                    event_type="turn_complete",
                    data={"history_size": len(session.context_manager.items)},
                )
            )

        # Increment turn counter and check for auto-save
        session.increment_turn()
        await session.auto_save_if_needed()

        return final_response

    @staticmethod
    async def undo(session: Session) -> None:
        """Remove the last complete turn and notify the frontend."""
        removed = session.context_manager.undo_last_turn()
        if not removed:
            logger.warning("Undo: no user message found to remove")
        await session.send_event(Event(event_type="undo_complete"))

    @staticmethod
    async def resume(session: Session, *, session_id: str) -> None:
        """Reload context from a saved trajectory into the active session.

        The CLI / frontend resolves ``session_id`` against the picker and
        passes it through here; the heavy lifting (Lakebase fetch +
        filesystem fallback + replay validation) lives in
        ``agent.core.session_resume``. We just emit the user-facing event.
        """
        from agent.core import session_resume

        # Single entry by id — the picker already filtered by user, we
        # just need to recover the entry shape so the rest of the resume
        # code reuses one path. Pull the row and synthesise.
        user_id = getattr(session, "user_email", None)
        entries = session_resume.list_session_logs(user_id=user_id)
        match = next((e for e in entries if e.session_id == session_id), None)
        if match is None:
            await session.send_event(Event(
                event_type="error",
                data={"error": f"Resume failed: session {session_id!r} not found"},
            ))
            return

        try:
            result = session_resume.restore_session_from_entry(session, match)
        except Exception as e:
            await session.send_event(Event(
                event_type="error", data={"error": f"Resume failed: {e}"},
            ))
            return
        await session.send_event(Event(event_type="resume_complete", data=result))

    @staticmethod
    async def exec_approval(session: Session, approvals: list[dict]) -> None:
        """Handle batch job execution approval"""
        if not session.pending_approval:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending approval to process"},
                )
            )
            return

        tool_calls = session.pending_approval.get("tool_calls", [])
        if not tool_calls:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending tool calls found"},
                )
            )
            return

        # Create a map of tool_call_id -> approval decision
        approval_map = {a["tool_call_id"]: a for a in approvals}
        for a in approvals:
            if a.get("edited_script"):
                logger.info(
                    f"Received edited script for tool_call {a['tool_call_id']} ({len(a['edited_script'])} chars)"
                )

        # Separate approved and rejected tool calls
        approved_tasks = []
        rejected_tasks = []

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                # Malformed arguments — treat as failed, notify agent
                logger.warning(f"Malformed tool arguments for {tool_name}: {e}")
                tool_msg = Message(
                    role="tool",
                    content=f"Malformed arguments: {e}",
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)
                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": f"Malformed arguments: {e}",
                            "success": False,
                        },
                    )
                )
                continue

            approval_decision = approval_map.get(tc.id, {"approved": False})

            if approval_decision.get("approved", False):
                edited_script = approval_decision.get("edited_script")
                was_edited = False
                if edited_script and "script" in tool_args:
                    tool_args["script"] = edited_script
                    was_edited = True
                    logger.info(f"Using user-edited script for {tool_name} ({tc.id})")
                approved_tasks.append((tc, tool_name, tool_args, was_edited))
            else:
                rejected_tasks.append((tc, tool_name, approval_decision))

        # Clear pending approval immediately so a page refresh during
        # execution won't re-show the approval dialog.
        session.pending_approval = None

        # Notify frontend of approval decisions immediately (before execution)
        for tc, tool_name, tool_args, _was_edited in approved_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "approved",
                    },
                )
            )
        for tc, tool_name, approval_decision in rejected_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "rejected",
                    },
                )
            )

        # Execute all approved tools concurrently
        async def execute_tool(tc, tool_name, tool_args, was_edited):
            """Execute a single tool and return its result.

            The TraceLog already exists on the frontend (created by
            approval_required), so we send tool_state_change instead of
            tool_call to avoid creating a duplicate.
            """
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "running",
                    },
                )
            )

            with tracing.trace_span(
                f"tool.{tool_name}",
                attributes={
                    "tool": tool_name,
                    "operation": (tool_args or {}).get("operation"),
                    "tool_call_id": tc.id,
                    "approved": True,
                },
            ):
                _tool_t0 = time.monotonic()
                output, success = await session.tool_router.call_tool(
                    tool_name, tool_args, session=session, tool_call_id=tc.id
                )
                tracing.record_tool_latency(
                    tool_name,
                    (time.monotonic() - _tool_t0) * 1000.0,
                    ok=success,
                )

            return (tc, tool_name, output, success, was_edited)

        # Execute all approved tools concurrently (cancellable)
        if approved_tasks:
            gather_task = asyncio.ensure_future(asyncio.gather(
                *[
                    execute_tool(tc, tool_name, tool_args, was_edited)
                    for tc, tool_name, tool_args, was_edited in approved_tasks
                ],
                return_exceptions=True,
            ))
            cancel_task = asyncio.ensure_future(session._cancelled.wait())

            done, _ = await asyncio.wait(
                [gather_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_task in done:
                gather_task.cancel()
                try:
                    await gather_task
                except asyncio.CancelledError:
                    pass
                # Notify frontend that approved tools were cancelled
                for tc, tool_name, _args, _was_edited in approved_tasks:
                    await session.send_event(Event(
                        event_type="tool_state_change",
                        data={"tool_call_id": tc.id, "tool": tool_name, "state": "cancelled"},
                    ))
                await _cleanup_on_cancel(session)
                await session.send_event(Event(event_type="interrupted"))
                session.increment_turn()
                await session.auto_save_if_needed()
                return

            cancel_task.cancel()
            results = gather_task.result()

            # Process results and add to context
            for result in results:
                if isinstance(result, Exception):
                    # Handle execution error
                    logger.error(f"Tool execution error: {result}")
                    continue

                tc, tool_name, output, success, was_edited = result

                if was_edited:
                    output = f"[Note: The user edited the script before execution. The output below reflects the user-modified version, not your original script.]\n\n{output}"

                # Add tool result to context
                tool_msg = Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)

                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": output,
                            "success": success,
                        },
                    )
                )

        # Process rejected tools
        for tc, tool_name, approval_decision in rejected_tasks:
            rejection_msg = "Job execution cancelled by user"
            user_feedback = approval_decision.get("feedback")
            if user_feedback:
                # Ensure feedback is a string and sanitize any problematic characters
                feedback_str = str(user_feedback).strip()
                # Remove any control characters that might break JSON parsing
                feedback_str = "".join(
                    char for char in feedback_str if ord(char) >= 32 or char in "\n\t"
                )
                rejection_msg += f". User feedback: {feedback_str}"

            # Ensure rejection_msg is a clean string
            rejection_msg = str(rejection_msg).strip()

            tool_msg = Message(
                role="tool",
                content=rejection_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={
                        "tool": tool_name,
                        "tool_call_id": tc.id,
                        "output": rejection_msg,
                        "success": False,
                    },
                )
            )

        # Continue agent loop with empty input to process the tool results
        await Handlers.run_agent(session, "")

    @staticmethod
    async def shutdown(session: Session) -> bool:
        """Handle shutdown (like shutdown in codex.rs:1329)"""
        if session.config.save_sessions:
            logger.info("Saving session trajectory...")
            session.save_trajectory_local()

        session.is_running = False
        await session.send_event(Event(event_type="shutdown"))
        return True


async def process_submission(session: Session, submission) -> bool:
    """
    Process a single submission and return whether to continue running.

    Returns:
        bool: True to continue, False to shutdown
    """
    op = submission.operation
    logger.debug("Received operation: %s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        await Handlers.run_agent(session, text)
        return True

    if op.op_type == OpType.COMPACT:
        await _compact_and_notify(session)
        return True

    if op.op_type == OpType.UNDO:
        await Handlers.undo(session)
        return True

    if op.op_type == OpType.RESUME:
        sid = op.data.get("session_id") if op.data else None
        if not sid:
            await session.send_event(Event(
                event_type="error", data={"error": "Resume requires a session_id"},
            ))
            return True
        await Handlers.resume(session, session_id=sid)
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = op.data.get("approvals", []) if op.data else []
        await Handlers.exec_approval(session, approvals)
        return True

    if op.op_type == OpType.SHUTDOWN:
        return not await Handlers.shutdown(session)

    logger.warning(f"Unknown operation: {op.op_type}")
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config | None = None,
    tool_router: ToolRouter | None = None,
    session_holder: list | None = None,
    hf_token: str | None = None,
    local_mode: bool = False,
    stream: bool = True,
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """

    # Create session with tool router
    session = Session(
        event_queue, config=config, tool_router=tool_router, hf_token=hf_token,
        local_mode=local_mode, stream=stream,
    )
    if session_holder is not None:
        session_holder[0] = session
    logger.info("Agent loop started")

    # MLflow Tracing — best-effort init. Failure (no workspace creds in unit
    # tests, etc.) is logged once and never blocks the loop.
    try:
        settings = db_client.resolve_settings(session.config)
        tracing.init_tracing(settings.experiment_path)
    except Exception as e:
        logger.debug("Tracing init skipped: %s", e)

    try:
        # Main processing loop
        async with tool_router:
            # Emit ready event after initialization
            await session.send_event(
                Event(event_type="ready", data={
                    "message": "Agent initialized",
                    "tool_count": len(tool_router.tools),
                })
            )

            while session.is_running:
                submission = await submission_queue.get()

                try:
                    should_continue = await process_submission(session, submission)
                    if not should_continue:
                        break
                except asyncio.CancelledError:
                    logger.warning("Agent loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}")
                    await session.send_event(
                        Event(event_type="error", data={"error": str(e)})
                    )

        logger.info("Agent loop exited")

    finally:
        if session.config.save_sessions and session.is_running:
            logger.info("Emergency save: preserving session before exit...")
            try:
                session.save_trajectory_local()
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
