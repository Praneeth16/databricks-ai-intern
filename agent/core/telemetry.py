"""All agent observability in one module.

Every telemetry signal the agent emits — LLM-call usage / cost, hf_jobs
lifecycle, sandbox lifecycle, user feedback, mid-turn heartbeat saves — is
defined here so business-logic files stay free of instrumentation noise.

Callsites are one-liners::

    await telemetry.record_llm_call(session, model=..., response=r, ...)
    await telemetry.record_hf_job_submit(session, job, args, image=..., job_type="Python")
    HeartbeatSaver.maybe_fire(session)

All ``record_*`` functions emit a single ``Event`` via ``session.send_event``
and never raise — telemetry is best-effort and must not break the agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── usage extraction ────────────────────────────────────────────────────────

def extract_usage(response_or_chunk: Any) -> dict:
    """Flat usage dict from a litellm response or final-chunk usage object.

    Normalizes across providers: Anthropic exposes cache tokens as
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens``; OpenAI uses
    ``prompt_tokens_details.cached_tokens``. Exposed under the stable keys
    ``cache_read_tokens`` / ``cache_creation_tokens``.
    """
    u = getattr(response_or_chunk, "usage", None)
    if u is None and isinstance(response_or_chunk, dict):
        u = response_or_chunk.get("usage")
    if u is None:
        return {}

    def _g(name, default=0):
        if isinstance(u, dict):
            return u.get(name, default) or default
        return getattr(u, name, default) or default

    prompt = _g("prompt_tokens")
    completion = _g("completion_tokens")
    total = _g("total_tokens") or (prompt + completion)

    cache_read = _g("cache_read_input_tokens")
    cache_creation = _g("cache_creation_input_tokens")

    if not cache_read:
        details = _g("prompt_tokens_details", None)
        if details is not None:
            if isinstance(details, dict):
                cache_read = details.get("cached_tokens", 0) or 0
            else:
                cache_read = getattr(details, "cached_tokens", 0) or 0

    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cache_read_tokens": int(cache_read),
        "cache_creation_tokens": int(cache_creation),
    }


# ── llm_call ────────────────────────────────────────────────────────────────

async def record_llm_call(
    session: Any,
    *,
    model: str,
    response: Any = None,
    latency_ms: int,
    finish_reason: str | None,
) -> dict:
    """Emit an ``llm_call`` event and return the extracted usage dict so
    callers can stash it on their result object if they want."""
    usage = extract_usage(response) if response is not None else {}
    # LiteLLM's catalog covers a lot of providers; for FMAPI it often
    # returns 0 because the endpoint name doesn't match an upstream
    # entry. Fall back to our static FMAPI catalog (issue #16) so the
    # YOLO budget gate has a non-zero number to work with on Claude /
    # Llama / DBRX served through Databricks.
    cost_usd = 0.0
    if response is not None:
        try:
            from litellm import completion_cost
            cost_usd = float(completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost_usd = 0.0
    if cost_usd == 0.0 and usage:
        try:
            from agent.core.cost_estimation import estimate_llm_cost

            est = estimate_llm_cost(
                model,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
            if est.estimated_cost_usd is not None:
                cost_usd = est.estimated_cost_usd
        except Exception as e:
            logger.debug("fmapi fallback price lookup failed: %s", e)

    # Accumulate into the session running total so the YOLO policy and
    # any UI surface that exposes "spent so far" both read the same field.
    try:
        if hasattr(session, "add_estimated_spend"):
            session.add_estimated_spend(cost_usd)
    except Exception as e:
        logger.debug("session.add_estimated_spend suppressed: %s", e)

    from agent.core.session import Event  # local import to avoid cycle
    try:
        await session.send_event(Event(
            event_type="llm_call",
            data={
                "model": model,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "cost_usd": cost_usd,
                **usage,
            },
        ))
    except Exception as e:
        logger.debug("record_llm_call failed (non-fatal): %s", e)
    return usage


# ── hf_jobs ────────────────────────────────────────────────────────────────

def _infer_push_to_hub(script_or_cmd: Any) -> bool:
    if not isinstance(script_or_cmd, str):
        return False
    return (
        "push_to_hub=True" in script_or_cmd
        or "push_to_hub=true" in script_or_cmd
        or "hub_model_id" in script_or_cmd
    )


async def record_hf_job_submit(
    session: Any,
    job: Any,
    args: dict,
    *,
    image: str,
    job_type: str,
) -> float:
    """Emit ``hf_job_submit``. Returns the monotonic start timestamp so the
    caller can pass it back into :func:`record_hf_job_complete`."""
    from agent.core.session import Event
    t_start = time.monotonic()
    try:
        script_text = args.get("script") or args.get("command") or ""
        await session.send_event(Event(
            event_type="hf_job_submit",
            data={
                "job_id": getattr(job, "id", None),
                "job_url": getattr(job, "url", None),
                "flavor": args.get("hardware_flavor", "cpu-basic"),
                "timeout": args.get("timeout", "30m"),
                "job_type": job_type,
                "image": image,
                "push_to_hub": _infer_push_to_hub(script_text),
            },
        ))
    except Exception as e:
        logger.debug("record_hf_job_submit failed (non-fatal): %s", e)
    return t_start


async def record_hf_job_complete(
    session: Any,
    job: Any,
    *,
    flavor: str,
    final_status: str,
    submit_ts: float,
) -> None:
    from agent.core.session import Event
    try:
        wall_time_s = int(time.monotonic() - submit_ts)
        await session.send_event(Event(
            event_type="hf_job_complete",
            data={
                "job_id": getattr(job, "id", None),
                "flavor": flavor,
                "final_status": final_status,
                "wall_time_s": wall_time_s,
            },
        ))
    except Exception as e:
        logger.debug("record_hf_job_complete failed (non-fatal): %s", e)


# ── sandbox ─────────────────────────────────────────────────────────────────

async def record_sandbox_create(
    session: Any,
    sandbox: Any,
    *,
    hardware: str,
    create_latency_s: int,
) -> None:
    from agent.core.session import Event
    try:
        # Pin created-at on the session so record_sandbox_destroy can diff.
        session._sandbox_created_at = time.monotonic() - create_latency_s
        await session.send_event(Event(
            event_type="sandbox_create",
            data={
                "sandbox_id": getattr(sandbox, "space_id", None),
                "hardware": hardware,
                "create_latency_s": int(create_latency_s),
            },
        ))
    except Exception as e:
        logger.debug("record_sandbox_create failed (non-fatal): %s", e)


async def record_sandbox_destroy(session: Any, sandbox: Any) -> None:
    from agent.core.session import Event
    try:
        created = getattr(session, "_sandbox_created_at", None)
        lifetime_s = int(time.monotonic() - created) if created else None
        await session.send_event(Event(
            event_type="sandbox_destroy",
            data={
                "sandbox_id": getattr(sandbox, "space_id", None),
                "lifetime_s": lifetime_s,
            },
        ))
    except Exception as e:
        logger.debug("record_sandbox_destroy failed (non-fatal): %s", e)


# ── feedback ───────────────────────────────────────────────────────────────

async def record_feedback(
    session: Any,
    *,
    rating: str,
    turn_index: int | None = None,
    message_id: str | None = None,
    comment: str | None = None,
) -> None:
    from agent.core.session import Event
    try:
        await session.send_event(Event(
            event_type="feedback",
            data={
                "rating": rating,
                "turn_index": turn_index,
                "message_id": message_id,
                "comment": (comment or "")[:500],
            },
        ))
    except Exception as e:
        logger.debug("record_feedback failed (non-fatal): %s", e)


# ── heartbeat ──────────────────────────────────────────────────────────────
#
# Removed in P4: previously flushed trajectory state to a HF dataset every
# heartbeat_interval_s. MLflow Tracing now persists spans server-side via its
# own background thread, so no manual mid-turn flush is needed. Local
# trajectory snapshots still happen on auto_save and shutdown
# (Session.save_trajectory_local).


class HeartbeatSaver:
    """No-op shim retained for callsite stability.

    Mid-turn telemetry is now provided by MLflow Tracing's automatic span
    flush. This class survives only so existing imports (``Session.send_event``
    calls ``HeartbeatSaver.maybe_fire``) keep compiling; it's removed entirely
    in P8 once the backend Session refactor settles.
    """

    @staticmethod
    def maybe_fire(session: Any) -> None:  # noqa: ARG004 - kept for ABI
        return
