"""MLflow Tracing for agent observability.

Replaces the previous HF-dataset trajectory upload + custom Delta KPI tables
with native MLflow Tracing. Each user turn becomes one trace; tool calls and
LLM calls hang off it as child spans. The MLflow UI under the configured
experiment shows the full agent loop tree, and Lakeview dashboards roll the
spans up into cost / latency / tool-mix charts.

Usage::

    init_tracing(settings.experiment_path)              # once per session

    with trace_span("agent_turn", {"turn_index": 4}):    # span every turn
        ...

    @traced(name="tool.run")                             # span every tool call
    async def call_tool(...): ...

All helpers are fail-soft: if MLflow init can't reach the workspace (e.g.
during unit tests), trace_span yields None and traced returns the function
unwrapped. The agent never breaks because telemetry can't connect.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Per-span attribute key MLflow's exporter checks first when resolving the
# trace's experiment binding (see mlflow.tracing.utils.get_experiment_id_for_trace).
# Hard-coded so we don't depend on importing ``mlflow.tracing.constant`` at
# init-time on builds that lack it.
_EXPERIMENT_ID_ATTR = "mlflow.experimentId"

_INITIALIZED = False
_EXPERIMENT_ID: str | None = None


def init_tracing(experiment_path: str | None) -> bool:
    """Configure MLflow tracking URI + experiment. Idempotent.

    Returns True if the configuration succeeded so callers can decide whether
    to skip span emission entirely. Failure (no host, no auth, network error)
    is logged at WARNING once and never raised.

    Beyond the tracking-URI + active-experiment setup, this binds the
    experiment id explicitly through every channel the MLflow tracing
    exporter might consult (``MLFLOW_EXPERIMENT_ID`` env var and, on builds
    that ship it, ``mlflow.tracing.set_destination``). Without that the
    v3 exporter logs ``trace_info.mlflow_experiment.experiment_id is
    missing`` for every flushed span — a span written outside an active
    ``start_run`` (the agent's normal case) has no run_id to anchor on.

    Workspace-collision fallback: ``mlflow.set_experiment(path)`` raises
    ``BAD_REQUEST: A node with name ... of type DIRECTORY already exists``
    when ``path`` collides with a pre-existing Workspace directory (legacy
    state, leftover artefacts). On that exact error class we retry under
    ``/Users/<email>/databricks-ai-intern`` so the agent never silently runs without
    traces just because of workspace housekeeping.
    """
    global _INITIALIZED, _EXPERIMENT_ID
    if _INITIALIZED:
        return True
    if not experiment_path:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        experiment = _set_experiment_with_fallback(mlflow, experiment_path)
        if experiment is None:
            return False
        exp_id = getattr(experiment, "experiment_id", None)
        if exp_id:
            _EXPERIMENT_ID = str(exp_id)
        _bind_tracing_destination(experiment)
        _INITIALIZED = True
        logger.info(
            "MLflow Tracing initialised at experiment=%s",
            getattr(experiment, "name", experiment_path),
        )
        return True
    except Exception as e:
        logger.warning("MLflow Tracing init failed (%s) — running without traces.", e)
        return False


def _is_workspace_collision(err: Exception) -> bool:
    """True when MLflow rejects set_experiment because of a path-name collision.

    Two error shapes seen in the field — both surface the same root cause
    (the parent already has a child node with that name, and it's a
    DIRECTORY rather than an MLFLOW_EXPERIMENT):
      * ``BAD_REQUEST: A node with name 'X' of type DIRECTORY already exists``
      * ``BAD_REQUEST: For input string: "None"`` (older API path returns
        a less informative message when the path collision happens inside
        ``MLFLOW_EXPERIMENT_NAME`` resolution).
    """
    msg = str(err)
    if "already exists" in msg and "DIRECTORY" in msg:
        return True
    if "BAD_REQUEST" in msg and 'For input string: "None"' in msg:
        return True
    return False


def _resolve_user_email() -> str | None:
    """Look up the current workspace user's email via the SDK.

    Returns None when the SDK can't authenticate (local CLI dev, unit
    tests). Caller falls through to a non-user-scoped fallback.
    """
    try:
        from agent.core.db_client import get_workspace_client

        wc = get_workspace_client()
        me = wc.current_user.me()
        return getattr(me, "user_name", None) or getattr(me, "userName", None)
    except Exception as e:
        logger.debug("user-email resolution failed: %s", e)
        return None


def _fallback_candidates(original: str) -> list[str]:
    """Cascade of fallback experiment paths to try in order.

    1. ``/Users/<email>/<leaf>`` — typical clean workspace; deterministic
       across sessions so the user reuses one experiment.
    2. ``/Users/<email>/<leaf>-mlflow`` — when the user already has a
       workspace directory named after the leaf (a real pattern we hit on
       fe-vm-lakebase-praneeth, where both /Shared/databricks-ai-intern AND
       /Users/<email>/databricks-ai-intern existed as DIRECTORY nodes from prior
       exploration). Suffix is deterministic so still session-stable.
    3. ``/Users/<email>/<leaf>-<uuid8>`` — last-resort escape; non-stable
       across sessions but breaks the chain when every nicer name is wedged.
    """
    email = _resolve_user_email()
    if not email:
        return []
    leaf = original.rsplit("/", 1)[-1] or "databricks-ai-intern"
    import uuid

    return [
        f"/Users/{email}/{leaf}",
        f"/Users/{email}/{leaf}-mlflow",
        f"/Users/{email}/{leaf}-{uuid.uuid4().hex[:8]}",
    ]


def _set_experiment_with_fallback(mlflow, experiment_path: str):
    """Run ``mlflow.set_experiment`` with workspace-collision recovery.

    Walks the fallback cascade from :func:`_fallback_candidates` because the
    first user-scoped path can itself collide with a stale workspace
    directory (observed on fe-vm-lakebase-praneeth). We keep trying until
    one succeeds, surface the path we settled on, or give up with a single
    warning.
    """
    try:
        return mlflow.set_experiment(experiment_path)
    except Exception as e:
        if not _is_workspace_collision(e):
            raise

    candidates = _fallback_candidates(experiment_path)
    if not candidates:
        logger.warning(
            "MLflow experiment %s collided with a workspace directory and "
            "no user-scoped fallback could be derived — running without traces.",
            experiment_path,
        )
        return None

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            experiment = mlflow.set_experiment(candidate)
            logger.warning(
                "MLflow experiment %s collided with a workspace directory; "
                "fell back to %s.",
                experiment_path, candidate,
            )
            return experiment
        except Exception as e:
            if not _is_workspace_collision(e):
                raise
            last_err = e
            logger.debug("Fallback candidate %s also collided; trying next.", candidate)

    logger.warning(
        "MLflow experiment %s and every fallback in %s collided with workspace "
        "directories (last error: %s) — running without traces.",
        experiment_path, candidates, last_err,
    )
    return None


def _bind_tracing_destination(experiment) -> None:
    """Pin trace export to ``experiment``'s id via env var + tracing API.

    Best-effort: any failure is logged at debug and tracing falls back to
    the active experiment lookup the exporter would have done anyway.
    """
    exp_id = getattr(experiment, "experiment_id", None)
    if not exp_id:
        return
    os.environ["MLFLOW_EXPERIMENT_ID"] = str(exp_id)
    try:
        from mlflow.tracing import set_destination  # type: ignore

        # Pick whichever destination class the installed MLflow ships.
        # Order: newer (MLflow 3.5+) → older → Databricks-flavored.
        dest = None
        for mod, attr in (
            ("mlflow.entities.trace_location", "MlflowExperimentLocation"),
            ("mlflow.tracing.destination", "MlflowExperiment"),
            ("mlflow.tracing.destination", "Databricks"),
        ):
            try:
                cls = getattr(__import__(mod, fromlist=[attr]), attr)
                dest = cls(experiment_id=str(exp_id))
                break
            except Exception:
                continue
        if dest is not None:
            set_destination(dest)
    except Exception as e:
        logger.debug("set_destination(%s) suppressed: %s", exp_id, e)


def _with_experiment_id(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Stamp ``mlflow.experimentId`` onto a span's attributes when known.

    The MLflow v3 trace exporter resolves the experiment binding through a
    fallback chain (per-span attr → user destination → active run → active
    experiment). Pinning the attribute makes the resolution deterministic
    even for spans flushed by the async exporter after globals churn at
    process exit — without it the backend rejects with
    ``trace_info.mlflow_experiment.experiment_id is missing``.
    """
    out = dict(attributes or {})
    if _EXPERIMENT_ID and _EXPERIMENT_ID_ATTR not in out:
        out[_EXPERIMENT_ID_ATTR] = _EXPERIMENT_ID
    return out


@contextlib.contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Start an MLflow span. Yields the span (or None on failure)."""
    if not _INITIALIZED:
        yield None
        return
    try:
        import mlflow

        with mlflow.start_span(
            name=name, attributes=_with_experiment_id(attributes),
        ) as span:
            yield span
    except Exception as e:
        logger.debug("trace_span(%s) suppressed: %s", name, e)
        yield None


def record_llm_span(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
) -> None:
    """Emit an ``llm.call`` span carrying per-call tokens / cost / latency.

    Makes each LLM round-trip a queryable row in the trace dataset: token
    counts (incl. prompt-cache splits), wall latency, and dollar cost. When
    ``cost_usd`` is omitted and the static FMAPI catalog can price the model,
    we backfill it from ``cost_estimation.estimate_llm_cost`` so the trace
    carries a number even when the caller didn't compute one.

    Fail-soft like the rest of the module: no-op when tracing isn't
    initialised, and any MLflow failure is swallowed at debug level.
    """
    if not _INITIALIZED:
        return
    if cost_usd is None:
        try:
            from agent.core.cost_estimation import estimate_llm_cost

            est = estimate_llm_cost(
                model, int(prompt_tokens or 0), int(completion_tokens or 0),
            )
            if est.estimated_cost_usd is not None:
                cost_usd = est.estimated_cost_usd
        except Exception as e:
            logger.debug("record_llm_span cost backfill suppressed: %s", e)
    attributes = {
        "llm.model": model,
        "llm.prompt_tokens": int(prompt_tokens or 0),
        "llm.completion_tokens": int(completion_tokens or 0),
        "llm.cache_read_tokens": int(cache_read_tokens or 0),
        "llm.cache_creation_tokens": int(cache_creation_tokens or 0),
    }
    if latency_ms is not None:
        attributes["llm.latency_ms"] = latency_ms
    if cost_usd is not None:
        attributes["llm.cost_usd"] = cost_usd
    try:
        import mlflow

        with mlflow.start_span(
            name="llm.call", attributes=_with_experiment_id(attributes),
        ):
            pass
    except Exception as e:
        logger.debug("record_llm_span suppressed: %s", e)


def record_tool_latency(tool_name: str, latency_ms: float, *, ok: bool = True) -> None:
    """Emit a ``tool.<name>`` span carrying call latency + ok/fail.

    Turns per-tool timing into a queryable trace dimension feeding the
    Lakeview tool-mix / latency charts. Fail-soft: no-op when tracing isn't
    initialised, and any MLflow failure is swallowed at debug level.
    """
    if not _INITIALIZED:
        return
    attributes = {
        "tool.name": tool_name,
        "tool.latency_ms": latency_ms,
        "tool.ok": ok,
    }
    try:
        import mlflow

        with mlflow.start_span(
            name=f"tool.{tool_name}", attributes=_with_experiment_id(attributes),
        ):
            pass
    except Exception as e:
        logger.debug("record_tool_latency(%s) suppressed: %s", tool_name, e)


def traced(name: str | None = None):
    """Decorator wrapping a sync or async function in an MLflow span.

    Falls back to a passthrough wrapper when tracing isn't initialised so
    decorated functions never crash from a missing MLflow session.
    """
    def deco(fn: Callable[..., Any]):
        if _is_coro(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                if not _INITIALIZED:
                    return await fn(*args, **kwargs)
                try:
                    import mlflow
                    with mlflow.start_span(
                        name=name or fn.__name__,
                        attributes=_with_experiment_id(None),
                    ):
                        return await fn(*args, **kwargs)
                except Exception:
                    return await fn(*args, **kwargs)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _INITIALIZED:
                return fn(*args, **kwargs)
            try:
                import mlflow
                with mlflow.start_span(
                    name=name or fn.__name__,
                    attributes=_with_experiment_id(None),
                ):
                    return fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)
        return wrapper
    return deco


def _is_coro(fn: Callable[..., Any]) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


def reset_for_tests() -> None:
    """Test hook: forget initialisation so the next call re-runs."""
    global _INITIALIZED, _EXPERIMENT_ID
    _INITIALIZED = False
    _EXPERIMENT_ID = None
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
