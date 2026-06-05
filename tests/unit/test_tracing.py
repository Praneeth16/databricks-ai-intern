"""Unit tests for agent.core.tracing.

Tracing must be fail-soft: when MLflow can't reach the workspace (no creds /
network during unit tests) span helpers should silently no-op instead of
breaking the agent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.core import tracing


def setup_function():
    tracing.reset_for_tests()


def test_init_returns_false_without_experiment_path():
    assert tracing.init_tracing(None) is False
    assert tracing.init_tracing("") is False


def test_init_returns_false_when_mlflow_raises():
    with patch("mlflow.set_tracking_uri", side_effect=RuntimeError("no creds")):
        ok = tracing.init_tracing("/Shared/ml-intern")
    assert ok is False


def test_init_idempotent_after_first_success():
    with patch("mlflow.set_tracking_uri") as set_uri, \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment"):
        assert tracing.init_tracing("/Shared/ml-intern") is True
        # Second call returns True without re-invoking MLflow setters.
        assert tracing.init_tracing("/Shared/ml-intern") is True
        assert set_uri.call_count == 1


def test_init_binds_experiment_id_for_trace_export():
    """The v3 trace exporter logs ``trace_info.mlflow_experiment.experiment_id
    is missing`` whenever a span flushes outside an active run. We sidestep
    by exporting ``MLFLOW_EXPERIMENT_ID`` from the experiment that
    ``set_experiment`` returns."""
    import os

    fake_experiment = type("E", (), {"experiment_id": "424242"})()
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        assert tracing.init_tracing("/Shared/ml-intern") is True
        assert os.environ.get("MLFLOW_EXPERIMENT_ID") == "424242"


def test_reset_for_tests_clears_experiment_id_env_var():
    import os

    os.environ["MLFLOW_EXPERIMENT_ID"] = "stale-from-prior-test"
    tracing.reset_for_tests()
    assert "MLFLOW_EXPERIMENT_ID" not in os.environ


def test_trace_span_stamps_experiment_id_attribute():
    """Per-span ``mlflow.experimentId`` attribute makes the v3 exporter's
    experiment lookup deterministic — without it traces flushed during
    process exit can race the active-experiment globals and the backend
    rejects them as missing experiment_id."""
    fake_experiment = type("E", (), {"experiment_id": "9001"})()
    captured: list[dict] = []

    class _FakeSpan:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_start_span(name, attributes=None, **_):
        captured.append({"name": name, "attributes": attributes})
        return _FakeSpan()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        tracing.init_tracing("/Shared/ml-intern")

    with patch("mlflow.start_span", side_effect=fake_start_span):
        with tracing.trace_span("foo", {"k": "v"}):
            pass

    assert captured, "trace_span never invoked mlflow.start_span"
    attrs = captured[0]["attributes"]
    assert attrs["mlflow.experimentId"] == "9001"
    # Caller's existing attributes survive.
    assert attrs["k"] == "v"


@pytest.mark.asyncio
async def test_traced_decorator_stamps_experiment_id_attribute():
    fake_experiment = type("E", (), {"experiment_id": "9002"})()
    captured: list[dict] = []

    class _FakeSpan:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_start_span(name, attributes=None, **_):
        captured.append({"name": name, "attributes": attributes})
        return _FakeSpan()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        tracing.init_tracing("/Shared/ml-intern")

    @tracing.traced("op")
    async def fn(x):
        return x + 1

    with patch("mlflow.start_span", side_effect=fake_start_span):
        assert await fn(2) == 3

    assert captured[0]["attributes"]["mlflow.experimentId"] == "9002"


def test_trace_span_yields_none_when_uninitialised():
    with tracing.trace_span("foo") as span:
        assert span is None


def test_trace_span_swallows_mlflow_failure():
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment"):
        tracing.init_tracing("/Shared/ml-intern")

    with patch("mlflow.start_span", side_effect=RuntimeError("boom")):
        with tracing.trace_span("foo") as span:
            assert span is None  # swallowed, not raised


@pytest.mark.asyncio
async def test_traced_decorator_passthrough_when_uninitialised():
    @tracing.traced("op")
    async def fn(x):
        return x * 2

    assert await fn(3) == 6


def test_traced_sync_decorator_passthrough_when_uninitialised():
    @tracing.traced("op")
    def fn(x):
        return x + 1

    assert fn(4) == 5


# ── record_llm_span / record_tool_latency (Phase 5 observability) ───────


def _init_tracing_with_fake_experiment(exp_id: str = "7000"):
    fake_experiment = type("E", (), {"experiment_id": exp_id})()
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        assert tracing.init_tracing("/Shared/ml-intern") is True


class _FakeSpan:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _capturing_start_span(captured: list[dict]):
    def fake_start_span(name, attributes=None, **_):
        captured.append({"name": name, "attributes": attributes})
        return _FakeSpan()
    return fake_start_span


def test_record_llm_span_noop_when_uninitialised():
    # No init, no mlflow patch — must not raise and must not touch mlflow.
    tracing.record_llm_span(
        model="databricks/databricks-claude-opus-4-7",
        prompt_tokens=100,
        completion_tokens=50,
    )


def test_record_tool_latency_noop_when_uninitialised():
    tracing.record_tool_latency("uc_volume_read", 12.5)


def test_record_llm_span_passes_attributes_through():
    _init_tracing_with_fake_experiment("7001")
    captured: list[dict] = []
    with patch("mlflow.start_span", side_effect=_capturing_start_span(captured)):
        tracing.record_llm_span(
            model="databricks/databricks-claude-opus-4-7",
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=20,
            cache_creation_tokens=10,
            latency_ms=1234.5,
            cost_usd=0.0099,
        )

    assert captured, "record_llm_span never invoked mlflow.start_span"
    span = captured[0]
    assert span["name"] == "llm.call"
    attrs = span["attributes"]
    assert attrs["llm.model"] == "databricks/databricks-claude-opus-4-7"
    assert attrs["llm.prompt_tokens"] == 100
    assert attrs["llm.completion_tokens"] == 50
    assert attrs["llm.cache_read_tokens"] == 20
    assert attrs["llm.cache_creation_tokens"] == 10
    assert attrs["llm.latency_ms"] == 1234.5
    assert attrs["llm.cost_usd"] == 0.0099
    # Experiment id still stamped for deterministic export.
    assert attrs["mlflow.experimentId"] == "7001"


def test_record_llm_span_backfills_cost_when_omitted(monkeypatch):
    """When ``cost_usd`` is None, the span should carry a cost computed
    from ``cost_estimation.estimate_llm_cost``."""
    from agent.core import cost_estimation

    sentinel = cost_estimation.CostEstimate(estimated_cost_usd=0.4242, billable=True)
    monkeypatch.setattr(
        cost_estimation, "estimate_llm_cost", lambda *a, **k: sentinel,
    )

    _init_tracing_with_fake_experiment("7002")
    captured: list[dict] = []
    with patch("mlflow.start_span", side_effect=_capturing_start_span(captured)):
        tracing.record_llm_span(
            model="databricks/databricks-claude-opus-4-7",
            prompt_tokens=1000,
            completion_tokens=500,
        )

    assert captured[0]["attributes"]["llm.cost_usd"] == 0.4242


def test_record_llm_span_omits_cost_when_uncatalogued(monkeypatch):
    """A model the catalog can't price (estimate returns None) leaves the
    cost attribute off entirely rather than emitting a bogus zero."""
    from agent.core import cost_estimation

    sentinel = cost_estimation.CostEstimate(estimated_cost_usd=None, billable=True)
    monkeypatch.setattr(
        cost_estimation, "estimate_llm_cost", lambda *a, **k: sentinel,
    )

    _init_tracing_with_fake_experiment("7003")
    captured: list[dict] = []
    with patch("mlflow.start_span", side_effect=_capturing_start_span(captured)):
        tracing.record_llm_span(
            model="databricks/some-unknown-model",
            prompt_tokens=10,
            completion_tokens=5,
        )

    assert "llm.cost_usd" not in captured[0]["attributes"]


def test_record_llm_span_swallows_mlflow_failure():
    _init_tracing_with_fake_experiment("7004")
    with patch("mlflow.start_span", side_effect=RuntimeError("boom")):
        # Must not raise.
        tracing.record_llm_span(
            model="databricks/databricks-claude-opus-4-7",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.01,
        )


def test_record_tool_latency_passes_attributes_through():
    _init_tracing_with_fake_experiment("7005")
    captured: list[dict] = []
    with patch("mlflow.start_span", side_effect=_capturing_start_span(captured)):
        tracing.record_tool_latency("databricks_jobs", 987.6, ok=False)

    span = captured[0]
    assert span["name"] == "tool.databricks_jobs"
    attrs = span["attributes"]
    assert attrs["tool.name"] == "databricks_jobs"
    assert attrs["tool.latency_ms"] == 987.6
    assert attrs["tool.ok"] is False
    assert attrs["mlflow.experimentId"] == "7005"


def test_record_tool_latency_swallows_mlflow_failure():
    _init_tracing_with_fake_experiment("7006")
    with patch("mlflow.start_span", side_effect=RuntimeError("boom")):
        tracing.record_tool_latency("uc_volume_read", 1.0)


# ── workspace-directory collision fallback (issue #14) ──────────────────


class _DirCollision(Exception):
    """Mock of the MLflow BAD_REQUEST surface seen in PTB-smoke run-D."""


def test_init_falls_back_to_user_scoped_experiment_on_directory_collision():
    """``/Shared/ml-intern`` collides with a workspace DIRECTORY on the
    test workspace (legacy state). ``init_tracing`` must catch the
    collision class, derive ``/Users/<email>/ml-intern`` via the SDK
    user lookup, and re-call ``set_experiment``. Verifies the fallback
    target is bound and the second call returns a real experiment.
    """
    fake_experiment = type("E", (), {"experiment_id": "9999", "name": "/Users/u@x/ml-intern"})()
    err = Exception(
        "BAD_REQUEST: A node with name 'ml-intern' of type DIRECTORY "
        "already exists under parent 3955935534744294, cannot create "
        "node of type MLFLOW_EXPERIMENT"
    )

    def fake_set_experiment(path):
        if path == "/Shared/ml-intern":
            raise err
        return fake_experiment

    fake_wc = MagicMock()
    fake_wc.current_user.me.return_value = type("M", (), {"user_name": "u@x"})()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=fake_set_experiment) as set_exp, \
         patch("agent.core.db_client.get_workspace_client", return_value=fake_wc):
        ok = tracing.init_tracing("/Shared/ml-intern")

    assert ok is True
    # Two attempts: original + fallback under user scope.
    assert set_exp.call_count == 2
    assert set_exp.call_args_list[1].args[0] == "/Users/u@x/ml-intern"


def test_init_falls_back_on_legacy_for_input_string_none_error():
    """Older MLflow path returns the less-informative ``For input string:
    "None"`` surface for the same collision. Must be treated identically.
    """
    fake_experiment = type("E", (), {"experiment_id": "1111"})()
    err = Exception('BAD_REQUEST: For input string: "None"')

    def fake_set_experiment(path):
        if path == "/Shared/ml-intern":
            raise err
        return fake_experiment

    fake_wc = MagicMock()
    fake_wc.current_user.me.return_value = type("M", (), {"user_name": "u@x"})()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=fake_set_experiment) as set_exp, \
         patch("agent.core.db_client.get_workspace_client", return_value=fake_wc):
        assert tracing.init_tracing("/Shared/ml-intern") is True
    assert set_exp.call_count == 2


def test_init_does_not_fallback_on_unrelated_failure():
    """Non-collision errors must bubble out to the warning log, not
    silently retry against a personal-scope path (where they'd succeed
    spuriously and mask the real problem).
    """
    err = Exception("NETWORK_ERROR: connection refused")
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=err) as set_exp:
        ok = tracing.init_tracing("/Shared/ml-intern")
    assert ok is False
    # Only one attempt — no fallback for unrelated errors.
    assert set_exp.call_count == 1


def _collision_err(name: str = "ml-intern", parent: str = "1") -> Exception:
    return Exception(
        f"BAD_REQUEST: A node with name '{name}' of type DIRECTORY "
        f"already exists under parent {parent}, cannot create "
        "node of type MLFLOW_EXPERIMENT"
    )


def test_init_walks_fallback_cascade_when_user_scoped_path_also_collides():
    """Observed on fe-vm-lakebase-praneeth (PTB-smoke run-E2): both
    ``/Shared/ml-intern`` AND ``/Users/<email>/ml-intern`` existed as
    DIRECTORY nodes from prior exploration. The first user-scoped
    fallback ALSO collided. The cascade must walk to the next candidate
    (``<leaf>-mlflow``) and use that.
    """
    fake_experiment = type("E", (), {"experiment_id": "5555"})()
    attempts: list[str] = []

    def fake_set_experiment(path):
        attempts.append(path)
        if len(attempts) <= 2:
            raise _collision_err()
        return fake_experiment

    fake_wc = MagicMock()
    fake_wc.current_user.me.return_value = type("M", (), {"user_name": "u@x"})()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=fake_set_experiment), \
         patch("agent.core.db_client.get_workspace_client", return_value=fake_wc):
        assert tracing.init_tracing("/Shared/ml-intern") is True

    assert attempts[0] == "/Shared/ml-intern"
    assert attempts[1] == "/Users/u@x/ml-intern"
    assert attempts[2] == "/Users/u@x/ml-intern-mlflow"
    assert len(attempts) == 3


def test_init_gives_up_after_full_cascade_exhausted():
    """When every candidate collides we surface ONE warning and return
    False so the agent runs without traces rather than retrying in a
    loop or masking the error.
    """
    fake_wc = MagicMock()
    fake_wc.current_user.me.return_value = type("M", (), {"user_name": "u@x"})()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=_collision_err()) as set_exp, \
         patch("agent.core.db_client.get_workspace_client", return_value=fake_wc):
        ok = tracing.init_tracing("/Shared/ml-intern")
    assert ok is False
    # Original + 3 cascade candidates = 4 attempts.
    assert set_exp.call_count == 4


def test_init_gives_up_cleanly_when_email_unresolvable():
    """No SDK auth (local dev, no DATABRICKS_HOST) → cannot build a
    user-scoped fallback. Must return False without raising, never
    re-attempting with an empty path.
    """
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", side_effect=_collision_err()) as set_exp, \
         patch("agent.core.db_client.get_workspace_client", side_effect=RuntimeError("no auth")):
        ok = tracing.init_tracing("/Shared/ml-intern")
    assert ok is False
    # Only the original attempt — no fallback path because email unresolved.
    assert set_exp.call_count == 1
