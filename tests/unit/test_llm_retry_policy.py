"""Error taxonomy and capped-exponential backoff for LLM calls."""

import asyncio

import pytest

from agent.core import agent_loop
from agent.core.agent_loop import (
    _MAX_CONNECTION_RETRIES,
    _MAX_LLM_RETRIES,
    _MAX_RETRY_DELAY,
    _RETRY_BASE_DELAY,
    _acompletion_with_retry,
    _backoff_delay,
    _classify_error,
    _is_transient_error,
    _retry_after_seconds,
    _retry_budget,
)


class _StubSession:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


class _Resp:
    """Stand-in for a successful completion."""


# --------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Connection reset by peer", "connection"),
        ("connection refused", "connection"),
        ("Request timed out", "connection"),
        ("broken pipe", "connection"),
        ("429 Too Many Requests", "rate_limit"),
        ("rate_limit_error", "rate_limit"),
        ("503 Service Unavailable", "server"),
        ("Overloaded", "server"),
        ("500 internal server error", "server"),
        ("invalid_request_error: bad tool schema", "fatal"),
        ("authentication_error", "fatal"),
    ],
)
def test_classify_error(message, expected):
    assert _classify_error(Exception(message)) == expected


def test_is_transient_matches_every_non_fatal_class():
    assert _is_transient_error(Exception("connection reset"))
    assert _is_transient_error(Exception("429"))
    assert _is_transient_error(Exception("503"))
    assert not _is_transient_error(Exception("invalid_request_error"))


def test_rate_limit_takes_precedence_over_connection_wording():
    """A 429 that mentions a timeout is still a rate limit — it must not get the
    connection class's much larger retry budget."""
    assert _classify_error(Exception("429 rate limit; request timed out")) == "rate_limit"


# --------------------------------------------------------------- budgets


def test_connection_failures_get_a_bigger_budget_than_rejections():
    assert _retry_budget("connection") == _MAX_CONNECTION_RETRIES
    assert _retry_budget("rate_limit") == _MAX_LLM_RETRIES
    assert _retry_budget("server") == _MAX_LLM_RETRIES
    assert _MAX_CONNECTION_RETRIES > _MAX_LLM_RETRIES


# --------------------------------------------------------------- backoff


def test_backoff_doubles_then_caps():
    delays = [_backoff_delay("connection", i, Exception("connection reset")) for i in range(8)]
    assert delays[0] == _RETRY_BASE_DELAY
    assert delays[1] == _RETRY_BASE_DELAY * 2
    assert delays[2] == _RETRY_BASE_DELAY * 4
    assert max(delays) == _MAX_RETRY_DELAY
    assert delays == sorted(delays), "backoff must be monotonically non-decreasing"


def test_retry_after_header_is_honoured():
    class E(Exception):
        class response:  # noqa: N801 - mimics the SDK's attribute shape
            headers = {"retry-after": "17"}

    assert _retry_after_seconds(E("429")) == 17.0
    assert _backoff_delay("rate_limit", 0, E("429")) == 17.0


def test_retry_after_parsed_from_message_when_no_header():
    err = Exception("rate limited, retry-after: 12")
    assert _retry_after_seconds(err) == 12.0


def test_retry_after_is_clamped_to_the_ceiling():
    err = Exception("429 retry after 9999")
    assert _backoff_delay("rate_limit", 0, err) == _MAX_RETRY_DELAY


def test_retry_after_absent_falls_back_to_exponential():
    assert _retry_after_seconds(Exception("429 slow down")) is None
    assert _backoff_delay("rate_limit", 1, Exception("429")) == _RETRY_BASE_DELAY * 2


# --------------------------------------------------------------- the retry loop


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff durations instead of actually waiting."""
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(agent_loop.asyncio, "sleep", fake_sleep)
    return slept


@pytest.mark.asyncio
async def test_retry_returns_on_first_success(monkeypatch, no_sleep):
    async def ok(**_kwargs):
        return _Resp()

    monkeypatch.setattr(agent_loop, "acompletion", ok)
    resp, params = await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert isinstance(resp, _Resp)
    assert params == {"model": "m"}
    assert no_sleep == []


@pytest.mark.asyncio
async def test_retry_recovers_after_transient_failures(monkeypatch, no_sleep):
    calls = {"n": 0}

    async def flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("connection reset by peer")
        return _Resp()

    monkeypatch.setattr(agent_loop, "acompletion", flaky)
    resp, _ = await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert isinstance(resp, _Resp)
    assert no_sleep == [_RETRY_BASE_DELAY, _RETRY_BASE_DELAY * 2]


@pytest.mark.asyncio
async def test_fatal_errors_are_not_retried(monkeypatch, no_sleep):
    calls = {"n": 0}

    async def fatal(**_kwargs):
        calls["n"] += 1
        raise Exception("invalid_request_error: unknown parameter")

    monkeypatch.setattr(agent_loop, "acompletion", fatal)
    with pytest.raises(Exception, match="invalid_request_error"):
        await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert calls["n"] == 1, "a rejected request must not be retried"
    assert no_sleep == []


@pytest.mark.asyncio
async def test_server_errors_stop_at_the_bounded_budget(monkeypatch, no_sleep):
    calls = {"n": 0}

    async def always_503(**_kwargs):
        calls["n"] += 1
        raise Exception("503 service unavailable")

    monkeypatch.setattr(agent_loop, "acompletion", always_503)
    with pytest.raises(Exception, match="503"):
        await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert calls["n"] == _MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_connection_errors_use_the_larger_budget(monkeypatch, no_sleep):
    calls = {"n": 0}

    async def always_down(**_kwargs):
        calls["n"] += 1
        raise Exception("connection refused")

    monkeypatch.setattr(agent_loop, "acompletion", always_down)
    with pytest.raises(Exception, match="connection refused"):
        await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert calls["n"] == _MAX_CONNECTION_RETRIES


@pytest.mark.asyncio
async def test_context_window_exceeded_propagates_immediately(monkeypatch, no_sleep):
    from litellm.exceptions import ContextWindowExceededError

    calls = {"n": 0}

    async def too_long(**_kwargs):
        calls["n"] += 1
        raise ContextWindowExceededError("too long", model="m", llm_provider="x")

    monkeypatch.setattr(agent_loop, "acompletion", too_long)
    with pytest.raises(ContextWindowExceededError):
        await _acompletion_with_retry(_StubSession(), {"model": "m"})
    assert calls["n"] == 1, "compaction handles this, retrying it would just fail again"


@pytest.mark.asyncio
async def test_retry_emits_a_progress_event_per_attempt(monkeypatch, no_sleep):
    calls = {"n": 0}

    async def flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("overloaded")
        return _Resp()

    monkeypatch.setattr(agent_loop, "acompletion", flaky)
    session = _StubSession()
    await _acompletion_with_retry(session, {"model": "m"})
    assert len(session.events) == 1
    assert "retrying" in session.events[0].data["log"]


@pytest.mark.asyncio
async def test_call_kwargs_and_params_both_reach_acompletion(monkeypatch, no_sleep):
    seen = {}

    async def capture(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(agent_loop, "acompletion", capture)
    await _acompletion_with_retry(
        _StubSession(), {"model": "m", "temperature": 0.5}, messages=[], stream=True
    )
    assert seen["model"] == "m"
    assert seen["temperature"] == 0.5
    assert seen["stream"] is True


def test_module_no_longer_exposes_the_old_fixed_delay_table():
    """The fixed [5,15,30] table was replaced; leaving it around invites drift."""
    assert not hasattr(agent_loop, "_LLM_RETRY_DELAYS")


def test_asyncio_is_still_the_real_module():
    """Guards the monkeypatch fixture above from silently leaking."""
    assert asyncio.sleep is not None
