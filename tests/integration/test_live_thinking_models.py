"""Opt-in live provider checks for thinking-metadata replay (issue #6).

Paid integration coverage for the gate logic landed in issue #3
(``_should_replay_thinking_state``). The default unit suite is fully
mock-based — without live coverage a LiteLLM upgrade or an FMAPI
served-model schema bump can silently regress tool-continuation
behavior for Claude-on-Databricks.

These tests intentionally call paid model APIs through Databricks
Foundation Model API and are skipped unless::

    ML_INTERN_LIVE_LLM_TESTS=1
    DATABRICKS_HOST=...
    DATABRICKS_TOKEN=...   (or rely on a configured profile)

are set. Default CI stays credential-free.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm import Message

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None  # type: ignore

from agent.core.agent_loop import (
    _assistant_message_from_result,
    _call_llm_streaming,
)
from agent.core.llm_params import _resolve_llm_params


# Honour an optional env file so contributors can keep paid creds out of
# the shell history (``ML_INTERN_LIVE_ENV_FILE=~/.ml-intern-live.env``).
if (env_file := os.environ.get("ML_INTERN_LIVE_ENV_FILE")) and load_dotenv:
    load_dotenv(Path(env_file))

LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"

# Foundation Model API endpoint ids on Databricks. Override per-workspace
# via env if the default endpoint names diverge (some workspaces ship
# ``databricks-claude-opus-4-7`` etc — keep the override path).
CLAUDE_MODEL = os.environ.get(
    "ML_INTERN_LIVE_CLAUDE_MODEL",
    "databricks/databricks-claude-opus-4-6",
)
NON_CLAUDE_MODEL = os.environ.get(
    "ML_INTERN_LIVE_NON_CLAUDE_MODEL",
    "databricks/databricks-meta-llama-3-3-70b-instruct",
)

REPORT_RESULT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_result",
            "description": "Report the final test result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The exact marker requested by the test.",
                    }
                },
                "required": ["answer"],
            },
        },
    }
]


def _skip_without_live_flag() -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip("set ML_INTERN_LIVE_LLM_TESTS=1 to run paid live LLM tests")


def _skip_without_databricks_auth() -> None:
    """Databricks SDK auth chain — host + token (or a profile / M2M cred)."""
    if not os.environ.get("DATABRICKS_HOST") and not os.environ.get(
        "DATABRICKS_CONFIG_PROFILE"
    ):
        pytest.skip(
            "set DATABRICKS_HOST + DATABRICKS_TOKEN (or DATABRICKS_CONFIG_PROFILE) "
            "to run live Databricks FMAPI tests"
        )


def _session(model_name: str):
    events = []

    async def send_event(event):
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        is_cancelled=False,
        send_event=send_event,
        events=events,
    )


@pytest.mark.asyncio
async def test_live_databricks_claude_preserves_thinking_metadata_for_replay():
    """Round-trip Claude-on-FMAPI with reasoning_effort='high' through the
    streaming path and confirm thinking_blocks survive the rebuild into
    the assistant-history Message. Regression coverage for issue #3.
    """
    _skip_without_live_flag()
    _skip_without_databricks_auth()

    session = _session(CLAUDE_MODEL)
    llm_params = _resolve_llm_params(
        CLAUDE_MODEL,
        reasoning_effort="high",
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content=(
                    "Use careful reasoning for this small check. "
                    "If 17 * 19 = 323, call report_result with answer CLAUDE_OK."
                ),
            )
        ],
        tools=REPORT_RESULT_TOOL,
        llm_params=llm_params,
    )

    replay = _assistant_message_from_result(
        result,
        model_name=CLAUDE_MODEL,
    )

    assert result.content or result.tool_calls_acc
    # FMAPI may or may not surface thinking_blocks depending on the served
    # model version's adaptive-thinking config. Assert the gate REPLAYS
    # whatever the response carried — empty is also fine; the regression
    # would be a mismatch between result.thinking_blocks and replay.
    assert getattr(replay, "thinking_blocks", None) == result.thinking_blocks
    assert getattr(replay, "reasoning_content", None) == result.reasoning_content


@pytest.mark.asyncio
async def test_live_databricks_non_claude_strips_reasoning_metadata():
    """Non-Claude FMAPI endpoints (Llama / DBRX / Mistral) reject
    ``reasoning_content`` on the next assistant-history turn. The gate in
    ``_should_replay_thinking_state`` MUST strip both fields even when
    the served model name carries the ``databricks/`` prefix.
    """
    _skip_without_live_flag()
    _skip_without_databricks_auth()

    session = _session(NON_CLAUDE_MODEL)
    llm_params = _resolve_llm_params(
        NON_CLAUDE_MODEL,
        reasoning_effort="low",
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content="Call report_result with answer LLAMA_OK.",
            )
        ],
        tools=REPORT_RESULT_TOOL,
        llm_params=llm_params,
    )

    # Force-populate reasoning_content so we exercise the strip path
    # regardless of whether the live model actually emitted it.
    result.reasoning_content = result.reasoning_content or "synthetic-reasoning"
    replay = _assistant_message_from_result(
        result,
        model_name=NON_CLAUDE_MODEL,
    )

    assert result.content or result.tool_calls_acc
    assert getattr(replay, "thinking_blocks", None) is None
    assert getattr(replay, "reasoning_content", None) is None
