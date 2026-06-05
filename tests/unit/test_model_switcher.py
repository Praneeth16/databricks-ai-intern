"""Tests for ``probe_and_switch_model`` outcome handling.

Focus: an inconclusive (transient) effort probe must NOT silently switch the
active model mid-conversation. On long unattended runs, committing the switch
on an unresolved probe risks a hard failure on the very next turn — keep the
known-good model and warn that the switch was deferred. A conclusive probe
must still switch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.core.effort_probe import ProbeInconclusive, ProbeOutcome
from agent.core.model_switcher import probe_and_switch_model


def _make_session():
    session = MagicMock()
    session.model_effective_effort = {}
    return session


def _make_config(model_name="databricks/databricks-claude-opus-4-6", effort="high"):
    return SimpleNamespace(model_name=model_name, reasoning_effort=effort)


@pytest.mark.asyncio
async def test_inconclusive_probe_leaves_model_unchanged_and_warns():
    config = _make_config()
    session = _make_session()
    console = MagicMock()
    target = "databricks/databricks-claude-sonnet-4-6"

    with (
        patch(
            "agent.core.model_switcher._print_routing_info", return_value=True
        ),
        patch(
            "agent.core.model_switcher.probe_effort",
            new=AsyncMock(side_effect=ProbeInconclusive("503 service unavailable")),
        ),
    ):
        await probe_and_switch_model(
            target, config, session, console, hf_token=None
        )

    # Active model untouched — no switch committed on an inconclusive probe.
    session.update_model.assert_not_called()
    assert config.model_name == "databricks/databricks-claude-opus-4-6"
    assert target not in session.model_effective_effort

    # A clear "deferred" warning was emitted.
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "deferred" in printed.lower()
    assert target in printed


@pytest.mark.asyncio
async def test_conclusive_probe_still_switches():
    config = _make_config()
    session = _make_session()
    console = MagicMock()
    target = "databricks/databricks-claude-sonnet-4-6"

    outcome = ProbeOutcome(
        effective_effort="high", attempts=1, elapsed_ms=120, note=None
    )

    with (
        patch(
            "agent.core.model_switcher._print_routing_info", return_value=True
        ),
        patch(
            "agent.core.model_switcher.probe_effort",
            new=AsyncMock(return_value=outcome),
        ),
    ):
        await probe_and_switch_model(
            target, config, session, console, hf_token=None
        )

    # Switch committed with the probed effort cached.
    session.update_model.assert_called_once_with(target)
    assert session.model_effective_effort[target] == "high"
