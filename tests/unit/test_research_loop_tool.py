"""Unit tests for the ``research_loop`` agent tool.

Monkeypatches the runner + ledger so no Databricks contact. Verifies dispatch,
validation, pass-through, registration, and approval gating.
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.research_loop import LoopResult, RoundSummary
from agent.tools import research_loop_tool


class _FakeSession:
    config = None
    session_id = "sess-1"
    databricks_user_token = None


@pytest.fixture
def session():
    return _FakeSession()


@pytest.fixture(autouse=True)
def _patched(tmp_path, monkeypatch):
    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")
    monkeypatch.setattr(research_loop_tool, "_get_ledger", lambda s: ledger)

    async def _no_jobs(s):
        return None

    monkeypatch.setattr(research_loop_tool, "_get_jobs_tool", _no_jobs)
    return ledger


def _args(**o):
    a = {
        "task_id": "t",
        "metric_name": "roc_auc",
        "script_template": "print('SWEEP_METRIC=0.9')",
        "hypotheses": [
            {"hypothesis": "a", "method": "m", "config": {"v": 1}},
            {"hypothesis": "b", "method": "m", "config": {"v": 2}},
        ],
    }
    a.update(o)
    return a


@pytest.mark.asyncio
async def test_happy_path_formats_summary(session, monkeypatch):
    captured = {}

    async def fake_loop(**kwargs):
        captured.update(kwargs)
        return LoopResult(
            rounds=[RoundSummary(0, 2, 0.95, "exp-1", True, "ok", 2.0)],
            best_experiment_id="exp-1", best_metric=0.95,
            total_cost_usd=2.0, accepted_count=1, stop_reason="exhausted",
        )

    monkeypatch.setattr(research_loop_tool, "run_research_loop", fake_loop)
    res = await research_loop_tool.research_loop_handler(
        _args(budget_usd=10.0, max_rounds=3, batch_size=2, target_metric=0.99), session=session
    )
    assert res["isError"] is False
    assert "stopped: exhausted" in res["formatted"]
    assert "roc_auc=0.95" in res["formatted"]
    # pass-through of control params
    assert captured["budget_usd"] == 10.0
    assert captured["max_rounds"] == 3
    assert captured["target_metric"] == 0.99
    assert captured["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_missing_field_is_error(session):
    res = await research_loop_tool.research_loop_handler(
        {"task_id": "t", "metric_name": "roc_auc"}, session=session
    )
    assert res["isError"] is True


@pytest.mark.asyncio
async def test_bad_hypothesis_item_is_error(session):
    res = await research_loop_tool.research_loop_handler(
        _args(hypotheses=[{"method": "m"}]), session=session
    )
    assert res["isError"] is True


def test_pool_source_yields_batches_then_empty():
    src = research_loop_tool._pool_source([1, 2, 3, 4, 5], batch_size=2)
    assert src(None) == [1, 2]
    assert src(None) == [3, 4]
    assert src(None) == [5]
    assert src(None) == []


def test_registered_and_approval_gated():
    from agent.core.tools import create_builtin_tools
    from agent.core.agent_loop import _needs_approval

    names = {t.name for t in create_builtin_tools()}
    assert "research_loop" in names
    # launches jobs → must require approval (not yolo)
    assert _needs_approval("research_loop", {}, None) is True
