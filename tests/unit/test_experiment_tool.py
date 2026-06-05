"""Unit tests for the ``experiment`` builtin tool.

Exercises dispatch + per-op validation against a real ``ExperimentLedger`` on a
tmp JSONL path (no workspace). ``experiment_tool._get_ledger`` is monkeypatched
so the handler never touches settings/db_client.
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.tools import experiment_tool


class _FakeSession:
    config = None
    session_id = "sess-1"
    databricks_user_token = None


@pytest.fixture
def session():
    return _FakeSession()


@pytest.fixture(autouse=True)
def _ledger_on_tmp(tmp_path, monkeypatch):
    """Route the handler at a JSONL ledger on a tmp path."""
    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")
    monkeypatch.setattr(experiment_tool, "_get_ledger", lambda session: ledger)
    return ledger


def _propose_args(**o):
    args = {
        "op": "propose",
        "task_id": "task-A",
        "hypothesis": "lightgbm beats baseline",
        "method": "lightgbm",
        "config": {"lr": 0.05, "n_estimators": 500},
        "metric_name": "roc_auc",
    }
    args.update(o)
    return args


async def _propose(session, **o) -> str:
    result = await experiment_tool.experiment_handler(_propose_args(**o), session=session)
    assert result["isError"] is False
    # "Proposed experiment <uuid> (status=proposed)."
    return result["formatted"].split()[2]


# ── propose ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_propose_returns_id_in_output(session):
    result = await experiment_tool.experiment_handler(_propose_args(), session=session)
    assert result["isError"] is False
    assert "Proposed experiment" in result["formatted"]
    # An id token must be present.
    exp_id = result["formatted"].split()[2]
    assert len(exp_id) > 0


@pytest.mark.asyncio
async def test_propose_missing_field_is_error(session):
    args = _propose_args()
    del args["hypothesis"]
    result = await experiment_tool.experiment_handler(args, session=session)
    assert result["isError"] is True
    assert "hypothesis is required" in result["formatted"]


# ── record ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_updates_experiment(session, _ledger_on_tmp):
    exp_id = await _propose(session, expected_metric=0.95)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id, "actual_metric": 0.93, "cost_usd": 1.2},
        session=session,
    )
    assert result["isError"] is False
    assert exp_id in result["formatted"]
    # Ledger row reflects the result + computed repro_gap.
    row = _ledger_on_tmp._get(exp_id)
    assert row.actual_metric == 0.93
    assert row.status == "done"
    assert row.repro_gap == pytest.approx(0.95 - 0.93)


@pytest.mark.asyncio
async def test_record_missing_actual_metric_is_error(session):
    exp_id = await _propose(session)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id}, session=session
    )
    assert result["isError"] is True
    assert "actual_metric is required" in result["formatted"]


# ── list ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_shows_the_row(session):
    exp_id = await _propose(session)
    result = await experiment_tool.experiment_handler(
        {"op": "list", "task_id": "task-A"}, session=session
    )
    assert result["isError"] is False
    assert exp_id[:8] in result["formatted"]
    assert "lightgbm" in result["formatted"]


@pytest.mark.asyncio
async def test_list_empty_task(session):
    result = await experiment_tool.experiment_handler(
        {"op": "list", "task_id": "nope"}, session=session
    )
    assert result["isError"] is False
    assert "No experiments" in result["formatted"]


# ── best ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_best_picks_higher_metric(session):
    low = await _propose(session, hypothesis="low")
    high = await _propose(session, hypothesis="high", config={"lr": 0.1})
    await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": low, "actual_metric": 0.90}, session=session
    )
    await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": high, "actual_metric": 0.96}, session=session
    )
    result = await experiment_tool.experiment_handler(
        {"op": "best", "task_id": "task-A", "metric_name": "roc_auc"}, session=session
    )
    assert result["isError"] is False
    assert high[:8] in result["formatted"]
    assert low[:8] not in result["formatted"]


@pytest.mark.asyncio
async def test_best_none_when_unscored(session):
    await _propose(session)  # proposed but never recorded
    result = await experiment_tool.experiment_handler(
        {"op": "best", "task_id": "task-A", "metric_name": "roc_auc"}, session=session
    )
    assert result["isError"] is False
    assert "No scored experiments" in result["formatted"]


# ── find_similar ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_similar_hits_identical_config(session):
    exp_id = await _propose(session)
    result = await experiment_tool.experiment_handler(
        {
            "op": "find_similar",
            "task_id": "task-A",
            "method": "lightgbm",
            "config": {"lr": 0.05, "n_estimators": 500},
        },
        session=session,
    )
    assert result["isError"] is False
    assert "Already tried" in result["formatted"]
    assert exp_id[:8] in result["formatted"]


@pytest.mark.asyncio
async def test_find_similar_misses_different_config(session):
    await _propose(session)
    result = await experiment_tool.experiment_handler(
        {
            "op": "find_similar",
            "task_id": "task-A",
            "method": "lightgbm",
            "config": {"lr": 0.2, "n_estimators": 500},
        },
        session=session,
    )
    assert result["isError"] is False
    assert "has not been tried" in result["formatted"]


# ── dispatch ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_op_is_error(session):
    result = await experiment_tool.experiment_handler({}, session=session)
    assert result["isError"] is True
    assert "op is required" in result["formatted"]


@pytest.mark.asyncio
async def test_unknown_op_is_error(session):
    result = await experiment_tool.experiment_handler({"op": "frobnicate"}, session=session)
    assert result["isError"] is True
    assert "Unknown op" in result["formatted"]


# ── repro-gap gate wired into record ─────────────────────────────────────


@pytest.mark.asyncio
async def test_record_major_gap_appends_reproduce_directive(session):
    exp_id = await _propose(session, expected_metric=0.95)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id, "actual_metric": 0.80},
        session=session,
    )
    assert result["isError"] is False
    # 0.95 - 0.80 = 0.15 >> major threshold → reproduce-first directive surfaces.
    assert "reproduce" in result["formatted"].lower()


@pytest.mark.asyncio
async def test_record_met_expectation_no_directive(session):
    exp_id = await _propose(session, expected_metric=0.95)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id, "actual_metric": 0.951},
        session=session,
    )
    assert result["isError"] is False
    assert "reproduce" not in result["formatted"].lower()


# ── registration ─────────────────────────────────────────────────────────


def test_experiment_registered_as_builtin_tool():
    from agent.core.tools import create_builtin_tools

    names = {t.name for t in create_builtin_tools()}
    assert "experiment" in names


@pytest.mark.asyncio
async def test_record_loss_metric_missed_triggers_directive(session):
    exp_id = await _propose(session, metric_name="eval_loss", expected_metric=0.30)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id, "actual_metric": 0.55},
        session=session,
    )
    assert result["isError"] is False
    # Loss far above target = underperformed → reproduce-first, not "ok".
    assert "reproduce" in result["formatted"].lower()


@pytest.mark.asyncio
async def test_record_loss_metric_beat_no_directive(session):
    exp_id = await _propose(session, metric_name="eval_loss", expected_metric=0.30)
    result = await experiment_tool.experiment_handler(
        {"op": "record", "experiment_id": exp_id, "actual_metric": 0.20},
        session=session,
    )
    assert result["isError"] is False
    assert "reproduce" not in result["formatted"].lower()
