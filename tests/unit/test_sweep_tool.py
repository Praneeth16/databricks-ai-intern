"""Unit tests for the ``sweep`` builtin tool.

Exercises the agent-facing wrapper without touching Databricks: ``_get_ledger``
points at a tmp JSONL ledger, ``_get_jobs_tool`` returns ``None``, and
``run_sweep_async`` is replaced by an async fake that returns a canned
:class:`SweepResult`. This isolates the handler's validation + formatting +
kwarg plumbing from the orchestrator and the Jobs API.
"""

from __future__ import annotations

import pytest

from agent.core.experiment_ledger import ExperimentLedger
from agent.core.sweep import SweepOutcome, SweepResult
from agent.tools import sweep_tool


class _FakeSession:
    config = None
    session_id = "sess-1"
    databricks_user_token = None
    user_email = None


@pytest.fixture
def session():
    return _FakeSession()


@pytest.fixture(autouse=True)
def _no_databricks(tmp_path, monkeypatch):
    """Route the ledger at a tmp JSONL and stub out the jobs-tool factory."""
    ledger = ExperimentLedger(local_path=tmp_path / "exp.jsonl")
    monkeypatch.setattr(sweep_tool, "_get_ledger", lambda session: ledger)

    async def _no_jobs(session):
        return None

    monkeypatch.setattr(sweep_tool, "_get_jobs_tool", _no_jobs)
    return ledger


def _canned_result() -> SweepResult:
    best = SweepOutcome(
        experiment_id="exp-2", method="lightgbm", config={"lr": 0.1},
        actual_metric=0.96, cost_usd=1.5, status="done",
    )
    worse = SweepOutcome(
        experiment_id="exp-1", method="lightgbm", config={"lr": 0.05},
        actual_metric=0.90, cost_usd=1.2, status="done",
    )
    failed = SweepOutcome(
        experiment_id="exp-3", method="xgboost", config={"lr": 0.3},
        actual_metric=None, cost_usd=None, status="failed", error="boom",
    )
    return SweepResult(
        outcomes=[best, worse, failed],
        best=best,
        total_cost_usd=2.7,
        n_submitted=3,
        n_skipped=0,
        n_failed=1,
    )


def _patch_run_sweep(monkeypatch, captured: dict):
    async def _fake_run_sweep_async(**kwargs):
        captured.update(kwargs)
        return _canned_result()

    monkeypatch.setattr(sweep_tool, "run_sweep_async", _fake_run_sweep_async)


def _sweep_args(**o):
    args = {
        "task_id": "task-A",
        "metric_name": "roc_auc",
        "script_template": "print(f'SWEEP_METRIC={metric}')",
        "hypotheses": [
            {"hypothesis": "lr=0.05", "method": "lightgbm", "config": {"lr": 0.05}},
            {"hypothesis": "lr=0.1", "method": "lightgbm", "config": {"lr": 0.1}},
        ],
    }
    args.update(o)
    return args


# ── happy path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_formats_ranked_best_and_totals(session, monkeypatch):
    captured: dict = {}
    _patch_run_sweep(monkeypatch, captured)

    result = await sweep_tool.sweep_handler(_sweep_args(), session=session)

    assert result["isError"] is False
    out = result["formatted"]
    # Ranked outcomes appear with method, config summary, metric, status.
    assert "ranked 3 variant(s)" in out
    assert "lr=0.1" in out and "roc_auc=0.96" in out and "done" in out
    assert "failed" in out and "boom" in out
    # Best line.
    assert "Best: lightgbm" in out
    assert "exp-2" in out
    # Totals.
    assert "submitted=3" in out
    assert "skipped=0" in out
    assert "failed=1" in out
    assert "total_cost_usd=2.7" in out


# ── validation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_required_field_is_error(session, monkeypatch):
    captured: dict = {}
    _patch_run_sweep(monkeypatch, captured)
    args = _sweep_args()
    del args["metric_name"]
    result = await sweep_tool.sweep_handler(args, session=session)
    assert result["isError"] is True
    assert "metric_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_empty_hypotheses_is_error(session, monkeypatch):
    captured: dict = {}
    _patch_run_sweep(monkeypatch, captured)
    result = await sweep_tool.sweep_handler(_sweep_args(hypotheses=[]), session=session)
    assert result["isError"] is True
    assert "hypotheses is required" in result["formatted"]


# ── kwarg plumbing ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_passes_hypotheses_and_budget_through(session, monkeypatch):
    captured: dict = {}
    _patch_run_sweep(monkeypatch, captured)

    hyps = _sweep_args()["hypotheses"]
    result = await sweep_tool.sweep_handler(
        _sweep_args(hypotheses=hyps, budget_usd=5.0, higher_is_better=False),
        session=session,
    )

    assert result["isError"] is False
    assert captured["task_id"] == "task-A"
    assert captured["metric_name"] == "roc_auc"
    assert captured["hypotheses"] == hyps
    assert captured["budget_usd"] == 5.0
    assert captured["higher_is_better"] is False
    assert captured["session_id"] == "sess-1"
    # submit_fn / score_fn were built and handed through.
    assert callable(captured["submit_fn"])
    assert callable(captured["score_fn"])


def test_sweep_and_critic_registered_as_builtin_tools():
    from agent.core.tools import create_builtin_tools

    names = {t.name for t in create_builtin_tools()}
    assert "sweep" in names
    assert "critic" in names


def test_render_script_injects_config_and_preserves_braces():
    from agent.tools.sweep_tool import _render_script

    # A script with dict literals + f-strings (brace-heavy) must survive intact.
    template = (
        "params = {'lr': CONFIG['lr']}\n"
        "auc = 0.9\n"
        "print(f'SWEEP_METRIC={auc}')\n"
    )
    out = _render_script(template, {"lr": 0.05, "name": "x'); import os"})
    assert "CONFIG = _json.loads(" in out
    assert template in out  # original body preserved verbatim, no .format mangling
    # The script compiles (no KeyError/injection from the malicious string value).
    compile(out, "<sweep>", "exec")


@pytest.mark.asyncio
async def test_bad_hypothesis_item_is_error(session):
    result = await experiment_handler_missing_fields(session)
    assert result["isError"] is True


async def experiment_handler_missing_fields(session):
    from agent.tools import sweep_tool

    return await sweep_tool.sweep_handler(
        {
            "task_id": "t",
            "metric_name": "roc_auc",
            "script_template": "print('SWEEP_METRIC=0.9')",
            "hypotheses": [{"method": "m"}],  # missing hypothesis + config
        },
        session=session,
    )
