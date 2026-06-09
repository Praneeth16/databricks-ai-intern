"""Unit tests for agent.tools.databricks_sandbox.

Coverage:
    - probe_compute prefers pool over on-demand when instance_pool_id is set.
    - probe_compute falls through to on-demand when no pool.
    - Cluster create polls clusters/get until RUNNING.
    - File ops route /Volumes/ to wc.files, /Workspace/ to wc.workspace.
    - bash builds the right Python wrapper and posts to commands/execute.
    - edit refuses unread files.
"""

from __future__ import annotations

import asyncio
import io
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.core import db_client
from agent.tools import databricks_sandbox as ds


def _settings(**o):
    d = dict(
        host="https://ws", warehouse_id="wh1",
        experiment_path="/Shared/databricks-ai-intern",
        uc_catalog="databricks_ai_intern", uc_schema="agent", uc_volume="scratch",
        secret_scope="databricks-ai-intern", lakebase_instance=None, instance_pool_id=None,
        default_node_type_id="g5.xlarge",
        default_runtime_version="15.4.x-gpu-ml-scala2.12",
        prompt_registry_name="databricks_ai_intern.agent.system_prompt",
    )
    d.update(o)
    return db_client.DatabricksSettings(**d)


def _mock_wc_for_cluster_lifecycle(running_after: int = 1):
    """Returns a mock wc whose api_client.do simulates cluster create + get."""
    wc = MagicMock()
    state = {"calls": 0}

    def _do(method, path, **kwargs):
        if path == "/api/2.1/clusters/create":
            return {"cluster_id": "cluster-abc"}
        if path == "/api/2.1/clusters/get":
            state["calls"] += 1
            if state["calls"] >= running_after:
                return {"state": "RUNNING"}
            return {"state": "PENDING"}
        if path == "/api/1.2/contexts/create":
            return {"id": "ctx-xyz"}
        if path == "/api/1.2/contexts/status":
            return {"status": "Running"}
        return {}

    wc.api_client.do.side_effect = _do
    return wc


# ---------------------------------------------------------------------------
# probe_compute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_compute_prefers_pool_when_set():
    wc = _mock_wc_for_cluster_lifecycle()
    settings = _settings(instance_pool_id="pool-xyz")
    choice = await ds.probe_compute(wc, settings, hardware="a10g-large")
    assert choice.kind == "pool"
    assert choice.pool_id == "pool-xyz"
    assert choice.cluster_id == "cluster-abc"


@pytest.mark.asyncio
async def test_probe_compute_falls_back_to_on_demand():
    wc = _mock_wc_for_cluster_lifecycle()
    settings = _settings(instance_pool_id=None)
    choice = await ds.probe_compute(wc, settings, hardware="a10g-large")
    assert choice.kind == "on_demand"
    # Hardware mapped to AWS node type.
    assert choice.node_type_id == "g5.4xlarge"


@pytest.mark.asyncio
async def test_create_cluster_raises_on_terminal_state():
    wc = MagicMock()

    def _do(method, path, **kwargs):
        if path == "/api/2.1/clusters/create":
            return {"cluster_id": "c1"}
        if path == "/api/2.1/clusters/get":
            return {"state": "ERROR", "state_message": "no quota"}
        return {}

    wc.api_client.do.side_effect = _do
    with pytest.raises(RuntimeError, match="no quota"):
        await ds._create_cluster(
            wc, settings=_settings(), instance_pool_id=None,
            node_type_id="g5.xlarge", hardware="a10g-small",
        )


# ---------------------------------------------------------------------------
# create_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_async_returns_handle_with_context():
    wc = _mock_wc_for_cluster_lifecycle()
    settings = _settings(instance_pool_id="pool-xyz")
    sb = await ds.DatabricksSandbox.create_async(
        settings, hardware="cpu-basic", wc=wc, user_email="alice@ex.com",
    )
    assert sb.cluster_id == "cluster-abc"
    assert sb.context_id == "ctx-xyz"
    assert sb.compute.kind == "pool"
    assert sb.work_dir.startswith("/Workspace/Users/alice@ex.com/")


# ---------------------------------------------------------------------------
# file ops
# ---------------------------------------------------------------------------


def test_download_routes_volume_path_to_files_api():
    wc = MagicMock()
    resp = MagicMock()
    resp.contents = io.BytesIO(b"hello")
    wc.files.download.return_value = resp
    data = ds._download(wc, "/Volumes/databricks_ai_intern/agent/scratch/x.txt")
    assert data == b"hello"
    wc.files.download.assert_called_once_with(file_path="/Volumes/databricks_ai_intern/agent/scratch/x.txt")


def test_download_routes_workspace_path_to_workspace_api():
    wc = MagicMock()
    import base64
    resp = MagicMock()
    resp.content = base64.b64encode(b"hi there").decode()
    wc.workspace.export.return_value = resp
    data = ds._download(wc, "/Workspace/Users/u/script.py")
    assert data == b"hi there"


def test_download_rejects_other_paths():
    with pytest.raises(ValueError):
        ds._download(MagicMock(), "/etc/passwd")


def test_upload_routes_volume_path():
    wc = MagicMock()
    ds._upload(wc, "/Volumes/databricks_ai_intern/agent/scratch/x.txt", b"data")
    kwargs = wc.files.upload.call_args.kwargs
    assert kwargs["file_path"] == "/Volumes/databricks_ai_intern/agent/scratch/x.txt"
    assert kwargs["overwrite"] is True


def test_upload_routes_workspace_path_creates_parents():
    wc = MagicMock()
    ds._upload(wc, "/Workspace/Users/u/x.py", b"data")
    wc.workspace.mkdirs.assert_called_once_with("/Workspace/Users/u")
    wc.workspace.upload.assert_called_once()


# ---------------------------------------------------------------------------
# Sandbox tool surface
# ---------------------------------------------------------------------------


def _make_sandbox(wc=None):
    wc = wc or MagicMock()
    return ds.DatabricksSandbox(
        wc=wc,
        settings=_settings(),
        compute=ds.ComputeChoice(kind="pool", cluster_id="c1", owns_cluster=True),
        context_id="ctx",
        user_email="alice@ex.com",
    )


def test_read_uses_download_and_marks_read():
    wc = MagicMock()
    resp = MagicMock()
    resp.contents = io.BytesIO(b"line one\nline two\n")
    wc.files.download.return_value = resp
    sb = _make_sandbox(wc)
    out = sb.read("/Volumes/databricks_ai_intern/agent/scratch/x.txt")
    assert out.success
    assert "1\tline one" in out.output
    assert "/Volumes/databricks_ai_intern/agent/scratch/x.txt" in sb._files_read


def test_write_uploads_and_marks_read():
    wc = MagicMock()
    sb = _make_sandbox(wc)
    out = sb.write("/Volumes/databricks_ai_intern/agent/scratch/x.txt", "hello")
    assert out.success
    wc.files.upload.assert_called_once()
    assert "/Volumes/databricks_ai_intern/agent/scratch/x.txt" in sb._files_read


def test_edit_refuses_unread_file():
    wc = MagicMock()
    sb = _make_sandbox(wc)
    out = sb.edit("/Volumes/x/y/z/a.txt", "old", "new")
    assert not out.success
    assert "has not been read" in out.error


def test_edit_replaces_after_read():
    wc = MagicMock()
    sb = _make_sandbox(wc)
    sb._files_read.add("/Volumes/x/y/z/a.txt")
    resp = MagicMock()
    resp.contents = io.BytesIO(b"foo bar foo")
    wc.files.download.return_value = resp
    out = sb.edit("/Volumes/x/y/z/a.txt", "foo", "baz", replace_all=True)
    assert out.success
    payload = wc.files.upload.call_args.kwargs["contents"].read()
    assert payload == b"baz bar baz"


def test_edit_refuses_when_old_eq_new():
    sb = _make_sandbox()
    sb._files_read.add("/Volumes/x.txt")
    out = sb.edit("/Volumes/x.txt", "x", "x")
    assert not out.success
    assert "identical" in out.error


def test_call_tool_dispatches():
    sb = _make_sandbox()
    sb._files_read.add("/Volumes/x.txt")
    sb.read = MagicMock(return_value=ds.ToolResult(True, output="r"))
    sb.write = MagicMock(return_value=ds.ToolResult(True, output="w"))
    assert sb.call_tool("read", {"path": "/Volumes/x.txt"}).output == "r"
    assert sb.call_tool("write", {"path": "/Volumes/x.txt", "content": "z"}).output == "w"


def test_call_tool_unknown():
    sb = _make_sandbox()
    out = sb.call_tool("nope", {})
    assert not out.success
    assert "Unknown tool" in out.error


# ---------------------------------------------------------------------------
# bash sync/async entry points
# ---------------------------------------------------------------------------


def test_bash_runs_via_asyncio_run_outside_event_loop():
    sb = _make_sandbox()
    with patch.object(
        sb, "_run_python", new=AsyncMock(return_value=ds.ToolResult(True, output="ok")),
    ) as rp:
        out = sb.bash("echo hi")
    assert out.success
    assert out.output == "ok"
    py = rp.call_args.args[0]
    assert "subprocess.run" in py
    assert "echo hi" in py


@pytest.mark.asyncio
async def test_bash_raises_inside_running_event_loop():
    sb = _make_sandbox()
    with pytest.raises(RuntimeError, match="bash_async"):
        sb.bash("echo hi")


@pytest.mark.asyncio
async def test_bash_async_awaits_run_python():
    sb = _make_sandbox()
    with patch.object(
        sb, "_run_python", new=AsyncMock(return_value=ds.ToolResult(True, output="ok")),
    ):
        out = await sb.bash_async("echo hi")
    assert out.success


# ---------------------------------------------------------------------------
# cluster delete retry
# ---------------------------------------------------------------------------


def test_delete_retries_permanent_delete_once(monkeypatch):
    wc = MagicMock()
    state = {"attempts": 0}

    def _do(method, path, **kwargs):
        if path == "/api/2.1/clusters/permanent-delete":
            state["attempts"] += 1
            if state["attempts"] == 1:
                raise Exception("transient")
        return {}

    wc.api_client.do.side_effect = _do
    sb = _make_sandbox(wc)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    sb.delete()
    assert state["attempts"] == 2


def test_delete_logs_manual_cleanup_when_retry_fails(monkeypatch, caplog):
    wc = MagicMock()

    def _do(method, path, **kwargs):
        if path == "/api/2.1/clusters/permanent-delete":
            raise Exception("still broken")
        return {}

    wc.api_client.do.side_effect = _do
    sb = _make_sandbox(wc)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    with caplog.at_level(logging.WARNING, logger="agent.tools.databricks_sandbox"):
        sb.delete()
    assert "manual cleanup required" in caplog.text
    assert "c1" in caplog.text


@pytest.mark.asyncio
async def test_safe_terminate_retries_then_warns(monkeypatch, caplog):
    wc = MagicMock()
    wc.api_client.do.side_effect = Exception("still broken")
    compute = ds.ComputeChoice(kind="pool", cluster_id="c9", owns_cluster=True)
    monkeypatch.setattr(ds.asyncio, "sleep", AsyncMock())
    with caplog.at_level(logging.WARNING, logger="agent.tools.databricks_sandbox"):
        await ds.DatabricksSandbox._safe_terminate(wc, compute)
    assert wc.api_client.do.call_count == 2
    assert "manual cleanup required" in caplog.text
    assert "c9" in caplog.text
