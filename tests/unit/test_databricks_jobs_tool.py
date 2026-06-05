"""Unit tests for agent.tools.databricks_jobs_tool.

Coverage:
    - _filter_agent_env: drops auth/cloud creds, keeps dynamic secret refs.
    - _parse_timeout: smoke.
    - dispatch: unknown ops surfaced as errors.
    - _build_submit_body: cluster spec wiring (pool vs node_type, env vars,
      serverless environment_key).
    - _resolve_or_stage_script: writes inline script via wc.workspace.upload.
    - _run_finetune: posts to the foundation-model-training endpoint and
      includes register_to + experiment_path defaults.
    - _ps / _cancel: thin REST wrappers, mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools import databricks_jobs_tool as djt
from agent.core import db_client


def _make_settings(**overrides):
    defaults = dict(
        host="https://ws.cloud.databricks.com",
        warehouse_id="abc123",
        experiment_path="/Shared/ml-intern",
        uc_catalog="ml_intern",
        uc_schema="agent",
        uc_volume="scratch",
        secret_scope="ml-intern",
        lakebase_instance=None,
        instance_pool_id=None,
        default_node_type_id="g5.xlarge",
        default_runtime_version="15.4.x-gpu-ml-scala2.12",
        prompt_registry_name="ml_intern.agent.system_prompt",
    )
    defaults.update(overrides)
    return db_client.DatabricksSettings(**defaults)


def _mock_wc():
    wc = MagicMock()
    wc.api_client.do = MagicMock()
    wc.workspace.mkdirs = MagicMock()
    wc.workspace.upload = MagicMock()
    # Default: nothing exists at the target path. Tests that exercise the
    # type-collision codepath replace this side_effect locally.
    wc.workspace.get_status = MagicMock(side_effect=Exception("not found"))
    wc.workspace.delete = MagicMock()
    return wc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_filter_agent_env_drops_blocklist():
    out = djt._filter_agent_env({
        "DATABRICKS_TOKEN": "leak",
        "AWS_SECRET_ACCESS_KEY": "leak",
        "OPENAI_API_KEY": "{{secrets/ml-intern/openai}}",
        "PYTHONUNBUFFERED": "1",
    })
    assert "DATABRICKS_TOKEN" not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out
    assert out["OPENAI_API_KEY"] == "{{secrets/ml-intern/openai}}"
    assert out["PYTHONUNBUFFERED"] == "1"


def test_filter_agent_env_dropped_databricks_prefixed_unless_secret_ref():
    out = djt._filter_agent_env({
        "DATABRICKS_RUNTIME_VERSION": "15.4",  # plaintext DB var → dropped
        "DATABRICKS_PROFILE": "{{secrets/ml-intern/profile}}",  # ref → kept
    })
    assert "DATABRICKS_RUNTIME_VERSION" not in out
    assert out["DATABRICKS_PROFILE"] == "{{secrets/ml-intern/profile}}"


def test_filter_agent_env_handles_none_and_non_string():
    assert djt._filter_agent_env(None) == {}
    out = djt._filter_agent_env({"FOO": 123, 42: "ignored"})
    assert out == {"FOO": "123"}


def test_parse_timeout_units():
    assert djt._parse_timeout("30m") == 1800
    assert djt._parse_timeout("4h") == 14400
    assert djt._parse_timeout("1d") == 86400
    assert djt._parse_timeout("45") == 45  # defaults to seconds
    assert djt._parse_timeout("garbage") == 0
    assert djt._parse_timeout(None) == 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_unknown_op_returns_error():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    result = await tool.execute({"operation": "frobnicate"})
    assert result.get("isError") is True
    assert "Unknown operation" in result["formatted"]


@pytest.mark.asyncio
async def test_execute_missing_op_errors():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    result = await tool.execute({})
    assert result.get("isError") is True


# ---------------------------------------------------------------------------
# Cluster + submit body
# ---------------------------------------------------------------------------


def test_build_cluster_uses_instance_pool_when_set():
    settings = _make_settings(instance_pool_id="pool-xyz")
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=settings)
    spec = tool._build_cluster({"hardware_flavor": "a10g-large"}, env={})
    assert spec["instance_pool_id"] == "pool-xyz"
    assert "node_type_id" not in spec


def test_build_cluster_explicit_node_type_overrides_pool():
    settings = _make_settings(instance_pool_id="pool-xyz")
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=settings)
    spec = tool._build_cluster({"node_type_id": "p4d.24xlarge"}, env={})
    assert spec["node_type_id"] == "p4d.24xlarge"
    assert "instance_pool_id" not in spec


def test_build_cluster_maps_hardware_flavor():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    spec = tool._build_cluster({"hardware_flavor": "a100-large"}, env={})
    assert spec["node_type_id"] == "p4d.24xlarge"


def test_build_cluster_threads_env_into_spark_env_vars():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    spec = tool._build_cluster({}, env={"FOO": "bar"})
    assert spec["spark_env_vars"] == {"FOO": "bar"}


@pytest.mark.asyncio
async def test_build_submit_body_serverless_uses_environment_key():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    body = await tool._build_submit_body(
        {"dependencies": ["transformers==4.45.2"], "timeout": "1h"},
        workspace_path="/Workspace/Users/u/x.py",
        kind="serverless",
    )
    task = body["tasks"][0]
    assert task["environment_key"] == "ml_intern_env"
    assert "new_cluster" not in task
    assert body["environments"][0]["spec"]["dependencies"] == ["transformers==4.45.2"]
    assert body["timeout_seconds"] == 3600


@pytest.mark.asyncio
async def test_build_submit_body_script_attaches_cluster():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    body = await tool._build_submit_body(
        {"hardware_flavor": "a10g-small", "env": {"FOO": "bar"}},
        workspace_path="/Workspace/Users/u/x.py",
        kind="script",
    )
    task = body["tasks"][0]
    assert task["new_cluster"]["node_type_id"] == "g5.xlarge"
    assert task["new_cluster"]["spark_env_vars"] == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# Script staging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_or_stage_script_uploads_inline_to_volume():
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    import base64

    path = await tool._resolve_or_stage_script({"script": "print('hi')"})
    # Scripts stage to /Workspace/.../files/<name> via /api/2.0/workspace/import
    # format=AUTO so spark_python_task can ``open()`` them as plain files. The
    # ``files/`` subdir keeps file uploads from colliding with notebook
    # uploads from the serverless_gpu path under the same session id.
    assert path.startswith("/Workspace/Users/alice@ex.com/ml-intern/")
    assert "/files/" in path
    assert path.endswith("/train.py")
    # Raw API call shape — POST to workspace/import with base64 content.
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    assert posts, "expected workspace/import POST"
    body = posts[0].kwargs["body"]
    assert body["path"] == path[len("/Workspace"):]
    assert body["format"] == "AUTO"
    assert body["overwrite"] is True
    assert base64.b64decode(body["content"]) == b"print('hi')"


@pytest.mark.asyncio
async def test_resolve_or_stage_script_notebook_path_uses_notebooks_subdir():
    """``as_notebook=True`` lands under .../notebooks/<name> so a later FILE
    upload of the same default filename in the same session can't collide
    with the NOTEBOOK asset."""
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    path = await tool._resolve_or_stage_script(
        {"script": "print('hi')"}, as_notebook=True,
    )
    assert path.startswith("/Workspace/Users/alice@ex.com/ml-intern/")
    assert "/notebooks/" in path
    assert path.endswith("/train.py")
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    body = posts[0].kwargs["body"]
    assert body["format"] == "SOURCE"
    assert body["language"] == "PYTHON"


@pytest.mark.asyncio
async def test_notebook_upload_wraps_user_script_with_stdout_capture():
    """Serverless GPU jobs use notebook_task; ``runs/get-output`` only
    surfaces ``dbutils.notebook.exit`` for those, not stdout. The wrapper
    must (a) tee stdout/stderr and (b) call ``dbutils.notebook.exit`` with
    the captured tail when the user didn't already exit."""
    import base64
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    await tool._resolve_or_stage_script(
        {"script": "print('hello')"}, as_notebook=True,
    )
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    decoded = base64.b64decode(posts[0].kwargs["body"]["content"]).decode()
    assert decoded.startswith("# Databricks notebook source\n")
    assert "_ML_INTERN_BUF" in decoded
    assert "_ML_INTERN_TEE" in decoded
    assert "dbutils.notebook.exit(_ML_INTERN_BUF.getvalue()[-4000:])" in decoded
    # User script is preserved inside the wrapper.
    assert "print('hello')" in decoded


@pytest.mark.asyncio
async def test_notebook_upload_preserves_user_explicit_exit():
    """If the user already calls ``dbutils.notebook.exit`` we must not
    append a second exit — that would override their intentional value."""
    import base64
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    user_script = "print('hi')\ndbutils.notebook.exit('user-value')"
    await tool._resolve_or_stage_script(
        {"script": user_script}, as_notebook=True,
    )
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    decoded = base64.b64decode(posts[0].kwargs["body"]["content"]).decode()
    # User's exit kept verbatim.
    assert "dbutils.notebook.exit('user-value')" in decoded
    # Wrapper's auto-exit not added when user already exits.
    assert "_ML_INTERN_BUF.getvalue()[-4000:]" not in decoded


@pytest.mark.asyncio
async def test_upload_workspace_file_clears_notebook_at_same_path():
    """Repro for the type-mismatch InvalidParameterValue: when a NOTEBOOK
    already lives at the target path, ``workspace/import`` with format=AUTO
    rejects with a ``type mismatch``. The tool must delete the prior asset
    before importing."""
    wc = _mock_wc()
    existing = MagicMock()
    existing.object_type = MagicMock()
    existing.object_type.name = "NOTEBOOK"
    wc.workspace.get_status = MagicMock(return_value=existing)
    wc.workspace.delete = MagicMock()

    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "s"

    await tool._upload_workspace_file(
        "/Workspace/Users/alice@ex.com/ml-intern/s/files/train.py",
        "print('x')",
    )
    wc.workspace.delete.assert_called_once()
    deleted_path = wc.workspace.delete.call_args.args[0]
    assert deleted_path == "/Users/alice@ex.com/ml-intern/s/files/train.py"


@pytest.mark.asyncio
async def test_upload_workspace_notebook_clears_file_at_same_path():
    """Mirror of the FILE-collision test for the serverless_gpu (notebook)
    direction."""
    wc = _mock_wc()
    existing = MagicMock()
    existing.object_type = MagicMock()
    existing.object_type.name = "FILE"
    wc.workspace.get_status = MagicMock(return_value=existing)
    wc.workspace.delete = MagicMock()

    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "s"

    await tool._upload_workspace_notebook(
        "/Workspace/Users/alice@ex.com/ml-intern/s/notebooks/train.py",
        "# Databricks notebook source\nprint('x')",
    )
    wc.workspace.delete.assert_called_once()


@pytest.mark.asyncio
async def test_upload_workspace_file_skips_delete_when_type_matches():
    """No delete when the existing asset is already the wanted type — the
    overwrite=True flag on workspace/import handles same-type rewrites."""
    wc = _mock_wc()
    existing = MagicMock()
    existing.object_type = MagicMock()
    existing.object_type.name = "FILE"
    wc.workspace.get_status = MagicMock(return_value=existing)
    wc.workspace.delete = MagicMock()

    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "s"

    await tool._upload_workspace_file(
        "/Workspace/Users/alice@ex.com/ml-intern/s/files/train.py",
        "print('x')",
    )
    wc.workspace.delete.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_or_stage_script_passthrough_workspace_path():
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(wc=wc, settings=_make_settings())
    path = await tool._resolve_or_stage_script(
        {"workspace_path": "/Workspace/existing/file.py"}
    )
    assert path == "/Workspace/existing/file.py"
    wc.workspace.upload.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_or_stage_script_rejects_bad_filename():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    with pytest.raises(ValueError):
        await tool._resolve_or_stage_script({"script": "x", "filename": "../evil.py"})


# ---------------------------------------------------------------------------
# Finetune
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_finetune_posts_to_endpoint_and_defaults_register_to():
    wc = _mock_wc()
    wc.api_client.do.return_value = {
        "name": "ft-abc",
        "mlflow_run_id": "mlrun-1",
    }
    tool = djt.DatabricksJobsTool(wc=wc, settings=_make_settings())
    result = await tool._run_finetune({
        "model": "meta-llama/Llama-3.2-1B",
        "train_data_path": "ml_intern.agent.sft_train",
    })
    assert not result.get("isError")
    method, path = wc.api_client.do.call_args[0][:2]
    assert method == "POST"
    assert path == djt._FINETUNE_API_PATH
    body = wc.api_client.do.call_args[1]["body"]
    assert body["model"] == "meta-llama/Llama-3.2-1B"
    assert body["train_data_path"] == "ml_intern.agent.sft_train"
    assert body["task_type"] == "INSTRUCTION_FINETUNE"
    # default register_to lives under the configured catalog.schema
    assert body["register_to"].startswith("ml_intern.agent.")
    assert body["experiment_path"] == "/Shared/ml-intern"


@pytest.mark.asyncio
async def test_run_finetune_missing_required_returns_error():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    result = await tool._run_finetune({"model": "x"})  # no train_data_path
    assert result.get("isError")
    assert "train_data_path" in result["formatted"]


# ---------------------------------------------------------------------------
# ps / cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ps_renders_table_from_runs_list():
    wc = _mock_wc()
    wc.api_client.do.return_value = {
        "runs": [
            {
                "run_id": 111,
                "run_name": "ml-intern-1",
                "state": {"life_cycle_state": "RUNNING", "result_state": ""},
                "job_id": 42,
            }
        ]
    }
    tool = djt.DatabricksJobsTool(wc=wc, settings=_make_settings())
    result = await tool._ps({})
    assert "111" in result["formatted"]
    assert "RUNNING" in result["formatted"]
    method, path = wc.api_client.do.call_args[0][:2]
    assert method == "GET"
    assert path == djt._JOBS_RUNS_LIST


@pytest.mark.asyncio
async def test_cancel_calls_runs_cancel():
    wc = _mock_wc()
    wc.api_client.do.return_value = {}
    tool = djt.DatabricksJobsTool(wc=wc, settings=_make_settings())
    result = await tool._cancel({"run_id": 999})
    assert not result.get("isError")
    method, path = wc.api_client.do.call_args[0][:2]
    assert method == "POST"
    assert path == djt._JOBS_RUNS_CANCEL
    assert wc.api_client.do.call_args[1]["body"] == {"run_id": 999}


@pytest.mark.asyncio
async def test_cancel_without_run_id_errors():
    tool = djt.DatabricksJobsTool(wc=_mock_wc(), settings=_make_settings())
    result = await tool._cancel({})
    assert result.get("isError")


@pytest.mark.asyncio
async def test_notebook_upload_injects_pre_install_block_by_default():
    """Issue #15: serverless_gpu image lacks datasets/trl/peft/accelerate.
    Every fine-tune job paid one round-trip per missing dep on PTB-smoke
    run-D. The notebook prelude must pip-install the common ML deps
    before the user script runs, no opt-in required."""
    import base64
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    await tool._resolve_or_stage_script(
        {"script": "import datasets"}, as_notebook=True,
    )
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    decoded = base64.b64decode(posts[0].kwargs["body"]["content"]).decode()
    # Prelude markers + every pinned spec must appear.
    assert "_ml_subprocess" in decoded
    for spec in djt._SERVERLESS_GPU_PRE_INSTALL:
        assert spec in decoded


@pytest.mark.asyncio
async def test_notebook_upload_pre_install_opt_out():
    """``pre_install=False`` keeps the env pristine. Useful for users who
    pin their own deps or want to avoid the install overhead per cold
    start."""
    import base64
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    await tool._resolve_or_stage_script(
        {"script": "import datasets", "pre_install": False}, as_notebook=True,
    )
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    decoded = base64.b64decode(posts[0].kwargs["body"]["content"]).decode()
    assert "_ml_subprocess" not in decoded
    for spec in djt._SERVERLESS_GPU_PRE_INSTALL:
        assert spec not in decoded
    # User script still present.
    assert "import datasets" in decoded


@pytest.mark.asyncio
async def test_pre_install_only_applies_to_notebook_path():
    """Non-notebook script kinds (script / serverless CPU) land via the
    plain workspace file path — no Databricks notebook source header, no
    stdout wrapper, no pre-install prelude. Adding one here would risk
    breaking spark_python_task by injecting top-level subprocess calls."""
    import base64
    wc = _mock_wc()
    tool = djt.DatabricksJobsTool(
        wc=wc, settings=_make_settings(), user_email="alice@ex.com",
    )
    tool.session = MagicMock()
    tool.session.session_id = "abcd-1234-5678-9012"

    await tool._resolve_or_stage_script(
        {"script": "print('cpu run')"}, as_notebook=False,
    )
    posts = [c for c in wc.api_client.do.call_args_list
             if c.args[0] == "POST" and c.args[1] == "/api/2.0/workspace/import"]
    decoded = base64.b64decode(posts[0].kwargs["body"]["content"]).decode()
    assert "_ml_subprocess" not in decoded
