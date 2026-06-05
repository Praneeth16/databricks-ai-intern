"""Unit tests for UC volume / dataset / model / hf_to_uc / repos tools.

All exercise the tool's dispatch logic + REST/SDK call shape using mocked
WorkspaceClients. No live network or warehouse traffic.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.core import db_client
from agent.tools import (
    hf_to_uc_tool,
    repos_tool,
    uc_dataset_tools,
    uc_model_tools,
    uc_volume_tools,
)


def _settings(**o):
    d = dict(
        host="https://ws", warehouse_id="wh1",
        experiment_path="/Shared/ml-intern",
        uc_catalog="ml_intern", uc_schema="agent", uc_volume="scratch",
        secret_scope="ml-intern", lakebase_instance=None, instance_pool_id=None,
        default_node_type_id="g5.xlarge",
        default_runtime_version="15.4.x-gpu-ml-scala2.12",
        prompt_registry_name="ml_intern.agent.system_prompt",
    )
    d.update(o)
    return db_client.DatabricksSettings(**d)


# ---------------------------------------------------------------------------
# uc_volume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uc_volume_ls_renders_listing():
    wc = MagicMock()
    item = MagicMock()
    item.path = "/Volumes/ml_intern/agent/scratch/foo.parquet"
    item.is_directory = False
    item.file_size = 1024
    wc.files.list_directory_contents.return_value = iter([item])
    tool = uc_volume_tools.UCVolumeTool(wc=wc, settings=_settings())
    out = await tool.execute({"operation": "ls", "path": "/Volumes/ml_intern/agent/scratch"})
    assert "foo.parquet" in out["formatted"]
    assert "1024" in out["formatted"]


@pytest.mark.asyncio
async def test_uc_volume_rejects_non_volume_path():
    tool = uc_volume_tools.UCVolumeTool(wc=MagicMock(), settings=_settings())
    out = await tool.execute({"operation": "ls", "path": "/etc/passwd"})
    assert out.get("isError")
    assert "/Volumes/" in out["formatted"]


@pytest.mark.asyncio
async def test_uc_volume_write_uploads_bytes():
    wc = MagicMock()
    tool = uc_volume_tools.UCVolumeTool(wc=wc, settings=_settings())
    out = await tool.execute({
        "operation": "write",
        "path": "/Volumes/ml_intern/agent/scratch/x.txt",
        "content": "hello",
    })
    assert not out.get("isError")
    kwargs = wc.files.upload.call_args[1]
    assert kwargs["file_path"] == "/Volumes/ml_intern/agent/scratch/x.txt"
    assert kwargs["overwrite"] is True
    body = kwargs["contents"].read()
    assert body == b"hello"


@pytest.mark.asyncio
async def test_uc_volume_read_decodes_utf8():
    wc = MagicMock()
    resp = MagicMock()
    stream = io.BytesIO(b"hello world")
    resp.contents = stream
    wc.files.download.return_value = resp
    tool = uc_volume_tools.UCVolumeTool(wc=wc, settings=_settings())
    out = await tool.execute({
        "operation": "read",
        "path": "/Volumes/ml_intern/agent/scratch/x.txt",
    })
    assert "hello world" in out["formatted"]


@pytest.mark.asyncio
async def test_uc_volume_rm_calls_files_delete():
    wc = MagicMock()
    tool = uc_volume_tools.UCVolumeTool(wc=wc, settings=_settings())
    out = await tool.execute({
        "operation": "rm",
        "path": "/Volumes/ml_intern/agent/scratch/x.txt",
    })
    assert not out.get("isError")
    wc.files.delete.assert_called_once_with(file_path="/Volumes/ml_intern/agent/scratch/x.txt")


# ---------------------------------------------------------------------------
# uc_dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uc_dataset_query_rejects_writes():
    tool = uc_dataset_tools.UCDatasetTool(wc=MagicMock(), settings=_settings())
    out = await tool.execute({"operation": "query", "sql": "DROP TABLE foo"})
    assert out.get("isError")


@pytest.mark.asyncio
async def test_uc_dataset_describe_runs_describe_sql():
    tool = uc_dataset_tools.UCDatasetTool(wc=MagicMock(), settings=_settings())
    seen = []

    async def fake_exec(sql):
        seen.append(sql)
        if sql.startswith("DESCRIBE"):
            return ["col_name", "data_type"], [("id", "int"), ("text", "string")]
        return ["n"], [(42,)]

    tool._execute_sql = fake_exec  # type: ignore
    out = await tool.execute({
        "operation": "describe",
        "table": "ml_intern.agent.alpaca",
    })
    assert "alpaca" in out["formatted"]
    assert "42 rows" in out["formatted"]
    assert any(s.startswith("DESCRIBE") for s in seen)


@pytest.mark.asyncio
async def test_uc_dataset_validates_table_name():
    tool = uc_dataset_tools.UCDatasetTool(wc=MagicMock(), settings=_settings())
    out = await tool.execute({"operation": "describe", "table": "not_three_levels"})
    assert out.get("isError")


# ---------------------------------------------------------------------------
# uc_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uc_model_list_renders_table():
    wc = MagicMock()
    m = MagicMock()
    m.full_name = "ml_intern.agent.llama_sft_v1"
    m.comment = "first try"
    m.updated_at = "2026-04-25"
    wc.registered_models.list.return_value = iter([m])
    tool = uc_model_tools.UCModelTool(wc=wc, settings=_settings())
    out = await tool.execute({"operation": "list"})
    assert "llama_sft_v1" in out["formatted"]


@pytest.mark.asyncio
async def test_uc_model_set_alias_calls_sdk():
    wc = MagicMock()
    tool = uc_model_tools.UCModelTool(wc=wc, settings=_settings())
    out = await tool.execute({
        "operation": "set_alias",
        "full_name": "ml_intern.agent.llama_sft_v1",
        "alias": "champion",
        "version": 3,
    })
    assert not out.get("isError")
    wc.registered_models.set_alias.assert_called_once_with(
        full_name="ml_intern.agent.llama_sft_v1", alias="champion", version_num=3,
    )


@pytest.mark.asyncio
async def test_uc_model_validates_name():
    tool = uc_model_tools.UCModelTool(wc=MagicMock(), settings=_settings())
    out = await tool.execute({"operation": "inspect", "full_name": "bad_name"})
    assert out.get("isError")


# ---------------------------------------------------------------------------
# hf_to_uc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hf_to_uc_ingest_dataset_uploads_each_file_and_creates_table(tmp_path):
    # Stub snapshot_download to write two parquet files into a tmp dir.
    src = tmp_path / "snap"
    src.mkdir()
    (src / "data").mkdir()
    (src / "data" / "train-0.parquet").write_bytes(b"PAR1\x00\x01")
    (src / "data" / "train-1.parquet").write_bytes(b"PAR1\x00\x02")

    wc = MagicMock()
    tool = hf_to_uc_tool.HfToUcTool(wc=wc, settings=_settings(), hf_token=None)

    sql_seen = []

    async def fake_run_sql(sql):
        sql_seen.append(sql)

    tool._run_sql = fake_run_sql  # type: ignore

    with patch.object(hf_to_uc_tool, "snapshot_download", create=True, return_value=str(src)), \
         patch("huggingface_hub.snapshot_download", return_value=str(src)):
        out = await tool.execute({
            "operation": "ingest_dataset",
            "repo_id": "tatsu-lab/alpaca",
            "create_table": True,
        })
    assert not out.get("isError")
    # Two files should have been uploaded.
    assert wc.files.upload.call_count == 2
    target_paths = [c.kwargs["file_path"] for c in wc.files.upload.call_args_list]
    assert all(p.startswith("/Volumes/ml_intern/agent/scratch/hf/datasets/") for p in target_paths)
    # CTAS issued.
    assert sql_seen and "CREATE OR REPLACE TABLE" in sql_seen[0]
    assert "format => 'parquet'" in sql_seen[0]


@pytest.mark.asyncio
async def test_hf_to_uc_aborts_on_size_limit(tmp_path):
    src = tmp_path / "huge"
    src.mkdir()
    (src / "big.bin").write_bytes(b"x" * 1024)
    wc = MagicMock()
    tool = hf_to_uc_tool.HfToUcTool(wc=wc, settings=_settings())
    with patch("huggingface_hub.snapshot_download", return_value=str(src)):
        out = await tool.execute({
            "operation": "ingest_dataset",
            "repo_id": "x/y",
            "max_size_gb": 1e-9,  # ~1 byte
        })
    assert out.get("isError")
    assert "exceeds max_size_gb" in out["formatted"]
    wc.files.upload.assert_not_called()


@pytest.mark.asyncio
async def test_hf_to_uc_ingest_file_uploads_single():
    wc = MagicMock()
    tool = hf_to_uc_tool.HfToUcTool(wc=wc, settings=_settings())
    fake_local = "/tmp/fake_download.parquet"
    with patch("huggingface_hub.hf_hub_download", return_value=fake_local), \
         patch.object(Path, "stat") as stat_mock, \
         patch("builtins.open", create=True) as open_mock:
        stat_mock.return_value.st_size = 12345
        open_mock.return_value.__enter__.return_value = MagicMock()
        out = await tool.execute({
            "operation": "ingest_file",
            "repo_id": "openai/gsm8k",
            "filename": "main/test.parquet",
        })
    assert not out.get("isError"), out
    wc.files.upload.assert_called_once()
    target = wc.files.upload.call_args.kwargs["file_path"]
    assert target.startswith("/Volumes/ml_intern/agent/scratch/hf/datasets/openai__gsm8k/")
    assert target.endswith("main/test.parquet")


# ---------------------------------------------------------------------------
# repos
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repos_clone_infers_provider_and_calls_create():
    wc = MagicMock()
    repo = MagicMock(id=99, path="/Workspace/Users/u/repos/x", branch="main")
    repo.head_commit_id = "abcdef"
    wc.repos.create.return_value = repo
    tool = repos_tool.ReposTool(wc=wc, settings=_settings(), user_email="alice@ex.com")
    out = await tool.execute({
        "operation": "clone",
        "url": "https://github.com/foo/bar",
    })
    assert not out.get("isError")
    kw = wc.repos.create.call_args.kwargs
    assert kw["url"] == "https://github.com/foo/bar"
    assert kw["provider"] == "gitHub"
    assert kw["path"] == "/Workspace/Users/alice@ex.com/repos/bar"


@pytest.mark.asyncio
async def test_repos_clone_unknown_host_requires_provider():
    tool = repos_tool.ReposTool(wc=MagicMock(), settings=_settings(), user_email="u@x")
    out = await tool.execute({
        "operation": "clone",
        "url": "https://git.internal.corp/team/repo",
    })
    assert out.get("isError")
    assert "provider" in out["formatted"]


@pytest.mark.asyncio
async def test_repos_pull_calls_update():
    wc = MagicMock()
    tool = repos_tool.ReposTool(wc=wc, settings=_settings())
    out = await tool.execute({"operation": "pull", "repo_id": 12, "branch": "main"})
    assert not out.get("isError")
    wc.repos.update.assert_called_once_with(repo_id=12, branch="main")


@pytest.mark.asyncio
async def test_repos_delete_calls_delete():
    wc = MagicMock()
    tool = repos_tool.ReposTool(wc=wc, settings=_settings())
    out = await tool.execute({"operation": "delete", "repo_id": 7})
    assert not out.get("isError")
    wc.repos.delete.assert_called_once_with(repo_id=7)
