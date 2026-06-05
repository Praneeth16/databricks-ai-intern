"""Hugging Face → Unity Catalog ingestion tool.

Pulls a HF dataset / model / single file into a UC Volume so subsequent
training jobs read from durable, governed storage instead of re-downloading
from the Hub on every run. Optionally registers ingested datasets as Delta
tables (CTAS over the parquet/json files).

Three modes:

- ``ingest_dataset`` — full dataset repo. Default: snapshot ``data/``,
  ``*.parquet``, ``*.jsonl``, ``*.json``. ``create_table=True`` runs CTAS
  into ``<catalog>.<schema>.<table>`` over the parquet/json files (Delta).
- ``ingest_model``   — model repo (weights / tokenizer / config) into a
  Volume subdir, suitable as ``custom_weights_path`` for fine-tune runs.
- ``ingest_file``    — single file via ``hf_hub_download``.

Implementation: ``huggingface_hub.snapshot_download`` writes to a temp
directory, then each file is streamed to ``wc.files.upload``. A size guard
(``max_size_gb``, default 10) aborts before upload if the snapshot exceeds
the limit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.core import db_client
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _safe_repo_segment(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _detect_format(filenames: List[str]) -> Optional[str]:
    ext_count: Dict[str, int] = {}
    for f in filenames:
        ext = Path(f).suffix.lower()
        if ext:
            ext_count[ext] = ext_count.get(ext, 0) + 1
    if ext_count.get(".parquet"):
        return "parquet"
    if ext_count.get(".jsonl") or ext_count.get(".json"):
        return "json"
    if ext_count.get(".csv"):
        return "csv"
    return None


class HfToUcTool:
    def __init__(self, wc, settings: db_client.DatabricksSettings, hf_token: Optional[str] = None):
        self.wc = wc
        self.settings = settings
        self.hf_token = hf_token

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        op = (args.get("operation") or "").lower().strip()
        try:
            if op == "ingest_dataset":
                return await self._ingest(args, repo_type="dataset")
            if op == "ingest_model":
                return await self._ingest(args, repo_type="model")
            if op == "ingest_file":
                return await self._ingest_file(args)
            return _err(f"Unknown operation {op!r}.")
        except Exception as e:
            logger.exception("hf_to_uc %s failed", op)
            return _err(f"{op} failed: {e}")

    async def _ingest(self, args: Dict[str, Any], repo_type: str) -> ToolResult:
        repo_id = args.get("repo_id")
        if not repo_id:
            return _err("repo_id is required (e.g. 'tatsu-lab/alpaca')")

        max_size_gb = float(args.get("max_size_gb", 10))
        allow_patterns = args.get("allow_patterns") or (
            ["data/*", "*.parquet", "*.jsonl", "*.json"] if repo_type == "dataset"
            else None  # model: take everything by default
        )
        revision = args.get("revision")

        dest = args.get("destination_path") or (
            f"{self.settings.volume_root}/hf/{repo_type}s/{_safe_repo_segment(repo_id)}"
        )
        if not dest.startswith("/Volumes/"):
            return _err(f"destination_path must be under /Volumes/ (got {dest!r})")

        # Snapshot to temp dir.
        from huggingface_hub import snapshot_download

        def _snapshot():
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=self.hf_token,
                allow_patterns=allow_patterns,
                local_dir=tempfile.mkdtemp(prefix="ml_intern_hf_"),
            )

        local_dir = await asyncio.to_thread(_snapshot)
        try:
            # Size guard.
            files = [p for p in Path(local_dir).rglob("*") if p.is_file()]
            total_bytes = sum(p.stat().st_size for p in files)
            limit = int(max_size_gb * (1 << 30))
            if total_bytes > limit:
                return _err(
                    f"Snapshot is {total_bytes / (1<<30):.2f} GB, exceeds max_size_gb={max_size_gb}. "
                    "Tighten allow_patterns or raise the limit."
                )

            # Upload each file relative to local_dir → dest.
            uploaded: List[str] = []
            for p in files:
                rel = p.relative_to(local_dir).as_posix()
                target = f"{dest.rstrip('/')}/{rel}"
                await self._upload_file(p, target)
                uploaded.append(target)

            msg_lines = [
                f"**Ingested {repo_id} ({repo_type}) → {dest}**",
                f"Files: {len(uploaded)}",
                f"Total size: {total_bytes / (1<<20):.2f} MB",
            ]

            if repo_type == "dataset" and args.get("create_table"):
                table = args.get("table_name") or (
                    f"{self.settings.full_schema}.{_safe_repo_segment(repo_id)}"
                )
                fmt = _detect_format([p.name for p in files])
                if not fmt:
                    msg_lines.append(
                        "create_table skipped: no parquet/json/csv files detected."
                    )
                else:
                    sql = (
                        f"CREATE OR REPLACE TABLE {table} AS "
                        f"SELECT * FROM read_files('{dest}/', format => '{fmt}')"
                    )
                    await self._run_sql(sql)
                    msg_lines.append(f"**Registered table:** {table} (format={fmt})")

            return _ok("\n".join(msg_lines), n=len(uploaded))
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)

    async def _ingest_file(self, args: Dict[str, Any]) -> ToolResult:
        repo_id = args.get("repo_id")
        filename = args.get("filename")
        if not repo_id or not filename:
            return _err("repo_id and filename are required")
        repo_type = args.get("repo_type", "dataset")
        revision = args.get("revision")

        dest = args.get("destination_path") or (
            f"{self.settings.volume_root}/hf/{repo_type}s/{_safe_repo_segment(repo_id)}/{filename}"
        )
        if not dest.startswith("/Volumes/"):
            return _err(f"destination_path must be under /Volumes/ (got {dest!r})")

        from huggingface_hub import hf_hub_download

        def _download():
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                revision=revision,
                token=self.hf_token,
            )

        local = await asyncio.to_thread(_download)
        await self._upload_file(Path(local), dest)
        size = Path(local).stat().st_size
        return _ok(
            f"**Ingested {repo_id}/{filename} → {dest}**\nSize: {size / (1<<20):.2f} MB",
            n=1,
        )

    async def _upload_file(self, local: Path, target: str) -> None:
        def _do():
            with open(local, "rb") as f:
                self.wc.files.upload(file_path=target, contents=f, overwrite=True)
        await asyncio.to_thread(_do)

    async def _run_sql(self, sql: str) -> None:
        def _do():
            conn = db_client.get_sql_connection(self.settings)
            try:
                cur = conn.cursor()
                cur.execute(sql)
            finally:
                conn.close()
        await asyncio.to_thread(_do)


HF_TO_UC_TOOL_SPEC = {
    "name": "hf_to_uc",
    "description": (
        "Ingest a Hugging Face dataset, model, or single file into a Unity Catalog Volume so jobs "
        "read from durable, governed storage. Optionally registers ingested datasets as Delta tables.\n\n"
        "Operations:\n"
        "- ingest_dataset: pull a HF dataset repo into /Volumes/<cat>/<schema>/<vol>/hf/datasets/<repo>/. "
        "Set create_table=true to also CTAS into <catalog>.<schema>.<table>.\n"
        "- ingest_model: pull a model repo (weights/tokenizer) — useful as custom_weights_path for finetune.\n"
        "- ingest_file: single file via hf_hub_download.\n\n"
        "Always validate dataset shape with uc_inspect_dataset after ingest. "
        "Size guard via max_size_gb (default 10) — tighten allow_patterns for big repos.\n\n"
        "Examples:\n"
        "{\"operation\":\"ingest_dataset\",\"repo_id\":\"tatsu-lab/alpaca\",\"create_table\":true}\n"
        "{\"operation\":\"ingest_model\",\"repo_id\":\"meta-llama/Llama-3.2-1B\"}\n"
        "{\"operation\":\"ingest_file\",\"repo_id\":\"openai/gsm8k\",\"filename\":\"main/test-00000-of-00001.parquet\"}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["ingest_dataset", "ingest_model", "ingest_file"]},
            "repo_id": {"type": "string", "description": "HF repo id, e.g. 'tatsu-lab/alpaca'."},
            "repo_type": {"type": "string", "enum": ["dataset", "model"], "description": "(ingest_file)"},
            "filename": {"type": "string", "description": "(ingest_file) Path within the repo."},
            "revision": {"type": "string", "description": "Commit / branch / tag."},
            "destination_path": {
                "type": "string",
                "description": "Override destination under /Volumes/. Default derived from repo_id.",
            },
            "allow_patterns": {
                "type": "array", "items": {"type": "string"},
                "description": "Glob filter (snapshot_download). Default for datasets: data/*, *.parquet, *.jsonl, *.json.",
            },
            "max_size_gb": {"type": "number", "description": "Abort if snapshot exceeds. Default 10."},
            "create_table": {
                "type": "boolean",
                "description": "(ingest_dataset) Also CTAS into <catalog>.<schema>.<table>.",
            },
            "table_name": {
                "type": "string",
                "description": "(ingest_dataset, create_table) Override target. Default derived from repo_id.",
            },
        },
        "required": ["operation", "repo_id"],
    },
}


async def hf_to_uc_handler(arguments: Dict[str, Any], session: Any = None,
                           tool_call_id: str | None = None) -> tuple[str, bool]:
    try:
        cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
        settings = db_client.resolve_settings(cfg)
        token = getattr(session, "databricks_user_token", None) if session else None
        if token and settings.host:
            wc = db_client.get_workspace_client_for_user(token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)
        hf_token = (
            (getattr(session, "hf_token", None) if session else None)
            or os.environ.get("HF_TOKEN")
        )
        tool = HfToUcTool(wc=wc, settings=settings, hf_token=hf_token)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("hf_to_uc handler crashed")
        return f"Error: {e}", False


def _load_default_config():
    from agent.config import load_config
    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)
