"""Unity Catalog Volume tool.

Read/write/list paths under ``/Volumes/<catalog>/<schema>/<volume>/...`` via
the Files API (``wc.files``). Volumes are the durable storage tier for
training data, model artifacts, and any file the agent stages outside
Workspace Files.

The agent must reference paths under the configured ``volume_root``. Other
``/Volumes/...`` paths are accepted (an explicit override) but logged.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any, Dict, Optional

from agent.core import db_client
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _validate_path(path: str) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return "path is required"
    if not path.startswith("/Volumes/"):
        return f"path must start with /Volumes/ (got {path!r})"
    if ".." in path.split("/"):
        return "path must not contain '..' segments"
    return None


class UCVolumeTool:
    def __init__(self, wc, settings: db_client.DatabricksSettings):
        self.wc = wc
        self.settings = settings

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        op = (args.get("operation") or "").lower().strip()
        try:
            if op == "ls":
                return await self._ls(args)
            if op == "read":
                return await self._read(args)
            if op == "write":
                return await self._write(args)
            if op == "rm":
                return await self._rm(args)
            if op == "mkdir":
                return await self._mkdir(args)
            return _err(f"Unknown operation {op!r}. Use ls | read | write | rm | mkdir.")
        except Exception as e:
            logger.exception("uc_volume %s failed", op)
            return _err(f"{op} failed: {e}")

    async def _ls(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path") or self.settings.volume_root
        err = _validate_path(path)
        if err:
            return _err(err)
        items = await asyncio.to_thread(
            lambda: list(self.wc.files.list_directory_contents(directory_path=path))
        )
        if not items:
            return _ok(f"(empty) {path}")
        rows = []
        for it in items:
            kind = "DIR" if getattr(it, "is_directory", False) else "FILE"
            size = getattr(it, "file_size", None)
            size_s = f"{size}" if size is not None else "-"
            rows.append(f"| {kind} | {it.path} | {size_s} |")
        body = "\n".join(rows)
        return _ok(
            f"**Listing {path}** ({len(items)} entries):\n\n| KIND | PATH | SIZE |\n|---|---|---|\n{body}",
            n=len(items),
        )

    async def _read(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        err = _validate_path(path)
        if err:
            return _err(err)
        max_bytes = int(args.get("max_bytes", 64 * 1024))

        def _do():
            resp = self.wc.files.download(file_path=path)
            stream = resp.contents
            data = stream.read(max_bytes + 1)
            stream.close()
            return data

        data = await asyncio.to_thread(_do)
        truncated = len(data) > max_bytes
        data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
            preview = text
        except UnicodeDecodeError:
            preview = f"(binary, {len(data)} bytes shown — base64)\n{base64.b64encode(data).decode()}"
        suffix = "\n\n[truncated]" if truncated else ""
        return _ok(f"**{path}**\n\n```\n{preview}{suffix}\n```")

    async def _write(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        err = _validate_path(path)
        if err:
            return _err(err)
        content = args.get("content")
        if content is None:
            return _err("content is required (string)")
        if not isinstance(content, (str, bytes)):
            content = str(content)
        data = content.encode("utf-8") if isinstance(content, str) else content
        overwrite = bool(args.get("overwrite", True))
        import io
        await asyncio.to_thread(
            self.wc.files.upload,
            file_path=path, contents=io.BytesIO(data), overwrite=overwrite,
        )
        return _ok(f"Wrote {len(data)} bytes to {path}.")

    async def _rm(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        err = _validate_path(path)
        if err:
            return _err(err)
        await asyncio.to_thread(self.wc.files.delete, file_path=path)
        return _ok(f"Deleted {path}.")

    async def _mkdir(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        err = _validate_path(path)
        if err:
            return _err(err)
        await asyncio.to_thread(self.wc.files.create_directory, directory_path=path)
        return _ok(f"Created {path}.")


UC_VOLUME_TOOL_SPEC = {
    "name": "uc_volume",
    "description": (
        "Read, write, list, and delete files in Unity Catalog Volumes. Volumes are the durable "
        "storage tier under /Volumes/<catalog>/<schema>/<volume>/... — use this for training data, "
        "checkpoints, and any artifact you want to persist beyond a single job run.\n\n"
        "All paths must start with /Volumes/. The default volume for this agent is configured via "
        "ML_INTERN_UC_CATALOG/SCHEMA/VOLUME.\n\n"
        "Operations:\n"
        "- ls: list directory contents\n"
        "- read: download file (up to max_bytes, default 64KB)\n"
        "- write: upload string content\n"
        "- rm: delete file\n"
        "- mkdir: create directory"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["ls", "read", "write", "rm", "mkdir"]},
            "path": {"type": "string", "description": "Volume path. Must start with /Volumes/."},
            "content": {"type": "string", "description": "(write) String content to upload."},
            "overwrite": {"type": "boolean", "description": "(write) Default true."},
            "max_bytes": {"type": "integer", "description": "(read) Max bytes returned. Default 65536."},
        },
        "required": ["operation"],
    },
}


async def uc_volume_handler(arguments: Dict[str, Any], session: Any = None,
                            tool_call_id: str | None = None) -> tuple[str, bool]:
    try:
        cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
        settings = db_client.resolve_settings(cfg)
        token = getattr(session, "databricks_user_token", None) if session else None
        if token and settings.host:
            wc = db_client.get_workspace_client_for_user(token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)
        tool = UCVolumeTool(wc=wc, settings=settings)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("uc_volume handler crashed")
        return f"Error: {e}", False


def _load_default_config():
    from agent.config import load_config
    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)
