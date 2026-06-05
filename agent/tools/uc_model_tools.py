"""Unity Catalog Registered Models tool.

Lists, inspects, and aliases UC registered models. Registration itself
happens inside training scripts via ``mlflow.<framework>.log_model(...,
registered_model_name="<catalog>.<schema>.<name>")`` — the agent surfaces
what's already there so it can reference versions and roll aliases.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional

from agent.core import db_client
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _validate_model_name(name: str) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return "model name is required"
    if not re.match(r"^[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*$", name):
        return f"model must be fully-qualified <catalog>.<schema>.<name> (got {name!r})"
    return None


class UCModelTool:
    def __init__(self, wc, settings: db_client.DatabricksSettings):
        self.wc = wc
        self.settings = settings

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        op = (args.get("operation") or "").lower().strip()
        try:
            if op == "list":
                return await self._list(args)
            if op == "inspect":
                return await self._inspect(args)
            if op == "list_versions":
                return await self._list_versions(args)
            if op == "set_alias":
                return await self._set_alias(args)
            if op == "delete_alias":
                return await self._delete_alias(args)
            return _err(f"Unknown operation {op!r}.")
        except Exception as e:
            logger.exception("uc_model %s failed", op)
            return _err(f"{op} failed: {e}")

    async def _list(self, args: Dict[str, Any]) -> ToolResult:
        catalog = args.get("catalog") or self.settings.uc_catalog
        schema = args.get("schema") or self.settings.uc_schema
        items = await asyncio.to_thread(
            lambda: list(self.wc.registered_models.list(
                catalog_name=catalog, schema_name=schema, max_results=100,
            ))
        )
        if not items:
            return _ok(f"No registered models in {catalog}.{schema}.")
        rows = []
        for m in items:
            rows.append(f"| {m.full_name} | {getattr(m, 'comment', '') or ''} | {getattr(m, 'updated_at', '') or ''} |")
        body = "\n".join(rows)
        return _ok(
            f"**Registered models in {catalog}.{schema}** ({len(items)}):\n\n| FULL NAME | COMMENT | UPDATED |\n|---|---|---|\n{body}",
            n=len(items),
        )

    async def _inspect(self, args: Dict[str, Any]) -> ToolResult:
        full_name = args.get("full_name")
        err = _validate_model_name(full_name)
        if err:
            return _err(err)
        m = await asyncio.to_thread(self.wc.registered_models.get, full_name=full_name)
        out = {
            "full_name": m.full_name,
            "comment": getattr(m, "comment", None),
            "owner": getattr(m, "owner", None),
            "created_at": getattr(m, "created_at", None),
            "updated_at": getattr(m, "updated_at", None),
            "aliases": [
                {"alias": a.alias_name, "version": a.version_num}
                for a in (getattr(m, "aliases", None) or [])
            ],
        }
        import json
        return _ok(f"**{full_name}**\n\n```json\n{json.dumps(out, default=str, indent=2)}\n```")

    async def _list_versions(self, args: Dict[str, Any]) -> ToolResult:
        full_name = args.get("full_name")
        err = _validate_model_name(full_name)
        if err:
            return _err(err)
        items = await asyncio.to_thread(
            lambda: list(self.wc.model_versions.list(full_name=full_name, max_results=100))
        )
        if not items:
            return _ok(f"No versions for {full_name}.")
        rows = []
        for v in items:
            rows.append(
                f"| {v.version} | {getattr(v, 'status', '') or ''} | "
                f"{getattr(v, 'run_id', '') or ''} | {getattr(v, 'created_at', '') or ''} |"
            )
        body = "\n".join(rows)
        return _ok(
            f"**Versions of {full_name}** ({len(items)}):\n\n"
            f"| VERSION | STATUS | RUN ID | CREATED |\n|---|---|---|---|\n{body}",
            n=len(items),
        )

    async def _set_alias(self, args: Dict[str, Any]) -> ToolResult:
        full_name = args.get("full_name")
        err = _validate_model_name(full_name)
        if err:
            return _err(err)
        alias = args.get("alias")
        version = args.get("version")
        if not alias or version is None:
            return _err("alias and version are required")
        await asyncio.to_thread(
            self.wc.registered_models.set_alias,
            full_name=full_name, alias=str(alias), version_num=int(version),
        )
        return _ok(f"Set alias {alias!r} → version {version} on {full_name}.")

    async def _delete_alias(self, args: Dict[str, Any]) -> ToolResult:
        full_name = args.get("full_name")
        err = _validate_model_name(full_name)
        if err:
            return _err(err)
        alias = args.get("alias")
        if not alias:
            return _err("alias is required")
        await asyncio.to_thread(
            self.wc.registered_models.delete_alias,
            full_name=full_name, alias=str(alias),
        )
        return _ok(f"Deleted alias {alias!r} from {full_name}.")


UC_MODEL_TOOL_SPEC = {
    "name": "uc_model",
    "description": (
        "Manage Unity Catalog registered models. Models are referenced as three-level names "
        "<catalog>.<schema>.<name> (e.g. ml_intern.agent.llama_sft_v1).\n\n"
        "Operations:\n"
        "- list: enumerate models in a schema\n"
        "- inspect: get model metadata + aliases\n"
        "- list_versions: per-version status, run id, created\n"
        "- set_alias: bind alias (e.g. 'champion', 'staging') to a version\n"
        "- delete_alias: remove alias\n\n"
        "Registration itself happens inside training scripts via "
        "mlflow.<framework>.log_model(..., registered_model_name='<catalog>.<schema>.<name>')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": [
                "list", "inspect", "list_versions", "set_alias", "delete_alias",
            ]},
            "catalog": {"type": "string"},
            "schema": {"type": "string"},
            "full_name": {"type": "string", "description": "<catalog>.<schema>.<name>"},
            "alias": {"type": "string"},
            "version": {"type": ["integer", "string"]},
        },
        "required": ["operation"],
    },
}


async def uc_model_handler(arguments: Dict[str, Any], session: Any = None,
                           tool_call_id: str | None = None) -> tuple[str, bool]:
    try:
        cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
        settings = db_client.resolve_settings(cfg)
        token = getattr(session, "databricks_user_token", None) if session else None
        if token and settings.host:
            wc = db_client.get_workspace_client_for_user(token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)
        tool = UCModelTool(wc=wc, settings=settings)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("uc_model handler crashed")
        return f"Error: {e}", False


def _load_default_config():
    from agent.config import load_config
    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)
