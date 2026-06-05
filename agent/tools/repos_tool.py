"""Databricks Git Folders (Repos) tool.

Clone arbitrary git repos into the workspace as Git Folders, list/pull/delete
them. The agent uses this to bring reference implementations into the same
workspace as training jobs (so a training script can ``%pip install -e
/Workspace/Repos/<u>/<repo>`` or import directly).

For private repos, callers must have linked their Git provider credentials
to Databricks Repos (Settings → Git Integration). The tool does not accept
PATs from agent-supplied env.
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


_PROVIDER_BY_HOST = {
    "github.com": "gitHub",
    "gitlab.com": "gitLab",
    "bitbucket.org": "bitbucketCloud",
    "dev.azure.com": "azureDevOpsServices",
}


def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _infer_provider(url: str) -> Optional[str]:
    m = re.match(r"https?://([^/]+)/", url)
    if not m:
        return None
    host = m.group(1).lower()
    return _PROVIDER_BY_HOST.get(host)


def _repo_name(url: str) -> str:
    base = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.git$", "", base) or "repo"


def _safe_segment(s: str | None, default: str = "user") -> str:
    if not s:
        return default
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", s)


class ReposTool:
    def __init__(self, wc, settings: db_client.DatabricksSettings, user_email: Optional[str] = None):
        self.wc = wc
        self.settings = settings
        self.user_email = user_email

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        op = (args.get("operation") or "").lower().strip()
        try:
            if op == "clone":
                return await self._clone(args)
            if op == "list":
                return await self._list(args)
            if op == "inspect":
                return await self._inspect(args)
            if op == "pull":
                return await self._pull(args)
            if op == "delete":
                return await self._delete(args)
            return _err(f"Unknown operation {op!r}.")
        except Exception as e:
            logger.exception("repos %s failed", op)
            return _err(f"{op} failed: {e}")

    async def _clone(self, args: Dict[str, Any]) -> ToolResult:
        url = args.get("url")
        if not url:
            return _err("url is required (e.g. https://github.com/org/repo)")
        provider = args.get("provider") or _infer_provider(url)
        if not provider:
            return _err(
                f"Could not infer provider from {url!r}. "
                f"Supply provider in {sorted(_PROVIDER_BY_HOST.values())}."
            )
        name = args.get("name") or _repo_name(url)
        user_seg = _safe_segment(self.user_email)
        path = args.get("path") or f"/Workspace/Users/{user_seg}/repos/{name}"
        branch = args.get("branch")

        kwargs: Dict[str, Any] = {"url": url, "provider": provider, "path": path}
        if branch:
            kwargs["branch"] = branch

        repo = await asyncio.to_thread(self.wc.repos.create, **kwargs)
        return _ok(
            f"**Cloned {url} → {repo.path}**\n"
            f"- repo_id: {repo.id}\n"
            f"- branch: {repo.branch}\n"
            f"- head_commit: {getattr(repo, 'head_commit_id', '?')}\n",
            n=1,
        )

    async def _list(self, args: Dict[str, Any]) -> ToolResult:
        prefix = args.get("path_prefix")
        kwargs: Dict[str, Any] = {}
        if prefix:
            kwargs["path_prefix"] = prefix
        items = await asyncio.to_thread(lambda: list(self.wc.repos.list(**kwargs)))
        if not items:
            return _ok("No git folders found.")
        rows = []
        for r in items:
            rows.append(f"| {r.id} | {r.path} | {getattr(r, 'branch', '')} | {getattr(r, 'url', '')} |")
        body = "\n".join(rows)
        return _ok(
            f"**Git folders ({len(items)}):**\n\n"
            f"| ID | PATH | BRANCH | URL |\n|---|---|---|---|\n{body}",
            n=len(items),
        )

    async def _inspect(self, args: Dict[str, Any]) -> ToolResult:
        repo_id = args.get("repo_id")
        if not repo_id:
            return _err("repo_id is required")
        r = await asyncio.to_thread(self.wc.repos.get, repo_id=int(repo_id))
        import json
        out = {
            "id": r.id,
            "path": r.path,
            "url": getattr(r, "url", None),
            "provider": getattr(r, "provider", None),
            "branch": getattr(r, "branch", None),
            "head_commit_id": getattr(r, "head_commit_id", None),
        }
        return _ok(f"**Repo {repo_id}:**\n\n```json\n{json.dumps(out, default=str, indent=2)}\n```")

    async def _pull(self, args: Dict[str, Any]) -> ToolResult:
        repo_id = args.get("repo_id")
        if not repo_id:
            return _err("repo_id is required")
        branch = args.get("branch")
        kwargs: Dict[str, Any] = {"repo_id": int(repo_id)}
        if branch:
            kwargs["branch"] = branch
        await asyncio.to_thread(self.wc.repos.update, **kwargs)
        return _ok(f"Pulled latest on repo {repo_id}" + (f" (branch={branch})" if branch else "."))

    async def _delete(self, args: Dict[str, Any]) -> ToolResult:
        repo_id = args.get("repo_id")
        if not repo_id:
            return _err("repo_id is required")
        await asyncio.to_thread(self.wc.repos.delete, repo_id=int(repo_id))
        return _ok(f"Deleted repo {repo_id}.")


REPOS_TOOL_SPEC = {
    "name": "repos",
    "description": (
        "Manage Databricks Git Folders (Repos). Clone any git repo into /Workspace/Users/<u>/repos/<name> "
        "so training scripts can import from it directly.\n\n"
        "Provider is inferred from URL host (github.com, gitlab.com, bitbucket.org, dev.azure.com); "
        "supply 'provider' for self-hosted.\n\n"
        "For private repos, link your Git provider credentials in Databricks Settings → Git Integration. "
        "PATs cannot be passed via this tool.\n\n"
        "Operations:\n"
        "- clone: create a new Git Folder from a URL\n"
        "- list: enumerate Git Folders\n"
        "- inspect: get metadata + HEAD commit\n"
        "- pull: update branch HEAD\n"
        "- delete: remove the folder"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["clone", "list", "inspect", "pull", "delete"]},
            "url": {"type": "string", "description": "(clone) Repo URL."},
            "provider": {"type": "string", "description": "(clone) Override inferred provider."},
            "name": {"type": "string", "description": "(clone) Override folder name."},
            "path": {"type": "string", "description": "(clone) Override workspace path."},
            "branch": {"type": "string", "description": "(clone, pull) Branch to checkout."},
            "repo_id": {"type": ["integer", "string"], "description": "(inspect, pull, delete)"},
            "path_prefix": {"type": "string", "description": "(list) Filter by prefix."},
        },
        "required": ["operation"],
    },
}


async def repos_handler(arguments: Dict[str, Any], session: Any = None,
                        tool_call_id: str | None = None) -> tuple[str, bool]:
    try:
        cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
        settings = db_client.resolve_settings(cfg)
        token = getattr(session, "databricks_user_token", None) if session else None
        if token and settings.host:
            wc = db_client.get_workspace_client_for_user(token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)
        user_email = getattr(session, "user_email", None) if session else None
        if not user_email:
            try:
                me = await asyncio.to_thread(wc.current_user.me)
                user_email = me.user_name or (me.emails[0].value if me.emails else None)
            except Exception:
                user_email = None
        tool = ReposTool(wc=wc, settings=settings, user_email=user_email)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("repos handler crashed")
        return f"Error: {e}", False


def _load_default_config():
    from agent.config import load_config
    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)
