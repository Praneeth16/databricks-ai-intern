"""Sandbox tools — expose the DatabricksSandbox as agent tools.

Five tools total:
    sandbox_create — explicit sandbox creation (requires approval).
    bash, read, write, edit — operations on the sandbox.

Calling an operation tool without an active sandbox auto-creates a
cpu-basic sandbox (no approval needed).

The HF-Space backend is gone; this is a thin wrapper over
``DatabricksSandbox`` from ``databricks_sandbox.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from agent.core import db_client
from agent.core.session import Event
from agent.tools.databricks_jobs_tool import HARDWARE_FLAVOR_TO_NODE_TYPE
from agent.tools.databricks_sandbox import DatabricksSandbox, ToolResult

logger = logging.getLogger(__name__)


def _looks_like_path(script: str) -> bool:
    return (
        isinstance(script, str)
        and script.strip() == script
        and not any(c in script for c in "\r\n\0")
        and (
            script.startswith("/Workspace/")
            or script.startswith("/Volumes/")
            or script.startswith("/")
            or script.startswith("./")
            or script.startswith("../")
        )
    )


async def resolve_sandbox_script(sandbox: Any, script: str) -> tuple[str | None, str | None]:
    """Read a file from the sandbox if *script* looks like a path.

    Returns ``(content, error)``. Both None ⇒ caller should treat *script*
    as inline code, not a path.
    """
    if not sandbox or not _looks_like_path(script):
        return None, None
    try:
        result = await asyncio.to_thread(sandbox.read, script, limit=100_000)
        if result.success and result.output:
            lines = []
            for line in result.output.split("\n"):
                parts = line.split("\t", 1)
                lines.append(parts[1] if len(parts) == 2 else line)
            return "\n".join(lines), None
        return None, f"Failed to read {script}: {result.error}"
    except Exception as e:
        return None, f"Failed to read {script}: {e}"


async def _ensure_sandbox(
    session: Any, hardware: str = "cpu-basic", **create_kwargs,
) -> tuple[DatabricksSandbox | None, str | None]:
    if session and getattr(session, "sandbox", None):
        return session.sandbox, None
    if not session:
        return None, "No session available."

    cfg = getattr(session, "config", None)
    if cfg is None:
        from agent.config import load_config
        cfg_path = os.environ.get(
            "ML_INTERN_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
        )
        cfg = load_config(cfg_path)

    settings = db_client.resolve_settings(cfg)
    user_token = getattr(session, "databricks_user_token", None)
    if user_token and settings.host:
        wc = db_client.get_workspace_client_for_user(user_token, settings.host)
    else:
        wc = db_client.get_workspace_client(settings)

    user_email = getattr(session, "user_email", None)
    if not user_email:
        try:
            me = await asyncio.to_thread(wc.current_user.me)
            user_email = me.user_name
        except Exception:
            user_email = None

    await session.send_event(Event(
        event_type="tool_log",
        data={"tool": "sandbox", "log": f"Creating Databricks sandbox ({hardware})…"},
    ))

    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.event_queue.put_nowait,
            Event(event_type="tool_log", data={"tool": "sandbox", "log": msg}),
        )

    cancel_flag = threading.Event()

    async def _watch_cancel():
        await session._cancelled.wait()
        cancel_flag.set()

    watcher = asyncio.create_task(_watch_cancel())

    import time as _t
    start = _t.monotonic()
    try:
        sb = await DatabricksSandbox.create_async(
            settings,
            hardware=hardware,
            wc=wc,
            user_email=user_email,
            log=_log,
            cancel_event=cancel_flag,
        )
    except DatabricksSandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    except Exception as e:
        return None, f"Sandbox creation failed: {e}"
    finally:
        watcher.cancel()

    session.sandbox = sb

    from agent.core import telemetry
    await telemetry.record_sandbox_create(
        session, sb, hardware=hardware,
        create_latency_s=int(_t.monotonic() - start),
    )

    await session.send_event(Event(
        event_type="tool_log",
        data={"tool": "sandbox", "log": f"Sandbox ready: cluster={sb.cluster_id}"},
    ))
    return sb, None


SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create a Databricks sandbox cluster for developing and testing scripts.\n\n"
        "Workflow: sandbox_create → write script → pip install → small test run → fix → "
        "databricks_jobs at scale.\n\n"
        "The sandbox is a single-node all-purpose cluster. State persists across tool calls "
        "until you delete it. /Workspace and /Volumes are durable across cluster restarts.\n\n"
        "Use this when iterating on training code (verify imports, run on a small slice). "
        "Skip for one-shot operations or scripts already validated elsewhere — submit them "
        "directly via databricks_jobs.\n\n"
        f"Hardware: {', '.join(HARDWARE_FLAVOR_TO_NODE_TYPE.keys())}. "
        "If ML_INTERN_INSTANCE_POOL_ID is configured, runs default to that pool (faster start)."
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": {
                "type": "string",
                "enum": list(HARDWARE_FLAVOR_TO_NODE_TYPE.keys()),
                "description": "Hardware tier for the sandbox (default cpu-basic).",
            },
        },
    },
}


async def sandbox_create_handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        return (
            f"Sandbox already active: cluster={sb.cluster_id}\n"
            f"URL: {sb.url}\n"
            f"Use bash/read/write/edit to interact."
        ), True

    hardware = args.get("hardware", "cpu-basic")
    try:
        sb, error = await _ensure_sandbox(session, hardware=hardware)
    except Exception as e:
        return f"Failed to create sandbox: {e}", False
    if error:
        return error, False
    return (
        f"Sandbox created: cluster={sb.cluster_id} ({sb.compute.kind})\n"
        f"URL: {sb.url}\n"
        f"Hardware: {hardware}\n"
        f"Use bash/read/write/edit to interact."
    ), True


def _make_tool_handler(sandbox_tool_name: str):
    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        sb, error = await _ensure_sandbox(session)
        if error:
            return error, False
        try:
            result = await asyncio.to_thread(sb.call_tool, sandbox_tool_name, args)
            if result.success:
                return result.output or "(no output)", True
            err = result.error or "Unknown error"
            return (f"{result.output}\n\nERROR: {err}" if result.output else f"ERROR: {err}"), False
        except Exception as e:
            return f"Sandbox operation failed: {e}", False
    return handler


def get_sandbox_tools():
    """Return the 5 sandbox ToolSpecs."""
    from agent.core.tools import ToolSpec

    tools = [
        ToolSpec(
            name=SANDBOX_CREATE_TOOL_SPEC["name"],
            description=SANDBOX_CREATE_TOOL_SPEC["description"],
            parameters=SANDBOX_CREATE_TOOL_SPEC["parameters"],
            handler=sandbox_create_handler,
        )
    ]
    for name, spec in DatabricksSandbox.TOOLS.items():
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=_make_tool_handler(name),
            )
        )
    return tools
