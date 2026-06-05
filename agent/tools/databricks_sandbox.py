"""Databricks-native sandbox.

Replaces the HF-Space-backed sandbox. Adaptive probe at session start picks
the cheapest compute the workspace exposes:

    1. Serverless GPU compute        — newest API; gated by ML_INTERN_ALLOW_SERVERLESS_GPU
    2. Serverless commands API       — TODO once stable; falls through today
    3. Pool-backed all-purpose cluster (instance_pool_id from settings)
    4. On-demand all-purpose cluster (created with default node + runtime)

Each backend exposes the same tool surface — ``bash`` / ``read`` / ``write``
/ ``edit`` — so the agent loop doesn't care which one is live.

`bash` runs the command in a Python kernel via ``command_execution`` (no
native shell on Databricks) by shelling out with ``subprocess.run`` and
echoing combined stdout/stderr. `read` / `write` work against Workspace
Files (``/Workspace/...``) and UC Volumes (``/Volumes/...``); the path
prefix decides which API is used.

Lifecycle::

    sb = await DatabricksSandbox.create_async(settings, hardware="a10g-large", log=...)
    sb.bash("uv run train.py")
    sb.read("/Workspace/Users/u/x.py")
    sb.delete()                               # tears down owned cluster
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from agent.core import db_client
from agent.tools.databricks_jobs_tool import HARDWARE_FLAVOR_TO_NODE_TYPE

logger = logging.getLogger(__name__)


# --- defaults ---------------------------------------------------------------
DEFAULT_TIMEOUT = 240
MAX_TIMEOUT = 1200
WAIT_INTERVAL_S = 5
CLUSTER_WAIT_TIMEOUT_S = 600
CONTEXT_WAIT_TIMEOUT_S = 120
COMMAND_POLL_INTERVAL_S = 2

OUTPUT_LIMIT = 25_000


# --- helpers ----------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _truncate(text: str, max_chars: int = OUTPUT_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n\n... ({omitted:,} of {len(text):,} chars omitted; head {head} + tail {tail}) ...\n\n"
        + text[-tail:]
    )


def _safe_segment(s: str | None, default: str = "x") -> str:
    if not s:
        return default
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", s)


# --- result -----------------------------------------------------------------


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return self.output or "(no output)"
        return f"ERROR: {self.error}"

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error}


# --- compute probe ----------------------------------------------------------


@dataclass
class ComputeChoice:
    """Resolved compute backend for a sandbox session."""

    kind: str  # "serverless_gpu" | "pool" | "on_demand"
    cluster_id: str
    owns_cluster: bool
    node_type_id: str | None = None
    pool_id: str | None = None


async def probe_compute(
    wc,
    settings: db_client.DatabricksSettings,
    hardware: str = "cpu-basic",
    log: Callable[[str], Any] | None = None,
) -> ComputeChoice:
    """Pick a compute backend the workspace can satisfy.

    Cascade (cheapest first), each step short-circuits on failure:

      1. Serverless GPU (if ``ML_INTERN_ALLOW_SERVERLESS_GPU=1`` and a
         serverless-capable cluster surfaces from ``wc.clusters.list``).
      2. Pool-backed (``instance_pool_id`` configured) — fast warm start.
      3. On-demand — slowest, always works if quota permits.
    """
    log = log or logger.info
    node_type = (
        HARDWARE_FLAVOR_TO_NODE_TYPE.get(hardware) or settings.default_node_type_id
    )

    if os.environ.get("ML_INTERN_ALLOW_SERVERLESS_GPU") == "1":
        try:
            log("Probing serverless GPU compute…")
            # Serverless GPU is exposed as a "serverless" cluster type when GA;
            # until then this just falls through. Kept here so swapping in the
            # real call later is a one-line change.
            raise NotImplementedError("serverless GPU probe not GA yet")
        except Exception as e:
            log(f"Serverless GPU unavailable: {e}")

    if settings.instance_pool_id:
        log(f"Provisioning pool-backed cluster (pool={settings.instance_pool_id})…")
        cluster_id = await _create_cluster(
            wc,
            settings=settings,
            instance_pool_id=settings.instance_pool_id,
            node_type_id=None,
            hardware=hardware,
        )
        return ComputeChoice(
            kind="pool",
            cluster_id=cluster_id,
            owns_cluster=True,
            pool_id=settings.instance_pool_id,
        )

    log(f"Provisioning on-demand cluster (node={node_type})…")
    cluster_id = await _create_cluster(
        wc,
        settings=settings,
        instance_pool_id=None,
        node_type_id=node_type,
        hardware=hardware,
    )
    return ComputeChoice(
        kind="on_demand",
        cluster_id=cluster_id,
        owns_cluster=True,
        node_type_id=node_type,
    )


async def _create_cluster(
    wc,
    *,
    settings: db_client.DatabricksSettings,
    instance_pool_id: str | None,
    node_type_id: str | None,
    hardware: str,
) -> str:
    """Create a single-node all-purpose cluster and wait for RUNNING."""
    body: dict[str, Any] = {
        "cluster_name": f"ml-intern-sandbox-{uuid.uuid4().hex[:8]}",
        "spark_version": settings.default_runtime_version,
        "num_workers": 0,
        "autotermination_minutes": 30,
        "spark_conf": {"spark.databricks.cluster.profile": "singleNode"},
        "custom_tags": {"ml_intern_purpose": "sandbox", "hardware": hardware},
    }
    if instance_pool_id:
        body["instance_pool_id"] = instance_pool_id
    elif node_type_id:
        body["node_type_id"] = node_type_id

    resp = await asyncio.to_thread(
        wc.api_client.do, "POST", "/api/2.1/clusters/create", body=body,
    )
    cluster_id = resp.get("cluster_id")
    if not cluster_id:
        raise RuntimeError(f"clusters/create returned no cluster_id: {resp}")

    deadline = time.monotonic() + CLUSTER_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        info = await asyncio.to_thread(
            wc.api_client.do, "GET", "/api/2.1/clusters/get", query={"cluster_id": cluster_id},
        )
        state = info.get("state") or ""
        if state == "RUNNING":
            return cluster_id
        if state in {"TERMINATED", "TERMINATING", "ERROR", "UNKNOWN"}:
            raise RuntimeError(
                f"Cluster {cluster_id} failed to start (state={state}, "
                f"reason={info.get('state_message','')})"
            )
        await asyncio.sleep(WAIT_INTERVAL_S)
    raise TimeoutError(f"Cluster {cluster_id} did not reach RUNNING in {CLUSTER_WAIT_TIMEOUT_S}s")


# --- file ops ---------------------------------------------------------------


def _is_volume(path: str) -> bool:
    return path.startswith("/Volumes/")


def _is_workspace(path: str) -> bool:
    return path.startswith("/Workspace/")


# --- Sandbox class ----------------------------------------------------------


@dataclass
class DatabricksSandbox:
    """Cluster-backed sandbox handle.

    Construct via ``DatabricksSandbox.create_async`` — it does the compute
    probe + execution-context setup in one shot. ``connect`` reattaches to
    an existing cluster id (no probe, no ownership transfer).
    """

    wc: Any
    settings: db_client.DatabricksSettings
    compute: ComputeChoice
    context_id: str
    user_email: str | None = None
    work_dir: str = "/Workspace/Users"
    timeout: int = DEFAULT_TIMEOUT
    _files_read: set[str] = field(default_factory=set, repr=False)

    class Cancelled(Exception):
        """Raised when sandbox creation is cancelled by the user."""

    @property
    def cluster_id(self) -> str:
        return self.compute.cluster_id

    @property
    def url(self) -> str:
        host = self.settings.host.rstrip("/")
        return f"{host}/#setting/clusters/{self.cluster_id}"

    @property
    def status(self) -> str:
        info = self.wc.api_client.do(
            "GET", "/api/2.1/clusters/get", query={"cluster_id": self.cluster_id},
        )
        return info.get("state") or "UNKNOWN"

    @property
    def space_id(self) -> str:
        # Compatibility shim with telemetry.record_sandbox_*.
        return self.cluster_id

    @classmethod
    async def create_async(
        cls,
        settings: db_client.DatabricksSettings,
        *,
        hardware: str = "cpu-basic",
        wc: Any = None,
        user_email: str | None = None,
        log: Callable[[str], Any] | None = None,
        cancel_event: Any | None = None,
    ) -> "DatabricksSandbox":
        log = log or logger.info
        wc = wc or db_client.get_workspace_client(settings)

        if cancel_event and cancel_event.is_set():
            raise cls.Cancelled("cancelled before compute probe")

        compute = await probe_compute(wc, settings, hardware=hardware, log=log)

        if cancel_event and cancel_event.is_set():
            await cls._safe_terminate(wc, compute)
            raise cls.Cancelled("cancelled after cluster start")

        context_id = await cls._create_context(wc, compute.cluster_id)
        log(f"Sandbox ready: cluster={compute.cluster_id} kind={compute.kind}")
        return cls(
            wc=wc,
            settings=settings,
            compute=compute,
            context_id=context_id,
            user_email=user_email,
            work_dir=f"/Workspace/Users/{_safe_segment(user_email, 'user')}/ml-intern",
        )

    @staticmethod
    async def _create_context(wc: Any, cluster_id: str) -> str:
        resp = await asyncio.to_thread(
            wc.api_client.do, "POST", "/api/1.2/contexts/create",
            body={"clusterId": cluster_id, "language": "python"},
        )
        ctx_id = resp.get("id")
        if not ctx_id:
            raise RuntimeError(f"contexts/create failed: {resp}")
        # Wait for context to enter Running state.
        deadline = time.monotonic() + CONTEXT_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            info = await asyncio.to_thread(
                wc.api_client.do, "GET", "/api/1.2/contexts/status",
                query={"clusterId": cluster_id, "contextId": ctx_id},
            )
            status = info.get("status") or ""
            if status == "Running":
                return ctx_id
            if status in {"Error"}:
                raise RuntimeError(f"Execution context error: {info}")
            await asyncio.sleep(2)
        raise TimeoutError("Execution context did not become Running")

    @staticmethod
    async def _safe_terminate(wc: Any, compute: ComputeChoice) -> None:
        if not compute.owns_cluster:
            return
        try:
            await asyncio.to_thread(
                wc.api_client.do, "POST", "/api/2.1/clusters/permanent-delete",
                body={"cluster_id": compute.cluster_id},
            )
        except Exception as e:
            logger.warning("cluster delete failed: %s", e)

    def delete(self) -> None:
        """Terminate the cluster (if we created it) and discard the context."""
        try:
            self.wc.api_client.do(
                "POST", "/api/1.2/contexts/destroy",
                body={"clusterId": self.cluster_id, "contextId": self.context_id},
            )
        except Exception as e:
            logger.debug("context destroy suppressed: %s", e)
        if self.compute.owns_cluster:
            try:
                self.wc.api_client.do(
                    "POST", "/api/2.1/clusters/permanent-delete",
                    body={"cluster_id": self.cluster_id},
                )
            except Exception as e:
                logger.warning("cluster delete failed: %s", e)

    def __enter__(self) -> "DatabricksSandbox":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.delete()
        except Exception as e:
            logger.warning("sandbox cleanup failed: %s", e)

    # --- command execution -------------------------------------------------

    async def _run_python(self, code: str, timeout: int | None = None) -> ToolResult:
        timeout = min(timeout or self.timeout, MAX_TIMEOUT)
        resp = await asyncio.to_thread(
            self.wc.api_client.do, "POST", "/api/1.2/commands/execute",
            body={
                "clusterId": self.cluster_id,
                "contextId": self.context_id,
                "language": "python",
                "command": code,
            },
        )
        cmd_id = resp.get("id")
        if not cmd_id:
            return ToolResult(False, error=f"commands/execute failed: {resp}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = await asyncio.to_thread(
                self.wc.api_client.do, "GET", "/api/1.2/commands/status",
                query={"clusterId": self.cluster_id, "contextId": self.context_id, "commandId": cmd_id},
            )
            status = info.get("status") or ""
            if status in {"Finished", "Error", "Cancelled"}:
                results = info.get("results") or {}
                if status == "Finished" and results.get("resultType") in ("text", "table"):
                    out = results.get("data") or ""
                    return ToolResult(True, output=_truncate(_strip_ansi(str(out))))
                cause = results.get("cause") or results.get("summary") or status
                return ToolResult(False, error=_strip_ansi(str(cause)))
            await asyncio.sleep(COMMAND_POLL_INTERVAL_S)
        # Timeout — best-effort cancel.
        try:
            await asyncio.to_thread(
                self.wc.api_client.do, "POST", "/api/1.2/commands/cancel",
                body={"clusterId": self.cluster_id, "contextId": self.context_id, "commandId": cmd_id},
            )
        except Exception:
            pass
        return ToolResult(False, error=f"Timeout after {timeout}s")

    # --- tool surface ------------------------------------------------------

    def bash(
        self, command: str, *,
        work_dir: str | None = None, timeout: int | None = None,
        description: str | None = None,
    ) -> ToolResult:
        cwd = work_dir or self.work_dir
        # Run shell on the driver via subprocess. Output goes to stdout for
        # commands/execute to capture.
        py = (
            "import subprocess, os\n"
            f"os.makedirs({cwd!r}, exist_ok=True)\n"
            f"r = subprocess.run({command!r}, shell=True, cwd={cwd!r}, "
            "capture_output=True, text=True)\n"
            "print(r.stdout, end='')\n"
            "if r.stderr:\n"
            "    print(r.stderr, end='')\n"
            "print(f'__exit_code__={r.returncode}')\n"
        )
        return asyncio.get_event_loop().run_until_complete(self._run_python(py, timeout)) \
            if not _in_event_loop() else _sync_via_thread(self._run_python(py, timeout))

    def read(self, path: str, *, offset: int | None = None, limit: int | None = None) -> ToolResult:
        try:
            data = _download(self.wc, path)
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            start = (offset or 1) - 1
            end = start + (limit or len(lines))
            sel = lines[start:end]
            numbered = "\n".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(sel))
            self._files_read.add(path)
            return ToolResult(True, output=numbered)
        except Exception as e:
            return ToolResult(False, error=str(e))

    def write(self, path: str, content: str) -> ToolResult:
        try:
            _upload(self.wc, path, content.encode("utf-8") if isinstance(content, str) else content)
            self._files_read.add(path)
            return ToolResult(True, output=f"Wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(False, error=str(e))

    def edit(
        self, path: str, old_str: str, new_str: str, *,
        replace_all: bool = False, mode: str = "replace",
    ) -> ToolResult:
        if old_str == new_str:
            return ToolResult(False, error="old_str and new_str are identical.")
        if path not in self._files_read:
            return ToolResult(False, error=f"File {path} has not been read this session. Read it first.")
        try:
            data = _download(self.wc, path)
            text = data.decode("utf-8", errors="replace")
            if old_str not in text:
                return ToolResult(False, error="old_str not found in file.")
            if mode == "replace":
                count = text.count(old_str)
                if count > 1 and not replace_all:
                    return ToolResult(False, error=f"old_str appears {count} times. Use replace_all.")
                new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
            elif mode == "append_after":
                new_text = text.replace(old_str, old_str + new_str, -1 if replace_all else 1)
            elif mode == "prepend_before":
                new_text = text.replace(old_str, new_str + old_str, -1 if replace_all else 1)
            else:
                return ToolResult(False, error=f"Unknown mode: {mode}")
            _upload(self.wc, path, new_text.encode("utf-8"))
            return ToolResult(True, output=f"Edited {path}")
        except Exception as e:
            return ToolResult(False, error=str(e))

    def kill_all(self) -> ToolResult:
        try:
            self.wc.api_client.do(
                "POST", "/api/1.2/commands/cancel",
                body={"clusterId": self.cluster_id, "contextId": self.context_id, "commandId": ""},
            )
        except Exception:
            pass
        return ToolResult(True, output="Cancel requested")

    # --- tool schemas (parallel to old Sandbox.TOOLS) ----------------------

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the Databricks sandbox cluster's driver shell.\n\n"
                "Each call runs in a fresh subprocess (state lives only in /Workspace, /Volumes, /tmp).\n"
                "Use 'read' for files, 'edit' for in-place changes, 'write' for new files — "
                "do NOT use cat/head/tail/sed/awk for those.\n\n"
                "For long-running training, launch via the databricks_jobs tool instead — bash is "
                "for development and small probes (timeout default 240s, max 1200s)."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "work_dir": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
            },
        },
        "read": {
            "description": (
                "Read a file from /Workspace/... or /Volumes/... Returns line-numbered "
                "text (cat -n format). Required before edit/write."
            ),
            "parameters": {
                "type": "object", "required": ["path"], "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
        },
        "write": {
            "description": (
                "Write a file to /Workspace/... or /Volumes/... Overwrites if exists. "
                "Must read first if file pre-exists."
            ),
            "parameters": {
                "type": "object", "required": ["path", "content"], "additionalProperties": False,
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        },
        "edit": {
            "description": (
                "In-place string replacement in a file. Read first. "
                "old_str/new_str must differ. Use replace_all for multi-occurrence."
            ),
            "parameters": {
                "type": "object", "required": ["path", "old_str", "new_str"], "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                    "mode": {"type": "string", "enum": ["replace", "append_after", "prepend_before"], "default": "replace"},
                },
            },
        },
    }

    @classmethod
    def tool_definitions(cls) -> list[dict]:
        return [{"name": n, **s} for n, s in cls.TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": lambda a: self.bash(a["command"], work_dir=a.get("work_dir"), timeout=a.get("timeout")),
            "read": lambda a: self.read(a["path"], offset=a.get("offset"), limit=a.get("limit")),
            "write": lambda a: self.write(a["path"], a["content"]),
            "edit": lambda a: self.edit(
                a["path"], a["old_str"], a["new_str"],
                replace_all=a.get("replace_all", False), mode=a.get("mode", "replace"),
            ),
        }
        fn = dispatch.get(name)
        if not fn:
            return ToolResult(False, error=f"Unknown tool: {name}")
        return fn(arguments)


# --- file IO helpers (Workspace Files vs UC Volumes) ------------------------


def _download(wc: Any, path: str) -> bytes:
    if _is_volume(path):
        resp = wc.files.download(file_path=path)
        return resp.contents.read()
    if _is_workspace(path):
        resp = wc.workspace.export(path=path, format="SOURCE")
        # SDK returns base64-encoded content for SOURCE format.
        import base64
        content = getattr(resp, "content", None)
        if content is None:
            return b""
        return base64.b64decode(content) if isinstance(content, str) else bytes(content)
    raise ValueError(f"Path must start with /Workspace/ or /Volumes/ (got {path!r})")


def _upload(wc: Any, path: str, data: bytes) -> None:
    if _is_volume(path):
        # Files API expects a file-like object.
        wc.files.upload(file_path=path, contents=io.BytesIO(data), overwrite=True)
        return
    if _is_workspace(path):
        parent = path.rsplit("/", 1)[0]
        wc.workspace.mkdirs(parent)
        wc.workspace.upload(path=path, content=data, overwrite=True)
        return
    raise ValueError(f"Path must start with /Workspace/ or /Volumes/ (got {path!r})")


# --- sync/async glue --------------------------------------------------------


def _in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _sync_via_thread(coro: Awaitable) -> Any:
    """Run an awaitable from a sync function while another loop is running."""
    import threading
    box: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            box["v"] = loop.run_until_complete(coro)
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    return box.get("v")
