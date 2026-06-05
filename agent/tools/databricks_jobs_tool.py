"""Databricks Jobs tool — replacement for the HF Jobs tool.

Dispatches by ``kind``:

- ``finetune``  → Mosaic AI Model Training REST API (foundation-model fine-tune,
                  registers the resulting model into Unity Catalog).
- ``script``    → Databricks Jobs ``runs/submit`` with a ``new_cluster`` spec
                  (GPU pool-backed when ``ML_INTERN_INSTANCE_POOL_ID`` is set,
                  otherwise on-demand at the configured node type).
- ``serverless``→ Databricks Jobs ``runs/submit`` with serverless compute (no
                  cluster spec, ``environment_key`` declares deps).

Inline scripts are written to Workspace Files at
``/Workspace/Users/<user>/ml-intern/<session>/<filename>`` and referenced by
path — never base64-wrapped into the job command. Secrets must be passed via
the dynamic-reference syntax ``{{secrets/<scope>/<key>}}``; the tool refuses
plaintext ``DATABRICKS_*`` / cloud-creds in agent-supplied env to keep the
LLM out of the auth path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from agent.core import db_client
from agent.core.session import Event
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


# Hardware-flavor convenience aliases. Default mapping is AWS — workspaces on
# Azure / GCP override via ``node_type_id`` arg or ML_INTERN_NODE_TYPE config.
HARDWARE_FLAVOR_TO_NODE_TYPE: Dict[str, str] = {
    "cpu-basic":     "m5.large",
    "cpu-upgrade":   "m5.2xlarge",
    "t4-small":      "g4dn.xlarge",
    "t4-medium":     "g4dn.2xlarge",
    "a10g-small":    "g5.xlarge",
    "a10g-large":    "g5.4xlarge",
    "a10g-largex2":  "g5.12xlarge",
    "a10g-largex4":  "g5.24xlarge",
    "a100-large":    "p4d.24xlarge",
    "a100x4":        "p4d.24xlarge",
    "a100x8":        "p4d.24xlarge",
    "l4x1":          "g6.xlarge",
    "l4x4":          "g6.12xlarge",
    "l40sx1":        "g6e.xlarge",
    "l40sx4":        "g6e.12xlarge",
    "l40sx8":        "g6e.48xlarge",
}

OperationType = Literal[
    "run", "ps", "logs", "inspect", "cancel",
    "scheduled run", "scheduled ps", "scheduled inspect",
    "scheduled delete", "scheduled suspend", "scheduled resume",
]

KindType = Literal["finetune", "script", "serverless", "serverless_gpu"]

# Mosaic AI Model Training endpoint. Stable since 2024-Q4. Override-able via
# env so we don't have to ship a release if Databricks renames the path.
_FINETUNE_API_PATH = os.environ.get(
    "ML_INTERN_FINETUNE_API_PATH",
    "/api/2.0/foundation-model-training/runs",
)

# Jobs API 2.2 — needed for per-task ``compute`` block (AI Runtime
# serverless GPU selector). 2.1 silently drops the field.
_JOBS_SUBMIT_PATH = "/api/2.2/jobs/runs/submit"
_JOBS_RUNS_GET = "/api/2.2/jobs/runs/get"
_JOBS_RUNS_LIST = "/api/2.2/jobs/runs/list"
_JOBS_RUNS_CANCEL = "/api/2.2/jobs/runs/cancel"
_JOBS_RUN_OUTPUT = "/api/2.2/jobs/runs/get-output"
_JOBS_CREATE = "/api/2.2/jobs/create"
_JOBS_LIST = "/api/2.2/jobs/list"
_JOBS_GET = "/api/2.2/jobs/get"
_JOBS_DELETE = "/api/2.2/jobs/delete"
_JOBS_UPDATE = "/api/2.2/jobs/update"

_TERMINAL_LIFECYCLES = {
    "TERMINATED", "INTERNAL_ERROR", "SKIPPED", "BLOCKED",
}

# Auth / cloud-creds the agent must NEVER set directly. Caught even when the
# LLM emits ``DATABRICKS_TOKEN: "$DATABRICKS_TOKEN"`` thinking it's a passthrough.
_AUTH_VAR_BLOCKLIST = {
    "DATABRICKS_TOKEN",
    "DATABRICKS_HOST",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_CONFIG_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "GOOGLE_APPLICATION_CREDENTIALS",
}


def _filter_agent_env(env: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Strip auth/cloud-cred vars; preserve dynamic secret refs.

    Vars in ``_AUTH_VAR_BLOCKLIST`` are dropped unconditionally. Other
    Databricks-prefixed vars are dropped unless their value is a dynamic
    secret reference ``{{secrets/scope/key}}``.
    """
    out: Dict[str, str] = {}
    if not env:
        return out
    for k, v in env.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            v = str(v)
        ku = k.upper()
        if ku in _AUTH_VAR_BLOCKLIST:
            logger.warning("Dropping agent-supplied auth var %s from job env", k)
            continue
        if ku.startswith("DATABRICKS_") and "{{secrets/" not in v:
            logger.warning(
                "Dropping agent-supplied %s — only dynamic secret refs allowed for DATABRICKS_*",
                k,
            )
            continue
        out[k] = v
    return out


def _parse_timeout(s: Optional[str]) -> int:
    """Parse ``"30m" / "8h" / "1d"`` into seconds. Returns 0 on bad input."""
    if not s:
        return 0
    m = re.match(r"^\s*(\d+)\s*([smhd]?)\s*$", str(s).lower())
    if not m:
        return 0
    n, u = int(m.group(1)), (m.group(2) or "s")
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]


def _safe_segment(s: Optional[str], default: str = "x") -> str:
    """Conservative slug for use in Workspace / UC Volume paths.

    Databricks file path resolvers reject some non-alnum characters in
    spark_python_task paths (notably ``@`` and ``.`` in segments). Strip
    everything outside ``[A-Za-z0-9_-]``.
    """
    if not s:
        return default
    out = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return out or default


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _run_url(host: str, run_id: int | str, job_id: int | str | None = None) -> str:
    base = host.rstrip("/")
    if job_id:
        return f"{base}/jobs/{job_id}/runs/{run_id}"
    return f"{base}/jobs/runs/{run_id}"


def _experiment_url(host: str, experiment_path: str) -> str:
    return f"{host.rstrip('/')}/#mlflow/experiments?path={experiment_path}"


# ---------------------------------------------------------------------------
# Notebook stdout-capture wrapper
# ---------------------------------------------------------------------------
#
# serverless_gpu jobs run via ``notebook_task``. The Jobs ``runs/get-output``
# endpoint returns ``notebook_output.result`` (the value of
# ``dbutils.notebook.exit(...)``) for notebook tasks — *not* stdout. Without a
# wrapper, ``logs`` op returns an empty string after a successful serverless_gpu
# run and the agent has to round-trip through MLflow to learn what happened.
#
# The wrapper tees ``sys.stdout``/``sys.stderr`` into an in-memory buffer and,
# on natural script exit, calls ``dbutils.notebook.exit`` with the last 4 KB so
# ``runs/get-output`` carries the tail. If the user's script already calls
# ``dbutils.notebook.exit`` (a textual check), we leave that path untouched —
# their explicit exit value wins.

_NOTEBOOK_STDOUT_WRAPPER_PRELUDE = '''\
import sys as _ml_sys, io as _ml_io
_ML_INTERN_BUF = _ml_io.StringIO()
class _ML_INTERN_TEE:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, b):
        for s in self._streams:
            try: s.write(b)
            except Exception: pass
        return len(b) if isinstance(b, str) else 0
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass
    def isatty(self):
        return False
_ml_sys.stdout = _ML_INTERN_TEE(_ml_sys.__stdout__, _ML_INTERN_BUF)
_ml_sys.stderr = _ML_INTERN_TEE(_ml_sys.__stderr__, _ML_INTERN_BUF)
'''


# Common ML deps that the serverless_gpu image (env_version=4, transformers 5.x)
# doesn't ship. Empirically derived from PTB-smoke run-D (4 iterations burned on
# missing/incompat packages). Each entry pinned to a range known to work with
# the runtime's transformers 5.x. Caller can disable with pre_install=False.
_SERVERLESS_GPU_PRE_INSTALL: tuple[str, ...] = (
    "datasets>=4.0,<5.0",
    "trl>=1.4,<2.0",
    "peft>=0.18,<0.30",
    "accelerate>=1.5,<2.0",
)


def _pre_install_prelude(specs: tuple[str, ...]) -> str:
    """Idempotent `pip install --quiet` for each spec; runs before the user
    script so missing packages don't burn a job round-trip per discovery."""
    if not specs:
        return ""
    spec_literal = ", ".join(repr(s) for s in specs)
    return f'''\
import subprocess as _ml_subprocess, sys as _ml_sys_pre
for _ml_spec in [{spec_literal}]:
    try:
        _ml_subprocess.check_call(
            [_ml_sys_pre.executable, "-m", "pip", "install", "--quiet", _ml_spec],
        )
    except Exception as _ml_pip_err:
        print(f"[ml-intern pre-install] {{_ml_spec}} failed: {{_ml_pip_err}}")
'''


def _wrap_user_script_with_stdout_capture(script: str) -> str:
    """Append a ``dbutils.notebook.exit`` finalizer that surfaces stdout tail.

    Skipped when the user's script already calls ``dbutils.notebook.exit``
    so we don't override an intentional exit value mid-flow.
    """
    if "dbutils.notebook.exit" in script:
        return script
    import textwrap
    return (
        "try:\n"
        + textwrap.indent(script, "    ")
        + "\nexcept BaseException as _ml_intern_err:\n"
        "    _ml_tail = _ML_INTERN_BUF.getvalue()[-4000:]\n"
        "    try:\n"
        "        dbutils.notebook.exit(_ml_tail + '\\n[ml-intern] error: ' + repr(_ml_intern_err))\n"
        "    except Exception:\n"
        "        pass\n"
        "    raise\n"
        "else:\n"
        "    try:\n"
        "        dbutils.notebook.exit(_ML_INTERN_BUF.getvalue()[-4000:])\n"
        "    except Exception:\n"
        "        pass\n"
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class DatabricksJobsTool:
    """Databricks Jobs + Mosaic AI Model Training dispatcher."""

    def __init__(
        self,
        wc,
        settings: db_client.DatabricksSettings,
        user_email: Optional[str] = None,
        log_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        session: Any = None,
        tool_call_id: Optional[str] = None,
    ):
        self.wc = wc
        self.settings = settings
        self.user_email = user_email
        self.log_callback = log_callback
        self.session = session
        self.tool_call_id = tool_call_id

    # ---- public dispatch -------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        operation = (params.get("operation") or "").lower().strip()
        if not operation:
            return _err("'operation' is required. See tool description.")
        try:
            if operation == "run":
                return await self._run(params)
            if operation == "ps":
                return await self._ps(params)
            if operation == "logs":
                return await self._logs(params)
            if operation == "inspect":
                return await self._inspect(params)
            if operation == "cancel":
                return await self._cancel(params)
            if operation == "scheduled run":
                return await self._scheduled_run(params)
            if operation == "scheduled ps":
                return await self._scheduled_ps(params)
            if operation == "scheduled inspect":
                return await self._scheduled_inspect(params)
            if operation == "scheduled delete":
                return await self._scheduled_delete(params)
            if operation == "scheduled suspend":
                return await self._scheduled_pause(params, pause=True)
            if operation == "scheduled resume":
                return await self._scheduled_pause(params, pause=False)
            return _err(f"Unknown operation: {operation!r}")
        except Exception as e:
            logger.exception("databricks_jobs %s failed", operation)
            return _err(f"{operation} failed: {e}")

    # ---- run / monitor ---------------------------------------------------

    async def _run(self, args: Dict[str, Any]) -> ToolResult:
        kind: KindType = (args.get("kind") or "script").lower()
        if kind == "finetune":
            return await self._run_finetune(args)
        if kind not in ("script", "serverless", "serverless_gpu"):
            return _err(
                f"Unsupported kind: {kind!r}. "
                "Use script | serverless | serverless_gpu | finetune."
            )

        workspace_path = await self._resolve_or_stage_script(
            args, as_notebook=(kind == "serverless_gpu"),
        )
        body = await self._build_submit_body(args, workspace_path, kind)

        resp = await asyncio.to_thread(
            self.wc.api_client.do, "POST", _JOBS_SUBMIT_PATH, body=body,
        )
        run_id = resp.get("run_id")
        if not run_id:
            return _err(f"runs/submit returned no run_id: {resp}")

        if self.session is not None:
            self.session._running_job_ids.add(str(run_id))

        url = _run_url(self.settings.host, run_id)
        await self._emit_state("running", run_id=run_id, url=url)
        await self._log(f"Job submitted: {url}")

        run = await self._wait_for_run(run_id)
        state = run.get("state") or {}
        life = state.get("life_cycle_state") or "UNKNOWN"
        result = state.get("result_state") or ""
        msg = state.get("state_message") or ""

        if self.session is not None:
            self.session._running_job_ids.discard(str(run_id))

        await self._emit_state(life.lower(), run_id=run_id, url=url, result=result)

        log_text = await self._fetch_run_output(run)
        return _ok(
            f"""**Databricks Job ({kind})**

**Run ID:** {run_id}
**Lifecycle:** {life}
**Result:** {result or "—"}
**Message:** {msg or "—"}
**View:** {url}

**Output:**
```
{log_text}
```""",
        )

    async def _build_submit_body(
        self, args: Dict[str, Any], workspace_path: str, kind: KindType,
    ) -> Dict[str, Any]:
        run_name = args.get("run_name") or f"ml-intern-{int(time.time())}"
        params = args.get("script_args") or []
        if not isinstance(params, list):
            params = [str(params)]

        task: Dict[str, Any] = {"task_key": "ml_intern_run"}

        body: Dict[str, Any] = {
            "run_name": run_name,
            "tasks": [task],
        }

        env = _filter_agent_env(args.get("env"))

        if kind == "serverless_gpu":
            # AI Runtime serverless GPU: notebook_task + per-task `compute`
            # selector + environment_version "4". Shape per Databricks docs:
            # https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting
            task["notebook_task"] = {
                "notebook_path": workspace_path,
                "base_parameters": {f"arg{i}": str(p) for i, p in enumerate(params)},
            }
            task["environment_key"] = "ml_intern_env"
            task["compute"] = {
                "hardware_accelerator": args.get("hardware_accelerator", "GPU_1xA10"),
            }
            deps = args.get("dependencies") or []
            env_spec: Dict[str, Any] = {"environment_version": "4"}
            if deps:
                env_spec["dependencies"] = list(deps)
            body["environments"] = [{
                "environment_key": "ml_intern_env",
                "spec": env_spec,
            }]
        elif kind == "serverless":
            task["spark_python_task"] = {
                "python_file": workspace_path,
                "parameters": [str(p) for p in params],
            }
            task["environment_key"] = "ml_intern_env"
            deps = args.get("dependencies") or []
            body["environments"] = [{
                "environment_key": "ml_intern_env",
                "spec": {"client": "1", "dependencies": list(deps)},
            }]
            if env:
                # Serverless tasks don't accept spark_env_vars; the agent must
                # bake env into the script or register secrets via UC.
                logger.info("Dropping env on serverless run (use UC secrets / args).")
        else:  # script
            task["spark_python_task"] = {
                "python_file": workspace_path,
                "parameters": [str(p) for p in params],
            }
            task["new_cluster"] = self._build_cluster(args, env)

        timeout = _parse_timeout(args.get("timeout", "30m"))
        if timeout:
            body["timeout_seconds"] = timeout
        return body

    def _build_cluster(self, args: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
        s = self.settings
        flavor = args.get("hardware_flavor", "cpu-basic")
        node_type = (
            args.get("node_type_id")
            or HARDWARE_FLAVOR_TO_NODE_TYPE.get(flavor)
            or s.default_node_type_id
        )
        spec: Dict[str, Any] = {
            "spark_version": args.get("runtime_version") or s.default_runtime_version,
            "num_workers": int(args.get("num_workers", 0)),
        }
        # Pool wins over explicit node_type_id only when caller didn't override.
        if s.instance_pool_id and not args.get("node_type_id"):
            spec["instance_pool_id"] = s.instance_pool_id
        else:
            spec["node_type_id"] = node_type
        if args.get("driver_node_type_id"):
            spec["driver_node_type_id"] = args["driver_node_type_id"]
        if env:
            spec["spark_env_vars"] = env
        return spec

    async def _resolve_or_stage_script(
        self, args: Dict[str, Any], *, as_notebook: bool = False,
    ) -> str:
        """Return the workspace path the job will execute.

        ``workspace_path`` is trusted as-is. Inline ``script`` content
        stages to ``/Workspace/Users/<user>/ml-intern/<session>/<subdir>/<filename>``,
        where ``<subdir>`` is ``notebooks`` for ``as_notebook=True`` and
        ``files`` otherwise. The split avoids the workspace API's hard
        rejection when re-uploading the same path with a different
        ``object_type`` (FILE vs NOTEBOOK) — the agent commonly reuses the
        default ``train.py`` filename across kinds.

        ``as_notebook=True`` (serverless GPU / AI Runtime path) prepends
        the Databricks notebook source magic plus a stdout-buffering
        wrapper and uploads via the workspace import API as
        ``format=SOURCE``. ``as_notebook=False`` (script / serverless CPU
        path) uses ``format=AUTO`` so the file lands as a plain Python
        file readable by ``spark_python_task``.
        """
        if args.get("workspace_path"):
            return args["workspace_path"]
        script = args.get("script")
        if not script:
            raise ValueError("Provide either `workspace_path` or `script`.")

        filename = args.get("filename") or "train.py"
        if not re.match(r"^[\w.-]+\.py$", filename):
            raise ValueError(f"Invalid filename {filename!r}: must end in .py")

        user_seg = _safe_segment(self.user_email, default="user")
        sid = _safe_segment(
            (getattr(self.session, "session_id", None) or "")[:24],
            default=uuid.uuid4().hex[:12],
        )
        subdir = "notebooks" if as_notebook else "files"
        path = (
            f"/Workspace/Users/{self.user_email or 'user'}/ml-intern/"
            f"{sid}/{subdir}/{filename}"
        )

        if as_notebook:
            # Imported with format=SOURCE + language=PYTHON; Databricks keeps
            # the full filename (incl. ``.py``) as the notebook's workspace
            # path. notebook_task.notebook_path matches that filename verbatim.
            # Pre-install prelude (datasets/trl/peft/accelerate) runs first so
            # the user script doesn't have to embed pip calls — PTB-smoke
            # run-D ate 4 job round-trips on this exact discovery loop.
            # Opt-out via args["pre_install"] = False for users that want a
            # pristine env or have already pinned their own deps.
            pre_install = args.get("pre_install", True)
            install_block = (
                _pre_install_prelude(_SERVERLESS_GPU_PRE_INSTALL)
                if pre_install else ""
            )
            content = (
                "# Databricks notebook source\n"
                + _NOTEBOOK_STDOUT_WRAPPER_PRELUDE
                + install_block
                + _wrap_user_script_with_stdout_capture(script)
            )
            await self._upload_workspace_notebook(path, content)
            return path

        await self._upload_workspace_file(path, script)
        return path

    def _clear_if_wrong_type(self, ws_path: str, wanted: str) -> None:
        """Delete the Workspace asset at ``ws_path`` when its ``object_type``
        doesn't match ``wanted`` (one of ``"FILE"``, ``"NOTEBOOK"``).

        ``/api/2.0/workspace/import`` with ``overwrite=True`` only overwrites
        objects of the same type — re-uploading ``train.py`` as a FILE when a
        prior call landed it as a NOTEBOOK fails with
        ``InvalidParameterValue: type mismatch``. Probe the path, delete on
        mismatch, swallow not-found / probe failures (the caller's import will
        surface any real error).
        """
        try:
            existing = self.wc.workspace.get_status(ws_path)
        except Exception as e:
            logger.debug("get_status(%s) suppressed: %s", ws_path, e)
            return
        existing_type = getattr(existing, "object_type", None)
        # SDK enum (.name) or plain string — both end up uppercased.
        existing_name = (
            getattr(existing_type, "name", None) or str(existing_type or "")
        ).upper()
        if not existing_name or wanted in existing_name:
            return
        try:
            self.wc.workspace.delete(ws_path, recursive=False)
            logger.debug(
                "Cleared %s at %s before re-upload as %s",
                existing_name, ws_path, wanted,
            )
        except Exception as e:
            logger.debug("delete(%s) suppressed: %s", ws_path, e)

    async def _upload_workspace_file(self, path: str, content: str) -> None:
        """Upload a plain Python file to a Workspace path.

        Threads the needle between three API surfaces that mostly don't
        cooperate:

        - ``wc.files.upload`` only accepts ``/Volumes/...`` paths today.
        - ``wc.workspace.upload`` always converts ``.py`` to a notebook
          (``spark_python_task`` then can't ``open()`` it → ``OSError 95``).
        - ``/api/2.0/workspace/import`` with ``format=AUTO`` lets us write
          a raw file when the content has no ``# Databricks notebook source``
          magic header — exactly what we want.

        We strip a workspace-relative path (drop ``/Workspace`` prefix) and
        send base64 content. The parent dir is created with mkdirs first
        because workspace/import doesn't auto-create parents. If a NOTEBOOK
        already lives at this path (e.g. from a prior serverless_gpu run that
        reused the default ``train.py``) we delete it first to dodge the
        ``type mismatch`` rejection.
        """
        import base64

        data = content.encode("utf-8") if isinstance(content, str) else content
        # workspace.* expects paths *without* the /Workspace prefix.
        ws_path = path[len("/Workspace"):] if path.startswith("/Workspace") else path
        parent = ws_path.rsplit("/", 1)[0]

        def _do():
            try:
                self.wc.workspace.mkdirs(parent)
            except Exception as e:
                logger.debug("mkdirs(%s) suppressed: %s", parent, e)
            self._clear_if_wrong_type(ws_path, wanted="FILE")
            self.wc.api_client.do(
                "POST", "/api/2.0/workspace/import",
                body={
                    "path": ws_path,
                    "format": "AUTO",
                    "overwrite": True,
                    "content": base64.b64encode(data).decode("ascii"),
                },
            )

        await asyncio.to_thread(_do)

    async def _upload_workspace_notebook(self, path: str, content: str) -> None:
        """Upload a Python notebook source file (serverless GPU path).

        Uses ``format=SOURCE`` + ``language=PYTHON`` so Databricks lands the
        file as a notebook bound to ``notebook_task.notebook_path``. The path
        is stripped of its ``.py`` suffix when referenced from the task body.
        Pre-deletes any FILE asset already at the path so cross-kind reuse
        doesn't trip ``InvalidParameterValue: type mismatch``.
        """
        import base64

        data = content.encode("utf-8") if isinstance(content, str) else content
        ws_path = path[len("/Workspace"):] if path.startswith("/Workspace") else path
        parent = ws_path.rsplit("/", 1)[0]

        def _do():
            try:
                self.wc.workspace.mkdirs(parent)
            except Exception as e:
                logger.debug("mkdirs(%s) suppressed: %s", parent, e)
            self._clear_if_wrong_type(ws_path, wanted="NOTEBOOK")
            self.wc.api_client.do(
                "POST", "/api/2.0/workspace/import",
                body={
                    "path": ws_path,
                    "format": "SOURCE",
                    "language": "PYTHON",
                    "overwrite": True,
                    "content": base64.b64encode(data).decode("ascii"),
                },
            )

        await asyncio.to_thread(_do)

    async def _wait_for_run(self, run_id: int | str) -> Dict[str, Any]:
        """Poll runs/get until terminal lifecycle. Cancels on session abort."""
        delay = 5
        while True:
            if self.session is not None and self.session.is_cancelled:
                try:
                    await asyncio.to_thread(
                        self.wc.api_client.do, "POST", _JOBS_RUNS_CANCEL,
                        body={"run_id": run_id},
                    )
                except Exception as e:
                    logger.warning("cancel_run on abort failed: %s", e)
                # Fall through to fetch terminal state.

            try:
                run = await asyncio.to_thread(
                    self.wc.api_client.do, "GET", _JOBS_RUNS_GET,
                    query={"run_id": run_id},
                )
            except Exception as e:
                logger.warning("runs/get failed (retry): %s", e)
                await asyncio.sleep(delay)
                continue

            life = (run.get("state") or {}).get("life_cycle_state") or "PENDING"
            if life in _TERMINAL_LIFECYCLES:
                return run
            await asyncio.sleep(delay)
            delay = min(delay + 5, 30)

    async def _fetch_run_output(self, run: Dict[str, Any]) -> str:
        """Best-effort task driver log fetch.

        ``runs/get-output`` returns task driver logs (truncated to 5MB) for
        notebook tasks unconditionally and for python tasks when the cluster
        had a log delivery destination configured.

        Some tasks have multiple entries when Databricks retries an attempt
        (e.g. dep install fails on cold env, succeeds on retry). Prefer the
        most-recent SUCCESS-state entry; otherwise fall through.
        """
        tasks = run.get("tasks") or []
        if not tasks:
            return (run.get("state") or {}).get("state_message") or "(no output)"

        def _attempt_score(t: Dict[str, Any]) -> tuple[int, int]:
            res = (t.get("state") or {}).get("result_state") or ""
            success = 1 if res == "SUCCESS" else 0
            return (success, t.get("start_time") or 0)

        sorted_tasks = sorted(tasks, key=_attempt_score, reverse=True)

        for task in sorted_tasks:
            task_run_id = task.get("run_id")
            if not task_run_id:
                continue
            try:
                out = await asyncio.to_thread(
                    self.wc.api_client.do, "GET", _JOBS_RUN_OUTPUT,
                    query={"run_id": task_run_id},
                )
            except Exception as e:
                logger.debug("get-output(%s) suppressed: %s", task_run_id, e)
                continue
            for key in ("notebook_output", "logs", "error_trace", "error"):
                v = out.get(key)
                if isinstance(v, dict):
                    v = v.get("result")
                if v:
                    return _strip_ansi(str(v))

        return (run.get("state") or {}).get("state_message") or "(no output)"

    # ---- ps / inspect / cancel / logs ------------------------------------

    async def _ps(self, args: Dict[str, Any]) -> ToolResult:
        active_only = not args.get("all", False)
        runs = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_RUNS_LIST,
            query={"active_only": "true"} if active_only else {},
        )
        items = runs.get("runs") or []
        if not items:
            tail = "" if not active_only else " Use `{\"operation\":\"ps\",\"all\":true}` for completed runs."
            return _ok(f"No runs found.{tail}")
        rows = "\n".join(_fmt_run_row(r, self.settings.host) for r in items)
        return _ok(
            f"**Runs ({len(items)}):**\n\n| RUN ID | NAME | LIFECYCLE | RESULT | URL |\n|---|---|---|---|---|\n{rows}",
            n=len(items),
        )

    async def _logs(self, args: Dict[str, Any]) -> ToolResult:
        run_id = args.get("run_id")
        if not run_id:
            return _err("run_id is required")
        run = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_RUNS_GET, query={"run_id": run_id},
        )
        text = await self._fetch_run_output(run)
        return _ok(f"**Logs for run {run_id}:**\n\n```\n{text}\n```")

    async def _inspect(self, args: Dict[str, Any]) -> ToolResult:
        run_id = args.get("run_id")
        if not run_id:
            return _err("run_id is required")
        run = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_RUNS_GET, query={"run_id": run_id},
        )
        return _ok(
            f"**Run {run_id}:**\n\n```json\n{_pretty(run)}\n```\n\nURL: {_run_url(self.settings.host, run_id)}",
        )

    async def _cancel(self, args: Dict[str, Any]) -> ToolResult:
        run_id = args.get("run_id")
        if not run_id:
            return _err("run_id is required")
        await asyncio.to_thread(
            self.wc.api_client.do, "POST", _JOBS_RUNS_CANCEL, body={"run_id": run_id},
        )
        return _ok(f"Cancel requested for run {run_id}.")

    # ---- scheduled jobs --------------------------------------------------

    async def _scheduled_run(self, args: Dict[str, Any]) -> ToolResult:
        cron = args.get("schedule")
        if not cron:
            return _err("schedule is required (cron expression e.g. '0 0 9 * * ?')")
        kind: KindType = (args.get("kind") or "script").lower()
        if kind == "finetune":
            return _err("Scheduled finetune runs not supported. Submit one-off finetune via `run`.")
        workspace_path = await self._resolve_or_stage_script(args)
        submit = await self._build_submit_body(args, workspace_path, kind)

        body = {
            "name": args.get("run_name") or f"ml-intern-sched-{int(time.time())}",
            "tasks": submit["tasks"],
            "schedule": {
                "quartz_cron_expression": cron,
                "timezone_id": args.get("timezone", "UTC"),
                "pause_status": "UNPAUSED",
            },
        }
        if "environments" in submit:
            body["environments"] = submit["environments"]
        if "timeout_seconds" in submit:
            body["timeout_seconds"] = submit["timeout_seconds"]

        resp = await asyncio.to_thread(
            self.wc.api_client.do, "POST", _JOBS_CREATE, body=body,
        )
        job_id = resp.get("job_id")
        return _ok(f"Scheduled job created: job_id={job_id}, schedule={cron!r}.")

    async def _scheduled_ps(self, args: Dict[str, Any]) -> ToolResult:
        resp = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_LIST, query={"limit": 50},
        )
        jobs = resp.get("jobs") or []
        # Only show jobs with a schedule.
        sched = [j for j in jobs if (j.get("settings") or {}).get("schedule")]
        if not args.get("all"):
            sched = [j for j in sched if (j["settings"]["schedule"].get("pause_status") != "PAUSED")]
        if not sched:
            return _ok("No active scheduled jobs.")
        rows = []
        for j in sched:
            s = j["settings"]
            sc = s.get("schedule", {})
            rows.append(
                f"| {j.get('job_id')} | {s.get('name','')} | {sc.get('quartz_cron_expression','')} | {sc.get('pause_status','')} |"
            )
        body = "\n".join(rows)
        return _ok(
            f"**Scheduled jobs ({len(sched)}):**\n\n| JOB ID | NAME | CRON | STATUS |\n|---|---|---|---|\n{body}",
            n=len(sched),
        )

    async def _scheduled_inspect(self, args: Dict[str, Any]) -> ToolResult:
        job_id = args.get("scheduled_job_id") or args.get("job_id")
        if not job_id:
            return _err("scheduled_job_id is required")
        resp = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_GET, query={"job_id": job_id},
        )
        return _ok(f"**Scheduled job {job_id}:**\n\n```json\n{_pretty(resp)}\n```")

    async def _scheduled_delete(self, args: Dict[str, Any]) -> ToolResult:
        job_id = args.get("scheduled_job_id") or args.get("job_id")
        if not job_id:
            return _err("scheduled_job_id is required")
        await asyncio.to_thread(
            self.wc.api_client.do, "POST", _JOBS_DELETE, body={"job_id": job_id},
        )
        return _ok(f"Scheduled job {job_id} deleted.")

    async def _scheduled_pause(self, args: Dict[str, Any], pause: bool) -> ToolResult:
        job_id = args.get("scheduled_job_id") or args.get("job_id")
        if not job_id:
            return _err("scheduled_job_id is required")
        # Read current settings, rewrite schedule.pause_status, send via update.
        cur = await asyncio.to_thread(
            self.wc.api_client.do, "GET", _JOBS_GET, query={"job_id": job_id},
        )
        settings = (cur.get("settings") or {}).copy()
        sched = (settings.get("schedule") or {}).copy()
        if not sched:
            return _err(f"Job {job_id} has no schedule.")
        sched["pause_status"] = "PAUSED" if pause else "UNPAUSED"
        settings["schedule"] = sched
        await asyncio.to_thread(
            self.wc.api_client.do, "POST", _JOBS_UPDATE,
            body={"job_id": job_id, "new_settings": settings},
        )
        verb = "suspended" if pause else "resumed"
        return _ok(f"Scheduled job {job_id} {verb}.")

    # ---- finetune (Mosaic AI Model Training) -----------------------------

    async def _run_finetune(self, args: Dict[str, Any]) -> ToolResult:
        required = ["model", "train_data_path"]
        missing = [k for k in required if not args.get(k)]
        if missing:
            return _err(f"finetune missing required: {missing}")

        register_to = args.get("register_to")
        if not register_to:
            register_to = f"{self.settings.full_schema}.ml_intern_finetune_{int(time.time())}"

        payload: Dict[str, Any] = {
            "model": args["model"],
            "train_data_path": args["train_data_path"],
            "task_type": args.get("task_type", "INSTRUCTION_FINETUNE"),
            "register_to": register_to,
            "experiment_path": args.get("experiment_path") or self.settings.experiment_path,
        }
        for opt in (
            "eval_data_path",
            "training_duration",
            "learning_rate",
            "context_length",
            "custom_weights_path",
            "data_prep_cluster_id",
            "validate_inputs",
        ):
            if args.get(opt) is not None:
                payload[opt] = args[opt]

        resp = await asyncio.to_thread(
            self.wc.api_client.do, "POST", _FINETUNE_API_PATH, body=payload,
        )
        run_name = resp.get("name") or resp.get("run_name") or resp.get("id")
        mlflow_run_id = resp.get("mlflow_run_id") or resp.get("experiment_run_id")
        exp_url = _experiment_url(self.settings.host, payload["experiment_path"])
        msg = (
            f"**Mosaic AI Model Training run submitted**\n\n"
            f"**Name:** {run_name}\n"
            f"**Model:** {payload['model']}\n"
            f"**Task:** {payload['task_type']}\n"
            f"**Train data:** {payload['train_data_path']}\n"
            f"**Register to:** {register_to}\n"
            f"**Experiment:** {exp_url}\n"
        )
        if mlflow_run_id:
            msg += f"**MLflow run:** {mlflow_run_id}\n"
        msg += (
            "\nMosaic AI Model Training is asynchronous; poll status via "
            f"`{{\"operation\":\"inspect\",\"kind\":\"finetune\",\"run_name\":\"{run_name}\"}}`."
        )
        return _ok(msg)

    # ---- helpers ---------------------------------------------------------

    async def _emit_state(self, state: str, **data: Any) -> None:
        if not (self.session and self.tool_call_id):
            return
        await self.session.send_event(Event(
            event_type="tool_state_change",
            data={
                "tool_call_id": self.tool_call_id,
                "tool": "databricks_jobs",
                "state": state,
                **data,
            },
        ))

    async def _log(self, line: str) -> None:
        if self.log_callback:
            await self.log_callback(line)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _pretty(d: Any) -> str:
    import json
    return json.dumps(d, indent=2, default=str)


def _fmt_run_row(r: Dict[str, Any], host: str) -> str:
    rid = r.get("run_id", "")
    name = (r.get("run_name") or "")[:30]
    state = r.get("state") or {}
    life = state.get("life_cycle_state", "")
    result = state.get("result_state", "")
    url = _run_url(host, rid, r.get("job_id"))
    return f"| {rid} | {name} | {life} | {result} | {url} |"


# ---------------------------------------------------------------------------
# Tool spec + handler
# ---------------------------------------------------------------------------

DATABRICKS_JOBS_TOOL_SPEC = {
    "name": "databricks_jobs",
    "description": (
        "Submit and manage compute jobs on Databricks. Three execution kinds:\n\n"
        "1. `kind=\"finetune\"` — Mosaic AI Model Training. Foundation-model fine-tune that registers "
        "the resulting model into Unity Catalog. Required: `model`, `train_data_path` (UC table or volume). "
        "Optional: `eval_data_path`, `task_type` (INSTRUCTION_FINETUNE | CONTINUED_PRETRAIN | CHAT_COMPLETION), "
        "`training_duration` (e.g. '5ep'), `learning_rate`, `register_to` (UC name; auto-named if omitted).\n\n"
        "2. `kind=\"script\"` (default) — Databricks Job with a GPU/CPU `new_cluster`. Provide `script` "
        "(inline Python — staged to Workspace Files automatically) OR `workspace_path` (existing file). "
        "Hardware via `hardware_flavor` (HF aliases mapped to AWS node types) or explicit `node_type_id`. "
        "If `ML_INTERN_INSTANCE_POOL_ID` is configured, runs default to that pool.\n\n"
        "3. `kind=\"serverless\"` — Serverless compute. Cheapest path for small CPU work. No cluster spec; "
        "deps declared via `dependencies`. Env vars not supported (use UC secrets).\n\n"
        "4. `kind=\"serverless_gpu\"` — AI Runtime serverless GPU (NVIDIA A10G default). "
        "Common ML deps (datasets, trl, peft, accelerate) are pre-installed via a prelude — "
        "your script can `import datasets` etc. directly. Set `pre_install=False` to opt out "
        "if you've pinned your own versions or want a clean env.\n\n"
        "BEFORE submitting training jobs:\n"
        "- Validate dataset format via `uc_inspect_dataset` (UC) or matching tool.\n"
        "- Models MUST be registered to Unity Catalog (`<catalog>.<schema>.<name>`). "
        "Without `register_to`, Mosaic AI auto-names but you lose discoverability.\n"
        "- For raw `script` runs, write training code that calls `mlflow.set_registry_uri('databricks-uc')` "
        "and `mlflow.<framework>.log_model(..., registered_model_name='ml_intern.agent.<name>')` so the "
        "trained artifact survives cluster termination.\n\n"
        "BATCH/ABLATION: submit ONE first, confirm it starts, THEN submit the rest. "
        "Cluster startup can take 3-5 min — don't fan out broken specs.\n\n"
        "Secrets: pass as values via dynamic refs only — `{\"OPENAI_API_KEY\": \"{{secrets/ml-intern/openai}}\"}`. "
        "Plaintext `DATABRICKS_*` / cloud credentials in `env` are silently dropped.\n\n"
        "Operations: run, ps, logs, inspect, cancel, scheduled run/ps/inspect/delete/suspend/resume.\n\n"
        f"Hardware aliases: {', '.join(HARDWARE_FLAVOR_TO_NODE_TYPE.keys())}. "
        "Override with `node_type_id` for non-AWS workspaces.\n\n"
        "Examples:\n"
        "Finetune: {\"operation\":\"run\",\"kind\":\"finetune\",\"model\":\"meta-llama/Llama-3.2-1B\","
        "\"train_data_path\":\"ml_intern.agent.sft_train\",\"task_type\":\"INSTRUCTION_FINETUNE\","
        "\"training_duration\":\"3ep\",\"register_to\":\"ml_intern.agent.llama_sft_v1\"}\n"
        "Script: {\"operation\":\"run\",\"kind\":\"script\",\"script\":\"import mlflow; ...\","
        "\"hardware_flavor\":\"a10g-large\",\"timeout\":\"4h\"}\n"
        "Monitor: {\"operation\":\"ps\"}, {\"operation\":\"logs\",\"run_id\":12345}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "run", "ps", "logs", "inspect", "cancel",
                    "scheduled run", "scheduled ps", "scheduled inspect",
                    "scheduled delete", "scheduled suspend", "scheduled resume",
                ],
                "description": "Operation to execute.",
            },
            "kind": {
                "type": "string",
                "enum": ["finetune", "script", "serverless", "serverless_gpu"],
                "description": "Execution backend. Defaults to 'script'.",
            },
            "script": {
                "type": "string",
                "description": (
                    "Inline Python to run. Staged to Workspace Files at "
                    "/Workspace/Users/<user>/ml-intern/<session>/<filename>. Mutually exclusive with workspace_path."
                ),
            },
            "workspace_path": {
                "type": "string",
                "description": "Existing Workspace Files path (e.g. /Workspace/Users/me/script.py). Mutually exclusive with script.",
            },
            "filename": {
                "type": "string",
                "description": "Override staged filename. Default: train.py.",
            },
            "script_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Args appended to the python file invocation.",
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pip deps for serverless kind. Ignored on script kind (use a init_script or %pip in code).",
            },
            "hardware_flavor": {
                "type": "string",
                "description": (
                    "HF-style alias mapped to AWS node types. "
                    "1-3B: t4-small/a10g-small. 7-13B: a10g-large. 30B+: a100-large. CPU: cpu-basic / cpu-upgrade."
                ),
            },
            "node_type_id": {
                "type": "string",
                "description": "Explicit Databricks node type (overrides hardware_flavor).",
            },
            "hardware_accelerator": {
                "type": "string",
                "description": (
                    "(serverless_gpu) AI Runtime accelerator selector. "
                    "Per HardwareAcceleratorType enum: GPU_1xA10 (1×A10, 24GB), "
                    "GPU_8xH100 (8×H100, distributed). Default: GPU_1xA10."
                ),
            },
            "driver_node_type_id": {
                "type": "string",
                "description": "Optional driver node type for multi-worker clusters.",
            },
            "num_workers": {
                "type": "integer",
                "description": "Worker count. 0 = single-node (recommended for fine-tune).",
            },
            "runtime_version": {
                "type": "string",
                "description": "Databricks runtime spark_version. Default: ML GPU runtime.",
            },
            "timeout": {
                "type": "string",
                "description": "Max runtime (e.g. '30m', '4h', '12h'). Training jobs need >2h.",
            },
            "env": {
                "type": "object",
                "description": (
                    "Cluster spark env vars. Use {{secrets/scope/key}} for secrets. "
                    "DATABRICKS_TOKEN and cloud creds are dropped."
                ),
            },
            "run_name": {
                "type": "string",
                "description": "Display name for the run. Auto-generated if omitted.",
            },
            "run_id": {
                "type": ["integer", "string"],
                "description": "Required for: logs, inspect, cancel.",
            },
            "scheduled_job_id": {
                "type": ["integer", "string"],
                "description": "Required for: scheduled inspect/delete/suspend/resume.",
            },
            "schedule": {
                "type": "string",
                "description": "Quartz cron expression for scheduled run (e.g. '0 0 9 * * ?').",
            },
            "timezone": {
                "type": "string",
                "description": "TZ id for the schedule. Default: UTC.",
            },
            "all": {
                "type": "boolean",
                "description": "ps / scheduled ps: include completed/paused entries.",
            },
            # Finetune-specific
            "model": {
                "type": "string",
                "description": "(finetune) Base model id, e.g. 'meta-llama/Llama-3.2-1B'.",
            },
            "train_data_path": {
                "type": "string",
                "description": "(finetune) UC table or volume path with training rows.",
            },
            "eval_data_path": {
                "type": "string",
                "description": "(finetune) UC eval split.",
            },
            "task_type": {
                "type": "string",
                "enum": ["INSTRUCTION_FINETUNE", "CONTINUED_PRETRAIN", "CHAT_COMPLETION"],
                "description": "(finetune) Training task type.",
            },
            "training_duration": {
                "type": "string",
                "description": "(finetune) e.g. '5ep' or '1000ba'.",
            },
            "learning_rate": {
                "type": "number",
                "description": "(finetune) Override default LR.",
            },
            "context_length": {
                "type": "integer",
                "description": "(finetune) Max sequence length.",
            },
            "register_to": {
                "type": "string",
                "description": "(finetune) UC name <catalog>.<schema>.<name>. Auto-generated if omitted.",
            },
            "custom_weights_path": {
                "type": "string",
                "description": "(finetune) Volume path of custom base weights.",
            },
            "experiment_path": {
                "type": "string",
                "description": "(finetune) MLflow experiment path. Defaults to ML_INTERN_EXPERIMENT_PATH.",
            },
        },
        "required": ["operation"],
    },
}


async def databricks_jobs_handler(
    arguments: Dict[str, Any], session: Any = None, tool_call_id: str | None = None,
) -> tuple[str, bool]:
    """Tool router entrypoint."""
    try:
        async def log_callback(line: str):
            if session:
                await session.send_event(
                    Event(event_type="tool_log", data={"tool": "databricks_jobs", "log": line})
                )

        # Pull a script from sandbox if the agent passed a sandbox path.
        script = arguments.get("script", "")
        sandbox = getattr(session, "sandbox", None) if session else None
        if sandbox and script and "\n" not in script and not arguments.get("workspace_path"):
            from agent.tools.sandbox_tool import resolve_sandbox_script
            content, error = await resolve_sandbox_script(sandbox, script)
            if error:
                return error, False
            if content:
                arguments = {**arguments, "script": content}

        # Resolve settings + WC. Prefer OBO when the backend stashed a token.
        from agent.config import load_config
        cfg_path = os.environ.get(
            "ML_INTERN_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
        )
        cfg = load_config(cfg_path) if (session is None or not getattr(session, "config", None)) else session.config
        settings = db_client.resolve_settings(cfg)

        user_token = getattr(session, "databricks_user_token", None) if session else None
        user_email = getattr(session, "user_email", None) if session else None
        if user_token and settings.host:
            wc = db_client.get_workspace_client_for_user(user_token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)

        # Best-effort: pull current user email if not stashed (for stage path).
        if not user_email:
            try:
                me = await asyncio.to_thread(wc.current_user.me)
                user_email = me.user_name or (me.emails[0].value if me.emails else None)
            except Exception:
                user_email = None

        tool = DatabricksJobsTool(
            wc=wc,
            settings=settings,
            user_email=user_email,
            log_callback=log_callback if session else None,
            session=session,
            tool_call_id=tool_call_id,
        )
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("databricks_jobs handler crashed")
        return f"Error executing databricks_jobs: {e}", False
