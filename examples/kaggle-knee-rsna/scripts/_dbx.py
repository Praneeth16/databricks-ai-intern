#!/usr/bin/env python3
"""Shared helpers for submitting a script to Databricks serverless compute.

Two lessons from earlier runs in this repo are baked in. The run poller dies on a
transient 60 second HTTP read timeout while the job itself keeps going and succeeds,
so run state is always confirmed through /api/2.2/jobs/runs/get rather than trusted
from a stream. And a notebook task only surfaces output through
dbutils.notebook.exit, so the wrapper captures stdout and returns its tail.
"""
from __future__ import annotations

import os
import pathlib
import textwrap
import time
import uuid

CATALOG = os.environ.get("KNEE_UC_CATALOG", "serverless_lakebase_praneeth_catalog")
SCHEMA = os.environ.get("KNEE_UC_SCHEMA", "ml_intern_test")
VOLUME = os.environ.get("KNEE_UC_VOLUME", "scratch")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
WORK = f"{VOLUME_ROOT}/rsna_knee"

STDOUT_PRELUDE = '''
import sys as _s, io as _io
_BUF = _io.StringIO()
class _Tee:
    def __init__(self, *st): self._st = st
    def write(self, b):
        for s in self._st:
            try: s.write(b)
            except Exception: pass
        return len(b) if isinstance(b, str) else 0
    def flush(self):
        for s in self._st:
            try: s.flush()
            except Exception: pass
    def isatty(self): return False
_s.stdout = _Tee(_s.__stdout__, _BUF)
_s.stderr = _Tee(_s.__stderr__, _BUF)
'''


def workspace_client():
    """Always go through the SDK's resolved auth chain, never a hand-built client."""
    from databricks.sdk import WorkspaceClient

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    return WorkspaceClient(profile=profile) if profile else WorkspaceClient()


def wrap(script: str) -> str:
    """Validate, then make the script's stdout tail readable from the run output."""
    import ast

    ast.parse(script)  # a syntax error here would otherwise cost a full job run
    return (
        "# Databricks notebook source\n"
        + STDOUT_PRELUDE
        + "\ntry:\n"
        + textwrap.indent(script, "    ")
        + "\nexcept BaseException as _e:\n"
        "    try: dbutils.notebook.exit(_BUF.getvalue()[-4000:] + '\\nERROR: ' + repr(_e))\n"
        "    except Exception: pass\n"
        "    raise\n"
        "else:\n"
        "    try: dbutils.notebook.exit(_BUF.getvalue()[-4000:])\n"
        "    except Exception: pass\n"
    )


def stage(wc, script: str, name: str) -> str:
    """Put the notebook in Workspace Files and return its path."""
    from databricks.sdk.service.workspace import ImportFormat, Language

    me = wc.current_user.me().user_name
    folder = f"/Workspace/Users/{me}/rsna-knee"
    wc.workspace.mkdirs(folder)
    path = f"{folder}/{name}"
    wc.workspace.upload(path, wrap(script).encode(), format=ImportFormat.SOURCE,
                        language=Language.PYTHON, overwrite=True)
    return path


def submit(wc, script: str, *, name: str | None = None, gpu: str | None = None,
           deps: list[str] | None = None, timeout_min: int = 120,
           secret_env: dict[str, str] | None = None) -> dict:
    """Run `script` on serverless compute. Pass `gpu` for a GPU tier such as GPU_1xA10.

    `secret_env` maps an environment variable name to "scope/key". Values are sent as
    Databricks dynamic secret references, so the secret is never in the job payload.
    """
    name = name or f"knee_{uuid.uuid4().hex[:8]}.py"
    path = stage(wc, script, name)

    task: dict = {"task_key": "main", "notebook_task": {"notebook_path": path,
                                                        "source": "WORKSPACE"}}
    if secret_env:
        task["notebook_task"]["base_parameters"] = {
            k: "{{secrets/" + v + "}}" for k, v in secret_env.items()
        }
    if gpu:
        task["environment_key"] = "gpuenv"
        env = {"environment_key": "gpuenv",
               "spec": {"client": "4", "dependencies": deps or []}}
        task["compute"] = {"gpu_node_type": {"gpu_type": gpu}}
        envs = [env]
    else:
        task["environment_key"] = "cpuenv"
        envs = [{"environment_key": "cpuenv",
                 "spec": {"client": "4", "dependencies": deps or []}}]

    body = {"run_name": name, "tasks": [task], "environments": envs,
            "timeout_seconds": timeout_min * 60}
    resp = wc.api_client.do("POST", "/api/2.2/jobs/runs/submit", body=body)
    return {"run_id": resp["run_id"], "notebook_path": path}


def wait(wc, run_id: int, poll_s: int = 30, timeout_min: int = 180) -> dict:
    """Poll runs/get. Transient read timeouts are retried rather than treated as failure."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        try:
            r = wc.api_client.do("GET", "/api/2.2/jobs/runs/get", query={"run_id": run_id})
        except Exception as e:
            print(f"  poll error, retrying: {type(e).__name__}")
            time.sleep(poll_s)
            continue
        state = r.get("status", {}).get("state") or r.get("state", {}).get("life_cycle_state")
        print(f"  run {run_id}: {state}", flush=True)
        if state in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR", "SUCCEEDED", "FAILED"}:
            return r
        time.sleep(poll_s)
    raise TimeoutError(f"run {run_id} still going after {timeout_min} minutes")


def run_output(wc, run_id: int) -> str:
    r = wc.api_client.do("GET", "/api/2.2/jobs/runs/get", query={"run_id": run_id})
    tasks = r.get("tasks") or []
    if not tasks:
        return ""
    out = wc.api_client.do("GET", "/api/2.2/jobs/runs/get-output",
                           query={"run_id": tasks[0]["run_id"]})
    return out.get("notebook_output", {}).get("result", "") or out.get("error", "")


def upload_file(wc, local: str | pathlib.Path, remote: str) -> None:
    with open(local, "rb") as fh:
        wc.files.upload(remote, fh, overwrite=True)
