"""System prompt source: MLflow Prompt Registry first, YAML fallback.

The agent's system prompt lives in two places:

1. **MLflow Prompt Registry** under ``ml_intern.agent.system_prompt`` —
   the source of truth in production. Versions are tied to bundle deploys
   so a rollback is one ``databricks bundle deploy --target prod
   --prompt-version <n>`` away.

2. **YAML at ``agent/prompts/system_prompt_v3.yaml``** — the fallback used
   in local CLI runs, unit tests, and any environment where the registry
   isn't reachable. Always shipped with the package so the agent has a
   prompt to start with even when the workspace is unreachable.

The registry takes precedence so prompt iteration doesn't require a code
deploy. ``load_system_prompt(name, version=...)`` tries the registry; on
any failure it falls back to the YAML and logs a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_YAML = (
    Path(__file__).resolve().parent.parent / "prompts" / "system_prompt_v3.yaml"
)


def _extract_template(prompt) -> Optional[str]:
    """Pull the rendered template string off whatever the loader returned."""
    if prompt is None:
        return None
    text = getattr(prompt, "template", None)
    if text:
        return text
    s = str(prompt)
    return s or None


def _try_load_uri(loader, uri: str) -> Optional[str]:
    """Try one ``prompts:/...`` URI through the loader. Returns None on any
    error so the caller can keep walking the resolution chain."""
    try:
        return _extract_template(loader(uri))
    except Exception as e:
        logger.debug("prompt load %s failed: %s", uri, e)
        return None


def _resolve_latest_version(client, name: str) -> Optional[str]:
    """Find the highest numeric version registered for ``name``.

    MLflow's URI parser tries ``int(version)`` before resolving — passing the
    literal string ``"latest"`` raises ``ValueError`` on builds that haven't
    learned the alias yet. We sidestep by listing versions and picking max.
    """
    try:
        find = (
            getattr(client, "search_prompt_versions", None)
            or getattr(client, "search_model_versions", None)
        )
        if not find:
            return None
        versions = list(find(f"name='{name}'"))
        if not versions:
            return None
        latest = max(
            versions,
            key=lambda v: int(getattr(v, "version", 0) or 0),
        )
        return str(latest.version)
    except Exception as e:
        logger.debug("prompt version search for %s failed: %s", name, e)
        return None


def load_from_registry(name: str, version: str | int | None = None) -> Optional[str]:
    """Pull the named prompt from the MLflow Prompt Registry.

    ``version`` may be a version number (str / int), an alias ("production",
    "champion"), or None for the latest. Returns the rendered template
    string, or None on any failure (network, auth, not-found).

    When ``version is None`` we walk a fallback chain: ``@production`` →
    ``@champion`` → highest numeric version found via the MLflow client.
    The literal URI ``prompts:/<name>/latest`` is *not* used because some
    MLflow builds raise ``ValueError: invalid literal for int() with base 10:
    'latest'`` when parsing it.
    """
    try:
        import mlflow
    except Exception as e:
        logger.debug("mlflow import failed: %s", e)
        return None

    try:
        mlflow.set_registry_uri("databricks-uc")
    except Exception as e:
        logger.debug("set_registry_uri failed: %s", e)

    # MLflow 2.21+ exposes ``mlflow.genai.load_prompt``; older builds have
    # ``mlflow.load_prompt``. Try both.
    loader = None
    try:
        from mlflow.genai import load_prompt as loader  # type: ignore
    except Exception:
        loader = getattr(mlflow, "load_prompt", None)
    if loader is None:
        logger.info("MLflow build has no load_prompt(); using YAML fallback.")
        return None

    if version is not None:
        return _try_load_uri(loader, f"prompts:/{name}/{version}")

    # Alias-first: most workspaces tag a stable version with @production.
    for alias in ("production", "champion"):
        text = _try_load_uri(loader, f"prompts:/{name}@{alias}")
        if text:
            return text

    # Fall back to the highest numeric version registered.
    try:
        client = mlflow.MlflowClient()
    except Exception as e:
        logger.debug("MlflowClient init failed: %s", e)
        return None

    latest = _resolve_latest_version(client, name)
    if latest is None:
        # Not registered (or registry unreachable). Caller falls back to
        # the bundled YAML — that's the documented contract.
        logger.debug(
            "Prompt %s not found in registry — using YAML fallback.", name,
        )
        return None
    return _try_load_uri(loader, f"prompts:/{name}/{latest}")


def load_from_yaml(path: Path | str | None = None) -> str:
    """Load the system_prompt template from the bundled YAML."""
    p = Path(path) if path else _DEFAULT_YAML
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return data.get("system_prompt", "")


def load_system_prompt(
    name: str,
    version: str | int | None = None,
    yaml_path: Path | str | None = None,
) -> str:
    """Resolve the system prompt, preferring the registry."""
    text = load_from_registry(name, version=version)
    if text:
        return text
    return load_from_yaml(yaml_path)
