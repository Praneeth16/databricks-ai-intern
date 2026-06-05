"""Register the bundled system prompt into MLflow's Prompt Registry.

Run on every bundle deploy (or manually) so production sessions resolve
``ml_intern.agent.system_prompt`` to the current YAML rather than the local
fallback. The agent's prompt loader prefers the registry; this script makes
sure the registry actually has the latest text.

Usage:
    python scripts/register_prompt.py                              # default name + YAML
    python scripts/register_prompt.py --name custom.prompt.id
    python scripts/register_prompt.py --file path/to/prompt.yaml
    python scripts/register_prompt.py --commit-message "tighten OOM advice"

Auth resolves via the standard Databricks SDK chain (DATABRICKS_HOST +
DATABRICKS_TOKEN, profile, or M2M). MLflow registry is set to
``databricks-uc`` so prompts land alongside UC-registered models.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "ml_intern.agent.system_prompt"
_DEFAULT_FILE = (
    Path(__file__).resolve().parent.parent / "agent" / "prompts" / "system_prompt_v3.yaml"
)


def _load_template(path: Path) -> str:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    text = (data or {}).get("system_prompt")
    if not text:
        raise ValueError(f"{path} has no top-level 'system_prompt' key")
    return text


def _register(name: str, template: str, commit_message: str) -> str:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    register = None
    try:
        from mlflow.genai import register_prompt as register  # type: ignore
    except Exception:
        register = getattr(mlflow, "register_prompt", None)
    if register is None:
        raise RuntimeError(
            "MLflow build has no register_prompt() — install mlflow[databricks]>=2.21."
        )

    prompt = register(name=name, template=template, commit_message=commit_message)
    version = getattr(prompt, "version", None) or "?"
    return str(version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=_DEFAULT_NAME)
    parser.add_argument("--file", default=str(_DEFAULT_FILE))
    parser.add_argument("--commit-message", default="bundle deploy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    template = _load_template(Path(args.file))
    try:
        version = _register(args.name, template, args.commit_message)
    except Exception as e:
        logger.error("register_prompt failed: %s", e)
        return 1
    logger.info("Registered %s version=%s", args.name, version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
