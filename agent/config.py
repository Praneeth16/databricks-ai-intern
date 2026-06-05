import json
import os
import re
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv

# Project root: two levels up from this file (agent/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from fastmcp.mcp_config import (
    RemoteMCPServer,
    StdioMCPServer,
)
from pydantic import BaseModel, field_validator

# These two are the canonical server config types for MCP servers.
MCPServerConfig = Union[StdioMCPServer, RemoteMCPServer]


class DatabricksConfig(BaseModel):
    """Databricks workspace binding. All fields resolved from env at load-time
    (see configs/main_agent_config.json). The WorkspaceClient reads host/auth
    from the Databricks SDK unified auth chain; this block carries the
    workspace-local choices (catalog, warehouse, experiment path, etc.)."""

    host: str | None = None
    warehouse_id: str | None = None
    experiment_path: str = "/Shared/ml-intern"
    uc_catalog: str = "ml_intern"
    uc_schema: str = "agent"
    uc_volume: str = "scratch"
    secret_scope: str = "ml-intern"
    lakebase_instance: str | None = None
    instance_pool_id: str | None = None
    default_node_type_id: str = "g5.xlarge"
    default_runtime_version: str = "15.4.x-gpu-ml-scala2.12"
    prompt_registry_name: str = "ml_intern.agent.system_prompt"

    @field_validator("host", "warehouse_id", "lakebase_instance", "instance_pool_id", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        # ${VAR:-} substitution yields "" for unset env vars; coerce to None
        # so downstream code can treat "missing" uniformly.
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class Config(BaseModel):
    """Configuration manager"""

    model_name: str
    databricks: DatabricksConfig = DatabricksConfig()
    mcpServers: dict[str, MCPServerConfig] = {}
    save_sessions: bool = True
    auto_save_interval: int = 1  # Save every N user turns (0 = disabled)
    yolo_mode: bool = False  # Auto-approve all tool calls without confirmation
    max_iterations: int = 300  # Max LLM calls per agent turn (-1 = unlimited)

    # Permission control parameters
    confirm_cpu_jobs: bool = True
    auto_file_upload: bool = False

    # Reasoning effort *preference* — the ceiling the user wants. The probe
    # on `/model` walks a cascade down from here (``max`` → ``xhigh`` → ``high``
    # → …) and caches per-model what the provider actually accepted in
    # ``Session.model_effective_effort``. Default ``max`` because we'd rather
    # burn tokens thinking than ship a wrong ML recipe; the cascade lands on
    # whichever level the model supports (``high`` for GPT-5 / HF router,
    # ``xhigh`` or ``max`` for Anthropic 4.6 / 4.7). ``None`` = thinking off.
    # Valid values: None | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    reasoning_effort: str | None = "max"


def substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in any data structure.

    Supports ${VAR_NAME} syntax for required variables and ${VAR_NAME:-default} for optional.
    """
    if isinstance(obj, str):
        pattern = r"\$\{([^}:]+)(?::(-)?([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif has_default:
                return default_value or ""
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Add it to your .env file."
                )

        return re.sub(pattern, replacer, obj)

    elif isinstance(obj, dict):
        return {key: substitute_env_vars(value) for key, value in obj.items()}

    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]

    return obj


def load_config(config_path: str = "config.json") -> Config:
    """
    Load configuration with environment variable substitution.

    Use ${VAR_NAME} in your JSON for any secret.
    Automatically loads from .env file.
    """
    # Load .env from project root first (so it works from any directory),
    # then CWD .env can override if present
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(override=False)

    with open(config_path, "r") as f:
        raw_config = json.load(f)

    config_with_env = substitute_env_vars(raw_config)
    return Config.model_validate(config_with_env)
