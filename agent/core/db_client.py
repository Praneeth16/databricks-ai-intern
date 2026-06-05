"""Databricks workspace gateway.

Single source of truth for talking to Databricks: WorkspaceClient factory,
SQL warehouse connections, MLflow client, Lakebase connection pool.

Auth precedence follows the Databricks SDK unified auth chain:
    1. Explicit DATABRICKS_HOST + DATABRICKS_TOKEN (PAT)
    2. OAuth U2M profile (~/.databrickscfg)
    3. M2M service principal (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
    4. Apps-injected identity (X-Forwarded-Access-Token, handled per-request)

OBO (on-behalf-of user) is a per-request concern — see
`get_workspace_client_for_user()` which threads the forwarded user token
through a fresh client so user-scoped actions carry user identity in the
audit log rather than the App's service principal.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config as SdkConfig

if TYPE_CHECKING:
    from agent.config import Config as AgentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabricksSettings:
    """Resolved workspace binding merged from agent Config + env vars."""

    host: str
    warehouse_id: str | None
    experiment_path: str
    uc_catalog: str
    uc_schema: str
    uc_volume: str
    secret_scope: str
    lakebase_instance: str | None
    instance_pool_id: str | None
    default_node_type_id: str
    default_runtime_version: str
    prompt_registry_name: str

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.uc_catalog}/{self.uc_schema}/{self.uc_volume}"

    @property
    def full_schema(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}"


def resolve_settings(agent_config: AgentConfig) -> DatabricksSettings:
    """Merge the config-file block with DATABRICKS_* env vars.

    Env wins when both are set — keeps the dev loop simple (override via
    shell) and matches SDK auth-chain behavior.
    """
    db = agent_config.databricks
    host = os.environ.get("DATABRICKS_HOST") or db.host
    if not host:
        # Defer the hard failure; some local workflows (e.g. unit tests with a
        # mocked WorkspaceClient) never actually need a host resolved.
        logger.debug("No DATABRICKS_HOST set — WorkspaceClient calls will fail.")
        host = ""

    return DatabricksSettings(
        host=host.rstrip("/"),
        warehouse_id=os.environ.get("DATABRICKS_WAREHOUSE_ID") or db.warehouse_id,
        experiment_path=db.experiment_path,
        uc_catalog=db.uc_catalog,
        uc_schema=db.uc_schema,
        uc_volume=db.uc_volume,
        secret_scope=db.secret_scope,
        lakebase_instance=db.lakebase_instance,
        instance_pool_id=os.environ.get("ML_INTERN_INSTANCE_POOL_ID") or db.instance_pool_id,
        default_node_type_id=db.default_node_type_id,
        default_runtime_version=db.default_runtime_version,
        prompt_registry_name=db.prompt_registry_name,
    )


_wc_lock = threading.Lock()
_wc_cache: WorkspaceClient | None = None


def get_workspace_client(settings: DatabricksSettings | None = None) -> WorkspaceClient:
    """Return the process-wide cached WorkspaceClient.

    Uses the SDK unified auth chain. `settings` is accepted for future
    multi-workspace support but currently only its host is consulted — env
    overrides take priority via the SDK itself.
    """
    global _wc_cache
    with _wc_lock:
        if _wc_cache is None:
            host = (settings.host if settings else None) or os.environ.get("DATABRICKS_HOST")
            _wc_cache = WorkspaceClient(host=host) if host else WorkspaceClient()
        return _wc_cache


def get_workspace_client_for_user(user_token: str, host: str) -> WorkspaceClient:
    """Build a per-request WorkspaceClient using an OBO user token.

    Apps runtime forwards the end-user's OAuth token via
    ``X-Forwarded-Access-Token``; passing it into the SDK makes subsequent API
    calls (Jobs submit, UC read, MLflow log) execute as the user, not as the
    App's service principal. That's the correct audit trail for user actions.
    Do not cache — each request carries its own token.
    """
    return WorkspaceClient(config=SdkConfig(host=host, token=user_token))


def get_sql_connection(settings: DatabricksSettings, user_token: str | None = None):
    """Open a databricks.sql connection against the configured warehouse.

    Caller is responsible for closing. Pass `user_token` for OBO; otherwise
    authenticates as the App SP (or local PAT) via the SDK auth chain.
    """
    from databricks import sql  # lazy — heavy import

    if not settings.warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID not set. Required for UC SQL reads."
        )
    if not settings.host:
        raise RuntimeError("DATABRICKS_HOST not resolvable.")

    http_path = f"/sql/1.0/warehouses/{settings.warehouse_id}"
    if user_token:
        return sql.connect(
            server_hostname=_hostname(settings.host),
            http_path=http_path,
            access_token=user_token,
        )
    # SDK auth chain: PAT env, profile, or M2M.
    cfg = get_workspace_client(settings).config
    return sql.connect(
        server_hostname=_hostname(settings.host),
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


def _hostname(host: str) -> str:
    # databricks.sql wants the bare hostname, not the https:// prefix.
    return host.replace("https://", "").replace("http://", "").rstrip("/")


@lru_cache(maxsize=1)
def get_mlflow_client(_marker: int = 0):
    """Return an MlflowClient bound to the workspace tracking server and the
    Unity Catalog model registry.

    The `_marker` is there so tests can bust the cache via
    `get_mlflow_client.cache_clear()`.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient()


# ---------------------------------------------------------------------------
# Lakebase (Postgres) — short-lived OAuth credentials
# ---------------------------------------------------------------------------
# Lakebase tokens expire after ~1h. The connection pool itself lives in the
# backend (backend/session_manager.py) so it can be tied to the FastAPI
# lifespan. Here we only expose a helper that materializes a fresh conninfo
# string; the pool's `max_lifetime` is set to 2700s (45 min) to recycle
# connections before tokens expire.


def build_lakebase_conninfo(settings: DatabricksSettings) -> str:
    """Resolve a fresh Lakebase conninfo string using the workspace SDK.

    Used both at pool construction time and by any caller that needs a
    one-off connection. Re-invoking this produces a fresh OAuth token.
    """
    if not settings.lakebase_instance:
        raise RuntimeError(
            "lakebase_instance not configured. Set ML_INTERN_LAKEBASE_INSTANCE."
        )
    wc = get_workspace_client(settings)
    instance = wc.database.get_database_instance(name=settings.lakebase_instance)
    cred = wc.database.generate_database_credential(
        instance_names=[settings.lakebase_instance],
        request_id=os.urandom(8).hex(),
    )
    me = wc.current_user.me()
    return (
        f"host={instance.read_write_dns} port=5432 dbname=databricks_postgres "
        f"user={me.user_name} password={cred.token} sslmode=require"
    )


def reset_clients_for_tests() -> None:
    """Bust every module-level cache. Call from pytest fixtures."""
    global _wc_cache
    with _wc_lock:
        _wc_cache = None
    get_mlflow_client.cache_clear()
