"""FastAPI auth dependencies.

Dual-mode:

- **Apps runtime** (detected via ``X-Forwarded-Access-Token`` + ``X-Forwarded-User``
  headers injected by the Databricks Apps proxy): the per-request user is the
  end user who hit the App, and their OAuth token is used for OBO calls via
  ``db_client.get_workspace_client_for_user``.

- **Local dev** (headers absent): fall back to the SDK unified auth chain —
  PAT env, `~/.databrickscfg` profile, or M2M service principal. The
  "current user" is whoever those creds resolve to via
  ``WorkspaceClient.current_user.me()``.

No HF OAuth, no cookies, no token validation round-trips — the Apps proxy
already validated the token before forwarding.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import HTTPException, Request, status

from agent.config import load_config
from agent.core import db_client

logger = logging.getLogger(__name__)

# Apps runtime signal. If set, we're running behind the Apps proxy.
_APP_MODE = bool(os.environ.get("DATABRICKS_APP_NAME") or os.environ.get("DATABRICKS_WORKSPACE_ID"))

_FORWARDED_TOKEN_HEADER = "X-Forwarded-Access-Token"
_FORWARDED_USER_HEADER = "X-Forwarded-User"
_FORWARDED_EMAIL_HEADER = "X-Forwarded-Email"


def _settings():
    # Resolved once per process; safe because Config is load-time-only.
    config = load_config(_config_path())
    return db_client.resolve_settings(config)


def _config_path() -> str:
    return os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "configs", "main_agent_config.json"),
    )


async def get_current_user(request: Request) -> dict[str, Any]:
    """Return a normalized user payload.

    Shape: ``{user_name, display_name, email, workspace_url, authenticated}``.
    In Apps mode pulls identity from forwarded headers; in dev mode calls
    ``WorkspaceClient.current_user.me()``.
    """
    token = request.headers.get(_FORWARDED_TOKEN_HEADER)
    user_name = request.headers.get(_FORWARDED_USER_HEADER)
    email = request.headers.get(_FORWARDED_EMAIL_HEADER)

    if token and user_name:
        # Apps-mode: trust the proxy's headers. Store token on request state
        # so route handlers can pick it up for OBO calls.
        request.state.user_token = token
        settings = _settings()
        return {
            "user_id": user_name,
            "user_name": user_name,
            "display_name": user_name,
            "email": email or user_name,
            "workspace_url": settings.host,
            "authenticated": True,
        }

    # Local/dev mode: SDK auth chain.
    try:
        settings = _settings()
        wc = db_client.get_workspace_client(settings)
        me = wc.current_user.me()
        return {
            "user_id": me.user_name,
            "user_name": me.user_name,
            "display_name": me.display_name or me.user_name,
            "email": me.emails[0].value if me.emails else me.user_name,
            "workspace_url": settings.host,
            "authenticated": True,
        }
    except Exception as e:
        logger.warning("Local auth resolution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Not authenticated. Set DATABRICKS_HOST + DATABRICKS_TOKEN, "
                "or configure a profile via `databricks auth login`."
            ),
        )


def extract_obo_token(request: Request) -> str | None:
    """Return the forwarded user token if running behind the Apps proxy."""
    return request.headers.get(_FORWARDED_TOKEN_HEADER)
