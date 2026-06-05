"""Minimal auth routes for Databricks-backed deploys.

The Apps runtime authenticates users at the proxy layer and forwards
identity headers. There is no OAuth dance to run here — this module only
exposes read-only identity endpoints for the frontend.

For local dev (no Apps proxy), identity is resolved from the SDK auth chain
in ``dependencies.get_current_user``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def auth_status() -> dict:
    """Report whether the App is running behind the Databricks Apps proxy."""
    from dependencies import _APP_MODE

    return {"auth_enabled": True, "mode": "apps" if _APP_MODE else "local"}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)) -> dict:
    """Return the authenticated user's normalized identity payload."""
    return user
