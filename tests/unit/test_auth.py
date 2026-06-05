"""Unit tests for backend auth dependency.

Two modes:
    - Apps-mode: X-Forwarded-* headers present → identity pulled from headers,
      token stashed on request.state for OBO use.
    - Local-mode: no forwarded headers → WorkspaceClient.current_user.me()
      resolved via the SDK auth chain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The backend package imports bare-module style (CWD=backend/). Add backend/
# to sys.path so `import dependencies` resolves.
_BACKEND = str(Path(__file__).resolve().parents[2] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


@pytest.fixture(autouse=True)
def _reset_db():
    from agent.core import db_client
    db_client.reset_clients_for_tests()
    yield
    db_client.reset_clients_for_tests()


def _fake_request(headers: dict | None = None):
    req = MagicMock()
    req.headers = headers or {}
    req.state = MagicMock()
    return req


@pytest.mark.asyncio
async def test_apps_mode_trusts_forwarded_headers():
    import dependencies

    headers = {
        "X-Forwarded-Access-Token": "user-token-xyz",
        "X-Forwarded-User": "alice@example.com",
        "X-Forwarded-Email": "alice@example.com",
    }
    req = _fake_request(headers)

    with patch.object(dependencies, "_settings") as mock_settings:
        mock_settings.return_value = MagicMock(host="https://ws.cloud.databricks.com")
        user = await dependencies.get_current_user(req)

    assert user["user_name"] == "alice@example.com"
    assert user["email"] == "alice@example.com"
    assert user["authenticated"] is True
    assert user["workspace_url"] == "https://ws.cloud.databricks.com"
    assert req.state.user_token == "user-token-xyz"


@pytest.mark.asyncio
async def test_local_mode_resolves_via_workspace_client():
    import dependencies

    req = _fake_request()  # no forwarded headers
    fake_me = MagicMock(
        user_name="dev@example.com",
        display_name="Dev",
        emails=[MagicMock(value="dev@example.com")],
    )
    fake_wc = MagicMock()
    fake_wc.current_user.me.return_value = fake_me
    fake_settings = MagicMock(host="https://ws")

    with patch.object(dependencies, "_settings", return_value=fake_settings), \
         patch("agent.core.db_client.get_workspace_client", return_value=fake_wc):
        user = await dependencies.get_current_user(req)

    assert user["user_name"] == "dev@example.com"
    assert user["display_name"] == "Dev"
    assert user["email"] == "dev@example.com"
    assert user["authenticated"] is True


@pytest.mark.asyncio
async def test_local_mode_failure_raises_401():
    from fastapi import HTTPException
    import dependencies

    req = _fake_request()
    with patch.object(dependencies, "_settings", side_effect=RuntimeError("no host")):
        with pytest.raises(HTTPException) as exc:
            await dependencies.get_current_user(req)
    assert exc.value.status_code == 401


def test_extract_obo_token_returns_header_value():
    import dependencies

    req = _fake_request({"X-Forwarded-Access-Token": "tok-123"})
    assert dependencies.extract_obo_token(req) == "tok-123"


def test_extract_obo_token_none_when_absent():
    import dependencies

    assert dependencies.extract_obo_token(_fake_request()) is None
