"""Databricks Foundation Model serving-endpoint catalog.

Replaces the old HF Inference Router catalog. Pulls the workspace's serving
endpoints (Foundation Model API + AI Gateway) live via the SDK and caches
them in-memory for a few minutes so ``/model`` switches don't repeatedly
round-trip the workspace.

Used to:

  • Validate ``/model`` switches against actual endpoints in this workspace.
  • Show the user which endpoints are READY, what entity they front, and the
    task type (chat / embedding / completions).
  • Suggest near-matches when a user pastes a typo'd endpoint name.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Optional

from agent.core import db_client

logger = logging.getLogger(__name__)


_CACHE_TTL_SECONDS = 300

_cache: list["EndpointInfo"] | None = None
_cache_time: float = 0.0


@dataclass
class EndpointInfo:
    name: str
    state: str  # READY | NOT_READY | UPDATING | etc.
    task: str  # llm/v1/chat | llm/v1/completions | llm/v1/embeddings | None
    served_entities: list[str] = field(default_factory=list)
    creator: str | None = None
    endpoint_type: str | None = None  # FOUNDATION_MODEL_API | EXTERNAL_MODEL | CUSTOM
    creation_time_ms: int | None = None

    @property
    def is_ready(self) -> bool:
        return self.state.upper() == "READY"

    @property
    def is_chat(self) -> bool:
        return (self.task or "").endswith("/chat")


def _to_info(ep: Any) -> EndpointInfo:
    state = ""
    s = getattr(ep, "state", None)
    if s is not None:
        ready = getattr(s, "ready", None)
        update = getattr(s, "config_update", None)
        # SDK exposes enum-like with .value or string directly.
        ready_val = getattr(ready, "value", ready) if ready else None
        update_val = getattr(update, "value", update) if update else None
        state = str(ready_val or update_val or "").upper()
    served: list[str] = []
    config = getattr(ep, "config", None)
    if config is not None:
        for se in (getattr(config, "served_entities", None) or []):
            name = getattr(se, "entity_name", None) or getattr(se, "name", None)
            if name:
                served.append(name)
    return EndpointInfo(
        name=getattr(ep, "name", "") or "",
        state=state or "UNKNOWN",
        task=getattr(ep, "task", None) or "",
        served_entities=served,
        creator=getattr(ep, "creator", None),
        endpoint_type=getattr(getattr(ep, "endpoint_type", None), "value", None)
            or getattr(ep, "endpoint_type", None),
        creation_time_ms=getattr(ep, "creation_timestamp", None),
    )


def _fetch(force: bool = False) -> list[EndpointInfo]:
    global _cache, _cache_time
    now = time.time()
    if not force and _cache is not None and now - _cache_time < _CACHE_TTL_SECONDS:
        return _cache
    try:
        wc = db_client.get_workspace_client()
        items = list(wc.serving_endpoints.list())
        _cache = [_to_info(ep) for ep in items]
        _cache_time = now
    except Exception as e:
        logger.warning("serving_endpoints.list failed: %s", e)
        if _cache is None:
            _cache = []
            _cache_time = now
    return _cache


def _strip_prefix(model_id: str) -> str:
    bare = model_id.split(":", 1)[0]
    return bare.removeprefix("databricks/")


def list_endpoints(force_refresh: bool = False) -> list[EndpointInfo]:
    return _fetch(force=force_refresh)


def lookup(model_id: str) -> Optional[EndpointInfo]:
    """Find a serving endpoint by ``databricks/<endpoint>`` model id.

    Accepts either the full ``databricks/<name>`` form or the bare endpoint
    name. Returns None when the endpoint isn't visible to the caller's auth
    chain (often because the App SP lacks CAN_QUERY).
    """
    name = _strip_prefix(model_id)
    for ep in _fetch():
        if ep.name == name:
            return ep
    return None


def fuzzy_suggest(model_id: str, limit: int = 3) -> list[str]:
    name = _strip_prefix(model_id)
    names = [ep.name for ep in _fetch() if ep.name]
    return get_close_matches(name, names, n=limit, cutoff=0.4)


def prewarm() -> None:
    """Populate the cache; safe to call from a background task."""
    try:
        _fetch(force=False)
    except Exception:
        pass


def reset_cache_for_tests() -> None:
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0
