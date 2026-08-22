"""Deferred tool loading — fetch full schemas for tools that aren't resident.

Ported from the `defer_loading` flag on codex's `ToolDefinition`
(`codex-rs/tools/src/tool_definition.rs`).

The problem this solves is measured, not theoretical: this repo's 25 builtin tools
serialize to ~52 KB of JSON schema, and every one of them was being sent on every
single LLM request. Most turns use a handful. So we keep a small resident set, list the
rest by name and one-line summary inside this tool's own description, and hand over full
schemas only when asked.

Deferral affects advertising only. `ToolRouter.call_tool` still resolves every
registered tool by name, so a model that names a deferred tool directly is not blocked —
it just gets promoted so the schema is visible on the next turn.
"""

from __future__ import annotations

import json
from typing import Any

TOOL_SEARCH_TOOL_SPEC: dict[str, Any] = {
    "name": "tool_search",
    # Description is rewritten at router init to append the live deferred-tool catalog.
    "description": (
        "Fetch the full parameter schemas for deferred tools so you can call them. "
        "Deferred tools are listed by name below; until you fetch one, you only know its "
        "name and summary, not its arguments. Query by keywords (\"kaggle dataset volume\") "
        "or select exact names with \"select:name_one,name_two\". Once fetched, a tool stays "
        "available for the rest of the session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords to match against deferred tool names and descriptions, or "
                    "'select:<name>[,<name>...]' to fetch specific tools by exact name."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum tools to return (default 5).",
            },
        },
        "required": ["query"],
    },
}

_MAX_RESULTS_CAP = 12


def _summary(description: str, limit: int = 140) -> str:
    """First sentence of a tool description, for the catalog listing."""
    text = " ".join((description or "").split())
    head = text.split(". ")[0]
    if len(head) > limit:
        head = head[: limit - 1].rstrip() + "…"
    return head


def build_catalog(deferred: dict[str, str]) -> str:
    """Render the deferred-tool catalog appended to this tool's description."""
    if not deferred:
        return ""
    lines = [f"  - {name}: {_summary(desc)}" for name, desc in sorted(deferred.items())]
    return (
        "\n\nDeferred tools available to fetch (name: what it does):\n"
        + "\n".join(lines)
    )


def _score(query_terms: list[str], name: str, description: str) -> int:
    name_l, desc_l = name.lower(), (description or "").lower()
    score = 0
    for t in query_terms:
        if t == name_l:
            score += 100
        elif t in name_l:
            score += 10
        if t in desc_l:
            score += 1
    return score


async def tool_search_handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
    query = (args.get("query") or "").strip()
    if not query:
        return "tool_search requires a non-empty 'query'.", False

    router = getattr(session, "tool_router", None)
    if router is None:
        return "tool_search is unavailable: no tool router on this session.", False

    deferred = router.deferred_tools()
    if not deferred:
        return "No deferred tools — every registered tool is already active.", True

    try:
        max_results = int(args.get("max_results") or 5)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, _MAX_RESULTS_CAP))

    if query.lower().startswith("select:"):
        wanted = [n.strip() for n in query[len("select:"):].split(",") if n.strip()]
        matched = [n for n in wanted if n in deferred]
        unknown = [n for n in wanted if n not in deferred]
        already = [n for n in unknown if n in router.tools]
        unknown = [n for n in unknown if n not in router.tools]
    else:
        terms = [t for t in query.lower().replace(",", " ").split() if t]
        ranked = sorted(
            ((_score(terms, n, d), n) for n, d in deferred.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        matched = [n for s, n in ranked if s > 0][:max_results]
        unknown, already = [], []

    if not matched:
        notes = []
        if already:
            notes.append(f"Already active (callable now, no fetch needed): {', '.join(already)}.")
        if unknown:
            notes.append(f"No such tool: {', '.join(unknown)}.")
        notes.append(
            "No deferred tool matched. Available deferred tools: " + ", ".join(sorted(deferred))
        )
        return " ".join(notes), False

    router.promote(matched)
    payload = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": router.tools[name].description,
                "parameters": router.tools[name].parameters,
            },
        }
        for name in matched
    ]
    out = [
        f"Loaded {len(matched)} tool schema(s): {', '.join(matched)}. "
        "These are now callable directly for the rest of the session.",
        "",
        "<functions>",
    ]
    out += [json.dumps(p) for p in payload]
    out.append("</functions>")
    if already:
        out.append(f"\nAlready active, no fetch needed: {', '.join(already)}.")
    if unknown:
        out.append(f"\nNo such tool: {', '.join(unknown)}.")
    return "\n".join(out), True
