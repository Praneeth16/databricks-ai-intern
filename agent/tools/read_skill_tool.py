"""``read_skill`` builtin — pull a skill playbook on demand.

The system prompt lists every available skill at boot (one line each
via :func:`agent.skills.loader.skill_catalog_for_prompt`). When the
agent's task matches a trigger, it calls this tool to fetch the full
playbook markdown into context. Same on-demand pattern as
``fetch_hf_docs`` — keeps the system prompt small and only pulls the
heavy content when relevant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.skills import list_skills, read_skill

logger = logging.getLogger(__name__)


READ_SKILL_TOOL_SPEC: Dict[str, Any] = {
    "name": "read_skill",
    "description": (
        "Pull the full playbook content for a named skill. Skills are "
        "domain-specific recipes (e.g., Kaggle tabular classification, "
        "computer vision warm-start, NLP fine-tune). The system prompt "
        "lists available skill names + triggers at session start — when "
        "your task matches a trigger, call this tool with the skill's "
        "name to load its playbook before writing code.\n\n"
        "Call this once per relevant skill. The playbook is markdown and "
        "is appended verbatim to the tool result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Exact skill name as printed in the system prompt's "
                    "'Available skills' section (e.g., "
                    "'kaggle-tabular-classification')."
                ),
            },
        },
        "required": ["name"],
    },
}


async def read_skill_handler(args: dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name") or ""
    if not name:
        return {"formatted": "Error: name is required.", "isError": True}
    playbook = read_skill(name)
    if playbook is None:
        catalog = ", ".join(s.name for s in list_skills())
        return {
            "formatted": (
                f"Error: skill {name!r} not found. Available: "
                f"{catalog or '(none registered)'}."
            ),
            "isError": True,
        }
    return {"formatted": playbook, "isError": False}
