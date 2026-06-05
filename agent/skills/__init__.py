"""Skill registry — discoverable playbooks the agent can pull at runtime.

Each subdirectory of ``agent/skills/<name>/`` is one skill. Contents:
  - ``skill.yaml`` (mandatory): name + summary + triggers + playbook ref.
  - ``playbook.md`` (mandatory): the operational content the agent reads
    when the skill is loaded.

Loader contract:
  * On session boot, the agent runtime calls :func:`list_skills` and
    injects a 1-line-per-skill catalog into the system prompt. The agent
    sees triggers + summary + skill name, NOT the full playbook (keeps
    the prompt small).
  * When the agent's task matches a skill's trigger, it calls the
    ``read_skill(name)`` builtin tool to pull the full playbook into
    the context. Same pattern as ``fetch_hf_docs`` — pull on demand.
"""

from agent.skills.loader import list_skills, read_skill, skill_catalog_for_prompt

__all__ = ["list_skills", "read_skill", "skill_catalog_for_prompt"]
