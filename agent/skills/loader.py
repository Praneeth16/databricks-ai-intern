"""Skill discovery + on-demand playbook loading.

Reads ``agent/skills/<name>/skill.yaml`` manifests and the paired
playbook markdown when the agent asks for it. Stays defensive — a
malformed skill file logs a warning and is skipped, never crashes the
boot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_SKILLS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Skill:
    """One skill manifest + its on-disk playbook path."""

    name: str
    version: str
    summary: str
    triggers: tuple[str, ...]
    playbook_path: Path
    applies_when: dict = field(default_factory=dict)


def _safe_load_yaml(path: Path) -> Optional[dict]:
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Skill manifest %s failed to load: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Skill manifest %s is not a YAML mapping; skipping.", path)
        return None
    return data


def _parse_skill(skill_dir: Path) -> Optional[Skill]:
    manifest_path = skill_dir / "skill.yaml"
    if not manifest_path.exists():
        return None
    data = _safe_load_yaml(manifest_path)
    if not data:
        return None
    name = data.get("name") or skill_dir.name
    playbook_rel = data.get("playbook") or "playbook.md"
    playbook_path = skill_dir / playbook_rel
    if not playbook_path.exists():
        logger.warning("Skill %s references missing playbook %s.", name, playbook_path)
        return None
    triggers_raw = data.get("triggers") or []
    if not isinstance(triggers_raw, list):
        triggers_raw = [str(triggers_raw)]
    return Skill(
        name=name,
        version=str(data.get("version") or "0.0.0"),
        summary=(data.get("summary") or "").strip(),
        triggers=tuple(str(t) for t in triggers_raw),
        playbook_path=playbook_path,
        applies_when=data.get("applies_when") or {},
    )


@lru_cache(maxsize=1)
def list_skills(root: Optional[Path] = None) -> tuple[Skill, ...]:
    """Discover all valid skills under ``agent/skills/``.

    Returns a tuple so the result is hashable + safe to cache. Cache is
    busted across processes; within a process, manifest edits won't be
    picked up until the next interpreter boot — that's the intended
    contract (skills are static config, not runtime state).
    """
    base = root or _SKILLS_ROOT
    skills: list[Skill] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        skill = _parse_skill(child)
        if skill is not None:
            skills.append(skill)
    return tuple(skills)


def read_skill(name: str) -> Optional[str]:
    """Return the full playbook content for ``name``, or None.

    The agent's ``read_skill`` builtin tool calls this. We keep the
    return type simple (string or None) so the tool handler doesn't
    need bespoke parsing.
    """
    for skill in list_skills():
        if skill.name == name:
            try:
                return skill.playbook_path.read_text()
            except Exception as e:
                logger.warning("read_skill(%s) failed: %s", name, e)
                return None
    return None


def skill_catalog_for_prompt() -> str:
    """One-line-per-skill catalog injected into the system prompt at boot.

    Keep it short so we don't burn context on skills the user may never
    invoke. The agent pulls the full playbook via ``read_skill(name)``
    when it sees a trigger match in the task description.
    """
    skills = list_skills()
    if not skills:
        return ""
    lines = ["# Available skills (pull full playbook via ``read_skill(<name>)``)"]
    for s in skills:
        triggers_preview = ", ".join(s.triggers[:4])
        if len(s.triggers) > 4:
            triggers_preview += ", …"
        summary_one_line = " ".join(s.summary.split())[:140]
        lines.append(
            f"- {s.name} (v{s.version}): {summary_one_line}\n"
            f"    triggers: {triggers_preview}"
        )
    return "\n".join(lines)
