"""Skill loader + ``read_skill`` tool tests."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skills import loader
from agent.tools.read_skill_tool import read_skill_handler


# ── helpers ────────────────────────────────────────────────────────────


def _make_skill(root: Path, name: str, body: str = "PLAYBOOK_BODY") -> None:
    """Materialise a fake skill on disk."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "skill.yaml").write_text(textwrap.dedent(f"""\
        name: {name}
        version: 1.2.3
        summary: |
            Test skill for {name}.
        triggers:
          - {name}
          - alpha
          - bravo
          - charlie
          - delta
        playbook: playbook.md
    """))
    (d / "playbook.md").write_text(body)


@pytest.fixture(autouse=True)
def _clear_lru(monkeypatch):
    """list_skills() is cached; clear it between tests."""
    loader.list_skills.cache_clear()
    yield
    loader.list_skills.cache_clear()


# ── discovery ──────────────────────────────────────────────────────────


def test_list_skills_returns_real_skill(tmp_path):
    """Default discovery picks up the shipped kaggle-tabular-classification."""
    # Use real root.
    loader.list_skills.cache_clear()
    skills = loader.list_skills()
    names = [s.name for s in skills]
    assert "kaggle-tabular-classification" in names


def test_list_skills_under_synthetic_root(tmp_path):
    _make_skill(tmp_path, "alpha-skill")
    _make_skill(tmp_path, "bravo-skill")
    skills = loader.list_skills(root=tmp_path)
    names = sorted(s.name for s in skills)
    assert names == ["alpha-skill", "bravo-skill"]
    assert all(s.version == "1.2.3" for s in skills)


def test_list_skills_skips_missing_playbook(tmp_path, caplog):
    d = tmp_path / "broken-skill"
    d.mkdir()
    (d / "skill.yaml").write_text("name: broken-skill\nplaybook: missing.md\n")
    skills = loader.list_skills(root=tmp_path)
    assert skills == ()


def test_list_skills_skips_malformed_yaml(tmp_path):
    d = tmp_path / "bad-yaml"
    d.mkdir()
    (d / "skill.yaml").write_text("::not valid yaml::\n: : :")
    (d / "playbook.md").write_text("body")
    skills = loader.list_skills(root=tmp_path)
    assert skills == ()


def test_list_skills_skips_hidden_dirs(tmp_path):
    _make_skill(tmp_path, ".hidden")
    _make_skill(tmp_path, "_underscored")
    _make_skill(tmp_path, "visible")
    names = [s.name for s in loader.list_skills(root=tmp_path)]
    assert names == ["visible"]


# ── read_skill ─────────────────────────────────────────────────────────


def test_read_skill_returns_playbook_content():
    """Pull the real kaggle skill's body and confirm a known marker is in it."""
    loader.list_skills.cache_clear()
    body = loader.read_skill("kaggle-tabular-classification")
    assert body is not None
    assert "Race-context" in body or "race-context" in body.lower()
    # Anti-pattern section must be present so the agent gets warned about
    # the wrong-target bug.
    assert "Wrong target column" in body or "wrong target" in body.lower()


def test_read_skill_returns_none_for_unknown_name():
    loader.list_skills.cache_clear()
    assert loader.read_skill("nonexistent-skill") is None


# ── catalog formatting ─────────────────────────────────────────────────


def test_catalog_for_prompt_includes_skill_name_and_triggers():
    loader.list_skills.cache_clear()
    catalog = loader.skill_catalog_for_prompt()
    assert "kaggle-tabular-classification" in catalog
    assert "kaggle" in catalog
    # Trigger overflow indicator must appear when > 4 triggers.
    assert "…" in catalog or "..." in catalog


def test_catalog_empty_when_no_skills(tmp_path):
    """Use a synthetic empty root via monkey-patching."""
    with patch.object(loader, "_SKILLS_ROOT", tmp_path):
        loader.list_skills.cache_clear()
        assert loader.skill_catalog_for_prompt() == ""


# ── tool handler ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_skill_handler_happy_path():
    result = await read_skill_handler({"name": "kaggle-tabular-classification"})
    assert result["isError"] is False
    assert "Race-context" in result["formatted"]


@pytest.mark.asyncio
async def test_read_skill_handler_missing_name():
    result = await read_skill_handler({})
    assert result["isError"] is True
    assert "name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_read_skill_handler_unknown_skill_lists_available():
    result = await read_skill_handler({"name": "does-not-exist"})
    assert result["isError"] is True
    # Error must list what IS available so the agent recovers.
    assert "kaggle-tabular-classification" in result["formatted"]


# ── tool registration ─────────────────────────────────────────────────


def test_read_skill_registered_as_builtin_tool():
    """The system prompt's "read_skill" reference must point at a real
    registered tool, otherwise the agent's tool call 404s."""
    from agent.core.tools import create_builtin_tools

    tools = create_builtin_tools()
    names = [t.name for t in tools]
    assert "read_skill" in names
