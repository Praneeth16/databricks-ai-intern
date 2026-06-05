"""Unit tests for agent.core.prompt_registry.

Behavior under test:
  - Registry hit → returns the registered template.
  - Registry miss / failure → falls back to YAML.
  - YAML loader returns ``system_prompt`` value.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

from agent.core import prompt_registry


def test_load_from_yaml(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: |\n  Hello {{ tools }}\n")
    out = prompt_registry.load_from_yaml(p)
    assert "Hello {{ tools }}" in out


def test_load_system_prompt_prefers_registry(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: 'YAML-FALLBACK'\n")
    fake = MagicMock()
    fake.template = "REGISTRY-WIN"
    with patch.object(prompt_registry, "load_from_registry", return_value="REGISTRY-WIN"):
        out = prompt_registry.load_system_prompt("ml_intern.agent.system_prompt", yaml_path=p)
    assert out == "REGISTRY-WIN"


def test_load_system_prompt_falls_back_to_yaml(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: 'YAML-FALLBACK'\n")
    with patch.object(prompt_registry, "load_from_registry", return_value=None):
        out = prompt_registry.load_system_prompt("ml_intern.agent.system_prompt", yaml_path=p)
    assert out == "YAML-FALLBACK"


def test_load_from_registry_returns_none_on_mlflow_import_failure():
    with patch("builtins.__import__", side_effect=ImportError("no mlflow")):
        out = prompt_registry.load_from_registry("anything")
    assert out is None


def test_load_from_registry_returns_template_attribute_when_present():
    fake_prompt = MagicMock()
    fake_prompt.template = "tpl-text"
    fake_loader = MagicMock(return_value=fake_prompt)
    fake_genai = MagicMock(load_prompt=fake_loader)
    with patch("mlflow.set_registry_uri"), \
         patch.dict(
             "sys.modules",
             {"mlflow.genai": fake_genai},
         ):
        out = prompt_registry.load_from_registry("name", version=3)
    assert out == "tpl-text"
    fake_loader.assert_called_once_with("prompts:/name/3")


def test_load_from_registry_latest_walks_alias_then_search():
    """``version=None`` must avoid the literal ``prompts:/<name>/latest`` URI
    (some MLflow builds raise ``ValueError: invalid literal for int()``) and
    instead try ``@production`` / ``@champion`` aliases before falling back
    to the highest numeric version found via the MLflow client."""
    seen_uris: list[str] = []

    fake_prompt = MagicMock()
    fake_prompt.template = "from-version-7"

    def loader(uri: str):
        seen_uris.append(uri)
        if uri.endswith("@production") or uri.endswith("@champion"):
            raise ValueError("alias not registered")
        if uri == "prompts:/name/7":
            return fake_prompt
        raise ValueError("unexpected uri")

    v1 = MagicMock(version="1")
    v7 = MagicMock(version="7")
    v3 = MagicMock(version="3")

    fake_client = MagicMock()
    fake_client.search_prompt_versions = MagicMock(return_value=[v1, v7, v3])

    fake_genai = MagicMock(load_prompt=loader)
    fake_mlflow = MagicMock()
    fake_mlflow.MlflowClient = MagicMock(return_value=fake_client)
    fake_mlflow.set_registry_uri = MagicMock()

    with patch.dict(
        "sys.modules",
        {"mlflow": fake_mlflow, "mlflow.genai": fake_genai},
    ):
        out = prompt_registry.load_from_registry("name")
    assert out == "from-version-7"
    # Aliases attempted first, never the literal /latest.
    assert "prompts:/name@production" in seen_uris
    assert "prompts:/name@champion" in seen_uris
    assert "prompts:/name/latest" not in seen_uris
    # Final URI used the highest numeric version.
    assert seen_uris[-1] == "prompts:/name/7"


def test_load_from_registry_latest_returns_none_when_nothing_registered():
    """When the prompt isn't registered we want a clean None (caller falls
    back to YAML) and crucially no ``int('latest')`` ValueError noise."""
    fake_genai = MagicMock(
        load_prompt=MagicMock(side_effect=ValueError("not registered")),
    )
    fake_client = MagicMock()
    fake_client.search_prompt_versions = MagicMock(return_value=[])
    fake_mlflow = MagicMock()
    fake_mlflow.MlflowClient = MagicMock(return_value=fake_client)
    fake_mlflow.set_registry_uri = MagicMock()

    with patch.dict(
        "sys.modules",
        {"mlflow": fake_mlflow, "mlflow.genai": fake_genai},
    ):
        out = prompt_registry.load_from_registry("missing")
    assert out is None
