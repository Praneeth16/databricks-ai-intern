"""Deferred tool loading, schema sanitization, and the global output cap."""

import json

import pytest

from agent.core.tools import RESIDENT_TOOLS, ToolRouter, ToolSpec, sanitize_schema
from agent.tools.tool_search import build_catalog
from agent.tools.truncation import MODEL_FACING_CHAR_CAP, truncate_output


@pytest.fixture
def router():
    return ToolRouter(mcp_servers={}, local_mode=True)


class _FakeSession:
    def __init__(self, router):
        self.tool_router = router


def _payload_bytes(specs):
    return sum(len(json.dumps(s)) for s in specs)


def _all_specs(router):
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in router.tools.values()
    ]


# --------------------------------------------------------------- deferral


def test_only_resident_tools_are_advertised(router):
    advertised = {s["function"]["name"] for s in router.get_tool_specs_for_llm()}
    assert advertised <= RESIDENT_TOOLS
    assert "bash" in advertised and "tool_search" in advertised
    # Specialised subsystems stay behind tool_search.
    assert "model_serving" not in advertised
    assert "hf_papers" not in advertised


def test_deferral_cuts_the_schema_payload_substantially(router):
    """The point of the feature is token cost, so assert the actual saving."""
    before = _payload_bytes(_all_specs(router))
    after = _payload_bytes(router.get_tool_specs_for_llm())
    assert after < before
    # Measured ~68% at time of writing; assert a floor so a regression is caught
    # without making the test brittle to adding a resident tool.
    assert (before - after) / before > 0.5, f"only saved {(before - after) / before:.1%}"


def test_deferred_tools_are_still_registered_and_callable(router):
    """Deferral withholds the schema, never the capability."""
    deferred = router.deferred_tools()
    assert "model_serving" in deferred
    assert router.tools["model_serving"].handler is not None


@pytest.mark.asyncio
async def test_tool_search_promotes_by_exact_name(router):
    out, ok = await router.tools["tool_search"].handler(
        {"query": "select:uc_inspect_dataset,sweep"}, session=_FakeSession(router)
    )
    assert ok
    assert "uc_inspect_dataset" in router.active_tools
    assert "sweep" in router.active_tools
    advertised = {s["function"]["name"] for s in router.get_tool_specs_for_llm()}
    assert {"uc_inspect_dataset", "sweep"} <= advertised
    # The returned payload must be a real schema the model can call from.
    assert '"parameters"' in out


@pytest.mark.asyncio
async def test_tool_search_ranks_keyword_matches(router):
    out, ok = await router.tools["tool_search"].handler(
        {"query": "papers", "max_results": 2}, session=_FakeSession(router)
    )
    assert ok
    assert "hf_papers" in router.active_tools
    assert "hf_papers" in out


@pytest.mark.asyncio
async def test_tool_search_reports_unknown_names(router):
    out, ok = await router.tools["tool_search"].handler(
        {"query": "select:no_such_tool"}, session=_FakeSession(router)
    )
    assert not ok
    assert "no_such_tool" in out


@pytest.mark.asyncio
async def test_tool_search_says_when_a_tool_is_already_active(router):
    out, ok = await router.tools["tool_search"].handler(
        {"query": "select:bash"}, session=_FakeSession(router)
    )
    assert not ok
    assert "already active" in out.lower()


@pytest.mark.asyncio
async def test_calling_a_deferred_tool_directly_promotes_it(router):
    """A model may name a deferred tool from the catalog without fetching it first."""
    assert "uc_model" not in router.active_tools
    await router.call_tool("uc_model", {"operation": "list"})
    assert "uc_model" in router.active_tools


# --------------------------------------------------------------- result contract


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_skill", {"name": "kaggle-tabular-classification"}),
        ("critic", {"operation": "list_detectors"}),
        ("experiment", {"operation": "list"}),
        ("sweep", {"operation": "status"}),
        ("research_loop", {"operation": "status"}),
        ("model_serving", {"operation": "plan_deployment"}),
    ],
)
@pytest.mark.asyncio
async def test_dict_returning_handlers_still_yield_real_output(router, name, args):
    """Regression: six handlers return a ToolResult dict, not (str, bool).

    Python unpacks a 2-key dict into its keys, so `out, ok = handler(...)` yielded
    ("formatted", "isError") — the model got the literal string "formatted" as the
    tool's entire output while the logs showed success. That silently disabled the
    skills system, critic, experiment ledger, sweeps, and the research loop.
    """
    out, ok = await router.call_tool(name, args)
    assert isinstance(ok, bool), f"{name} success flag must be a bool, got {ok!r}"
    assert out not in ("formatted", "isError"), f"{name} leaked a ToolResult key as output"
    assert len(out) > 20, f"{name} returned suspiciously little: {out!r}"


@pytest.mark.asyncio
async def test_read_skill_returns_the_whole_playbook(router):
    out, ok = await router.call_tool("read_skill", {"name": "kaggle-tabular-classification"})
    assert ok
    assert out.startswith("# Skill: kaggle-tabular-classification")
    assert len(out) > 5_000, "a playbook this short is not the real thing"


def test_normalize_tool_result_shapes():
    from agent.core.tools import _normalize_tool_result as norm

    assert norm("t", ("hello", True)) == ("hello", True)
    assert norm("t", {"formatted": "body", "isError": False}) == ("body", True)
    assert norm("t", {"formatted": "bad", "isError": True}) == ("bad", False)
    # A dict with no `formatted` key must not silently become an empty result.
    out, ok = norm("t", {"other": 1, "isError": False})
    assert "other" in out and ok is True
    # Unexpected types degrade to a string rather than crashing the turn.
    out, ok = norm("t", 42)
    assert out == "42" and ok is True


def test_subagent_tool_lists_are_not_filtered_by_deferral(router):
    """Regression: deferral must not silently shrink a sub-agent's curated tool list.

    `research_tool` used to build its read-only set by filtering
    `get_tool_specs_for_llm()`. Once most tools were deferred that returned just
    bash + read, gutting the research sub-agent. It resolves by name instead now.
    """
    from agent.tools.research_tool import RESEARCH_TOOL_NAMES

    specs = router.get_tool_specs_by_name(RESEARCH_TOOL_NAMES)
    names = {s["function"]["name"] for s in specs}
    # Every named tool that is actually registered must come back, deferred or not.
    expected = {n for n in RESEARCH_TOOL_NAMES if n in router.tools}
    assert names == expected
    assert len(names) > 2, "research sub-agent should get far more than bash + read"
    # And these really are deferred, so the test is exercising the interesting path.
    assert "hf_papers" not in router.active_tools


def test_get_tool_specs_by_name_ignores_unknown_names(router):
    specs = router.get_tool_specs_by_name({"bash", "definitely_not_a_tool"})
    assert {s["function"]["name"] for s in specs} == {"bash"}


def test_get_tool_specs_by_name_does_not_promote(router):
    """Handing a sub-agent a tool must not change what the main loop advertises."""
    before = set(router.active_tools)
    router.get_tool_specs_by_name({"hf_papers"})
    assert set(router.active_tools) == before


def test_catalog_lists_every_deferred_tool(router):
    catalog = build_catalog(router.deferred_tools())
    for name in router.deferred_tools():
        assert name in catalog, f"{name} missing from catalog — model could not discover it"


# --------------------------------------------------------------- sanitization


def test_sanitize_schema_drops_unreferenced_defs():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Used"}},
        "$defs": {
            "Used": {"type": "string"},
            "Unused": {"type": "object", "properties": {"x": {"type": "number"}}},
        },
    }
    out = sanitize_schema(schema)
    assert "Used" in out["$defs"]
    assert "Unused" not in out["$defs"]


def test_sanitize_schema_keeps_transitively_referenced_defs():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Outer"}},
        "$defs": {
            "Outer": {"type": "object", "properties": {"i": {"$ref": "#/$defs/Inner"}}},
            "Inner": {"type": "string"},
            "Orphan": {"type": "boolean"},
        },
    }
    out = sanitize_schema(schema)
    assert set(out["$defs"]) == {"Outer", "Inner"}


def test_sanitize_schema_removes_the_bucket_when_nothing_survives():
    out = sanitize_schema({"type": "object", "$defs": {"Unused": {"type": "string"}}})
    assert "$defs" not in out


def test_sanitize_schema_leaves_plain_schemas_alone():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert sanitize_schema(schema) == schema


# --------------------------------------------------------------- truncation


def test_truncate_output_passes_short_output_through():
    assert truncate_output("hello") == "hello"


def test_truncate_output_caps_and_keeps_head_and_tail():
    body = "H" * 50_000 + "TAILMARKER"
    out = truncate_output(body, max_chars=1000)
    assert len(out) < 2000
    assert out.startswith("H")
    assert "TAILMARKER" in out, "tail must survive — errors appear at the end of output"
    assert "omitted" in out


def test_truncate_output_spills_full_body_to_a_readable_file(tmp_path):
    out = truncate_output("Z" * 40_000, max_chars=500)
    assert "Full output saved to" in out
    path = out.split("Full output saved to ")[1].split(" ")[0]
    with open(path) as f:
        assert len(f.read()) == 40_000


@pytest.mark.asyncio
async def test_router_caps_oversized_tool_output(router):
    async def flood(_args):
        return "x" * (MODEL_FACING_CHAR_CAP * 3), True

    router.register_tool(
        ToolSpec(name="flood", description="d", parameters={"type": "object"}, handler=flood)
    )
    out, ok = await router.call_tool("flood", {})
    assert ok
    assert len(out) < MODEL_FACING_CHAR_CAP * 1.1
    assert "Full output saved to" in out


@pytest.mark.asyncio
async def test_router_does_not_touch_output_under_the_cap(router):
    async def small(_args):
        return "fine", True

    router.register_tool(
        ToolSpec(name="small", description="d", parameters={"type": "object"}, handler=small)
    )
    out, ok = await router.call_tool("small", {})
    assert (out, ok) == ("fine", True)
